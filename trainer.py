# trainer.py
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os

from lghrec_project.config import (DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, K_EVAL,
                    NUM_NEGATIVE_SAMPLES_HGPO, TEMPERATURE_INITIAL) 
from models.lghrec_model import LGHRec
from utils.loss import bpr_loss, contrastive_loss_for_nodes 
from utils.metrics import evaluate_model
from utils.logger import setup_logger
from rl.hgpo_agent import HGPOAgent
from rl.group_manager import GroupManager
from rl.hgpo_utils import construct_rl_state 
from rl.rewards import calculate_total_reward
from data.graph_utils import get_graph_node_degrees


class Trainer:
    def __init__(self, lghrec_model: LGHRec, hgpo_agent: HGPOAgent, 
                 train_loader: DataLoader, test_loader: DataLoader,
                 num_users: int, num_items: int, all_item_ids_for_eval: list,
                 user_degrees_tensor: torch.Tensor, 
                 item_degrees_tensor: torch.Tensor, 
                 user_history_for_eval: dict,
                 logger, device=DEVICE):
        
        self.lghrec_model = lghrec_model
        self.hgpo_agent = hgpo_agent
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_users = num_users
        self.num_items = num_items
        self.all_item_ids_for_eval = all_item_ids_for_eval
        self.user_history_for_eval = user_history_for_eval
        self.logger = logger
        self.device = device

        trainable_params = list(lghrec_model.sgl_gnn.parameters()) + list(lghrec_model.dseg_module.parameters()) + list(lghrec_model.user_gnn_embeddings.parameters()) + list(lghrec_model.item_id_for_dseg_embeddings.parameters())
        
        self.optimizer_main = optim.Adam(filter(lambda p: p.requires_grad, trainable_params),
                                         lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.user_group_manager = GroupManager(num_groups=hgpo_agent.num_groups, device=self.device) 
        self.user_group_manager.build_group_boundaries(user_degrees_tensor.cpu()) 
        
        self.user_degrees_tensor = user_degrees_tensor.to(device)
        self.item_degrees_tensor = item_degrees_tensor.to(device)
        self.global_candidate_pool_size = self.hgpo_agent.num_candidate_neg_samples

    def _sample_candidate_negative_pool(self, batch_size, positive_item_indices):

        candidate_pool = torch.zeros((batch_size, self.global_candidate_pool_size), dtype=torch.long, device=self.device)
        for i in range(batch_size):
            pos_item = positive_item_indices[i].item()
            neg_count = 0
            seen_negs = {pos_item}
            temp_candidates = []
            while len(temp_candidates) < self.global_candidate_pool_size:
                neg_item_candidate = random.randint(0, self.num_items - 1)
                if neg_item_candidate != pos_item and neg_item_candidate not in temp_candidates:
                    temp_candidates.append(neg_item_candidate)
            candidate_pool[i] = torch.tensor(temp_candidates, dtype=torch.long, device=self.device)
        return candidate_pool


    def train_epoch(self, epoch_num):

        self.lghrec_model.train()
        self.hgpo_agent.policy_net.train() 

        total_bpr_loss = 0.0
        total_contrastive_loss = 0.0
        total_rl_policy_loss = 0.0
        total_rl_entropy_loss = 0.0
        total_rl_harm_loss = 0.0
        
        num_batches = len(self.train_loader)
        rl_update_count = 0

        for batch_idx, (users, pos_items) in enumerate(self.train_loader):
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)

            # 1. Main recommendation task optimization (BPR Loss)
            neg_items_bpr = torch.randint(0, self.num_items, (len(users),), device=self.device)

            self.optimizer_main.zero_grad()
            bpr_pos_scores, bpr_neg_scores, _ = self.lghrec_model(users, pos_items, neg_items_bpr, for_contrastive=False)
            loss_bpr = bpr_loss(bpr_pos_scores, bpr_neg_scores)
            
            # 2. Contrastive learning task (InfoNCE Loss)
            anchor_user_embeds_initial, current_item_embeds_initial = \
                self.lghrec_model.get_all_learned_embeddings(after_gnn_propagation=False)
            
            batch_anchor_user_embeds_initial = anchor_user_embeds_initial[users]
            batch_positive_item_embeds_initial = current_item_embeds_initial[pos_items]
            
            candidate_neg_pool_indices = self._sample_candidate_negative_pool(len(users), pos_items)
            candidate_neg_pool_embeds_initial = current_item_embeds_initial[candidate_neg_pool_indices]

            batch_user_degrees = self.user_degrees_tensor[users]
            
            rl_states = construct_rl_state(
                batch_anchor_user_embeds_initial,
                batch_positive_item_embeds_initial,
                candidate_neg_pool_embeds_initial, 
                batch_user_degrees 
            )

            self.hgpo_agent.policy_net.eval() 
            with torch.no_grad():
                selected_neg_indices_in_pool, neg_log_probs_old, \
                chosen_temp_tau, temp_log_probs_old, _ = \
                    self.hgpo_agent.policy_net.select_action(rl_states, NUM_NEGATIVE_SAMPLES_HGPO)
            self.hgpo_agent.policy_net.train()

            actual_selected_neg_item_ids = torch.gather(candidate_neg_pool_indices, 1, selected_neg_indices_in_pool)

            (users_v1, items_v1_pos), (users_v2, items_v2_pos) = \
                self.lghrec_model(users, pos_items, for_contrastive=True)
            
            # --- Use HGPO selected temperature ---
            loss_contrastive = contrastive_loss_for_nodes(
                users_v1, users_v2, 
                items_v1_pos, items_v2_pos, 
                temperature=chosen_temp_tau.squeeze() 
            )
            # ------------------------------------------

            # Get the propagated embeddings of the actually selected M negative samples (for reward calculation)
            _, all_items_v1_propagated = self.lghrec_model.sgl_gnn(user_indices=None, item_indices=torch.arange(self.num_items, device=self.device), for_contrastive=False) 
            
            actual_selected_neg_embeds_v1 = all_items_v1_propagated[actual_selected_neg_item_ids.view(-1)]
            actual_selected_neg_embeds_v1 = actual_selected_neg_embeds_v1.view(len(users), NUM_NEGATIVE_SAMPLES_HGPO, -1)


            rewards_rl = calculate_total_reward(
                anchor_embed = users_v1, 
                positive_embed = items_v1_pos, 
                selected_negative_embeds = actual_selected_neg_embeds_v1,
                chosen_temp = chosen_temp_tau, 
                node_degree = batch_user_degrees, 
                device = self.device
            )

            batch_user_group_indices = self.user_group_manager.get_group_idx(batch_user_degrees) 
            
            sum_neg_log_probs_old = neg_log_probs_old.sum(dim=1, keepdim=True)

            for i in range(len(users)):
                self.hgpo_agent.buffer.push(
                    rl_states[i].unsqueeze(0), 
                    selected_neg_indices_in_pool[i].unsqueeze(0), 
                    chosen_temp_tau[i].unsqueeze(0), 
                    sum_neg_log_probs_old[i].unsqueeze(0), 
                    temp_log_probs_old[i].unsqueeze(0), 
                    rewards_rl[i].unsqueeze(0), 
                    batch_user_degrees[i].unsqueeze(0), 
                    batch_user_group_indices[i].unsqueeze(0) 
                )
            
            # Total loss (BPR + Contrastive)

            lambda_cl =  0.1 # Weight for contrastive loss
            total_main_loss = loss_bpr + lambda_cl * loss_contrastive
            total_main_loss.backward()
            self.optimizer_main.step()

            total_bpr_loss += loss_bpr.item()
            total_contrastive_loss += loss_contrastive.item()

            if len(self.hgpo_agent.buffer) >= BATCH_SIZE : 
                p_loss, e_loss, h_loss = self.hgpo_agent.update_policy()
                total_rl_policy_loss += p_loss
                total_rl_entropy_loss += e_loss
                total_rl_harm_loss += h_loss
                rl_update_count +=1
            
            if (batch_idx + 1) % (num_batches // 10 + 1) == 0: 
                 self.logger.info(f"Epoch {epoch_num+1}/{EPOCHS}, Batch {batch_idx+1}/{num_batches} | "
                               f"BPR Loss: {loss_bpr.item():.4f}, CL Loss: {loss_contrastive.item():.4f} (temp avg: {chosen_temp_tau.mean().item():.3f}) | "
                               f"Avg Reward (this batch): {rewards_rl.mean().item():.4f}")

        avg_bpr_loss = total_bpr_loss / num_batches
        avg_cl_loss = total_contrastive_loss / num_batches
        avg_rl_p_loss = total_rl_policy_loss / rl_update_count if rl_update_count > 0 else 0
        avg_rl_e_loss = total_rl_entropy_loss / rl_update_count if rl_update_count > 0 else 0
        avg_rl_h_loss = total_rl_harm_loss / rl_update_count if rl_update_count > 0 else 0

        self.logger.info(f"Epoch {epoch_num+1} Summary: Avg BPR Loss: {avg_bpr_loss:.4f}, Avg CL Loss: {avg_cl_loss:.4f}")
        if rl_update_count > 0:
            self.logger.info(f"Epoch {epoch_num+1} RL Summary: Avg Policy Loss: {avg_rl_p_loss:.4f}, "
                           f"Avg Entropy Loss: {avg_rl_e_loss:.4f}, Avg Harm Loss: {avg_rl_h_loss:.4f}")
        

    def evaluate(self, epoch_num):
        """
        Evaluate the model on the test set and log the results.
        """
        self.lghrec_model.eval() 
        recall_k, ndcg_k = evaluate_model(
            model=self.lghrec_model, 
            test_loader=self.test_loader,
            all_item_ids=self.all_item_ids_for_eval, 
            k=K_EVAL,
            device=self.device,
            user_history_for_exclusion=self.user_history_for_eval
        )
        
        self.logger.info(f"Epoch {epoch_num+1} Evaluation: Recall@{K_EVAL}: {recall_k:.4f}, NDCG@{K_EVAL}: {ndcg_k:.4f}")
        return recall_k, ndcg_k

    def run(self):
        best_ndcg = 0.0
        best_epoch = 0

        for epoch in range(EPOCHS):
            start_time = time.time()
            self.train_epoch(epoch)
            
            if (epoch + 1) % 1 == 0: 
                recall, ndcg = self.evaluate(epoch)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_epoch = epoch + 1
                    save_dir = './saved_models'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model_save_path = os.path.join(save_dir, f"lghrec_model_best_ep{best_epoch}.pth")
                    agent_save_path = os.path.join(save_dir, f"hgpo_policy_best_ep{best_epoch}.pth")
                    
                    torch.save(self.lghrec_model.state_dict(), model_save_path)
                    self.hgpo_agent.save_model(agent_save_path)
                    self.logger.info(f"New best model saved to {model_save_path} and {agent_save_path} at epoch {epoch+1} with NDCG@{K_EVAL}: {ndcg:.4f}")

            end_time = time.time()
            self.logger.info(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")

        self.logger.info(f"Training finished. Best NDCG@{K_EVAL}: {best_ndcg:.4f} at epoch {best_epoch}.")


