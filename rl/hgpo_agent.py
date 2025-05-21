# rl/hgpo_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.policy_network import PolicyNetwork
from rl.replay_buffer import ReplayBuffer
from rl.rewards import calculate_total_reward # For actual reward calculation
from rl.group_manager import GroupManager
from lghrec_project.config import (POLICY_LEARNING_RATE, PPO_CLIP_EPSILON, ENTROPY_COEFF, 
                    HARMONIZATION_COEFF, DEVICE, NUM_NEGATIVE_SAMPLES_HGPO, NUM_GROUPS_HGPO)

class HGPOAgent:
    def __init__(self, state_dim, num_candidate_neg_samples, num_groups, device=DEVICE):
        """
        HGPO Agent.
        Args:
            state_dim (int): State dimension.
            num_candidate_neg_samples (int): Size of candidate negative sample pool.
            num_groups (int): Number of node groups K.
            device (torch.device): Device.
        """
        self.device = device
        self.num_groups = num_groups
        self.num_candidate_neg_samples = num_candidate_neg_samples # Candidate pool size

        # Policy network (Actor)
        self.policy_net = PolicyNetwork(state_dim, num_candidate_neg_samples).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=POLICY_LEARNING_RATE)
        
        # Replay buffer (for storing a batch of data for update)
        self.buffer = ReplayBuffer(capacity=4096 * 2)

        self.ppo_clip_epsilon = PPO_CLIP_EPSILON
        self.entropy_coeff = ENTROPY_COEFF
        self.harmonization_coeff = HARMONIZATION_COEFF

    def select_actions_and_collect_experience(self, rl_state_batch, # (B, state_dim)
                                              anchor_embeds_batch, # (B, D_emb)
                                              positive_embeds_batch, # (B, D_emb)
                                              candidate_neg_embeds_batch, # (B, Num_candidates, D_emb)
                                              anchor_degrees_batch, # (B,)
                                              anchor_group_indices_batch, # (B,)
                                              num_neg_to_select_m=NUM_NEGATIVE_SAMPLES_HGPO):
        """
        Use current policy to select actions, compute rewards, and store experience in buffer.
        Args:
            rl_state_batch (torch.Tensor): Batch of states.
            anchor_embeds_batch, positive_embeds_batch: Embeddings for reward calculation.
            candidate_neg_embeds_batch: Embeddings of candidate negatives, used to gather actual embeddings by selected indices.
            anchor_degrees_batch: Anchor degrees, for reward and grouping.
            anchor_group_indices_batch: Group indices for anchors.
            num_neg_to_select_m (int): Number of negative samples M to select.
        """
        self.policy_net.eval() # Ensure no gradient during action selection
        with torch.no_grad():
            # selected_neg_indices: (B, M), indices within the candidate pool
            # selected_neg_log_probs: (B, M), log_prob of choosing these M neg samples
            # sampled_temp: (B, 1), chosen temperature
            # temp_log_prob: (B, 1), log_prob of choosing this temperature
            # policy_entropy: scalar, average entropy of the policy for this batch
            selected_neg_indices, selected_neg_log_probs_old, \
            sampled_temp, temp_log_prob_old, _ = \
                self.policy_net.select_action(rl_state_batch, num_neg_to_select_m)

            # Gather actual selected negative embeddings from candidate pool
            # candidate_neg_embeds_batch: (B, Num_candidates, D_emb)
            # selected_neg_indices: (B, M)
            # We need to gather along dim 1.
            # Unsqueeze selected_neg_indices to (B, M, 1) and expand to (B, M, D_emb)
            idx_expanded = selected_neg_indices.unsqueeze(-1).expand(-1, -1, candidate_neg_embeds_batch.shape[-1])
            actual_selected_negative_embeds = torch.gather(candidate_neg_embeds_batch, 1, idx_expanded) # (B,M,D)

            # Compute rewards
            rewards_batch = calculate_total_reward(
                anchor_embeds_batch,
                positive_embeds_batch,
                actual_selected_negative_embeds, # (B, M, D_emb)
                sampled_temp, # (B, 1)
                anchor_degrees_batch, # (B,)
                self.device
            ) # (B,)

        self.policy_net.train() # Switch back to train mode

        # Sum log_probs of M negative samples.
        sum_neg_log_probs_old = selected_neg_log_probs_old.sum(dim=1, keepdim=True) # (B,1)

        # Push individual experiences if rl_state_batch represents multiple independent decisions
        for i in range(rl_state_batch.shape[0]):
            self.buffer.push(
                rl_state_batch[i],
                selected_neg_indices[i], # (M,)
                sampled_temp[i],         # (1,)
                sum_neg_log_probs_old[i],# (1,) - log_prob for the set of M negative samples
                temp_log_prob_old[i],    # (1,) - log_prob for temperature
                rewards_batch[i].unsqueeze(0),       # (1,)
                anchor_degrees_batch[i].unsqueeze(0),# (1,)
                anchor_group_indices_batch[i].unsqueeze(0) # (1,)
            )
        
        return rewards_batch.mean().item() # Return average reward as monitoring metric

    def update_policy(self):
        """
        Update policy network using data in buffer.
        Implements HGPO objective.
        """
        if len(self.buffer) == 0:
            return 0.0, 0.0, 0.0 # policy_loss, entropy_loss, harm_loss

        # Get all data from buffer
        data = self.buffer.get_all_data(self.device)
        if data is None:
            return 0.0, 0.0, 0.0
            
        states, neg_actions_indices_old, temp_actions_values_old, \
        neg_log_probs_old, temp_log_probs_old, \
        rewards, degrees, group_indices = data
        
        # 1. Compute log_prob of old policy for combined actions
        # log_p_old(a_neg, a_temp) = log_p_old(a_neg) + log_p_old(a_temp)
        # neg_log_probs_old is already sum for M samples.
        log_probs_old_combined = neg_log_probs_old + temp_log_probs_old # (TotalSamples, 1)

        # 2. Compute log_prob and entropy of current policy for these actions
        # neg_log_probs_new: (TotalSamples, M)
        # temp_log_probs_new: (TotalSamples, 1)
        # entropy_new: scalar
        neg_log_probs_new_indiv, temp_log_probs_new, entropy_new = \
            self.policy_net.evaluate_actions(states, neg_actions_indices_old, temp_actions_values_old)
        
        sum_neg_log_probs_new = neg_log_probs_new_indiv.sum(dim=1, keepdim=True) # (TotalSamples, 1)
        log_probs_new_combined = sum_neg_log_probs_new + temp_log_probs_new # (TotalSamples, 1)

        # 3. Compute relative advantage A_t^rel = r_t - R_bar_g(s_t)
        # First compute average reward R_bar_g for each group
        group_avg_rewards = torch.zeros(self.num_groups, device=self.device)
        group_counts = torch.zeros(self.num_groups, device=self.device)

        for g_idx in range(self.num_groups):
            mask = (group_indices.squeeze() == g_idx)
            if mask.sum() > 0:
                group_avg_rewards[g_idx] = rewards[mask].mean()
                group_counts[g_idx] = mask.sum()
            else:
                group_avg_rewards[g_idx] = 0 

        # Get group average reward for each sample
        # rewards: (TotalSamples, 1)
        # group_indices: (TotalSamples, 1)
        # R_bar_g_for_samples: (TotalSamples, 1)
        R_bar_g_for_samples = group_avg_rewards[group_indices.squeeze().long()] 
        if R_bar_g_for_samples.ndim == 1:
            R_bar_g_for_samples = R_bar_g_for_samples.unsqueeze(1)
            
        relative_advantages = rewards - R_bar_g_for_samples # (TotalSamples, 1)
        # Normalize advantage
        relative_advantages = (relative_advantages - relative_advantages.mean()) / (relative_advantages.std() + 1e-8)

        # 4. Compute policy loss L^POLICY (Eq. 9)
        # r_t(theta) = exp(log_prob_new - log_prob_old)
        ratios = torch.exp(log_probs_new_combined - log_probs_old_combined.detach()) # (TotalSamples, 1)
        
        surr1 = ratios * relative_advantages.detach()
        surr2 = torch.clamp(ratios, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * relative_advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean() # Negative because we minimize -L^POLICY

        # 5. Compute entropy regularization S[pi_theta] (Eq. 10)
        entropy_loss = -self.entropy_coeff * entropy_new # Minimize negative entropy to maximize entropy

        # 6. Compute harmonization loss L^HARM (theta) 
        # L^HARM = lambda_harm * Var_g [R_bar_g]
        valid_group_avg_rewards = group_avg_rewards[group_counts > 0]
        if len(valid_group_avg_rewards) > 1:
            harmonization_loss = self.harmonization_coeff * torch.var(valid_group_avg_rewards)
        else:
            harmonization_loss = torch.tensor(0.0, device=self.device)

        # 7. Total loss (Eq. 8)
        # L_HGPO = -L^POLICY + c1 * S[pi_theta] + L^HARM
        total_loss = policy_loss + entropy_loss + harmonization_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return policy_loss.item(), entropy_loss.item(), harmonization_loss.item()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"HGPO Agent policy network saved to {path}")

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"HGPO Agent policy network loaded from {path}")

