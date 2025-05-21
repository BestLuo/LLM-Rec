# models/lghrec_model.py
import torch
import torch.nn as nn
import dgl
from lghrec_project.config import EMBEDDING_DIM, GNN_LAYERS, TEMPERATURE_INITIAL, DROPOUT_RATE, DEVICE
from models.dseg import DeepSemanticEmbeddingGenerator
from models.sgl_gnn import SGL_GNN # SGL_GNN is the graph learning backbone

class LGHRec(nn.Module):
    def __init__(self, num_users, num_items, gnn_embedding_dim, cot_embedding_dim,
                 dgl_graph, initial_cot_embeddings_tensor, 
                 sgl_temp=TEMPERATURE_INITIAL, sgl_dropout=DROPOUT_RATE, num_gnn_layers=GNN_LAYERS,
                 device=DEVICE):
        """
        LGHRec main model.
        Integrates DSEG (for item embedding initialization) and SGL_GNN (for graph learning and contrastive framework).
        HGPO Agent will interact with this model to optimize contrastive learning parameters.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            gnn_embedding_dim (int): Final embedding dimension of users and items in GNN.
            cot_embedding_dim (int): Dimension of raw semantic embeddings from LLM CoT.
            dgl_graph (dgl.DGLGraph): Complete user-item interaction graph.
            initial_cot_embeddings_tensor (torch.Tensor): (num_items, cot_embedding_dim), preloaded CoT embeddings.
            sgl_temp (float): Initial temperature for contrastive loss (overridden by HGPO).
            sgl_dropout (float): Dropout rate for SGL graph augmentation.
            num_gnn_layers (int): Number of layers in GNN.
            device (torch.device): Computation device.
        """
        super(LGHRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.gnn_embedding_dim = gnn_embedding_dim
        self.device = device
        self.dgl_graph = dgl_graph.to(device)

        # 1. Item ID embedding layer (for DSEG fusion)
        # One input to DSEG is the item's ID embedding. These ID embeddings are learned by the model.
        self.item_id_for_dseg_embeddings = nn.Embedding(num_items, gnn_embedding_dim).to(device) # d_id
        nn.init.xavier_uniform_(self.item_id_for_dseg_embeddings.weight)

        # 2. DSEG module
        # DSEG input: item_id_embeddings (d_id), item_cot_embeddings (d_c)
        # DSEG output: initial_item_gnn_embeddings (gnn_embedding_dim)
        self.dseg_module = DeepSemanticEmbeddingGenerator(
            num_items=num_items,
            id_embedding_dim=gnn_embedding_dim, # ID embedding dimension for DSEG input
            cot_embedding_dim=cot_embedding_dim,
            output_embedding_dim=gnn_embedding_dim # DSEG output, used as initial item embedding for GNN
        ).to(device)
        
        # Preprocess CoT embeddings, ensure on correct device and no gradient needed
        self.initial_cot_embeddings_tensor = initial_cot_embeddings_tensor.clone().detach().to(device)

        # Get all item ID embeddings (from self.item_id_for_dseg_embeddings)
        all_item_indices = torch.arange(num_items, device=device)
        id_embeds_for_dseg_input = self.item_id_for_dseg_embeddings(all_item_indices)
        
        # 3. User ID embedding layer (initial user embedding for GNN)
        self.user_gnn_embeddings = nn.Embedding(num_users, gnn_embedding_dim).to(device)
        nn.init.xavier_uniform_(self.user_gnn_embeddings.weight)

        # 4. SGL-based GNN module
        # SGL_GNN needs initial user embeddings and (DSEG-fused) item embeddings
        # For modularity, SGL_GNN receives initial embeddings as parameters.
        # DSEG output will be used as SGL_GNN's initial item embedding.
        # SGL_GNN has its own nn.Parameter to store and update these initial embeddings.
        
        # Dynamically get DSEG-fused item embeddings as initial item embeddings for SGL_GNN
        initial_gnn_item_embeds = self.dseg_module(
            id_embeds_for_dseg_input, # (num_items, gnn_embedding_dim)
            self.initial_cot_embeddings_tensor[:num_items, :] # (num_items, cot_embedding_dim)
        )

        self.sgl_gnn = SGL_GNN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=gnn_embedding_dim,
            initial_user_embeddings=self.user_gnn_embeddings.weight.data.clone().detach(), # initial value
            initial_item_embeddings=initial_gnn_item_embeds.data.clone().detach(), # initial DSEG output
            graph=self.dgl_graph,
            num_layers=num_gnn_layers,
            sgl_temp=sgl_temp, # This temperature will be dynamically overridden by HGPO Agent
            sgl_dropout_rate=sgl_dropout
        ).to(device)
        

    def get_trainable_dseg_item_id_embeddings(self):
        """Return trainable item ID embeddings used by DSEG for optimizer."""
        return self.item_id_for_dseg_embeddings.weight

    def forward(self, user_indices_batch, positive_item_indices_batch, negative_item_indices_batch=None,
                for_contrastive=False, contrastive_temp_override=None):
        """
        Forward pass of LGHRec model.
        Args:
            user_indices_batch (torch.Tensor): Batch of user IDs.
            positive_item_indices_batch (torch.Tensor): Corresponding positive item IDs.
            negative_item_indices_batch (torch.Tensor, optional): Corresponding negative item IDs (for BPR loss, etc.).
            for_contrastive (bool): Whether to generate dual-view embeddings for contrastive learning.
            contrastive_temp_override (float, optional): If provided, overrides SGL_GNN's internal temperature for InfoNCE, provided by HGPO Agent.
        Returns:
            If for_contrastive=False (for recommendation task):
                pos_scores (torch.Tensor): Scores for positive pairs.
                neg_scores (torch.Tensor): Scores for negative pairs
                (user_embeds, pos_item_embeds, neg_item_embeds)
            If for_contrastive=True:
                (user_embeds_v1, item_embeds_v1), (user_embeds_v2, item_embeds_v2) # from SGL_GNN
        """
        # Dynamically update initial item embeddings used in SGL_GNN (latest output from DSEG)
        # Because DSEG's item_id_for_dseg_embeddings and fusion_layer are learnable
        all_item_idxs = torch.arange(self.num_items, device=self.device)
        current_id_embeds_for_dseg = self.item_id_for_dseg_embeddings(all_item_idxs)
        current_fused_item_embeds_for_gnn = self.dseg_module(
            current_id_embeds_for_dseg,
            self.initial_cot_embeddings_tensor[:self.num_items, :]
        )
        # Update SGL_GNN's internal item embedding parameter
        self.sgl_gnn.item_embeds.data = current_fused_item_embeds_for_gnn.data
        
        # Update SGL_GNN's user embeddings
        self.sgl_gnn.user_embeds.data = self.user_gnn_embeddings.weight.data

        if for_contrastive:
            # What we need is the contrastive embeddings of users and items in the current batch
            (batch_users_v1, batch_items_v1), (batch_users_v2, batch_items_v2) = \
                self.sgl_gnn(user_indices=user_indices_batch, 
                             item_indices=positive_item_indices_batch, 
                             for_contrastive=True)

            return (batch_users_v1, batch_items_v1), (batch_users_v2, batch_items_v2)
        else:
            # For recommendation task (BPR loss)
            # Get final embeddings of users, positive items, and negative items
            user_embeds_final_propagated, pos_item_embeds_final_propagated = \
                self.sgl_gnn(user_indices=user_indices_batch, 
                             item_indices=positive_item_indices_batch, 
                             for_contrastive=False)

            pos_scores = torch.sum(user_embeds_final_propagated * pos_item_embeds_final_propagated, dim=1)
            
            neg_scores = None
            neg_item_embeds_final_propagated = None
            if negative_item_indices_batch is not None:
                _, neg_item_embeds_final_propagated = \
                    self.sgl_gnn(user_indices=None, 
                                 item_indices=negative_item_indices_batch,
                                 for_contrastive=False)
                neg_scores = torch.sum(user_embeds_final_propagated * neg_item_embeds_final_propagated, dim=1)

            return pos_scores, neg_scores, (user_embeds_final_propagated, pos_item_embeds_final_propagated, neg_item_embeds_final_propagated)

    def get_all_learned_embeddings(self, after_gnn_propagation=True):
        """
        Get all learned final embeddings of users and items.
        Args:
            after_gnn_propagation (bool): True to return embeddings after GNN propagation, False for initial GNN embeddings.
        Returns:
            torch.Tensor: All user embeddings (num_users, gnn_embedding_dim)
            torch.Tensor: All item embeddings (num_items, gnn_embedding_dim)
        """
        if after_gnn_propagation:
            all_item_idxs = torch.arange(self.num_items, device=self.device)
            current_id_embeds_for_dseg = self.item_id_for_dseg_embeddings(all_item_idxs)
            current_fused_item_embeds_for_gnn = self.dseg_module(
                current_id_embeds_for_dseg,
                self.initial_cot_embeddings_tensor[:self.num_items, :]
            )
            self.sgl_gnn.item_embeds.data = current_fused_item_embeds_for_gnn.data
            self.sgl_gnn.user_embeds.data = self.user_gnn_embeddings.weight.data
            
            return self.sgl_gnn.get_all_embeddings() 
        else:
            # Return initial GNN embeddings (DSEG-fused item embeddings and initial user ID embeddings)
            all_item_idxs = torch.arange(self.num_items, device=self.device)
            id_embeds_for_dseg_input = self.item_id_for_dseg_embeddings(all_item_idxs)
            initial_item_gnn_embeds = self.dseg_module(
                id_embeds_for_dseg_input,
                self.initial_cot_embeddings_tensor[:self.num_items, :]
            )
            initial_user_gnn_embeds = self.user_gnn_embeddings.weight
            return initial_user_gnn_embeds, initial_item_gnn_embeds

    def get_user_item_scores_for_eval(self, user_ids_tensor, item_ids_tensor):
        """
        Compute user-item scores.
        Use SGL_GNN's internal get_user_item_scores method.
        """
        all_item_idxs = torch.arange(self.num_items, device=self.device)
        current_id_embeds_for_dseg = self.item_id_for_dseg_embeddings(all_item_idxs)
        current_fused_item_embeds_for_gnn = self.dseg_module(
            current_id_embeds_for_dseg,
            self.initial_cot_embeddings_tensor[:self.num_items, :]
        )
        self.sgl_gnn.item_embeds.data = current_fused_item_embeds_for_gnn.data
        self.sgl_gnn.user_embeds.data = self.user_gnn_embeddings.weight.data
        
        return self.sgl_gnn.get_user_item_scores(user_ids_tensor, item_ids_tensor)

