# models/sgl_gnn.py
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from lghrec_project.config import GNN_LAYERS, TEMPERATURE_INITIAL, DROPOUT_RATE, DEVICE

class LightGCNLayer(nn.Module):
    """
    Single LightGCN convolution layer.
    LightGCN propagation rule: e_u^(k+1) = sum_{i in N(u)} (1/sqrt(|N(u)||N(i)|)) * e_i^(k)
                               e_i^(k+1) = sum_{u in N(i)} (1/sqrt(|N(u)||N(i)|)) * e_u^(k)
    """
    def __init__(self):
        super(LightGCNLayer, self).__init__()

    def forward(self, graph, h):
        """
        Args:
            graph (dgl.DGLGraph): DGL graph.
            h (torch.Tensor): Input node features (embeddings of all users and items).
        Returns:
            torch.Tensor: Node features after one LightGCN propagation layer.
        """
        with graph.local_scope():
            degs_src = graph.out_degrees().float().clamp(min=1) # Source node out-degree
            degs_dst = graph.in_degrees().float().clamp(min=1)  # Target node in-degree
            
            # Prepare normalization factors for edge weights
            graph.srcdata['h'] = h
            graph.srcdata['norm_coeff_sqrt_deg'] = 1.0 / torch.sqrt(degs_src) # 1/sqrt(|N_u|) for u
            graph.dstdata['norm_coeff_sqrt_deg'] = 1.0 / torch.sqrt(degs_dst) # 1/sqrt(|N_i|) for i

            # Send source node sqrt_deg to edges
            graph.apply_edges(lambda edges: {'norm_u': edges.src['norm_coeff_sqrt_deg']})
            # Send target node sqrt_deg to edges
            graph.apply_edges(lambda edges: {'norm_v': edges.dst['norm_coeff_sqrt_deg']})
            
            # Compute edge weights 1 / (sqrt(deg_u) * sqrt(deg_v))
            graph.edata['weight'] = graph.edata['norm_u'] * graph.edata['norm_v']
            
            # Message passing: weight * h_src
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h_out'))
            
            h_new = graph.dstdata['h_out']
            return h_new

class SGL_GNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, 
                 initial_user_embeddings, initial_item_embeddings,
                 graph, num_layers=GNN_LAYERS, 
                 sgl_temp=TEMPERATURE_INITIAL, sgl_dropout_rate=DROPOUT_RATE):
        """
        SGL-based GNN model (core is LightGCN).
        SGL (Self-supervised Graph Learning) generates different views via graph augmentation for contrastive learning.
        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_dim (int): Embedding dimension.
            initial_user_embeddings (torch.Tensor): Initial user embeddings (usually ID embeddings).
            initial_item_embeddings (torch.Tensor): Initial item embeddings (from DSEG fusion).
            graph (dgl.DGLGraph): Complete user-item interaction graph.
            num_layers (int): Number of LightGCN layers.
            sgl_temp (float): InfoNCE loss temperature coefficient (dynamically adjusted by HGPO).
            sgl_dropout_rate (float): Dropout rate for SGL graph augmentation.
        """
        super(SGL_GNN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.graph = graph
        self.sgl_temp = sgl_temp # This temperature will be controlled by HGPO
        self.sgl_dropout_rate = sgl_dropout_rate

        # Embedding layers (use provided initial embeddings)
        self.user_embeds = nn.Parameter(initial_user_embeddings)
        self.item_embeds = nn.Parameter(initial_item_embeddings) # from DSEG

        # LightGCN layers
        self.lightgcn_layers = nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])
        
        # SGL graph augmentation dropout
        # More complex SGL may include multiple augmentation strategies (node_drop, edge_drop, random_walk)
        # Here we only prepare an edge_drop mechanism for contrastive learning
        self.edge_dropout = dgl.DropEdge(p=sgl_dropout_rate) if sgl_dropout_rate > 0 else None

    def _graph_augmentation(self, graph):
        """
        SGL graph augmentation. Here, simply implement edge dropout.
        Returns an augmented graph view.
        """
        if self.edge_dropout:
            augmented_graph = self.edge_dropout(graph)
            return augmented_graph
        return graph.clone() # If no dropout, return a clone to avoid modification

    def forward_gnn(self, graph_view, current_user_embeds, current_item_embeds):
        """
        Perform LightGCN propagation on a graph view.
        Args:
            graph_view (dgl.DGLGraph): Graph for propagation (original or augmented).
            current_user_embeds (torch.Tensor): User embeddings for current layer.
            current_item_embeds (torch.Tensor): Item embeddings for current layer.
        Returns:
            torch.Tensor: Final user embeddings.
            torch.Tensor: Final item embeddings.
        """
        all_embeddings = torch.cat([current_user_embeds, current_item_embeds], dim=0)
        layer_embeddings = [all_embeddings] # Store embeddings for each layer (including layer 0)

        for i in range(self.num_layers):
            all_embeddings = self.lightgcn_layers[i](graph_view, all_embeddings)
            layer_embeddings.append(all_embeddings)
        
        # Final embedding is the mean of all layer embeddings
        final_embeddings_tensor = torch.stack(layer_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings_tensor, dim=1)
        
        final_user_embeds, final_item_embeds = torch.split(final_embeddings, [self.num_users, self.num_items], dim=0)
        return final_user_embeds, final_item_embeds

    def forward(self, user_indices=None, item_indices=None, for_contrastive=False):
        """
        Model forward pass.
        Args:
            user_indices (torch.Tensor): User IDs to get embeddings for. If None, return all user embeddings.
            item_indices (torch.Tensor): Item IDs to get embeddings for. If None, return all item embeddings.
            for_contrastive (bool): If True, perform two graph augmentations and GNN propagations to generate two views for contrastive learning.
        Returns:
            If for_contrastive=False:
                user_embeds (torch.Tensor): Final user embeddings.
                item_embeds (torch.Tensor): Final item embeddings.
            If for_contrastive=True:
                (user_embeds_v1, item_embeds_v1), (user_embeds_v2, item_embeds_v2)
        """
        # Get current learned ID embeddings and DSEG-fused item embeddings
        current_learned_user_embeds = self.user_embeds
        current_learned_item_embeds = self.item_embeds

        if for_contrastive:
            # Generate two views for contrastive learning
            graph_v1 = self._graph_augmentation(self.graph)
            graph_v2 = self._graph_augmentation(self.graph) # Augment again for a different view

            # Propagate on view 1
            users_v1, items_v1 = self.forward_gnn(graph_v1, current_learned_user_embeds, current_learned_item_embeds)
            # Propagate on view 2
            users_v2, items_v2 = self.forward_gnn(graph_v2, current_learned_user_embeds, current_learned_item_embeds)
            
            # If indices provided, return embeddings for those indices
            if user_indices is not None:
                users_v1 = users_v1[user_indices]
                users_v2 = users_v2[user_indices]
            if item_indices is not None:
                items_v1 = items_v1[item_indices]
                items_v2 = items_v2[item_indices]
            
            return (users_v1, items_v1), (users_v2, items_v2)
        else:
            final_user_embeds, final_item_embeds = self.forward_gnn(self.graph, current_learned_user_embeds, current_learned_item_embeds)
            
            if user_indices is not None:
                final_user_embeds = final_user_embeds[user_indices]
            if item_indices is not None:
                final_item_embeds = final_item_embeds[item_indices]
                
            return final_user_embeds, final_item_embeds

    def get_all_embeddings(self, graph_to_use=None):
        """Helper function to get all user and item final embeddings on a specified graph."""
        g = graph_to_use if graph_to_use is not None else self.graph
        return self.forward_gnn(g, self.user_embeds, self.item_embeds)

    def get_user_item_scores(self, user_ids_tensor, item_ids_tensor):
        """
        Compute predicted scores for given user-item pairs.
        Args:
            user_ids_tensor (torch.Tensor): User ID tensor.
            item_ids_tensor (torch.Tensor): Item ID tensor.
        Returns:
            torch.Tensor: Predicted scores.
        """
        # Get all user and item final embeddings (on original graph)
        all_user_embeds, all_item_embeds = self.get_all_embeddings()
        
        user_embeds = all_user_embeds[user_ids_tensor] 
        item_embeds = all_item_embeds[item_ids_tensor] 

        # If item_ids_tensor is a batch of items (one-to-one with user_ids_tensor)
        if user_embeds.shape[0] == item_embeds.shape[0] and user_embeds.ndim == item_embeds.ndim == 2 :
            scores = torch.sum(user_embeds * item_embeds, dim=1)
        # If item_ids_tensor is all items, user_ids_tensor is a single user (or multiple users, need broadcasting)
        elif user_embeds.ndim == 2 and item_embeds.ndim == 2 and user_embeds.shape[0] == 1: # Single user vs all items
            scores = torch.matmul(user_embeds, item_embeds.transpose(0, 1)).squeeze()
        elif user_embeds.ndim == 2 and item_embeds.ndim == 2 and user_embeds.shape[0] > 1 and item_embeds.shape[0] > user_embeds.shape[0]: # Multiple users vs all items
             scores = torch.matmul(user_embeds, item_embeds.transpose(0, 1)) # 
        else:
            raise ValueError("Shape of user and item embeddings incompatible for score computation.")
            
        return scores

