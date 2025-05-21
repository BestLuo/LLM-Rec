# data/graph_utils.py
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from lghrec_project.config import DEVICE

def build_dgl_graph(num_users, num_items, train_dataset):
    """
    Build user-item interaction graph from training dataset.

    Args:
        num_users (int): Total number of users.
        num_items (int): Total number of items.
        train_dataset (RecommendationDataset): Training dataset.

    Returns:
        dgl.DGLGraph: Constructed DGL graph.
    """
    # Extract user-item interactions from training data
    user_ids = train_dataset.users.numpy().copy()
    item_ids = train_dataset.items.numpy().copy()

    graph_data = {
        ('user', 'interacts', 'item'): (torch.tensor(user_ids), torch.tensor(item_ids)),
        ('item', 'interacted_by', 'user'): (torch.tensor(item_ids), torch.tensor(user_ids)) # Add reverse edges
    }

    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil() 

    # Edges from users to items
    adj_mat[user_ids, item_ids + num_users] = 1
    # Edges from items to users (symmetric)
    adj_mat[item_ids + num_users, user_ids] = 1
    
    adj_mat = adj_mat.tocsr() # convert to CSR for DGL graph construction
    
    # Create DGL graph from Scipy sparse matrix
    graph = dgl.from_scipy(adj_mat)
    graph = graph.to(DEVICE)

    print(f"DGL graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    return graph

def get_graph_node_degrees(graph, num_users, num_items):
    """
    Compute degrees of user and item nodes in the graph.
    Used for node grouping in HGPO.

    Args:
        graph (dgl.DGLGraph): Constructed DGL graph.
        num_users (int): Number of users.
        num_items (int): Number of items.

    Returns:
        tuple: (user_degrees, item_degrees)
               user_degrees (torch.Tensor): Degree of each user node.
               item_degrees (torch.Tensor): Degree of each item node.
    """
    if graph.is_homogeneous:
        degrees = graph.in_degrees().float() # In-degree for all nodes
        user_degrees = degrees[:num_users]
        item_degrees = degrees[num_users:] # Item nodes start from num_users
    else: # Heterogeneous graph handling (if using heterogeneous graph scheme)
        user_degrees = graph.in_degrees(etype='interacts', ntype='user')
        item_degrees = graph.in_degrees(etype='interacted_by', ntype='item')
    
    return user_degrees.to(DEVICE), item_degrees.to(DEVICE)

