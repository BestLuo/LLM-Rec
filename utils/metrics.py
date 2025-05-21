# utils/metrics.py
import torch
import numpy as np

def recall_at_k(y_true_ranks, k):
    """
    Compute Recall@K.
    Args:
        y_true_ranks (torch.Tensor): Shape (num_users,).
        k (int): Value of K in Top-K.
    Returns:
        float: Recall@K value.
    """
    hits_at_k = (y_true_ranks < k) & (y_true_ranks != -1) 
    recall = torch.sum(hits_at_k).float() / len(y_true_ranks)
    return recall.item()

def ndcg_at_k(y_true_ranks, k):
    """
    Compute NDCG@K.
    Args:
        y_true_ranks (torch.Tensor): Shape (num_users,).
        k (int): Value of K in Top-K.
    Returns:
        float: NDCG@K value.
    """
    
    dcg_values = torch.zeros(len(y_true_ranks), device=y_true_ranks.device)
    hits_mask = (y_true_ranks < k) & (y_true_ranks != -1)
    
    dcg_values[hits_mask] = 1.0 / torch.log2(y_true_ranks[hits_mask].float() + 2.0)
    dcg = torch.sum(dcg_values)

    # Calculate IDCG (Ideal DCG)
    idcg = torch.sum(hits_mask).float() 

    if idcg == 0:
        return 0.0
        
    ndcg = dcg / idcg
    
    ndcgs_per_user = torch.zeros(len(y_true_ranks), device=y_true_ranks.device)

    valid_hits_mask = (y_true_ranks < k) & (y_true_ranks != -1)
    
    if torch.sum(valid_hits_mask) == 0:
        return 0.0

    ndcgs_per_user[valid_hits_mask] = 1.0 / torch.log2(y_true_ranks[valid_hits_mask].float() + 2.0)
    
    return torch.mean(ndcgs_per_user).item()


def get_user_positive_items(df):
    """
    Get positive items for each user from DataFrame.
    Args:
        df (pd.DataFrame): Contains 'user_id' and 'item_id' columns.
    Returns:
        dict: {user_id: [item_id_1, item_id_2, ...]}
    """
    user_positive_items = df.groupby('user_id')['item_id'].apply(list).to_dict()
    return user_positive_items

def evaluate_model(model, test_loader, all_item_ids, k, device, user_history_for_exclusion=None):
    """
    Evaluate model performance.
    Args:
        model: Trained model, should have predict(users, items) method.
        test_loader: DataLoader for test data, each sample is (user_id_tensor, item_id_tensor).
        all_item_ids (torch.Tensor): Tensor of all item IDs for recommendation.
        k (int): Value of K for Top-K evaluation.
        device: torch.device.
        user_history_for_exclusion (dict, optional): {user_id: set(historically_interacted_item_ids)} for excluding interacted items during evaluation.
    Returns:
        tuple: (recall@k, ndcg@k)
    """
    model.eval()
    recalls = []
    ndcgs = []
    
    # Collect all test users and their true positive items from test_loader
    test_user_item_pairs = []
    for users, items in test_loader:
        for user, item in zip(users.tolist(), items.tolist()):
            test_user_item_pairs.append((user, item))

    # Group true items by user
    test_user_true_items = {}
    for user, item in test_user_item_pairs:
        if user not in test_user_true_items:
            test_user_true_items[user] = []
        test_user_true_items[user].append(item)

    unique_test_users = sorted(list(test_user_true_items.keys()))
    
    all_item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(device)

    ranks_list = []

    with torch.no_grad():
        for user_id in unique_test_users:
            user_tensor = torch.tensor([user_id] * len(all_item_ids_tensor), dtype=torch.long).to(device)
            
            # Predict scores for all items for the user
            try:
                # Return preference scores for all items
                scores = model.get_user_item_scores(user_tensor, all_item_ids_tensor) # (num_all_items,)
            except AttributeError:
                 # If get_user_item_scores not available, use forward or compute embeddings externally
                user_embed, _ = model.get_embeddings(user_tensor, all_item_ids_tensor) # Assume returns user and item embeddings
                _, item_embeds = model.get_embeddings(None, all_item_ids_tensor, is_item_ids_global=True)
                current_user_embed = user_embed[0].unsqueeze(0) # (1, dim)
                scores = torch.matmul(current_user_embed, item_embeds.transpose(0, 1)).squeeze() # (num_all_items)

            # Exclude historically interacted items
            if user_history_for_exclusion and user_id in user_history_for_exclusion:
                exclude_items_indices = [idx for idx, item in enumerate(all_item_ids) if item in user_history_for_exclusion[user_id]]
                if exclude_items_indices:
                    scores[exclude_items_indices] = -np.inf # Set scores of interacted items to -inf

            # Get sorted item indices
            _, ranked_indices = torch.topk(scores, k=len(all_item_ids_tensor)) # Sort all items
            ranked_item_ids = all_item_ids_tensor[ranked_indices].cpu().tolist()

            # Find the rank of true positive items in the recommendation list
            user_true_pos_items = test_user_true_items[user_id]
            for true_pos_item_id in user_true_pos_items: # A user may have multiple positives in test set
                try:
                    rank = ranked_item_ids.index(true_pos_item_id)
                    ranks_list.append(rank)
                except ValueError:
                    ranks_list.append(-1) # Not found (not in all items or excluded)

    if not ranks_list:
        return 0.0, 0.0

    y_true_ranks_tensor = torch.tensor(ranks_list, dtype=torch.long)
    
    recall_k = recall_at_k(y_true_ranks_tensor, k)
    ndcg_k = ndcg_at_k(y_true_ranks_tensor, k)
    
    return recall_k, ndcg_k

