# utils/loss.py
import torch
import torch.nn.functional as F

def bpr_loss(positive_scores, negative_scores):
    """
    Compute BPR (Bayesian Personalized Ranking) Loss.
    L_BPR = - sum_{u,i,j} ln(sigmoid(score_ui - score_uj))
    Args:
        positive_scores (torch.Tensor): Predicted scores for positive items.
        negative_scores (torch.Tensor): Predicted scores for negative items.
    Returns:
        torch.Tensor: BPR loss value (scalar).
    """
    loss = -torch.log(torch.sigmoid(positive_scores - negative_scores) + 1e-8) # add small value to avoid log(0)
    return torch.mean(loss)

def info_nce_loss(view1_embeds, view2_embeds, temperature):
    """
    Compute InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss for contrastive learning.
    Args:
        view1_embeds (torch.Tensor): Embeddings of view 1, shape (batch_size, dim).
        view2_embeds (torch.Tensor): Embeddings of view 2, shape (batch_size, dim).
        temperature (torch.Tensor or float): Temperature parameter.
    Returns:
        torch.Tensor: InfoNCE loss value (scalar).
    """
    batch_size = view1_embeds.shape[0]
    
    # L2 normalize embeddings so dot product is cosine similarity
    view1_norm = F.normalize(view1_embeds, p=2, dim=1)
    view2_norm = F.normalize(view2_embeds, p=2, dim=1)

    # Compute similarity: (batch_size, dim) @ (dim, batch_size) -> (batch_size, batch_size)
    similarity_matrix_unscaled = torch.matmul(view1_norm, view2_norm.t())
    
    # Adjust by temperature
    if isinstance(temperature, torch.Tensor):
        if temperature.ndim == 1:
            temperature = temperature.unsqueeze(1) # (B,) -> (B,1) for broadcasting
        similarity_matrix_scaled = similarity_matrix_unscaled / temperature # (B,B) / (B,1)
    else:
        similarity_matrix_scaled = similarity_matrix_unscaled / temperature

    # Positive similarity (already scaled)
    diag_elements_scaled = torch.diag(similarity_matrix_scaled) # (B,)
    exp_pos_sim = torch.exp(diag_elements_scaled)
    
    # All similarities (for each anchor)
    exp_all_sim_sum_for_anchor = torch.sum(torch.exp(similarity_matrix_scaled), dim=1) 
    
    # Loss per anchor
    loss_per_anchor = -torch.log(exp_pos_sim / (exp_all_sim_sum_for_anchor + 1e-8))
    
    return torch.mean(loss_per_anchor)


def contrastive_loss_for_nodes(user_embeds_v1, user_embeds_v2, 
                               item_embeds_v1, item_embeds_v2, 
                               temperature, lambda_coeff=1.0):
    """
    Compute node-level contrastive loss (users and items).
    L_node = L_user + lambda * L_item
    Args:
        user_embeds_v1, user_embeds_v2: Two views of user embeddings.
        item_embeds_v1, item_embeds_v2: Two views of item embeddings.
        temperature (torch.Tensor or float): InfoNCE temperature.
        lambda_coeff: Weight for item contrastive loss.
    Returns:
        torch.Tensor: Total node contrastive loss.
    """
    
    loss_user = info_nce_loss(user_embeds_v1, user_embeds_v2, temperature)
    
    loss_item = torch.tensor(0.0, device=user_embeds_v1.device)
    if item_embeds_v1 is not None and item_embeds_v1.shape[0] > 0 and \
       item_embeds_v2 is not None and item_embeds_v2.shape[0] > 0:
        temp_for_item = temperature
        if isinstance(temperature, torch.Tensor) and temperature.shape[0] != item_embeds_v1.shape[0]:
             pass 

        loss_item = info_nce_loss(item_embeds_v1, item_embeds_v2, temp_for_item)
        
    total_loss = loss_user + lambda_coeff * loss_item
    return total_loss


