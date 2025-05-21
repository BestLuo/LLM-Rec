# rl/hgpo_utils.py
import torch
from lghrec_project.config import DEVICE

def get_ideal_temperature_for_degrees(node_degrees):
    """
    Compute ideal temperature for a batch of nodes: T_ideal(d_u) = 1 / (1 + log(1 + d_u))
    Args:
        node_degrees (torch.Tensor): Node degrees, shape (batch_size,)
    Returns:
        torch.Tensor: Ideal temperature, shape (batch_size, 1)
    """
    node_degrees_float = node_degrees.float().to(DEVICE)
    # log1p(x) = log(1+x)
    # Ensure denominator is not zero (log1p(0)=0, 1+0=1. For degree=0)
    # clamp(min=0) to handle potential negative degrees if data is noisy, though degrees should be non-negative.
    ideal_temp = 1.0 / (1.0 + torch.log1p(node_degrees_float.clamp(min=0)))
    return ideal_temp.unsqueeze(1) # (batch_size, 1)

def construct_rl_state(anchor_embed, positive_embed, candidate_negative_pool_embeds, anchor_degree):
    """
    Construct RL state representation.
    Here we concatenate these features into a state vector.
    Candidate negative pool is represented by mean embedding.

    Args:
        anchor_embed (torch.Tensor): Anchor embedding (B, D_emb)
        positive_embed (torch.Tensor): Positive sample embedding (B, D_emb)
        candidate_negative_pool_embeds (torch.Tensor): Candidate negative pool embeddings (B, D_emb)
        anchor_degree (torch.Tensor): Anchor degree (B, 1)
    Returns:
        torch.Tensor: State vector (B, D_state)
    """
    if candidate_negative_pool_embeds.ndim == 3: # (B, Num_candidates, D_emb)
        # Use mean pooling to aggregate candidate negative pool
        agg_negative_pool_embed = torch.mean(candidate_negative_pool_embeds, dim=1) # (B, D_emb)
    elif candidate_negative_pool_embeds.ndim == 2: # Already aggregated (B, D_emb)
        agg_negative_pool_embed = candidate_negative_pool_embeds
    else:
        raise ValueError("Dimension of candidate negative embeddings is incorrect.")

    if anchor_degree.ndim == 1:
        anchor_degree = anchor_degree.unsqueeze(1) # (B, 1)

    # Concatenate all state features
    # (B, D_emb), (B, D_emb), (B, D_emb), (B, 1)
    state = torch.cat([anchor_embed, 
                       positive_embed, 
                       agg_negative_pool_embed, 
                       anchor_degree.float()], dim=1)
    return state

