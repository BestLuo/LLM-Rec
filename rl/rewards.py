# rl/rewards.py
import torch
import torch.nn.functional as F
from lghrec_project.config import (DEVICE, THETA_FN, THETA_EASY, THETA_FP, THETA_EASY_LOW,
                    REWARD_W1, REWARD_W2, REWARD_W3, REWARD_W4,
                    TEMPERATURE_REWARD_COEFF) 

def calculate_similarity(emb1, emb2, dim=1, eps=1e-8):
    """Calculate cosine similarity"""
    return F.cosine_similarity(emb1, emb2, dim=dim, eps=eps)

def get_ideal_temperature(node_degree, device):
    """
    Calculate the ideal temperature for a node: T_ideal(d_u) = 1 / (1 + log(1 + d_u))
    Args:
        node_degree (torch.Tensor): Node degree, shape (batch_size,)
        device: torch.device
    Returns:
        torch.Tensor: Ideal temperature, shape (batch_size,)
    """
    node_degree_float = node_degree.float().to(device)
    # log(1+x) for x=0 is 0, so denominator is 1. For large x, log(1+x) increases, T decreases.
    ideal_temp = 1.0 / (1.0 + torch.log1p(node_degree_float)) # log1p(x) = log(1+x)
    return ideal_temp.unsqueeze(1) # (batch_size, 1)

def rule_based_reward_for_negative_samples(anchor_embed, positive_embed, selected_negative_embeds):
    """
    Compute rule-based reward for selected negative samples. Eq. 1, 2, 3, 4 from paper.
    Args:
        anchor_embed (torch.Tensor): Anchor user/item embedding, shape (batch_size, embed_dim).
        positive_embed (torch.Tensor): Positive sample embedding, shape (batch_size, embed_dim).
        selected_negative_embeds (torch.Tensor): M negative samples selected by policy network,
                                                 shape (batch_size, M, embed_dim).
                                                 M is NUM_NEGATIVE_SAMPLES_HGPO.
    Returns:
        torch.Tensor: Total reward for each negative sample, shape (batch_size,).
    """
    batch_size, m_neg_samples, embed_dim = selected_negative_embeds.shape
    
    # To calculate similarity, expand anchor_embed and positive_embed M times
    # anchor_embed: (B, D) -> (B, M, D)
    # positive_embed: (B, D) -> (B, M, D)
    anchor_embed_expanded = anchor_embed.unsqueeze(1).expand(-1, m_neg_samples, -1)
    positive_embed_expanded = positive_embed.unsqueeze(1).expand(-1, m_neg_samples, -1)

    # sim_anchor_neg: sim(z_u^(0), z_n^*)
    sim_anchor_neg = calculate_similarity(anchor_embed_expanded, selected_negative_embeds, dim=2) # (B, M)
    # sim_pos_neg: sim(z_u^(k), z_n^*)
    sim_pos_neg = calculate_similarity(positive_embed_expanded, selected_negative_embeds, dim=2) # (B, M)

    # 1. Reward Hard Negatives (Eq. 1)
    # theta_easy < sim(anchor, neg) < theta_FN AND sim(pos, neg) < theta_FP
    is_hard_neg = (sim_anchor_neg > THETA_EASY) & \
                  (sim_anchor_neg < THETA_FN) & \
                  (sim_pos_neg < THETA_FP)
    R_hard = torch.where(is_hard_neg, torch.tensor(REWARD_W1, device=anchor_embed.device), torch.tensor(0.0, device=anchor_embed.device))

    # 2. Punish False Negatives (Eq. 2)
    # sim(anchor, neg) >= theta_FN OR sim(pos, neg) >= theta_FP
    # Paper formulas are separate, here combined
    R_false_anchor = torch.where(sim_anchor_neg >= THETA_FN, torch.tensor(-REWARD_W2, device=anchor_embed.device), torch.tensor(0.0, device=anchor_embed.device))
    R_false_pos = torch.where(sim_pos_neg >= THETA_FP, torch.tensor(-REWARD_W3, device=anchor_embed.device), torch.tensor(0.0, device=anchor_embed.device))
    # If a sample meets both conditions, penalties are accumulated:
    R_false = R_false_anchor + R_false_pos
    # 3. Punish Easy Negatives (Eq. 3)
    # sim(anchor, neg) <= theta_easy_low
    is_easy_neg = sim_anchor_neg <= THETA_EASY_LOW
    R_easy = torch.where(is_easy_neg, torch.tensor(-REWARD_W4, device=anchor_embed.device), torch.tensor(0.0, device=anchor_embed.device))

    # Total reward for negative samples (Eq. 4)
    # Calculate for each negative sample, then sum to get the reward for this (anchor, positive) pair's negative sampling action.
    total_reward_per_neg_sample = R_hard + R_false + R_easy # (B, M)
    
    # Aggregate rewards of M negative samples, take the mean
    aggregated_neg_sample_reward = torch.mean(total_reward_per_neg_sample, dim=1) # (B,)
    
    return aggregated_neg_sample_reward


def self_adaptive_temperature_reward(chosen_temp, node_degree, device):
    """
    Compute self-adaptive temperature reward R_tau (Eq. 5).
    R_tau(u, tau_u^(t)) = -w5 * |tau_u^(t) - T_ideal(d_u)|
    Args:
        chosen_temp (torch.Tensor): Temperature coefficient chosen by policy network, shape (batch_size, 1).
        node_degree (torch.Tensor): Anchor user/item degree, shape (batch_size,).
        device: torch.device
    Returns:
        torch.Tensor: Temperature reward, shape (batch_size,).
    """
    w5 = TEMPERATURE_REWARD_COEFF
    ideal_temp = get_ideal_temperature(node_degree, device) # (batch_size, 1)
    

    # ideal_temp is (batch_size, 1)
    # Ensure shape consistency
    if chosen_temp.ndim == 1:
        chosen_temp = chosen_temp.unsqueeze(1)
        
    abs_diff = torch.abs(chosen_temp - ideal_temp)
    R_tau = -w5 * abs_diff
    return R_tau.squeeze(1) # (batch_size,)

def calculate_total_reward(anchor_embed, positive_embed, selected_negative_embeds,
                           chosen_temp, node_degree, device):
    """
    Compute total RL reward R_total = R_neg_samples + R_tau.
    Args:
        anchor_embed (torch.Tensor): (B, D)
        positive_embed (torch.Tensor): (B, D)
        selected_negative_embeds (torch.Tensor): (B, M, D)
        chosen_temp (torch.Tensor): (B, 1) 
        node_degree (torch.Tensor): (B,)
        device: torch.device
    Returns:
        torch.Tensor: Total reward, shape (batch_size,).
    """
    neg_reward = rule_based_reward_for_negative_samples(anchor_embed, positive_embed, selected_negative_embeds)
    temp_reward = self_adaptive_temperature_reward(chosen_temp, node_degree, device)
    
    total_reward = neg_reward + temp_reward
    return total_reward
