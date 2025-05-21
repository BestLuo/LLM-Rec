# rl/replay_buffer.py
import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Experience replay buffer.
        Stores tuples of (state, neg_action_indices, temp_action_value, neg_log_prob, temp_log_prob, reward, next_state, done, group_idx).

        Args:
            capacity (int): Maximum buffer capacity.
        """
        self.capacity = capacity
        self.buffer = [] # Store trajectory data as a list
        self.position = 0

    def push(self, state, neg_action_indices, temp_action_value,
             neg_action_log_prob, temp_action_log_prob,
             reward, anchor_degree, group_idx
            ):
        """
        Add an experience to the buffer.
        Args:
            state (torch.Tensor): Current state.
            neg_action_indices (torch.Tensor): Selected negative sample indices.
            temp_action_value (torch.Tensor): Selected temperature value.
            neg_action_log_prob (torch.Tensor): Log probability of negative actions (from old policy).
            temp_action_log_prob (torch.Tensor): Log probability of temperature action (from old policy).
            reward (torch.Tensor): Immediate reward.
            anchor_degree (torch.Tensor): Degree of anchor node.
            group_idx (torch.Tensor): Group index of anchor node.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        experience = (
            state.cpu().detach(),
            neg_action_indices.cpu().detach(),
            temp_action_value.cpu().detach(),
            neg_action_log_prob.cpu().detach(),
            temp_action_log_prob.cpu().detach(),
            reward.cpu().detach(),
            anchor_degree.cpu().detach(),
            group_idx.cpu().detach()
        )
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            tuple: Tensors of states, actions, rewards, etc.
        """
        if batch_size > len(self.buffer):
            # If requested batch_size is greater than current buffer size,
            batch_size = len(self.buffer)
            print(f"Warning: Requested sample size {batch_size} exceeds buffer size {len(self.buffer)}. Returning all available data.")

        batch = random.sample(self.buffer, batch_size)
        
        states, neg_indices, temp_vals, neg_log_probs, temp_log_probs, rewards, degrees, group_indices = zip(*batch)
        
        return (
            torch.stack(states),
            torch.stack(neg_indices),
            torch.stack(temp_vals),
            torch.stack(neg_log_probs),
            torch.stack(temp_log_probs),
            torch.stack(rewards),
            torch.stack(degrees),
            torch.stack(group_indices)
        )

    def get_all_data(self, device):
        """
        Get all data in the buffer and move to the specified device.
        Returns:
            tuple: Tensors of all states, actions, rewards, etc.
        """
        if not self.buffer:
            return None

        states, neg_indices, temp_vals, neg_log_probs, temp_log_probs, rewards, degrees, group_indices = zip(*self.buffer)
        
        valid_experiences = [exp for exp in self.buffer if exp is not None]
        if not valid_experiences:
            return None
        
        states, neg_indices, temp_vals, neg_log_probs, temp_log_probs, rewards, degrees, group_indices = zip(*valid_experiences)

        return (
            torch.cat(states, dim=0).to(device) if isinstance(states[0], torch.Tensor) else torch.tensor(states, device=device).to(device), 
            torch.cat(neg_indices, dim=0).to(device) if isinstance(neg_indices[0], torch.Tensor) else torch.tensor(neg_indices, device=device).to(device),
            torch.cat(temp_vals, dim=0).to(device) if isinstance(temp_vals[0], torch.Tensor) else torch.tensor(temp_vals, device=device).to(device),
            torch.cat(neg_log_probs, dim=0).to(device) if isinstance(neg_log_probs[0], torch.Tensor) else torch.tensor(neg_log_probs, device=device).to(device),
            torch.cat(temp_log_probs, dim=0).to(device) if isinstance(temp_log_probs[0], torch.Tensor) else torch.tensor(temp_log_probs, device=device).to(device),
            torch.cat(rewards, dim=0).to(device) if isinstance(rewards[0], torch.Tensor) else torch.tensor(rewards, device=device).to(device),
            torch.cat(degrees, dim=0).to(device) if isinstance(degrees[0], torch.Tensor) else torch.tensor(degrees, device=device).to(device),
            torch.cat(group_indices, dim=0).to(device) if isinstance(group_indices[0], torch.Tensor) else torch.tensor(group_indices, device=device).to(device)
        )


    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0
        print("Replay buffer cleared.")

    def __len__(self):
        return len(self.buffer)

