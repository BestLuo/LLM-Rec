# models/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from lghrec_project.config import EMBEDDING_DIM, DEVICE

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_candidate_neg_samples, temp_action_continuous=True, temp_log_std_min=-20, temp_log_std_max=2):
        """
        Policy network for HGPO.
        This network receives state as input and outputs two actions:
        1. Negative sample selection: probability distribution for selecting M negative samples from candidate pool.
        2. Temperature coefficient selection: select a temperature coefficient tau for the current anchor.

        Args:
            state_dim (int): Dimension of input state. State includes: user embedding z_u, positive sample embedding z_k, candidate negative pool N_u, user degree d_u.
            num_candidate_neg_samples (int): Size of candidate negative sample pool. Policy will select from this.
            temp_action_continuous (bool): Whether temperature action is continuous.
                                          True: output mean and std of Gaussian distribution.
                                          False: output probability distribution over discrete temperature intervals.
            temp_log_std_min (float): Minimum log std for continuous temperature action (for stability).
            temp_log_std_max (float): Maximum log std for continuous temperature action.
        """
        super(PolicyNetwork, self).__init__()
        self.temp_action_continuous = temp_action_continuous
        self.temp_log_std_min = temp_log_std_min
        self.temp_log_std_max = temp_log_std_max
        self.num_candidate_neg_samples = num_candidate_neg_samples # Candidate pool size

        # Shared feature extraction layers
        self.fc_shared1 = nn.Linear(state_dim, 256)
        self.fc_shared2 = nn.Linear(256, 128)

        # Output unnormalized log probabilities (logits) for each sample in candidate pool.
        self.neg_sample_head = nn.Linear(128, num_candidate_neg_samples)

        # Temperature coefficient selection branch
        if self.temp_action_continuous:
            # Output mean mu and log_std of Gaussian distribution
            self.temp_mu_head = nn.Linear(128, 1)
            self.temp_log_std_head = nn.Linear(128, 1)
        else:
            pass
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state):
        """
        Forward pass.
        Args:
            state (torch.Tensor): Input state, shape (batch_size, state_dim).
        Returns:
            tuple: Returns different content depending on action type.
                   neg_sample_logits (torch.Tensor): shape: (batch_size, num_candidate_neg_samples).
                   temp_action_params (tuple or torch.Tensor):
                       If temp_action_continuous: (temp_mu, temp_std)
                       If not temp_action_continuous: temp_probs
        """
        shared_features = F.relu(self.fc_shared1(state))
        shared_features = F.relu(self.fc_shared2(shared_features))

        # Negative sample selection
        neg_sample_logits = self.neg_sample_head(shared_features) # (batch_size, num_candidate_neg_samples)

        # Temperature coefficient selection
        if self.temp_action_continuous:
            temp_mu = self.temp_mu_head(shared_features)
            temp_log_std = self.temp_log_std_head(shared_features)
            temp_log_std = torch.clamp(temp_log_std, self.temp_log_std_min, self.temp_log_std_max) # Clamp log_std range
            temp_std = torch.exp(temp_log_std)
            temp_action_params = (temp_mu, temp_std)
        else:
            pass
            
        return neg_sample_logits, temp_action_params

    def select_action(self, state, num_neg_to_select_m):
        """
        Select actions from state according to current policy.
        Args:
            state (torch.Tensor): Input state, shape (batch_size, state_dim).
            num_neg_to_select_m (int): Number of negative samples M to select from candidate pool.
        Returns:
            tuple:
                selected_neg_indices (torch.Tensor): shape (batch_size, num_neg_to_select_m), indices of selected negatives in candidate pool.
                selected_neg_log_probs (torch.Tensor): shape (batch_size, num_neg_to_select_m), log probs of selected negatives.
                sampled_temp (torch.Tensor): shape (batch_size, 1), sampled temperature coefficient.
                temp_log_prob (torch.Tensor): shape (batch_size, 1), log prob of sampled temperature.
                entropy (torch.Tensor): entropy of policy
        """
        neg_sample_logits, temp_action_params = self.forward(state)
        # 1. Select negative samples
        neg_probs = F.softmax(neg_sample_logits, dim=-1)
        # torch.multinomial samples M without replacement
        # selected_neg_indices shape: (batch_size, num_neg_to_select_m)
        try:
            selected_neg_indices = torch.multinomial(neg_probs, num_samples=num_neg_to_select_m, replacement=False)
        except RuntimeError as e:
            # If probabilities are close to zero or num_samples > num_classes, may error
            # Add small epsilon for numerical stability
            print(f"Multinomial sampling error: {e}. Try adding epsilon to probabilities.")
            neg_probs = neg_probs + 1e-9 # Add stability
            neg_probs = neg_probs / neg_probs.sum(dim=-1, keepdim=True) # Renormalize
            selected_neg_indices = torch.multinomial(neg_probs, num_samples=num_neg_to_select_m, replacement=False)

        # Get log probs of selected negatives
        # log_probs[b, i] = log(probs[b, selected_indices[b, i]])
        # Use gather to collect corresponding indices
        gathered_probs = torch.gather(neg_probs, dim=1, index=selected_neg_indices)
        selected_neg_log_probs = torch.log(gathered_probs + 1e-8) # (batch_size, num_neg_to_select_m)
        
        # Entropy of negative sampling action H_neg = - sum(p_j * log p_j) for each batch entry
        # Here p_j is probability of selecting j-th sample in candidate pool
        neg_entropy_per_batch = Categorical(probs=neg_probs).entropy() # (batch_size,)

        # 2. Select temperature coefficient
        if self.temp_action_continuous:
            temp_mu, temp_std = temp_action_params
            temp_dist = Normal(temp_mu, temp_std)
            sampled_temp_raw = temp_dist.sample() # (batch_size, 1)
           
            sampled_temp = torch.clamp(sampled_temp_raw, min=0.01, max=2.0) # Ensure positive and upper bound

            temp_log_prob = temp_dist.log_prob(sampled_temp_raw) # (batch_size, 1)
            temp_entropy_per_batch = temp_dist.entropy().squeeze(-1) # (batch_size,)
        else:
            pass

        total_entropy = torch.mean(neg_entropy_per_batch + temp_entropy_per_batch) # Mean entropy

        return selected_neg_indices, selected_neg_log_probs, sampled_temp, temp_log_prob, total_entropy

    def evaluate_actions(self, state, neg_action_indices, temp_action_values):
        """
        Evaluate log probability and entropy of taking specific actions in given state.
        Used for computing old policy logprob in HGPO algorithm.
        Args:
            state (torch.Tensor): Input state (batch_size, state_dim).
            neg_action_indices (torch.Tensor): Indices of negative sampling actions (batch_size, num_neg_to_select_m).
            temp_action_values (torch.Tensor): Values of temperature actions (batch_size, 1).
        Returns:
            tuple:
                selected_neg_log_probs (torch.Tensor): Log probs of selected negatives (batch_size, num_neg_to_select_m).
                temp_log_prob (torch.Tensor): Log prob of selected temperature (batch_size, 1).
                entropy (torch.Tensor): Entropy of policy (scalar).
        """
        neg_sample_logits, temp_action_params = self.forward(state)
        neg_probs = F.softmax(neg_sample_logits, dim=-1)

        # Evaluate negative sampling actions
        gathered_probs = torch.gather(neg_probs, dim=1, index=neg_action_indices)
        selected_neg_log_probs = torch.log(gathered_probs + 1e-8)
        neg_entropy_per_batch = Categorical(probs=neg_probs).entropy()

        # Evaluate temperature actions
        if self.temp_action_continuous:
            temp_mu, temp_std = temp_action_params
            temp_dist = Normal(temp_mu, temp_std)
            temp_log_prob = temp_dist.log_prob(temp_action_values)
            temp_entropy_per_batch = temp_dist.entropy().squeeze(-1)
        else:
            raise NotImplementedError()

        total_entropy = torch.mean(neg_entropy_per_batch + temp_entropy_per_batch)
        return selected_neg_log_probs, temp_log_prob, total_entropy

