# rl/group_manager.py
import torch
import numpy as np
from lghrec_project.config import NUM_GROUPS_HGPO, DEVICE # NUM_GROUPS_HGPO should be 5

class GroupManager:
    def __init__(self, num_groups=NUM_GROUPS_HGPO, device=DEVICE):
        """
        Manage node group partitioning based on node degree.

        Args:
            num_groups (int): Number of groups K to partition.
        """
        self.expected_num_groups = 5 
        self.num_actual_groups = self.expected_num_groups
        self.device = device
        # deg < 10 (group 0)
        # 10 <= deg < 30 (group 1)
        # 30 <= deg < 70 (group 2)
        # 70 <= deg < 100 (group 3)
        # deg >= 100 (group 4)
        self.fixed_boundaries_tensor = torch.tensor([10.0, 30.0, 70.0, 100.0], dtype=torch.float32, device=self.device)
        self.group_boundaries = self.fixed_boundaries_tensor.cpu().numpy()

    def build_group_boundaries(self, node_degrees_tensor=None):
        """
        Set fixed group boundaries.
        """
        pass
        # Group 0: deg < 10
        # Group 1: 10 <= deg < 30
        # Group 2: 30 <= deg < 70
        # Group 3: 70 <= deg < 100
        # Group 4: deg >= 100

    def get_group_idx(self, node_degree_value):
        """
        Get group index (0 to K-1) for a given node degree value.
        Args:
            node_degree_value (torch.Tensor): A batch of node degrees.
        Returns:
            torch.Tensor: Group indices, same shape as node_degree_value.
        """
        if not isinstance(node_degree_value, torch.Tensor):
            node_degree_value = torch.tensor(node_degree_value, dtype=torch.float32)
        
        node_degree_value = node_degree_value.to(self.device).float()

        # deg < 10                     -> bucketize result 0 (Group 0)
        # 10 <= deg < 30               -> bucketize result 1 (Group 1)
        # 30 <= deg < 70               -> bucketize result 2 (Group 2)
        # 70 <= deg < 100              -> bucketize result 3 (Group 3)
        # deg >= 100                   -> bucketize result 4 (Group 4)
        
        group_indices = torch.bucketize(node_degree_value, self.fixed_boundaries_tensor, right=False)
        
        return group_indices.long() 

