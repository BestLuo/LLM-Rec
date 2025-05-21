# models/dseg.py
import torch
import torch.nn as nn
from lghrec_project.config import EMBEDDING_DIM

class DeepSemanticEmbeddingGenerator(nn.Module):
    def __init__(self, num_items, id_embedding_dim, cot_embedding_dim, output_embedding_dim):
        """
        Deep Semantic Embedding Generator (DSEG).
        According to the paper, DSEG fuses item ID embeddings and LLM CoT-generated semantic IDs (embeddings).
        The fusion method is concatenation followed by a linear layer.

        Args:
            num_items (int): Total number of items.
            id_embedding_dim (int): Dimension of item ID embeddings.
            cot_embedding_dim (int): Dimension of CoT semantic embeddings.
            output_embedding_dim (int): Output embedding dimension of DSEG (usually equal to GNN embedding dimension).
        """
        super(DeepSemanticEmbeddingGenerator, self).__init__()
        
        # Fusion layer: linear transformation (ID_dim + CoT_dim) -> output_dim
        self.fusion_layer = nn.Linear(id_embedding_dim + cot_embedding_dim, output_embedding_dim)
        
        self.id_embedding_dim = id_embedding_dim
        self.cot_embedding_dim = cot_embedding_dim
        self.output_embedding_dim = output_embedding_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        if self.fusion_layer.bias is not None:
            nn.init.zeros_(self.fusion_layer.bias)

    def forward(self, item_id_embeddings, item_cot_embeddings):
        """
        Forward pass.

        Args:
            item_id_embeddings (torch.Tensor): Item ID embeddings
            item_cot_embeddings (torch.Tensor): Item CoT semantic embeddings, precomputed and loaded.

        Returns:
            torch.Tensor: Fused item embeddings.
        """
        if item_id_embeddings.shape[0] != item_cot_embeddings.shape[0]:
            raise ValueError(f"ID embeddings and CoT embeddings count mismatch: "
                             f"{item_id_embeddings.shape[0]} vs {item_cot_embeddings.shape[0]}")
        
        # Concatenate ID and CoT embeddings
        concatenated_embeddings = torch.cat([item_id_embeddings, item_cot_embeddings], dim=1)
        
        # Pass through fusion layer
        fused_embeddings = self.fusion_layer(concatenated_embeddings)
        
        return fused_embeddings

