import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.
        This function should accept the following parameters:
        num_embeddings: int
        Size of the vocabulary
        embedding_dim: int
        9Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None
        Device to store the parameters on
        dtype: torch.dtype | None = None
        Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        trunc_normal_(self.weight, std=1., a=-3., b=3.)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        # ipdb.set_trace()
        return self.weight[token_ids]