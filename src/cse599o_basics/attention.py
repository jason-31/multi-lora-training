import math
from .softmax import softmax
import torch
from einops import einsum, rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_
from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .linear_multi_lora import MultiLoraLinear

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """ Scaled Dot-Product Attention

    Args:
        query: (batch_size, ..., seq_len, d_k)
        key: (batch_size, ..., seq_len, d_k)
        value: (batch_size, ..., seq_len, d_v)
        mask: (seq_len, seq_len), this can only be a boolean mask

    Returns:
        Float[Tensor, "...", seq_len_q, d_v"]: Output of the attention mechanism.
    """
    d_k = query.size(-1)
    scores = einsum(query, key, '... q d, ... k d -> ... q k')
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output

class MultiHeadSelfAttention(torch.nn.Module):
    """ Multi-Head Self-Attention module.

    Args:
        d_model: int, # Dimensionality of the Transformer block inputs
        num_heads: int # Number of heads to use in multi-head self-attention
    """

    def __init__(
        self, 
        d_model: int, # Dimensionality of the Transformer block inputs
        num_heads: int, # Number of heads to use in multi-head self-attention
        max_seq_len: int = None,
        theta: float = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kv = d_model // num_heads
        
        self.to_q = MultiLoraLinear(d_model, d_model)
        self.to_k = MultiLoraLinear(d_model, d_model)
        self.to_v = MultiLoraLinear(d_model, d_model)
        self.to_out = MultiLoraLinear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(theta, self.d_kv, max_seq_len, device=None) if theta and max_seq_len else None

    def forward(self, x: torch.Tensor, lora_start_indices) -> torch.Tensor:
        Q = self.to_q(x, lora_start_indices)
        K = self.to_k(x, lora_start_indices)
        V = self.to_v(x, lora_start_indices)
        
        Q = rearrange(Q, '... seq_len (heads d_k) -> ... heads seq_len d_k', heads=self.num_heads)
        K = rearrange(K, '... seq_len (heads d_k) -> ... heads seq_len d_k', heads=self.num_heads)
        V = rearrange(V, '... seq_len (heads d_v) -> ... heads seq_len d_v', heads=self.num_heads)  
        
        # Apply RoPE if applicable
        if self.rope is not None:
            seq_len = x.size(-2)
            # token_positions has shape [seq_len]
            token_positions = torch.arange(seq_len, device=x.device)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # create the mask
        seq_len = x.size(-2)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()

        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, '... heads seq_len d_v -> ... seq_len (heads d_v)')
        out = self.to_out(out, lora_start_indices)
        
        return out