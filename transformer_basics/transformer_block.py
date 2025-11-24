import math
from softmax import softmax
from einops import einsum, rearrange, parse_shape
import ipdb
import torch
from torch.nn.init import trunc_normal_
from attention import MultiHeadSelfAttention
from rms import RMSNorm
from linear import Linear
from swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):

    def __init__(
        self, 
        d_model: int, # Dimensionality of the Transformer block inputs
        num_heads: int, # Number of heads to use in multi-head self-attention
        d_ff: int, # Dimensionality of the position-wise feed-forward inner layer
        max_seq_len=None,
        theta=None
    ):
        
        """ 
        Args:
            d_model: int # Dimensionality of the Transformer block inputs
            num_heads: int # Number of heads to use in multi-head self-attention
            d_ff: int # Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ipdb.set_trace()
        x_norm_1 = self.norm_1(x)
        attn_output = self.attention(x_norm_1)
        x = x + attn_output
        
        x_norm_2 = self.norm_2(x)
        ffn_output = self.ffn(x_norm_2)
        out = x + ffn_output
        
        return out