import math
import torch
from einops import rearrange, parse_shape, repeat
import ipdb
from torch.nn.init import trunc_normal_

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: Theta value for RoPE.
            d_k: Dimension of query/key vectors (should be even).
            max_seq_len: Maximum sequence length that will be inputted.
            device: torch.device | None. Device to store the buffers on.
        """
        
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Create the cos and sin buffers
        # ipdb.set_trace()
        inv_freq = 1. / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos_seq = torch.arange(max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum('i , j -> i j', pos_seq, inv_freq)
        self.register_buffer('cos', torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(-1))  # Shape: (1, max_seq_len, d_k/2, 1)
        self.register_buffer('sin', torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(-1))  # Shape: (1, max_seq_len, d_k/2, 1)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.
        Notes:
        - Accept x with an arbitrary number of batch dimensions.
        - token_positions has shape (seq_len) and gives absolute
        positions per token along the sequence dimension.
        - Use token_positions to slice (precomputed) cos/sin tensors
        along the sequence dimension.
        """

        # ipdb.set_trace()

        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 2)])
        x_reshaped = rearrange(x, f'{patern} seq_len (d_k1 d_k2) -> {patern} seq_len d_k1 d_k2', d_k1=self.d_k//2, d_k2=2)
        token_positions = repeat(token_positions, f'seq_len -> {patern} seq_len', **{f'dim_{i}': x.shape[i] for i in range(dims - 2)})
        
        cos_slice = self.cos[:, token_positions, :, :]  # Shape: (..., seq_len, d_k/2, 1)
        sin_slice = self.sin[:, token_positions, :, :]  # Shape: (..., seq_len, d_k/2, 1)
        
        x1 = x_reshaped[..., :, :, 0:1]  # Shape: (..., seq_len, d_k/2, 1)
        x2 = x_reshaped[..., :, :, 1:2]  # Shape: (..., seq_len, d_k/2, 1)
        
        x_rotated_1 = x1 * cos_slice - x2 * sin_slice
        x_rotated_2 = x1 * sin_slice + x2 * cos_slice
        
        x_rotated = torch.cat([x_rotated_1, x_rotated_2], dim=-1)  # Shape: (..., seq_len, d_k/2, 2)
        x_rotated = x_rotated[0]
        out = rearrange(x_rotated, f'{patern} seq_len d_k1 d_k2 -> {patern} seq_len (d_k1 d_k2)')
        
        return out