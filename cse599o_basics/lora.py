import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_

class LoraBlock(torch.nn.Module):
    """
    A single LoRA adapter block.
    """
    def __init__(self, in_features, out_features, lora_rank: int, lora_alpha: float, lora_dropout: float, device=None, dtype=None):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        self.lora_A = torch.nn.Parameter(torch.empty((lora_rank, in_features), device=device, dtype=dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros((out_features, lora_rank), device=device, dtype=dtype))
        self.lora_dropout = torch.nn.Dropout(lora_dropout) if lora_dropout > 0.0 else torch.nn.Identity()

        # Initialize LoRA A with Kaiming uniform and B with zeros
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # x shape: (..., in_features)
        result = self.lora_dropout(x)  # Apply dropout
        result = rearrange(result, '... d_in -> d_in (...)')  # (in_features, N)
        result = self.lora_A @ result  # (lora_rank, N)
        result = self.lora_B @ result  # (out_features, N)
        result = rearrange(result, f'd_out ({ " ".join([f"dim_{i}" for i in range(len(x.shape)-1)]) }) -> {" ".join([f"dim_{i}" for i in range(len(x.shape)-1)])} d_out', **parse_shape(x, f'{" ".join([f"dim_{i}" for i in range(len(x.shape)-1)])} _'))
        return result * self.scaling