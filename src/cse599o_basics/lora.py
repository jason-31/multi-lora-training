import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_

class LoraBlock(torch.nn.Module):
    """
    A single LoRA adapter block.
    """
    def __init__(self, in_features, out_features, lora_rank: int, lora_alpha: float, device=None, dtype=None):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        self.lora_A = torch.nn.Parameter(torch.empty((lora_rank, in_features), device=device, dtype=dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros((out_features, lora_rank), device=device, dtype=dtype))

        # Initialize LoRA A with Kaiming uniform and B with zeros
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Naive implementation of LoRA forward pass.
        """
        # print("Lora forward called")
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        x_column = rearrange(x, '... d_in -> d_in (...)')
        lora_y_column = self.lora_B @ (self.lora_A @ x_column) * self.scaling
        lora_y = rearrange(lora_y_column, f'd_out ({patern}) -> {patern} d_out', **parse_shape(x, f'{patern} _'))
        return lora_y