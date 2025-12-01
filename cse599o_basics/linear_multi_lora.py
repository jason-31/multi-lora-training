import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_
from lora import LoraBlock

# a Linear class that inherits from torch.nn.Module
class MultiLoraLinear(torch.nn.Module):
    """
    A linear layer that supprorts multiple LoRA adapters.
    """
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.loras = {}  # Dictionary to hold multiple LoRA adapters, idx -> LoraBlock

        trunc_normal_(self.weight, std=1./math.sqrt(in_features), a=-3./math.sqrt(in_features), b=3./math.sqrt(in_features))
            
    def forward(self, x):
        # ipdb.set_trace()
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        x_column = rearrange(x, '... d_in -> d_in (...)')
        y_column = self.weight @ x_column + self.bias[:, None]
        y = rearrange(y_column, f'd_out ({patern}) -> {patern} d_out', **parse_shape(x, f'{patern} _'))
        return y
    
    def add_lora(self, 
                 lora_index: int,
                 lora_rank: int = None,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.0,
        ):
        """
        Add LoRA adapters to this linear layer.
        
        """
        raise NotImplementedError("LoRA addition not implemented in this snippet.")