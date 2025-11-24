import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_
from typing import Optional, Dict

# a Linear class that inherits from torch.nn.Module
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))

        trunc_normal_(self.weight, std=1./math.sqrt(in_features), a=-3./math.sqrt(in_features), b=3./math.sqrt(in_features))
        
        # LoRA parameters (initialized as None)
        self.lora_A = None
        self.lora_B = None
        self.lora_rank = 0
        self.lora_alpha = 1.0
        self.lora_dropout = None
        self.scaling = 1.0
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def add_lora(self, 
                 lora_rank: Optional[int] = None,
                 lora_alpha: float = 1.0,
                 lora_dropout: float = 0.0):
        """
        Add LoRA adaptation to this linear layer.
        
        Can be used in two ways:
        1. Initialize new LoRA parameters:
           add_lora(lora_rank=8, lora_alpha=16, lora_dropout=0.1)
           
        2. Load pre-trained LoRA weights:
           add_lora(lora_A=pretrained_A, lora_B=pretrained_B, lora_alpha=16)
        
        Args:
            lora_rank: Rank for LoRA decomposition (required if lora_A/lora_B not provided)
            lora_alpha: Scaling parameter for LoRA
            lora_dropout: Dropout probability for LoRA layers
            lora_A: Pre-trained LoRA A matrix (rank, in_features)
            lora_B: Pre-trained LoRA B matrix (out_features, rank)
        """
        if lora_rank is not None and lora_rank > 0:
            self.lora_rank = lora_rank
            self.lora_A = torch.nn.Parameter(torch.empty((lora_rank, self.in_features), device=self._get_device(), dtype=self._get_dtype()))
            self.lora_B = torch.nn.Parameter(torch.zeros((self.out_features, lora_rank), device=self._get_device(), dtype=self._get_dtype()))
            # Standard LoRA initialization: A with kaiming, B with zeros
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        else:
            raise ValueError("Either provide lora_rank for initialization, or both lora_A and lora_B for loading pre-trained weights")
        
        # Set alpha and scaling
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank
        
        # Setup dropout
        if lora_dropout > 0.0:
            self.lora_dropout = torch.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = None
        
        return self
    
    def merge_lora(self):
        """Merge LoRA weights into base weights permanently."""
        if self.lora_A is not None and self.lora_B is not None:
            with torch.no_grad():
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.remove_lora()
        return self
            
    def forward(self, x):
        # ipdb.set_trace()
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        x_column = rearrange(x, '... d_in -> d_in (...)')
        y_column = self.weight @ x_column + self.bias[:, None]
        
        # Add LoRA adaptation if present
        if self.lora_A is not None and self.lora_B is not None:
            lora_input = x_column
            if self.lora_dropout is not None and self.training:
                lora_input_reshaped = rearrange(lora_input, 'd_in batch -> batch d_in')
                lora_input_reshaped = self.lora_dropout(lora_input_reshaped)
                lora_input = rearrange(lora_input_reshaped, 'batch d_in -> d_in batch')
            
            # LoRA: (B @ A) @ x * scaling
            lora_output = self.lora_B @ (self.lora_A @ lora_input) * self.scaling
            y_column = y_column + lora_output
        
        y = rearrange(y_column, f'd_out ({patern}) -> {patern} d_out', **parse_shape(x, f'{patern} _'))
        return y