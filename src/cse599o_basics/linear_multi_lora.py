import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_
from .lora import LoraBlock

# a Linear class that inherits from torch.nn.Module
class MultiLoraLinear(torch.nn.Module):
    """
    A linear layer that supprorts multiple LoRA adapters.
    """
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    @property
    def device(self):
        return next(self.parameters()).device
    
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        self.loras = torch.nn.ModuleDict()  # ModuleDict to hold multiple LoRA adapters, idx -> LoraBlock

        trunc_normal_(self.weight, std=1./math.sqrt(in_features), a=-3./math.sqrt(in_features), b=3./math.sqrt(in_features))
            
    def forward(self, x, lora_start_indices):
        """
        Forward pass through the linear layer with multiple LoRA adapters.
        
        Args:
            x: Input tensor of shape (..., in_features)
            lora_start_indices: An int list containing the starting index of input in the batch for each LoRA. For example, if batch size is 4, and lora_start_indices = [0, 2], it means samples 0 and 1 use LoRA adapter 1, and samples 2 and 3 use LoRA adapter 2.
            
        Returns:
            Output tensor of shape (..., out_features)        
        """
        # ipdb.set_trace()
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        x_column = rearrange(x, '... d_in -> d_in (...)')
        y_column = self.weight @ x_column + self.bias[:, None]
        y = rearrange(y_column, f'd_out ({patern}) -> {patern} d_out', **parse_shape(x, f'{patern} _'))
        
        # Skip LoRA logic if no LoRA adapters have been added
        if len(self.loras) == 0:
            return y
        
        # separate each of the batch
        lora_grouped_inputs = {}
        for i in range(len(lora_start_indices)):
            start_idx = lora_start_indices[i]
            end_idx = lora_start_indices[i + 1] if i + 1 < len(lora_start_indices) else x.shape[0]
            lora_key = str(i)  # lora indices start from 1
            if lora_key in self.loras or i == 0:  # include base model
                if start_idx != end_idx:
                    lora_grouped_inputs[lora_key] = x[start_idx:end_idx]

        # pass each group to their corresponding LoRA adapter parallely
        # TODO: optimize this part later with pytorch stream
        for lora_key, group_input in lora_grouped_inputs.items():
            if lora_key == '0':
                continue
            lora_adapter = self.loras[lora_key]
            lora_output = lora_adapter(group_input)
            start_idx = lora_start_indices[int(lora_key)]
            end_idx = lora_start_indices[int(lora_key) + 1] if int(lora_key) + 1 < len(lora_start_indices) else x.shape[0]
            y[start_idx:end_idx] = y[start_idx:end_idx] + lora_output

        return y
    
    def add_lora(self, 
                 lora_index: int,
                 lora_rank: int,
                 lora_alpha: int = 16,
        ):
        """
        Add a LoRA adapters to this linear layer.
        """
        lora_key = str(lora_index)
        
        # check if lora_index already exists
        if lora_key in self.loras:
            raise ValueError(f"LoRA adapter with index {lora_index} already exists.")
        
        if lora_index == 0:
            raise ValueError(f"Index 0 is reserved for no LoRA adapter.")
        
        # add the lora block
        self.loras[lora_key] = LoraBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device = self.device,
            dtype = self.dtype,
        )
        