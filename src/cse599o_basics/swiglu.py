import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_
from .linear import Linear
from .linear_multi_lora import MultiLoraLinear

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the SiLU activation function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying SiLU activation.
    """
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU module.
        
        Args:
            d_model: int 
                Hidden dimension of the model
            d_ff: int 
                Dimension of the feedforward layer
            device: torch.device | None
                Device to store the parameters on
            dtype: torch.dtype | None 
                Data type of the parameters
        """
        
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        """
        w1_weight: Float[Tensor, " d_ff d_model"]
        w2_weight: Float[Tensor, " d_model d_ff"]
        w3_weight: Float[Tensor, " d_ff d_model"]
        """
        self.w1 = MultiLoraLinear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = MultiLoraLinear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = MultiLoraLinear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, in_features: torch.Tensor, lora_start_indices) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        
        in_features: Tensor [... d_model]
        lora_start_indices: List of starting indices for each LoRA adapter in the batch
        """
        # ipdb.set_trace()
        x1 = self.w1(in_features, lora_start_indices) #[... d_ff]
        silu_x1 = silu(x1) #[... d_ff]
        x3 = self.w3(in_features, lora_start_indices) #[... d_ff]
        silu_x1_x3 = silu_x1 * x3 #[... d_ff]
        out_features = self.w2(silu_x1_x3, lora_start_indices) #[... d_model]        
        
        return out_features
        