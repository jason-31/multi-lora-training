import math
import torch
from einops import rearrange, parse_shape, reduce
import ipdb
from torch.nn.init import trunc_normal_


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model: int 
                Hidden dimension of the model
            eps: float = 1e-5 
                Epsilon value for numerical stability
            device: torch.device | None
                Device to store the parameters on
            dtype: torch.dtype | None 
                Data type of the parameters
        """
        
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # ipdb.set_trace()
        x_squared = x**2
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        rms_mean = reduce(x_squared, f'{patern} d_model -> {patern} 1', 'mean')
        rms = torch.sqrt(rms_mean + self.eps)
        
        x_norm = x / rms
        x_scaled = x_norm * self.weight
        
        return x_scaled.to(in_dtype)