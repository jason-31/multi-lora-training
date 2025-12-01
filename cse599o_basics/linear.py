import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_

# a Linear class that inherits from torch.nn.Module
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))

        trunc_normal_(self.weight, std=1./math.sqrt(in_features), a=-3./math.sqrt(in_features), b=3./math.sqrt(in_features))
            
    def forward(self, x):
        # ipdb.set_trace()
        dims = len(x.shape)
        patern = ' '.join([f'dim_{i}' for i in range(dims - 1)])
        x_column = rearrange(x, '... d_in -> d_in (...)')
        y_column = self.weight @ x_column + self.bias[:, None]
        y = rearrange(y_column, f'd_out ({patern}) -> {patern} d_out', **parse_shape(x, f'{patern} _'))
        return y