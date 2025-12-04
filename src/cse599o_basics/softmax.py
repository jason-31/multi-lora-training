import math
import torch
from einops import rearrange, parse_shape
import ipdb
from torch.nn.init import trunc_normal_


def softmax(x, dim=-1):
    """ Numerically stable softmax. """
    x = x - x.amax(dim=dim, keepdim=True)
    x = x.exp()
    x = x / x.sum(dim=dim, keepdim=True)
    return x