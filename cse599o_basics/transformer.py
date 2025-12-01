import math
import torch
from einops import rearrange, parse_shape
import ipdb
from transformer_block import TransformerBlock
from rms import RMSNorm
from linear import Linear
from embedding import Embedding

class Transformer(torch.nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 theta=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.theta = theta
        self.profile_mode = False  # Flag to enable timing breakdown
        
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, return_timings: bool = False):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            return_timings: If True, return a dict with timing breakdowns
            
        Returns:
            If return_timings is False: logits tensor
            If return_timings is True: (logits, timings_dict)
        """
        if return_timings and torch.cuda.is_available():
            return self._forward_with_timings(x)
        else:
            return self._forward_regular(x)
    
    def _forward_regular(self, x: torch.Tensor) -> torch.Tensor:
        """Regular forward pass without timing."""
        x = self.token_embedding(x) 
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.output_linear(x)

        return logits
    
    def _forward_with_timings(self, x: torch.Tensor):
        """Forward pass with detailed timing breakdown using CUDA events."""
        timings = {
            'embedding': 0.0,
            'attention_total': 0.0,
            'ffn_total': 0.0,
            'norm_total': 0.0,
            'output_linear': 0.0,
            'attention_per_layer': [],
            'ffn_per_layer': []
        }
        
        # Time embedding
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.token_embedding(x)
        end.record()
        torch.cuda.synchronize()
        timings['embedding'] = start.elapsed_time(end) / 1000.0
        
        # Time each layer
        for layer in self.layers:
            # Time attention
            attn_start = torch.cuda.Event(enable_timing=True)
            attn_end = torch.cuda.Event(enable_timing=True)
            
            x_norm_1 = layer.norm_1(x)
            attn_start.record()
            attn_output = layer.attention(x_norm_1)
            attn_end.record()
            x = x + attn_output
            
            torch.cuda.synchronize()
            attn_time = attn_start.elapsed_time(attn_end) / 1000.0
            timings['attention_per_layer'].append(attn_time)
            timings['attention_total'] += attn_time
            
            # Time FFN
            ffn_start = torch.cuda.Event(enable_timing=True)
            ffn_end = torch.cuda.Event(enable_timing=True)
            
            x_norm_2 = layer.norm_2(x)
            ffn_start.record()
            ffn_output = layer.ffn(x_norm_2)
            ffn_end.record()
            x = x + ffn_output
            
            torch.cuda.synchronize()
            ffn_time = ffn_start.elapsed_time(ffn_end) / 1000.0
            timings['ffn_per_layer'].append(ffn_time)
            timings['ffn_total'] += ffn_time
        
        # Time final norm and output
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = self.norm(x)
        logits = self.output_linear(x)
        end.record()
        torch.cuda.synchronize()
        timings['output_linear'] = start.elapsed_time(end) / 1000.0
        
        return logits, timings