import math
import torch
from einops import rearrange, parse_shape
import ipdb
import os
import json
from pathlib import Path
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
    
    def save_checkpoint(self, save_dir: str):
        """
        Save model checkpoint including configuration and weights.
        
        Args:
            save_dir: Directory path to save the checkpoint
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'num_layers': self.num_layers,
            'theta': self.theta,
        }
        
        config_path = save_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save model weights (excluding LoRA parameters)
        state_dict = self.state_dict()
        # Filter out LoRA parameters
        base_state_dict = {k: v for k, v in state_dict.items() if 'lora_A' not in k and 'lora_B' not in k}
        
        model_path = save_path / 'model.pt'
        torch.save(base_state_dict, model_path)
        
        print(f"Checkpoint saved to {save_dir} (LoRA parameters excluded)")
        return save_dir
    
    @classmethod
    def load_checkpoint(cls, load_dir: str, device=None, dtype=None):
        """
        Load model checkpoint from directory.
        
        Args:
            load_dir: Directory path containing the checkpoint
            device: Device to load model onto (optional)
            dtype: Data type for model parameters (optional)
            
        Returns:
            Loaded Transformer model
        """
        load_path = Path(load_dir)
        
        # Load configuration
        config_path = load_path / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            vocab_size=config['vocab_size'],
            context_length=config['context_length'],
            num_layers=config['num_layers'],
            theta=config.get('theta', None),
        )
        
        # Load model weights
        model_path = load_path / 'model.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights file not found at {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu')
        # Load with strict=False to allow missing LoRA parameters
        model.load_state_dict(state_dict, strict=False)
        
        # Move to specified device/dtype if provided
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)
        
        print(f"Checkpoint loaded from {load_dir} (base weights only)")
        return model
    
    def save_lora(self, save_dir: str):
        """
        Save only LoRA parameters from all Linear layers in the model.
        
        Args:
            save_dir: Directory path to save the LoRA weights
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Collect LoRA parameters and metadata
        lora_state = {}
        lora_config = {}
        
        state_dict = self.state_dict()
        for name, param in state_dict.items():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state[name] = param
        
        # Collect LoRA metadata from all Linear modules
        for name, module in self.named_modules():
            if isinstance(module, Linear) and module.lora_A is not None:
                lora_config[name] = {
                    'lora_rank': module.lora_rank,
                    'lora_alpha': module.lora_alpha,
                    'lora_dropout': module.lora_dropout.p if module.lora_dropout is not None else 0.0
                }
        
        if not lora_state:
            print("No LoRA parameters found in the model")
            return None
        
        # Save LoRA weights
        lora_path = save_path / 'lora.pt'
        torch.save(lora_state, lora_path)
        
        # Save LoRA configuration
        config_path = save_path / 'lora_config.json'
        with open(config_path, 'w') as f:
            json.dump(lora_config, f, indent=2)
        
        print(f"LoRA weights saved to {save_dir} ({len(lora_state)} parameters)")
        return save_dir
    
    def load_lora(self, load_dir: str):
        """
        Load LoRA parameters into all Linear layers in the model.
        
        Args:
            load_dir: Directory path containing the LoRA weights
        """
        load_path = Path(load_dir)
        
        # Load LoRA configuration
        config_path = load_path / 'lora_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"LoRA configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            lora_config = json.load(f)
        
        # Load LoRA weights
        lora_path = load_path / 'lora.pt'
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights file not found at {lora_path}")
        
        lora_state = torch.load(lora_path, map_location='cpu')
        
        # First, add LoRA to all Linear modules using the config
        for name, module in self.named_modules():
            if isinstance(module, Linear) and name in lora_config:
                config = lora_config[name]
                module.add_lora(
                    lora_rank=config['lora_rank'],
                    lora_alpha=config['lora_alpha'],
                    lora_dropout=config['lora_dropout']
                )
        
        # Then load the LoRA state dict
        # We need to load only the LoRA parameters, so use strict=False
        current_state = self.state_dict()
        current_state.update(lora_state)
        self.load_state_dict(current_state, strict=False)
        
        print(f"LoRA weights loaded from {load_dir} ({len(lora_state)} parameters)")
        return self