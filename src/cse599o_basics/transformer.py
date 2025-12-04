import math
import torch
from einops import rearrange, parse_shape
import ipdb
from .transformer_block import TransformerBlock
from .rms import RMSNorm
from .linear import Linear
from .embedding import Embedding
from .linear_multi_lora import MultiLoraLinear

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
        self.output_linear = MultiLoraLinear(d_model, vocab_size)
        
        self.lora_modules = [0]  # List to hold LoRA modules if needed
        
    def forward(self, 
                x: torch.Tensor, 
                lora_indices = None,
                lora_start_indices = None
    ):
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            lora_indices: Optional list of LoRA indices for each sample in the batch.
                         If None, all samples use LoRA index 0 (base model).
                         Length must equal batch_size.
            lora_start_indices: Optional list of starting indices for each LoRA adapter group.
                               If provided, assumes the batch is already ordered and grouped by LoRA index,
                               and skips the grouping/reordering logic.
                               If None, will compute it from lora_indices.
        
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # If lora_start_indices is provided, assume batch is already grouped and ordered
        reordered = lora_start_indices is not None and len(self.lora_modules) > 1
        if len(self.lora_modules) > 1:
            if lora_start_indices is None:
                # initialize lora_indices if not provided
                if lora_indices is None:
                    lora_indices = [0] * x.shape[0]
                    
                # group samples by their LoRA indices
                lora_groups = {}
                for batch_idx, lora_idx in enumerate(lora_indices):
                    if lora_idx not in lora_groups:
                        lora_groups[lora_idx] = []
                    lora_groups[lora_idx].append(batch_idx)
                    
                # regroup the batched input
                lora_start_indices = []
                new_x = []
                count = 0
                for lora_idx in range(len(self.lora_modules)):
                    lora_start_indices.append(count)
                    if lora_idx in lora_groups:
                        count += len(lora_groups[lora_idx])
                        for batch_idx in lora_groups[lora_idx]:
                            new_x.append(x[batch_idx])
                new_x = torch.stack(new_x, dim=0)
                # assert new_x has same shape as x
                assert new_x.shape == x.shape, f"new_x shape {new_x.shape} does not match x shape {x.shape}"
                x = new_x
            
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x, lora_start_indices)
        
        x = self.norm(x)
        logits = self.output_linear(x, lora_start_indices)
        
        # TODO: restore the original order of logits based on lora_indices, maybe don't need since we don't care about correctness for testing throughput
        if reordered:
            print("Warning: restoring original order of logits is not implemented yet.")

        return logits
    
    def get_named_linear_modules(self):
        """
        Get all linear modules with their hierarchical names.
        Returns a dictionary mapping module names to the actual modules.
        
        Example names:
            - "layers.0.attention.to_q"
            - "layers.0.attention.to_k"
            - "layers.0.attention.to_v"
            - "layers.0.attention.to_out"
            - "layers.0.ffn.w1"
            - "layers.0.ffn.w2"
            - "layers.0.ffn.w3"
            - "layers.1.attention.to_q"
            - ...
            - "output_linear"
        """
        named_modules = {}
        
        # Add transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Attention layers
            if hasattr(layer, 'attention'):
                attention = layer.attention
                if hasattr(attention, 'to_q'):
                    named_modules[f"layers.{layer_idx}.attention.to_q"] = attention.to_q
                if hasattr(attention, 'to_k'):
                    named_modules[f"layers.{layer_idx}.attention.to_k"] = attention.to_k
                if hasattr(attention, 'to_v'):
                    named_modules[f"layers.{layer_idx}.attention.to_v"] = attention.to_v
                if hasattr(attention, 'to_out'):
                    named_modules[f"layers.{layer_idx}.attention.to_out"] = attention.to_out
            
            # FFN layers
            if hasattr(layer, 'ffn'):
                ffn = layer.ffn
                if hasattr(ffn, 'w1'):
                    named_modules[f"layers.{layer_idx}.ffn.w1"] = ffn.w1
                if hasattr(ffn, 'w2'):
                    named_modules[f"layers.{layer_idx}.ffn.w2"] = ffn.w2
                if hasattr(ffn, 'w3'):
                    named_modules[f"layers.{layer_idx}.ffn.w3"] = ffn.w3
        
        # Add output linear
        if hasattr(self, 'output_linear'):
            named_modules["output_linear"] = self.output_linear
        
        return named_modules
    
    def find_modules_by_pattern(self, patterns: list[str]):
        """
        Find modules matching the given patterns.
        Supports wildcards and patterns like:
            - "layers.*.attention.to_q" (all to_q in all layers)
            - "layers.0.*" (all modules in layer 0)
            - "layers.*.ffn.*" (all ffn modules in all layers)
        
        Args:
            patterns: List of string patterns to match
        
        Returns:
            Dictionary mapping matched module names to modules
        """
        import re
        
        all_modules = self.get_named_linear_modules()
        matched_modules = {}
        
        for pattern in patterns:
            # Convert wildcard pattern to regex
            # Replace * with .*? for non-greedy matching
            regex_pattern = pattern.replace(".", r"\.").replace("*", ".*?")
            regex_pattern = f"^{regex_pattern}$"
            
            for name, module in all_modules.items():
                if re.match(regex_pattern, name):
                    matched_modules[name] = module
        
        return matched_modules
    
    def add_lora(self,
                 rank: int,
                 lora_alpha: int =16,
                 layers: list[str] = None
    ):
        """
        Add LoRA adapters to specified layers.
        
        Args:
            rank: Rank of the LoRA adapters
            layers: List of layer names or patterns to add LoRA to.
                    If None, adds LoRA to all supported linear layers.
                    Examples:
                        - ["layers.0.attention.to_q", "layers.0.attention.to_v"]
                        - ["layers.*.attention.*"] (all attention layers)
                        - ["layers.0.*"] (all layers in block 0)
        """
        lora_index = len(self.lora_modules)
        
        # Default to all linear layers if no specific layers are provided
        if layers is None:
            layers = ["layers.*.*.*", "output_linear"]
        
        # Find modules matching the patterns
        matched_modules = self.find_modules_by_pattern(layers)
        
        print(f"Found {len(matched_modules)} modules matching patterns:")
        for name in matched_modules.keys():
            print(f"  - {name}")
        
        # TODO: Implement actual LoRA addition logic
        
        for name, module in matched_modules.items():
            # assert module is instance of MultiLoraLinear
            if not isinstance(module, MultiLoraLinear):
                raise ValueError(f"Module {name} is not an instance of MultiLoraLinear.")
            
            module.add_lora(
                lora_index=lora_index,
                lora_rank=rank,
                lora_alpha=lora_alpha,
            )
        self.lora_modules.append(lora_index)
        
        print(f"Added LoRA adapters with rank {rank} to {len(matched_modules)} modules as LoRA index {lora_index}.")