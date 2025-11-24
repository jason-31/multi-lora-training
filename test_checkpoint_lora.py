import torch
import tempfile
import shutil
from pathlib import Path
import sys

# Add transformer_basics to path
sys.path.insert(0, str(Path(__file__).parent / 'transformer_basics'))

from transformer import Transformer

def test_checkpoint_and_lora():
    """Test saving/loading base model checkpoint and LoRA adapters."""
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / "checkpoint"
    lora_dir = Path(temp_dir) / "lora"
    
    try:
        print("=" * 60)
        print("Test: Checkpoint and LoRA Save/Load")
        print("=" * 60)
        
        # 1. Create a small transformer model
        print("\n1. Creating model...")
        model = Transformer(
            d_model=128,
            num_heads=4,
            d_ff=256,
            vocab_size=1000,
            context_length=64,
            num_layers=2,
            theta=10000.0
        )
        print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # 2. Save base model checkpoint
        print("\n2. Saving base model checkpoint...")
        model.save_checkpoint(str(checkpoint_dir))
        print(f"   ✓ Checkpoint saved to {checkpoint_dir}")
        
        # Verify checkpoint files exist
        assert (checkpoint_dir / "config.json").exists(), "config.json not found"
        assert (checkpoint_dir / "model.pt").exists(), "model.pt not found"
        print("   ✓ Checkpoint files verified")
        
        # 3. Add LoRA to specific layers
        print("\n3. Adding LoRA adapters to model...")
        lora_rank = 8
        lora_alpha = 16
        
        # Add LoRA to attention layers
        for layer_idx, layer in enumerate(model.layers):
            layer.attention.to_q.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha)
            layer.attention.to_k.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha)
            layer.attention.to_v.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha)
            layer.attention.to_out.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha)
            print(f"   ✓ Added LoRA to layer {layer_idx} attention")
        
        # Add LoRA to output linear
        model.output_linear.add_lora(lora_rank=lora_rank, lora_alpha=lora_alpha)
        print(f"   ✓ Added LoRA to output linear")
        
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad and 
                         any(name in n for n, p2 in model.named_parameters() if p2 is p 
                             for name in ['lora_A', 'lora_B']))
        print(f"   Total parameters: {total_params} ({lora_params} LoRA parameters)")
        
        # 4. Run a forward pass and store output
        print("\n4. Running forward pass with LoRA...")
        input_ids = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        with torch.no_grad():
            output_with_lora = model(input_ids)
        print(f"   ✓ Output shape: {output_with_lora.shape}")
        
        # 5. Save LoRA weights
        print("\n5. Saving LoRA weights...")
        model.save_lora(str(lora_dir))
        print(f"   ✓ LoRA saved to {lora_dir}")
        
        # Verify LoRA files exist
        assert (lora_dir / "lora.pt").exists(), "lora.pt not found"
        assert (lora_dir / "lora_config.json").exists(), "lora_config.json not found"
        print("   ✓ LoRA files verified")
        
        # 6. Load base model from checkpoint (no LoRA)
        print("\n6. Loading base model from checkpoint...")
        loaded_model = Transformer.load_checkpoint(str(checkpoint_dir))
        print(f"   ✓ Model loaded with {sum(p.numel() for p in loaded_model.parameters())} parameters")
        
        # 7. Run forward pass without LoRA
        print("\n7. Running forward pass without LoRA...")
        with torch.no_grad():
            output_without_lora = loaded_model(input_ids)
        print(f"   ✓ Output shape: {output_without_lora.shape}")
        
        # Verify outputs are different (original had LoRA)
        diff = torch.abs(output_with_lora - output_without_lora).max().item()
        print(f"   Max difference from LoRA model: {diff:.6f}")
        assert diff > 1e-6, "Outputs should differ (LoRA was active)"
        
        # 8. Load LoRA into the loaded model
        print("\n8. Loading LoRA weights into model...")
        loaded_model.load_lora(str(lora_dir))
        print("   ✓ LoRA weights loaded")
        
        # 9. Run forward pass with loaded LoRA
        print("\n9. Running forward pass with loaded LoRA...")
        with torch.no_grad():
            output_loaded_lora = loaded_model(input_ids)
        print(f"   ✓ Output shape: {output_loaded_lora.shape}")
        
        # 10. Verify outputs match original LoRA model
        print("\n10. Verifying LoRA weights match...")
        diff = torch.abs(output_with_lora - output_loaded_lora).max().item()
        print(f"    Max difference: {diff:.6f}")
        
        if diff < 1e-5:
            print("   ✓ PASS: Outputs match! LoRA loaded correctly")
        else:
            print(f"   ✗ FAIL: Outputs differ by {diff}")
            return False
        
        # 11. Test that base weights weren't affected
        print("\n11. Verifying base weights unchanged...")
        original_state = torch.load(checkpoint_dir / "model.pt")
        loaded_state = loaded_model.state_dict()
        
        # Check a few base weight parameters
        base_weight_keys = [k for k in original_state.keys() if 'lora' not in k][:5]
        all_match = True
        for key in base_weight_keys:
            if not torch.allclose(original_state[key], loaded_state[key]):
                print(f"   ✗ Base weight mismatch: {key}")
                all_match = False
        
        if all_match:
            print("   ✓ PASS: Base weights unchanged")
        else:
            print("   ✗ FAIL: Base weights were modified")
            return False
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n✓ Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    success = test_checkpoint_and_lora()
    sys.exit(0 if success else 1)
