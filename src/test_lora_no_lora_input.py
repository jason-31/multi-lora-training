import torch
from cse599o_basics.transformer import Transformer
from test_utils import load_test_config

# Load test configuration
config = load_test_config()
batch_size = config["batch_size"]
seq_len = config["seq_len"]
d_model = config["d_model"]
num_heads = config["num_heads"]
d_ff = config["d_ff"]
vocab_size = config["vocab_size"]
context_length = config["context_length"]
num_layers = config["num_layers"]
theta = config["theta"]
lora_rank = config["lora_rank"]
lora_alpha = config["lora_alpha"]

print("Initializing transformer...")
model = Transformer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    vocab_size=vocab_size,
    context_length=context_length,
    num_layers=num_layers,
    theta=theta
)

print(f"Model parameters before LoRA: {sum(p.numel() for p in model.parameters()):,}")

# Add two LoRA adapters
print("\nAdding LoRA adapter 1...")
model.add_lora(rank=lora_rank, lora_alpha=lora_alpha)

print("\nAdding LoRA adapter 2...")
model.add_lora(rank=lora_rank, lora_alpha=lora_alpha)

print(f"\nModel parameters after LoRA: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of LoRA modules: {len(model.lora_modules) -1}")

# Freeze base model and enable LoRA gradients
print("\nFreezing base model and enabling LoRA gradients...")
model.set_base_model_grad(False)
model.set_all_loras_grad(True)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# Create fake input with mixed LoRA indices
# Batch of 8: All base model (no LoRA usage)
print("\nCreating fake input without using LoRA (all base model)...")
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
lora_indices = None
target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"Input shape: {input_ids.shape}")
print(f"LoRA indices: {lora_indices}")

# Store initial base model weights for verification
print("\nStoring initial base model weights for verification...")
initial_weights = {}
for name, param in model.named_parameters():
    if 'loras' not in name:  # Base model parameters only
        initial_weights[name] = param.data.clone()

# Store initial LoRA weights for verification
print("Storing initial LoRA weights for verification...")
initial_lora_weights = {}
for name, param in model.named_parameters():
    if 'loras' in name:  # LoRA parameters only
        initial_lora_weights[name] = param.data.clone()

# Test forward pass with automatic grouping
print("\nRunning forward passes without using LoRA (automatic grouping)...")
for i in range(3):
    print(f"\nPass {i+1}:")
    
    # Forward pass - let the model handle grouping
    logits = model(input_ids, lora_indices=lora_indices)
    print(f"  Output shape: {logits.shape}")
    
    # Compute loss
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target_ids.view(-1)
    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"  Backward pass completed")
    
    # Clear gradients
    model.zero_grad()

# Verify base model weights haven't changed
print("\nVerifying base model weights haven't changed...")
weights_unchanged = True
for name, param in model.named_parameters():
    if 'loras' not in name:  # Base model parameters only
        if not torch.allclose(param.data, initial_weights[name]):
            print(f"  ✗ Weight changed: {name}")
            weights_unchanged = False

if weights_unchanged:
    print("  ✓ All base model weights unchanged!")
else:
    raise AssertionError("Base model weights changed during training!")

# Verify LoRA weights have NOT changed (since they weren't used)
print("\nVerifying LoRA weights have NOT changed (since they weren't used)...")
lora_weights_unchanged = True
for name, param in model.named_parameters():
    if 'loras' in name:  # LoRA parameters only
        if not torch.allclose(param.data, initial_lora_weights[name]):
            print(f"  ✗ LoRA weight changed: {name}")
            lora_weights_unchanged = False

if lora_weights_unchanged:
    print("  ✓ All LoRA weights unchanged (as expected)!")
else:
    raise AssertionError("LoRA weights changed even though they weren't used!")

print("\n✓ All tests passed!")
print("✓ Model successfully processed batch using only base model (no LoRA adapters used)!")
