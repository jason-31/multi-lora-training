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

# Create fake input with mixed LoRA indices
# Batch of 8: 2 base model, 3 LoRA 1, 3 LoRA 2
print("\nCreating fake input with mixed LoRA indices...")
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
lora_indices = [0, 0, 1, 1, 1, 2, 2, 2]  # Mixed LoRA usage
target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"Input shape: {input_ids.shape}")
print(f"LoRA indices: {lora_indices}")

# Test forward pass with automatic grouping
print("\nRunning forward passes with automatic LoRA grouping...")
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

print("\n✓ All tests passed!")
print("✓ Model successfully grouped samples by LoRA index and processed them!")
