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

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create fake input and target
print("\nCreating fake input and target...")
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Test forward pass without LoRA
print("\nRunning forward passes without LoRA...")
for i in range(3):
    print(f"\nPass {i+1}:")
    
    # Forward pass
    logits = model(input_ids)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Compute loss (cross entropy)
    # Reshape for loss computation: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target_ids.view(-1)
    loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"  Backward pass completed")
    
    # Clear gradients for next iteration
    model.zero_grad()

print("\n✓ All tests passed!")
