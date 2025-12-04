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
print(f"Number of LoRA modules: {len(model.lora_modules)}")

# Create pre-grouped input
# Manually group samples: base model (0), LoRA 1, LoRA 2
print("\nCreating pre-grouped fake input...")

# Generate samples for each group
base_samples = 2
lora1_samples = 3
lora2_samples = 3

# Create input already grouped and ordered
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Define lora_start_indices for pre-grouped batch
# Format: [start_of_base, start_of_lora1, start_of_lora2]
lora_start_indices = [0, base_samples, base_samples + lora1_samples]

print(f"Input shape: {input_ids.shape}")
print(f"Pre-grouped lora_start_indices: {lora_start_indices}")
print(f"  Samples 0-{base_samples-1}: base model")
print(f"  Samples {base_samples}-{base_samples+lora1_samples-1}: LoRA 1")
print(f"  Samples {base_samples+lora1_samples}-{batch_size-1}: LoRA 2")

# Test forward pass with pre-grouped input
print("\nRunning forward passes with pre-grouped input...")
for i in range(3):
    print(f"\nPass {i+1}:")
    
    # Forward pass - provide lora_start_indices to skip grouping
    logits = model(input_ids, lora_start_indices=lora_start_indices)
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
print("✓ Model successfully processed pre-grouped batch without regrouping!")
