# Multi-LoRA Training

A PyTorch implementation of transformer models with multi-LoRA (Low-Rank Adaptation) support for efficient fine-tuning.

## Requirements

- Python 3.11 (recommended) or 3.9+
- CUDA-capable GPU (recommended)

## Installation

### 1. Create a Conda Environment

```bash
conda create -n 599o-project python=3.11
conda activate 599o-project
```

### 2. Install PyTorch

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Required Packages

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes PyTorch, but you should install it separately first using the command above to ensure CUDA 12.1 support.

## Project Structure

```
src/
├── cse599o_basics/          # Core implementation modules
│   ├── transformer.py       # Main transformer model
│   ├── attention.py         # Multi-head attention
│   ├── lora.py             # LoRA implementation
│   ├── linear_multi_lora.py # Multi-LoRA linear layers
│   └── ...
├── test_transformer.py      # Basic transformer tests
├── test_lora_*.py          # LoRA-specific tests
└── test_config.json        # Test configuration
```

## Usage

Run tests to verify the installation:

```bash
cd src
python test_transformer.py
```

## Features

- Custom transformer implementation with RoPE positional embeddings
- Multi-LoRA support for efficient multi-task training
- SwiGLU activation functions
- RMSNorm normalization
- Custom tokenizer with tiktoken backend
