import os
from typing import IO, BinaryIO

import torch
import numpy.typing as npt
import numpy as np
import json
from transformer import Transformer

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    n = dataset.shape[0]
    max_start_index = n - context_length - 1
    if max_start_index <= 0:
        raise ValueError("Dataset is too small for the requested context length.")

    start_indices = np.random.randint(0, max_start_index + 1, size=batch_size)
    inputs = np.stack(
        [dataset[i : i + context_length] for i in start_indices], axis=0
    )
    labels = np.stack(
        [dataset[i + 1 : i + context_length + 1] for i in start_indices], axis=0
    )

    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    return inputs_tensor, labels_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    obj = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": int(iteration),
    }
    # torch.save accepts path-like or file-like objects
    # create the directory if it doesn't exist
    if isinstance(out, (str, os.PathLike)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(obj, out)
    
    if isinstance(model, Transformer):
        model_config = {
            "d_model": model.d_model,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "num_layers": model.num_layers,
            "theta": model.theta,
        }
        
        # make config a json and save it in the same directory as the checkpoint
        if isinstance(out, (str, os.PathLike)):
            config_path = os.path.join(os.path.dirname(out), "model_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=4)

def init_model_from_config(config_path: str | os.PathLike):
    with open(config_path, "r") as f:
        model_config = json.load(f)
    model = Transformer(
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        theta=model_config.get("theta", None),
    )
    return model

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:

    checkpoint = torch.load(src, map_location="cpu")

    if model is None:
        config_path = os.path.join(os.path.dirname(src), "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config file not found at {config_path} and model is None."
            )
        model = init_model_from_config(config_path)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return int(checkpoint["iteration"])

def load_model_only(
    src: str | os.PathLike | BinaryIO | IO[bytes], 
    model: torch.nn.Module
) -> int:
    checkpoint = torch.load(src, map_location="cpu")

    if model is None:
        config_path = os.path.join(os.path.dirname(src), "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found at {config_path} and model is None.")        
        model = init_model_from_config(config_path)
        
    model.load_state_dict(checkpoint["model_state"])
    return model
    
