"""
Utility functions for testing.
"""
import json
from pathlib import Path


def load_test_config(config_path="test_config.json"):
    """
    Load test hyperparameters from a JSON configuration file.
    
    Args:
        config_path: Path to the configuration file relative to the src directory.
    
    Returns:
        Dictionary containing test configuration parameters.
    """
    # Get the directory where this file is located (src/)
    src_dir = Path(__file__).parent
    config_file = src_dir / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config
