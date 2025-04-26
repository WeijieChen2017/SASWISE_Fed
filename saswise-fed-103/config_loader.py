import json
import os
import torch
from datetime import datetime

def get_device():
    """Determines and prints the execution device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=========================================")
    print(f"EXECUTION INFORMATION:")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"=========================================")
    return device

def load_config(config_path="config.json"):
    """Loads configuration from a JSON file."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration loaded.")
    # Optionally print loaded config parameters here if desired
    # num_clients = config["training"]["num_clients"]
    # num_rounds = config["training"]["num_rounds"]
    # epochs = config["training"]["epochs"]
    # batch_size = config["training"]["batch_size"]
    # learning_rate = config["training"]["learning_rate"]
    # print(f"  Clients: {num_clients}, Rounds: {num_rounds}, Epochs: {epochs}")
    # print(f"  Batch size: {batch_size}, Learning rate: {learning_rate}")
    return config

def setup_logging(config, config_path="config.json"):
    """Sets up the logging directory and saves the configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/run_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logs will be stored in: {log_dir}")

    # Save config to the log directory
    config_save_path = os.path.join(log_dir, os.path.basename(config_path))
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")

    return log_dir, timestamp 