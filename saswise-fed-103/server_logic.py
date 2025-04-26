import os
import time
import json
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, parameters_to_ndarrays

from models import ResNet20, set_weights, get_weights

# Try to import log from utils2, otherwise create a basic logger
try:
    from utils2 import log
except ImportError:
    from logging import INFO, ERROR
    # Simple print logger if utils2.log is not available
    def log(level, msg, *args):
        if level >= INFO: # Basic filtering
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg % args}")

def evaluate_fn_factory(testset, test_subsets, device, num_rounds, log_dir, client_metrics_ref):
    """
    Factory function for creating server-side evaluation function.
    
    Args:
        testset: The global test dataset
        test_subsets: Dictionary of named test subsets for specialized evaluation
        device: Device to use for evaluation (CPU/GPU)
        num_rounds: Total number of federation rounds
        log_dir: Directory to save evaluation logs
        client_metrics_ref: Reference to shared metrics dictionary
    
    Returns:
        evaluate_fn: Server-side evaluation function
    """
    
    def evaluate_fn(server_round, parameters, config):
        """Evaluate the global model on the test dataset and subsets."""
        log(INFO, f"[Server] Evaluating global model on round {server_round}")
        
        # Create model and load parameters
        model = ResNet20().to(device)
        set_weights(model, parameters)
        model.eval()
        
        # Define criterion
        criterion = nn.CrossEntropyLoss()
        
        # Create a function to evaluate on a dataset
        def evaluate_dataset(dataset, name="global"):
            """Helper function to evaluate on a specific dataset."""
            dataloader = DataLoader(dataset, batch_size=128)
            correct, total, loss = 0, 0, 0.0
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss += criterion(outputs, targets).item()
                    
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = 100.0 * correct / total
            avg_loss = loss / len(dataloader)
            
            return {
                "accuracy": float(accuracy),
                "loss": float(avg_loss),
                "dataset_size": total,
                "name": name
            }
        
        # Record start time
        start_time = time.time()
        
        # Evaluate on global test set
        global_metrics = evaluate_dataset(testset, "global")
        
        # Evaluate on each test subset
        subset_metrics = {}
        for name, subset in test_subsets.items():
            subset_metrics[name] = evaluate_dataset(subset, name)
        
        # Record evaluation time
        eval_time = time.time() - start_time
        
        # Create result dictionary
        results = {
            "global": global_metrics,
            "subsets": subset_metrics,
            "round": server_round,
            "eval_time": eval_time
        }
        
        # Store metrics in the shared dictionary
        if client_metrics_ref is not None:
            if server_round not in client_metrics_ref:
                client_metrics_ref[server_round] = {}
            
            client_metrics_ref[server_round]["server_evaluation"] = results
        
        # Save metrics to file if log_dir is provided
        if log_dir:
            metrics_dir = os.path.join(log_dir, "server_evaluation")
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_file = os.path.join(metrics_dir, f"round_{server_round}.json")
            
            with open(metrics_file, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        # Log global results
        log(INFO, f"[Server] Round {server_round} evaluation complete:")
        log(INFO, f"  Global accuracy: {global_metrics['accuracy']:.2f}%")
        log(INFO, f"  Global loss: {global_metrics['loss']:.4f}")
        
        # Log subset results
        for name, metrics in subset_metrics.items():
            log(INFO, f"  Subset '{name}' accuracy: {metrics['accuracy']:.2f}%")
        
        # Flower expects loss, then a dictionary of metrics
        return global_metrics["loss"], {
            "accuracy": global_metrics["accuracy"],
            "eval_time": eval_time
        }
    
    return evaluate_fn

def server_fn_factory(initial_parameters, evaluate_fn, num_clients, num_rounds):
    """
    Factory function for creating server function.
    
    Args:
        initial_parameters: Initial model parameters
        evaluate_fn: Server-side evaluation function
        num_clients: Number of clients in the federation
        num_rounds: Total number of federation rounds
    
    Returns:
        server_fn: Function that configures and returns a ServerApp
    """
    
    def server_fn(context=None) -> ServerApp:
        """Configure and create a ServerApp instance."""
        
        # Use FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=num_clients,  # Minimum number of clients to sample during training
            min_evaluate_clients=num_clients,  # Minimum number of clients to sample during evaluation
            min_available_clients=num_clients,  # Minimum number of clients that need to be connected
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,  # Pass the server-side evaluation function
        )
        
        # Create server configuration
        server_config = ServerConfig(num_rounds=num_rounds)
        
        return ServerApp(
            config=server_config, 
            strategy=strategy
        )
    
    return server_fn 