#!/usr/bin/env python3
"""
Runner script for the SASWISE Fed-101 simulation.
This script serves as a direct entry point for running the simulation in Docker.
"""

import os
import sys

# Ensure we use the right Python path
package_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_root)

# Create a standalone simulation runner that doesn't rely on importing
# This avoids Python package/import complexities in different environments
def run_simulation():
    """Standalone implementation to run the simulation"""
    print("Setting up simulation environment...")
    
    # Import the required modules directly
    import torch
    from flwr.client import ClientApp
    from flwr.common import Metrics, Context
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    
    # Add saswise_fed_101 to the Python path
    saswise_path = os.path.join(package_root, 'saswise_fed_101')
    if saswise_path not in sys.path:
        sys.path.insert(0, saswise_path)
    
    # Import saswise modules
    from saswise_fed_101.task import Net, load_data, test, train
    from saswise_fed_101.client_app import FlowerClient
    
    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 1
    
    # GPU setup for local simulation
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    
    # If using GPU, optimize memory usage
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        # Enable memory efficient methods if available
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
    
    # Define client_fn
    def client_fn(context):
        # Load model and ensure it's on the right device
        net = Net().to(DEVICE)
        
        # Load data
        partition_id = context.node_config["partition-id"]
        trainloader, valloader = load_data(partition_id, NUM_CLIENTS)
        
        # Return client with device specification
        return FlowerClient(net, trainloader, valloader, LOCAL_EPOCHS).to_client()
    
    # Define weighted_average
    def weighted_average(metrics):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # Define the server function that returns ServerAppComponents
    def server_fn(context: Context) -> ServerAppComponents:
        # Create strategy
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=5,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        
        # Configure server
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        
        # Return the components
        return ServerAppComponents(strategy=strategy, config=config)
    
    # Create ClientApp and ServerApp properly
    client = ClientApp(client_fn=client_fn)
    server = ServerApp(server_fn=server_fn)
    
    # Backend configuration optimized for local simulation on a single GPU
    if DEVICE.type == "cuda":
        # For local GPU simulation, we can efficiently share the GPU
        # Each client gets a portion of the GPU memory
        gpu_memory_per_client = 0.1  # 10% of GPU memory per client
        
        # Configure resources for local simulation on a single GPU
        backend_config = {
            "client_resources": {
                "num_cpus": 0.5,  # Share CPU cores
                "num_gpus": gpu_memory_per_client  # Each client gets a portion of the GPU
            },
            "server_resources": {
                "num_cpus": 1.0,
                "num_gpus": 0.1  # Server also needs some GPU resources
            },
            # Ray configuration for better GPU sharing
            "ray_init_args": {
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "_memory": 1024 * 1024 * 1024,  # 1GB memory limit per client
                "_redis_max_memory": 1024 * 1024 * 1024,  # 1GB for redis
            }
        }
    else:
        # CPU-only configuration
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    
    # Run simulation
    print(f"Starting local simulation with {NUM_CLIENTS} clients on {DEVICE}...")
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
    print("Simulation completed!")

if __name__ == "__main__":
    print("Starting SASWISE Fed-101 simulation...")
    run_simulation() 