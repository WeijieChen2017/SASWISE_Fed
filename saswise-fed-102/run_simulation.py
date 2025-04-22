#!/usr/bin/env python3
"""
Runner script for the SASWISE Fed-102 simulation.
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
    from flwr.common import Metrics, Context, ndarrays_to_parameters
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    import numpy as np
    from collections import defaultdict
    
    # Add saswise_fed_102 to the Python path
    saswise_path = os.path.join(package_root, 'saswise_fed_102')
    if saswise_path not in sys.path:
        sys.path.insert(0, saswise_path)
    
    # Import saswise modules
    from saswise_fed_102.task import Net, load_data, test, train, get_weights
    from saswise_fed_102.client_app import FlowerClient
    
    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 100  # Increased to 100 epochs
    LOCAL_EPOCHS = 1
    
    # Storage for client-specific training losses
    client_losses = defaultdict(list)
    
    # GPU setup for local simulation
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    
    # If using GPU, optimize memory usage
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        # Enable memory efficient methods if available
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
    
    # Custom FlowerClient that logs training loss
    class DetailedFlowerClient(FlowerClient):
        def __init__(self, net, trainloader, valloader, local_epochs, client_id):
            super().__init__(net, trainloader, valloader, local_epochs)
            self.client_id = client_id
            
        def fit(self, parameters, config):
            set_weights(self.net, parameters)
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            )
            
            # Save and print client-specific loss
            client_losses[self.client_id].append(train_loss)
            print(f"Client {self.client_id} - Round {len(client_losses[self.client_id])} - Loss: {train_loss:.4f}")
            
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss, "client_id": self.client_id},
            )
    
    # Define client_fn with client ID tracking
    def client_fn(context):
        # Get the client ID from the context
        partition_id = context.node_config["partition-id"]
        
        # Load model and ensure it's on the right device
        net = Net().to(DEVICE)
        
        # Load data
        trainloader, valloader = load_data(partition_id, NUM_CLIENTS)
        
        # Return client with device specification and client ID
        return DetailedFlowerClient(
            net, 
            trainloader, 
            valloader, 
            LOCAL_EPOCHS, 
            client_id=partition_id
        ).to_client()
    
    # Define weighted_average
    def weighted_average(metrics):
        # Extract client-specific metrics
        client_metrics = {}
        for num_examples, met in metrics:
            if "client_id" in met:
                client_id = met["client_id"]
                if "train_loss" in met:
                    loss = met["train_loss"]
                    client_metrics[client_id] = loss
        
        # Compute the standard metrics
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        # Print client metrics
        print("\n=== Client Training Losses for this round ===")
        for client_id, loss in sorted(client_metrics.items()):
            print(f"Client {client_id}: Loss = {loss:.4f}")
        print("============================================\n")
        
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # Import task's set_weights to use in DetailedFlowerClient
    from saswise_fed_102.task import set_weights
    
    # Initialize model parameters - IMPORTANT to fix the model initialization issue
    initial_model = Net()
    initial_parameters = get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_parameters)
    
    # Define the server function that returns ServerAppComponents
    def server_fn(context: Context) -> ServerAppComponents:
        # Create strategy with initial parameters
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=5,
            min_available_clients=NUM_CLIENTS,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,  # Use initialized parameters
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
    print(f"Starting 100-epoch local simulation with {NUM_CLIENTS} clients on {DEVICE}...")
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
    
    # Print final loss summary for all clients
    print("\n=== Final Client Loss Summary ===")
    print("Client ID | Epochs | Initial Loss | Final Loss | Improvement")
    print("-" * 65)
    for client_id, losses in sorted(client_losses.items()):
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            improvement = initial_loss - final_loss
            print(f"Client {client_id:7d} | {len(losses):6d} | {initial_loss:12.4f} | {final_loss:10.4f} | {improvement:11.4f}")
    
    print("\nSimulation completed!")

if __name__ == "__main__":
    print("Starting SASWISE Fed-102 simulation...")
    run_simulation() 