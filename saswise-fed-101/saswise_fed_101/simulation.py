"""SASWISE Fed-101: Federated Learning Simulation Experiment

This script runs a federated learning simulation based on the SASWISE Fed-101 client and server.
"""

import os
import sys
import argparse

# Import torch and other libraries first
import torch
import torch.nn as nn
from typing import List, Tuple

from flwr.client import Client, ClientApp
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import torchvision.models as models

# Dynamically determine where we are relative to the package and import accordingly
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))  # Add parent directory

# Direct absolute imports - avoid relative imports
try:
    from saswise_fed_101.task import load_data, test, train, set_weights, get_weights
    from saswise_fed_101.client_app import FlowerClient
except ImportError:
    # Fall back to direct imports if package structure doesn't work
    sys.path.insert(0, script_dir)  # Add current directory
    from task import load_data, test, train, set_weights, get_weights
    from client_app import FlowerClient

# Define ResNet model for CIFAR-10
class ResNetCIFAR10(nn.Module):
    def __init__(self):
        super(ResNetCIFAR10, self).__init__()
        # Use a pre-trained ResNet model but modify for CIFAR-10
        self.model = models.resnet18(pretrained=False)
        # Modify the first conv layer to accept 3-channel 32x32 inputs
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool as CIFAR-10 images are much smaller than ImageNet
        self.model.maxpool = nn.Identity()
        # Adjust the final layer for 10 classes
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
    def forward(self, x):
        return self.model(x)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SASWISE Fed-101 Simulation")
    parser.add_argument("--num_rounds", type=int, default=10000,
                       help="Number of federated learning rounds")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Number of clients to simulate")
    parser.add_argument("--local_epochs", type=int, default=10,
                       help="Number of local training epochs on each client")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training and evaluation")
    parser.add_argument("--use_resnet", action="store_true", default=True,
                       help="Use ResNet model instead of simple CNN")
    
    return parser.parse_args()

# Configuration
args = parse_args()
NUM_CLIENTS = args.num_clients
NUM_ROUNDS = args.num_rounds
LOCAL_EPOCHS = args.local_epochs
BATCH_SIZE = args.batch_size
USE_RESNET = args.use_resnet
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")
print(f"Configuration: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs, batch size {BATCH_SIZE}")
print(f"Using {'ResNet' if USE_RESNET else 'Simple CNN'} model")

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    if USE_RESNET:
        net = ResNetCIFAR10()
    else:
        # Import Net only if needed
        try:
            from saswise_fed_101.task import Net
        except ImportError:
            from task import Net
        net = Net()

    # Load data (CIFAR-10)
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = NUM_CLIENTS  # Use the same number of partitions as clients
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size=BATCH_SIZE)

    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, LOCAL_EPOCHS).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics across clients."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behavior."""

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,            # Sample 100% of available clients for training
        fraction_evaluate=0.5,       # Sample 50% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS, # Never sample less than all clients for training
        min_evaluate_clients=5,      # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS, # Wait until all clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # Use custom metrics aggregation
    )

    # Configure the server
    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)

def main():
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

    # Specify the resources each client needs
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if torch.cuda.is_available():
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    # Client run configuration to pass parameters to clients
    client_config = {
        "batch-size": BATCH_SIZE,
        "local-epochs": LOCAL_EPOCHS,
    }

    # Run simulation
    print(f"Starting simulation with {NUM_CLIENTS} clients...")
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        client_run_config=client_config,
    )
    print("Simulation completed!")

if __name__ == "__main__":
    main() 