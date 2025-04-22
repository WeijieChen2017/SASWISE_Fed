"""SASWISE Fed-101: Federated Learning Simulation Experiment

This script runs a federated learning simulation based on the SASWISE Fed-101 client and server.
"""

import torch
from typing import List, Tuple
import sys
import os

# Add the parent directory to sys.path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flwr.client import Client, ClientApp
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# Import relative to where the script is being run
from saswise_fed_101.task import Net, load_data, test, train
from saswise_fed_101.client_app import FlowerClient

# Configuration
NUM_CLIENTS = 10
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net()

    # Load data (CIFAR-10)
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = NUM_CLIENTS  # Use the same number of partitions as clients
    trainloader, valloader = load_data(partition_id, num_partitions)

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

    # Run simulation
    print(f"Starting simulation with {NUM_CLIENTS} clients...")
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )
    print("Simulation completed!")

if __name__ == "__main__":
    main() 