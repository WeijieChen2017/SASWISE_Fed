"""SASWISE Fed-101: Federated Learning Simulation Experiment

This script runs a federated learning simulation based on the SASWISE Fed-101 client and server.
"""

import os
import sys
import argparse
import datetime
import logging
import json

# Import torch and other libraries first
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional

from flwr.client import Client, ClientApp
from flwr.common import Metrics, Context, Parameters, FitRes, MetricsAggregationFn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, Strategy
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
    parser.add_argument("--eval_every", type=int, default=20,
                       help="Evaluate model every N rounds")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save logs and models")
    
    return parser.parse_args()

# Set up logging
def setup_logging(log_dir):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"federated_learning_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("federated_learning")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_dir, timestamp

# Custom FedAvg strategy with model saving and logging
class CustomFedAvg(FedAvg):
    def __init__(
        self,
        *args,
        evaluate_every: int = 20,
        logger: Optional[logging.Logger] = None,
        model_dir: str = "models",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.evaluate_every = evaluate_every
        self.logger = logger
        self.model_dir = model_dir
        self.best_accuracy = 0.0
        self.round_metrics = {}
        os.makedirs(model_dir, exist_ok=True)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Client, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model weights and log client training losses."""
        if not results:
            return None, {}
            
        # Log client training losses
        if self.logger:
            client_losses = {}
            for i, (_, fit_res) in enumerate(results):
                client_losses[f"client_{i}"] = fit_res.metrics.get("train_loss", 0.0)
            
            self.logger.info(f"Round {server_round} client losses: {json.dumps(client_losses)}")
            self.round_metrics[f"round_{server_round}"] = {"client_losses": client_losses}
            
        # Call parent's aggregate_fit to aggregate parameters
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Save model if evaluation is due in the next round
        if parameters_aggregated is not None and server_round % self.evaluate_every == 0:
            # Save current model
            save_path = os.path.join(self.model_dir, f"model_round_{server_round}.pth")
            weights_ndarrays = parameters_aggregated.tensors
            
            if self.logger:
                self.logger.info(f"Saving model after round {server_round} to {save_path}")
                
            # Convert weights to a model and save
            if USE_RESNET:
                model = ResNetCIFAR10()
            else:
                try:
                    from saswise_fed_101.task import Net
                except ImportError:
                    from task import Net
                model = Net()
                
            set_weights(model, weights_ndarrays)
            torch.save(model.state_dict(), save_path)
        
        return parameters_aggregated, metrics_aggregated
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Client, Tuple[float, int, Dict[str, float]]]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics and save best model."""
        if not results:
            return None, {}
            
        # Call parent's aggregate_evaluate to aggregate metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Log evaluation metrics
        if self.logger and metrics_aggregated:
            self.logger.info(f"Round {server_round} evaluation: {json.dumps(metrics_aggregated)}")
            
            # Add to round metrics
            if f"round_{server_round}" in self.round_metrics:
                self.round_metrics[f"round_{server_round}"]["evaluation"] = metrics_aggregated
            else:
                self.round_metrics[f"round_{server_round}"] = {"evaluation": metrics_aggregated}
                
            # Save metrics to file
            with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
                json.dump(self.round_metrics, f, indent=4)
            
            # Check if this is the best model so far
            current_accuracy = metrics_aggregated.get("accuracy", 0.0)
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                # Save best model
                if server_round > 1:  # Avoid saving initial model
                    best_path = os.path.join(self.model_dir, "best_model.pth")
                    prev_path = os.path.join(self.model_dir, f"model_round_{server_round}.pth")
                    
                    if os.path.exists(prev_path):
                        # Copy the model to best_model
                        self.logger.info(f"New best model at round {server_round} with accuracy {current_accuracy:.4f}")
                        model_state = torch.load(prev_path)
                        torch.save(model_state, best_path)
        
        return loss_aggregated, metrics_aggregated

# Configuration
args = parse_args()
NUM_CLIENTS = args.num_clients
NUM_ROUNDS = args.num_rounds
LOCAL_EPOCHS = args.local_epochs
BATCH_SIZE = args.batch_size
USE_RESNET = args.use_resnet
EVAL_EVERY = args.eval_every
LOG_DIR = args.log_dir
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup logging
logger, log_dir, timestamp = setup_logging(LOG_DIR)
model_dir = os.path.join(log_dir, f"models_{timestamp}")
os.makedirs(model_dir, exist_ok=True)

logger.info(f"Training on {DEVICE}")
logger.info(f"Configuration: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs, batch size {BATCH_SIZE}")
logger.info(f"Using {'ResNet' if USE_RESNET else 'Simple CNN'} model")
logger.info(f"Evaluating every {EVAL_EVERY} rounds")
logger.info(f"Models and logs will be saved to {model_dir}")

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
    trainloader, valloader, testloader = load_data(partition_id, num_partitions, batch_size=BATCH_SIZE)

    # Create a single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, testloader, LOCAL_EPOCHS).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics across clients."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behavior."""

    # Create custom FedAvg strategy with logging and model saving
    strategy = CustomFedAvg(
        fraction_fit=1.0,                # Sample 100% of available clients for training
        fraction_evaluate=0.5,           # Sample 50% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,     # Never sample less than all clients for training
        min_evaluate_clients=5,          # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS, # Wait until all clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # Use custom metrics aggregation
        evaluate_every=EVAL_EVERY,       # Evaluate only every N rounds
        logger=logger,                   # Pass logger to strategy
        model_dir=model_dir,             # Pass model directory to strategy
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
    logger.info(f"Starting simulation with {NUM_CLIENTS} clients...")
    history = run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        client_run_config=client_config,
    )
    logger.info("Simulation completed!")
    
    # Final evaluation on test set
    logger.info("Performing final evaluation on test set...")
    
    # Load the best model
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        if USE_RESNET:
            final_model = ResNetCIFAR10()
        else:
            try:
                from saswise_fed_101.task import Net
            except ImportError:
                from task import Net
            final_model = Net()
            
        # Load the best model weights
        final_model.load_state_dict(torch.load(best_model_path))
        
        # Create a special evaluation config
        eval_config = {
            "batch-size": BATCH_SIZE,
            "local-epochs": LOCAL_EPOCHS,
            "eval_on": "test"  # Signal to use test set
        }
        
        # Evaluate on the test set
        test_results = server.evaluate_round(
            server_round=NUM_ROUNDS + 1,
            timeout=None,
            client_instructions=[eval_config] * NUM_CLIENTS,
        )
        
        if test_results:
            test_loss, test_metrics = test_results
            logger.info(f"Final test evaluation results: {json.dumps(test_metrics)}")
            
            # Save final results
            final_results = {
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "best_model_path": best_model_path,
                "total_rounds": NUM_ROUNDS
            }
            
            with open(os.path.join(model_dir, "final_results.json"), "w") as f:
                json.dump(final_results, f, indent=4)
        else:
            logger.warning("Final test evaluation failed")
    else:
        logger.warning(f"Best model not found at {best_model_path}")

if __name__ == "__main__":
    main() 