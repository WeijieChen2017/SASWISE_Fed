#!/usr/bin/env python3
"""
Runner script for the SASWISE Fed-101 simulation.
This script serves as a direct entry point for running the simulation in Docker.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import json
import datetime

# Ensure we use the right Python path
package_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_root)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="SASWISE Fed-101 Simulation")
    parser.add_argument("--num_rounds", type=int, default=500,
                       help="Number of federated learning rounds (default: 500)")
    parser.add_argument("--num_clients", type=int, default=10,
                       help="Number of clients to simulate (default: 10)")
    parser.add_argument("--local_epochs", type=int, default=1,
                       help="Number of local training epochs on each client (default: 1)")
    parser.add_argument("--use_resnet", action="store_true",
                       help="Use ResNet model instead of simple CNN")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    return parser.parse_args()

# Create a standalone simulation runner that doesn't rely on importing
# This avoids Python package/import complexities in different environments
def run_simulation():
    """Standalone implementation to run the simulation"""
    print("Setting up simulation environment...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Import the required modules directly
    import torch
    from flwr.client import ClientApp
    from flwr.common import Metrics, Context, ndarrays_to_parameters
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
    import numpy as np
    from collections import defaultdict
    
    # Add saswise_fed_101 to the Python path
    saswise_path = os.path.join(package_root, 'saswise_fed_101')
    if saswise_path not in sys.path:
        sys.path.insert(0, saswise_path)
    
    # Import saswise modules
    from saswise_fed_101.task import Net, load_data, test, train, get_weights
    from saswise_fed_101.client_app import FlowerClient
    
    # Add ResNet support
    class ResNetCIFAR10(torch.nn.Module):
        def __init__(self):
            super(ResNetCIFAR10, self).__init__()
            # Use a pre-trained ResNet model but modify for CIFAR-10
            import torchvision.models as models
            self.model = models.resnet18(pretrained=False)
            # Modify the first conv layer to accept 3-channel 32x32 inputs
            self.model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove maxpool as CIFAR-10 images are much smaller than ImageNet
            self.model.maxpool = torch.nn.Identity()
            # Adjust the final layer for 10 classes
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
            
        def forward(self, x):
            return self.model(x)
    
    # Configuration
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    LOCAL_EPOCHS = args.local_epochs
    USE_RESNET = args.use_resnet
    BATCH_SIZE = args.batch_size
    
    # Storage for client-specific training losses
    client_losses = defaultdict(list)
    val_accuracies = defaultdict(list)
    
    # GPU setup for local simulation
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")
    print(f"Configuration: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs, batch size {BATCH_SIZE}")
    print(f"Using {'ResNet' if USE_RESNET else 'Simple CNN'} model")
    
    # If using GPU, optimize memory usage
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        # Enable memory efficient methods if available
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
    
    # Import task's set_weights to use in DetailedFlowerClient
    from saswise_fed_101.task import set_weights as original_set_weights
    
    # Create a safe version of set_weights that handles model type mismatches
    def safe_set_weights(net, parameters):
        try:
            # Attempt to set weights using the original function
            original_set_weights(net, parameters)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"ERROR: Model architecture mismatch detected. Make sure you're using the same model type (CNN or ResNet) consistently.")
                print(f"Current model type: {MODEL_TYPE}")
                print(f"Error details: {str(e)}")
                # Reinitialize the model instead of failing
                print("Reinitializing model with random weights...")
                # Model keeps its original random weights
            else:
                # Re-raise if it's not a size mismatch error
                raise e

    # Custom FlowerClient that logs training loss
    class DetailedFlowerClient(FlowerClient):
        def __init__(self, net, trainloader, valloader, testloader, local_epochs, client_id):
            super().__init__(net, trainloader, valloader, testloader, local_epochs)
            self.client_id = client_id
            
        def fit(self, parameters, config):
            safe_set_weights(self.net, parameters)
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            )
            
            # Evaluate on validation set after training
            val_loss, val_accuracy = test(self.net, self.valloader, self.device)
            
            # Save and print client-specific loss
            client_losses[self.client_id].append(train_loss)
            print(f"Client {self.client_id} - Round {len(client_losses[self.client_id])} - Loss: {train_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {
                    "train_loss": train_loss, 
                    "client_id": self.client_id,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
            )
        
        def evaluate(self, parameters, config):
            safe_set_weights(self.net, parameters)
            
            # By default, evaluate on validation set for regular federated evaluation
            # The config here is from server.evaluate_round, not node_config
            # Get eval_on from node_config (stored in context) if available
            eval_on = "val"
            
            # Try to get the partition ID for this client
            try:
                from flwr.common.context import Context
                ctx = Context.get()
                if ctx and ctx.node_config:
                    eval_on = ctx.node_config.get("eval_on", "val")
            except Exception:
                # If Context.get() fails, use the old approach from config
                eval_on = config.get("eval_on", "val")
            
            if eval_on == "test":
                # If specifically asked to evaluate on test set
                loss, accuracy = test(self.net, self.testloader, self.device)
                return loss, len(self.testloader.dataset), {"accuracy": accuracy, "dataset": "test", "client_id": self.client_id}
            else:
                # Regular evaluation on validation set
                loss, accuracy = test(self.net, self.valloader, self.device)
                return loss, len(self.valloader.dataset), {"accuracy": accuracy, "dataset": "val", "client_id": self.client_id}
    
    # Define client_fn with client ID tracking
    def client_fn(context):
        # Get the client ID from the context
        partition_id = context.node_config["partition-id"]
        
        # Check if we're in evaluation and a model_type was specified
        config_model_type = context.node_config.get("model_type", MODEL_TYPE)
        use_resnet = config_model_type == "resnet"
        
        # Get batch size and local epochs from node_config
        batch_size = context.node_config.get("batch_size", BATCH_SIZE)
        local_epochs = context.node_config.get("local_epochs", LOCAL_EPOCHS)
        
        # Load model and ensure it's on the right device
        if use_resnet:
            net = ResNetCIFAR10().to(DEVICE)
            print(f"Client {partition_id} using ResNet model")
        else:
            net = Net().to(DEVICE)
            print(f"Client {partition_id} using Simple CNN model")
        
        # Load data with train/val/test splits
        trainloader, valloader, testloader = load_data(partition_id, NUM_CLIENTS, batch_size=batch_size)
        
        # Return client with device specification and client ID
        return DetailedFlowerClient(
            net, 
            trainloader, 
            valloader,
            testloader,
            local_epochs, 
            client_id=partition_id
        ).to_client()
    
    # Define weighted_average
    def weighted_average(metrics):
        # Extract client-specific metrics
        client_metrics = {}
        for num_examples, met in metrics:
            if "client_id" in met:
                client_id = met["client_id"]
                client_metrics[client_id] = {
                    "loss": met.get("train_loss", 0.0),
                    "val_accuracy": met.get("val_accuracy", 0.0),
                    "val_loss": met.get("val_loss", 0.0),
                    "dataset": met.get("dataset", "val")
                }
                
                # Store validation accuracy for tracking
                if "val_accuracy" in met:
                    val_accuracies[client_id].append(met["val_accuracy"])
        
        # Compute the standard metrics
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        # Print client metrics
        print("\n=== Client Training Metrics for this round ===")
        for client_id, metrics_dict in sorted(client_metrics.items()):
            dataset_type = metrics_dict.get("dataset", "val")
            if dataset_type == "test":
                print(f"Client {client_id}: Test Accuracy = {metrics_dict.get('accuracy', 0.0):.4f}")
            else:
                print(f"Client {client_id}: Loss = {metrics_dict.get('loss', 0.0):.4f}, Val Accuracy = {metrics_dict.get('val_accuracy', 0.0):.4f}")
        print("============================================\n")
        
        return {"accuracy": sum(accuracies) / sum(examples)}
    
    # Initialize model parameters - IMPORTANT to fix the model initialization issue
    if USE_RESNET:
        initial_model = ResNetCIFAR10()
    else:
        initial_model = Net()
    
    initial_parameters = get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_parameters)
    
    # Save the model type to ensure consistency during evaluation
    MODEL_TYPE = "resnet" if USE_RESNET else "cnn"
    
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
    
    # Add logging for training/validation curves
    import matplotlib.pyplot as plt
    import json
    import datetime
    
    # Create a timestamped log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Function to log metrics to file and create plots
    def log_metrics_and_plot():
        # Save all metrics to a JSON file
        metrics_data = {
            "client_losses": {str(client_id): losses for client_id, losses in client_losses.items()},
            "val_accuracies": {str(client_id): accs for client_id, accs in val_accuracies.items()}
        }
        
        with open(f"{log_dir}/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=4)
        
        # Create training loss curve
        plt.figure(figsize=(10, 6))
        for client_id, losses in client_losses.items():
            plt.plot(losses, label=f"Client {client_id}")
        plt.title("Training Loss per Client")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{log_dir}/training_loss_curve.png")
        
        # Create validation accuracy curve
        plt.figure(figsize=(10, 6))
        for client_id, accs in val_accuracies.items():
            plt.plot(accs, label=f"Client {client_id}")
        plt.title("Validation Accuracy per Client")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{log_dir}/validation_accuracy_curve.png")
        
        print(f"Saved metrics and plots to {log_dir}/")

    # Run simulation
    print(f"Starting {NUM_ROUNDS}-epoch local simulation with {NUM_CLIENTS} clients on {DEVICE}...")
    
    # Set node configs for each client
    client_node_configs = []
    for i in range(NUM_CLIENTS):
        # Pass the model type in the node config instead
        node_config = {
            "partition-id": i,
            "model_type": MODEL_TYPE,
            "batch_size": BATCH_SIZE,
            "local_epochs": LOCAL_EPOCHS
        }
        client_node_configs.append(node_config)
    
    # Remove client_run_config as it's not supported in this version
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        # Pass model type and other settings via node_configs
        client_node_configs=client_node_configs
    )
    
    # Print final loss summary for all clients
    print("\n=== Final Client Loss Summary ===")
    print("Client ID | Epochs | Initial Loss | Final Loss | Improvement | Final Val Acc")
    print("-" * 80)
    for client_id, losses in sorted(client_losses.items()):
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            improvement = initial_loss - final_loss
            final_val_acc = val_accuracies[client_id][-1] if client_id in val_accuracies and val_accuracies[client_id] else 0.0
            print(f"Client {client_id:7d} | {len(losses):6d} | {initial_loss:12.4f} | {final_loss:10.4f} | {improvement:11.4f} | {final_val_acc:.4f}")
    
    # Generate plots and save metrics
    log_metrics_and_plot()
    
    # Add a final test evaluation
    print("\n=== Running final test evaluation ===")
    # Create a special evaluation config for test set
    try:
        # Set up node config for test evaluation
        test_node_configs = []
        for i in range(NUM_CLIENTS):
            test_config = {
                "partition-id": i,
                "model_type": MODEL_TYPE,
                "eval_on": "test"
            }
            test_node_configs.append(test_config)
            
        test_results = server.evaluate_round(
            server_round=NUM_ROUNDS + 1,  # Use a round number after training
            timeout=None,
            node_configs=test_node_configs  # Use node_configs instead of client_instructions
        )
        
        if test_results:
            test_loss, test_metrics = test_results
            print(f"Final test evaluation - Loss: {test_loss:.4f}, Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
            
            # Save test results
            with open(f"{log_dir}/test_results.json", "w") as f:
                json.dump({
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_metrics.get("accuracy", 0.0))
                }, f, indent=4)
        else:
            print("Final test evaluation failed")
    except Exception as e:
        print(f"Error during final evaluation: {str(e)}")
        print("This may be due to model architecture mismatch. Try rerunning with the same model type.")
    
    print("\nSimulation completed!")

if __name__ == "__main__":
    print("Starting SASWISE Fed-101 simulation...")
    run_simulation() 