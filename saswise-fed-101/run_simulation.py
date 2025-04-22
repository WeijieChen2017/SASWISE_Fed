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

# Use a class to encapsulate the simulation to avoid global variables
class FederatedSimulation:
    def __init__(self, args):
        # Import the required modules
        import torch
        from flwr.client import ClientApp, NumPyClient
        from flwr.common import Metrics, Context, ndarrays_to_parameters
        from flwr.server import ServerApp, ServerConfig, ServerAppComponents
        from flwr.server.strategy import FedAvg
        from flwr.simulation import run_simulation
        import numpy as np
        from collections import defaultdict
        
        # Store imported modules as instance variables
        self.torch = torch
        self.ClientApp = ClientApp
        self.NumPyClient = NumPyClient
        self.Context = Context
        self.ndarrays_to_parameters = ndarrays_to_parameters
        self.ServerApp = ServerApp
        self.ServerConfig = ServerConfig
        self.ServerAppComponents = ServerAppComponents
        self.FedAvg = FedAvg
        self.run_simulation = run_simulation
        self.np = np
        self.defaultdict = defaultdict
        
        # Add saswise_fed_101 to the Python path
        saswise_path = os.path.join(package_root, 'saswise_fed_101')
        if saswise_path not in sys.path:
            sys.path.insert(0, saswise_path)
        
        # Import saswise modules
        from saswise_fed_101.task import Net, load_data, test, train, get_weights, set_weights
        self.Net = Net
        self.load_data = load_data
        self.test = test
        self.train = train
        self.get_weights = get_weights
        self.set_weights = set_weights
        
        # Configuration
        self.NUM_CLIENTS = args.num_clients
        self.NUM_ROUNDS = args.num_rounds
        self.LOCAL_EPOCHS = args.local_epochs
        self.USE_RESNET = args.use_resnet
        self.BATCH_SIZE = args.batch_size
        self.MODEL_TYPE = "resnet" if self.USE_RESNET else "cnn"
        self.eval_mode = "val"  # Default evaluation mode ("val" or "test")
        
        # Storage for client-specific training losses
        self.client_losses = self.defaultdict(list)
        self.val_accuracies = self.defaultdict(list)
        
        # GPU setup for local simulation
        self.DEVICE = self.torch.device("cuda:0" if self.torch.cuda.is_available() else "cpu")
        
        # Setup logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Add ResNet support
        self.setup_resnet_model()
    
    def setup_resnet_model(self):
        # Add ResNet support as a class
        class ResNetCIFAR10(self.torch.nn.Module):
            def __init__(self):
                super(ResNetCIFAR10, self).__init__()
                # Use a pre-trained ResNet model but modify for CIFAR-10
                import torchvision.models as models
                self.model = models.resnet18(pretrained=False)
                # Modify the first conv layer to accept 3-channel 32x32 inputs
                self.model.conv1 = self.torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                # Remove maxpool as CIFAR-10 images are much smaller than ImageNet
                self.model.maxpool = self.torch.nn.Identity()
                # Adjust the final layer for 10 classes
                self.model.fc = self.torch.nn.Linear(self.model.fc.in_features, 10)
                
            def forward(self, x):
                return self.model(x)
        
        self.ResNetCIFAR10 = ResNetCIFAR10
    
    def safe_set_weights(self, net, parameters):
        try:
            # Attempt to set weights using the original function
            self.set_weights(net, parameters)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"ERROR: Model architecture mismatch detected. Make sure you're using the same model type (CNN or ResNet) consistently.")
                print(f"Current model type: {self.MODEL_TYPE}")
                print(f"Error details: {str(e)}")
                # Reinitialize the model instead of failing
                print("Reinitializing model with random weights...")
                # Model keeps its original random weights
            else:
                # Re-raise if it's not a size mismatch error
                raise e
    
    def create_client(self):
        # Create a client class that captures the simulation instance
        sim = self  # Capture the simulation instance
        
        class DetailedFlowerClient(self.NumPyClient):
            def __init__(self, net, trainloader, valloader, testloader, local_epochs, client_id):
                self.net = net
                self.trainloader = trainloader
                self.valloader = valloader
                self.testloader = testloader
                self.local_epochs = local_epochs
                self.client_id = client_id
                self.device = sim.DEVICE
                self.net.to(self.device)
                
            def fit(self, parameters, config):
                sim.safe_set_weights(self.net, parameters)
                train_loss = sim.train(
                    self.net,
                    self.trainloader,
                    self.local_epochs,
                    self.device,
                )
                
                # Evaluate on validation set after training
                val_loss, val_accuracy = sim.test(self.net, self.valloader, self.device)
                
                # Save and print client-specific loss
                sim.client_losses[self.client_id].append(train_loss)
                print(f"Client {self.client_id} - Round {len(sim.client_losses[self.client_id])} - Loss: {train_loss:.4f} - Val Acc: {val_accuracy:.4f}")
                
                return (
                    sim.get_weights(self.net),
                    len(self.trainloader.dataset),
                    {
                        "train_loss": train_loss, 
                        "client_id": self.client_id,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy
                    },
                )
            
            def evaluate(self, parameters, config):
                sim.safe_set_weights(self.net, parameters)
                
                # Use the simulation's eval_mode to determine which dataset to evaluate on
                if sim.eval_mode == "test":
                    # Evaluate on test set
                    loss, accuracy = sim.test(self.net, self.testloader, self.device)
                    return loss, len(self.testloader.dataset), {"accuracy": accuracy, "dataset": "test", "client_id": self.client_id}
                else:
                    # Regular evaluation on validation set
                    loss, accuracy = sim.test(self.net, self.valloader, self.device)
                    # Store validation accuracy for tracking
                    sim.val_accuracies[self.client_id].append(accuracy)
                    return loss, len(self.valloader.dataset), {"accuracy": accuracy, "dataset": "val", "client_id": self.client_id}
        
        return DetailedFlowerClient
    
    def client_fn(self, context):
        # Get the client ID from the context
        partition_id = context.node_config["partition-id"]
        
        # Load model and ensure it's on the right device
        if self.USE_RESNET:
            net = self.ResNetCIFAR10().to(self.DEVICE)
            print(f"Client {partition_id} using ResNet model")
        else:
            net = self.Net().to(self.DEVICE)
            print(f"Client {partition_id} using Simple CNN model")
        
        # Load data with train/val/test splits
        trainloader, valloader, testloader = self.load_data(partition_id, self.NUM_CLIENTS, batch_size=self.BATCH_SIZE)
        
        # Get the client class
        DetailedFlowerClient = self.create_client()
        
        # Return client with device specification and client ID
        return DetailedFlowerClient(
            net, 
            trainloader, 
            valloader,
            testloader,
            self.LOCAL_EPOCHS, 
            client_id=partition_id
        ).to_client()
    
    def weighted_average(self, metrics):
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
    
    def log_metrics_and_plot(self):
        # Save all metrics to a JSON file
        metrics_data = {
            "client_losses": {str(client_id): losses for client_id, losses in self.client_losses.items()},
            "val_accuracies": {str(client_id): accs for client_id, accs in self.val_accuracies.items()}
        }
        
        with open(f"{self.log_dir}/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=4)
        
        # Create training loss curve
        plt.figure(figsize=(10, 6))
        for client_id, losses in self.client_losses.items():
            plt.plot(losses, label=f"Client {client_id}")
        plt.title("Training Loss per Client")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.log_dir}/training_loss_curve.png")
        
        # Create validation accuracy curve
        plt.figure(figsize=(10, 6))
        for client_id, accs in self.val_accuracies.items():
            plt.plot(accs, label=f"Client {client_id}")
        plt.title("Validation Accuracy per Client")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{self.log_dir}/validation_accuracy_curve.png")
        
        print(f"Saved metrics and plots to {self.log_dir}/")
    
    def run(self):
        print(f"Training on {self.DEVICE}")
        print(f"Configuration: {self.NUM_CLIENTS} clients, {self.NUM_ROUNDS} rounds, {self.LOCAL_EPOCHS} local epochs, batch size {self.BATCH_SIZE}")
        print(f"Using {'ResNet' if self.USE_RESNET else 'Simple CNN'} model")
        
        # If using GPU, optimize memory usage
        if self.DEVICE.type == "cuda":
            self.torch.cuda.empty_cache()
            # Enable memory efficient methods if available
            if hasattr(self.torch.backends, 'cudnn'):
                self.torch.backends.cudnn.benchmark = True
        
        # Create ray_init_args
        ray_init_args = {
            "include_dashboard": False,
            "ignore_reinit_error": True,
            "_memory": 1024 * 1024 * 1024,  # 1GB memory limit per client
            "_redis_max_memory": 1024 * 1024 * 1024,  # 1GB for redis
        }
        
        # Initialize model parameters
        if self.USE_RESNET:
            initial_model = self.ResNetCIFAR10()
        else:
            initial_model = self.Net()
        
        initial_parameters = self.get_weights(initial_model)
        initial_parameters = self.ndarrays_to_parameters(initial_parameters)
        
        # Define the server function
        def server_fn(context: self.Context) -> self.ServerAppComponents:
            # Create strategy with initial parameters
            strategy = self.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.5,
                min_fit_clients=self.NUM_CLIENTS,
                min_evaluate_clients=5,
                min_available_clients=self.NUM_CLIENTS,
                evaluate_metrics_aggregation_fn=self.weighted_average,
                initial_parameters=initial_parameters,
            )
            
            # Configure server
            config = self.ServerConfig(num_rounds=self.NUM_ROUNDS)
            
            # Return the components
            return self.ServerAppComponents(strategy=strategy, config=config)
        
        # Create ClientApp and ServerApp
        client = self.ClientApp(client_fn=self.client_fn)
        server = self.ServerApp(server_fn=server_fn)
        
        # Backend configuration
        if self.DEVICE.type == "cuda":
            # For GPU simulation
            gpu_memory_per_client = 0.1  # 10% of GPU memory per client
            backend_config = {
                "client_resources": {
                    "num_cpus": 0.5,
                    "num_gpus": gpu_memory_per_client
                },
                "server_resources": {
                    "num_cpus": 1.0,
                    "num_gpus": 0.1
                },
                "ray_init_args": ray_init_args
            }
        else:
            # CPU-only configuration
            backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}, "ray_init_args": ray_init_args}
        
        # Run simulation
        print(f"Starting {self.NUM_ROUNDS}-epoch local simulation with {self.NUM_CLIENTS} clients on {self.DEVICE}...")
        self.run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=self.NUM_CLIENTS,
            backend_config=backend_config
        )
        
        # Print final loss summary
        print("\n=== Final Client Loss Summary ===")
        print("Client ID | Epochs | Initial Loss | Final Loss | Improvement | Final Val Acc")
        print("-" * 80)
        for client_id, losses in sorted(self.client_losses.items()):
            if losses:
                initial_loss = losses[0]
                final_loss = losses[-1]
                improvement = initial_loss - final_loss
                final_val_acc = self.val_accuracies[client_id][-1] if client_id in self.val_accuracies and self.val_accuracies[client_id] else 0.0
                print(f"Client {client_id:7d} | {len(losses):6d} | {initial_loss:12.4f} | {final_loss:10.4f} | {improvement:11.4f} | {final_val_acc:.4f}")
        
        # Generate plots and save metrics
        self.log_metrics_and_plot()
        
        # Final test evaluation
        print("\n=== Running final test evaluation ===")
        try:
            # Set evaluation mode to test
            self.eval_mode = "test"
            
            test_results = server.evaluate_round(
                server_round=self.NUM_ROUNDS + 1,
                timeout=None
            )
            
            if test_results:
                test_loss, test_metrics = test_results
                print(f"Final test evaluation - Loss: {test_loss:.4f}, Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
                
                # Save test results
                with open(f"{self.log_dir}/test_results.json", "w") as f:
                    json.dump({
                        "test_loss": float(test_loss),
                        "test_accuracy": float(test_metrics.get("accuracy", 0.0))
                    }, f, indent=4)
            else:
                print("Final test evaluation failed")
                
            # Reset the evaluation mode
            self.eval_mode = "val"
        except Exception as e:
            print(f"Error during final evaluation: {str(e)}")
            print("This may be due to model architecture mismatch. Try rerunning with the same model type.")
        
        print("\nSimulation completed!")

# Main function to run the simulation
def run_simulation():
    """Standalone implementation to run the simulation"""
    print("Setting up simulation environment...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Create and run the simulation
    simulation = FederatedSimulation(args)
    simulation.run()

if __name__ == "__main__":
    print("Starting SASWISE Fed-101 simulation...")
    run_simulation() 