# remember to install ray and flwr
# pip install "numpy<2"

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import json
import torch
import numpy as np
from collections import Counter
import os
import time
from datetime import datetime
from logging import INFO, ERROR

from utils2 import *

# Set device and print clear information about it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"=========================================")
print(f"EXECUTION INFORMATION:")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"=========================================")

# Create logs directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/run_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
print(f"Logs will be stored in: {log_dir}")

# Save a copy of the config to the log directory
with open("config.json", "r") as f:
    config = json.load(f)

# Save config to the log directory
with open(f"{log_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Training parameters
num_clients = config["training"]["num_clients"]
num_rounds = config["training"]["num_rounds"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
momentum = config["training"]["momentum"]
data_fraction = config.get("training", {}).get("data_fraction", 0.8)  # Default to 80% if not specified

print(f"[{datetime.now().strftime('%H:%M:%S')}] Configuration loaded:")
print(f"  Clients: {num_clients}, Rounds: {num_rounds}, Epochs: {epochs}")
print(f"  Batch size: {batch_size}, Learning rate: {learning_rate}")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CIFAR10 dataset...")
start_time = time.time()
trainset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=True, transform=transform
)
load_time = time.time() - start_time
print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded in {load_time:.2f} seconds")
print(f"Total dataset size: {len(trainset)}")

# Prepare client datasets - each client starts with the full dataset
print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing client datasets...")
train_sets = []
for i, client_config in enumerate(config["clients"]):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Setting up Client {i}...")
    # Create a copy of the full dataset for this client
    client_dataset = trainset
    
    # Apply the data fraction if needed (e.g., 80% of data)
    if data_fraction < 1.0:
        full_size = len(client_dataset)
        subset_size = int(full_size * data_fraction)
        # Use a different random seed for each client
        torch.manual_seed(42 + i)  
        indices = torch.randperm(full_size)[:subset_size]
        client_dataset = Subset(client_dataset, indices)
        print(f"  Using {data_fraction*100}% of data. Size after fraction: {len(client_dataset)}")
    
    # Apply exclusions based on config
    client_dataset = exclude_classes(client_dataset, excluded_classes=client_config["excluded_classes"])
    print(f"  Size after excluding classes {client_config['excluded_classes']}: {len(client_dataset)}")
    
    # Analyze class distribution
    labels = []
    for j in range(len(client_dataset)):
        # Get the label directly from the dataset item
        label = client_dataset[j][1]
        labels.append(label)
    
    class_counts = Counter(labels)
    # Ensure all classes 0-9 are represented, defaulting to 0 if not present
    full_class_counts = {i: class_counts.get(i, 0) for i in range(10)}
    
    print(f"  Final dataset size: {len(client_dataset)}")
    print(f"  Excluded classes: {client_config['excluded_classes']}")
    print(f"  Class distribution: {dict(full_class_counts)}")
    print(f"  Classes present: {sorted(class_counts.keys())}")
    
    train_sets.append(client_dataset)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading test dataset...")
testset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=False, transform=transform
)
print("Number of examples in `testset`:", len(testset))

# Create test subsets based on config
print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating test subsets...")
test_subsets = {}
for subset in config["evaluation"]["test_subsets"]:
    test_subsets[subset["name"]] = include_classes(testset, subset["classes"])
    print(f"Test subset '{subset['name']}': {len(test_subsets[subset['name']])} samples, classes: {subset['classes']}")

# Sets the parameters of the model
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v).to(device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

# Retrieves the parameters from the model
def get_weights(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

# Dictionary to store client metrics
client_metrics = {}

class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset, client_id):
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.client_id = client_id

    # Train the model
    def fit(self, parameters, config):
        try:
            start_time = time.time()
            current_round = config.get("server_round", 0)
            
            # Get server-provided config parameters or use defaults
            local_epochs = config.get("local_epochs", epochs)
            local_lr = config.get("learning_rate", learning_rate)
            local_momentum = config.get("momentum", momentum)
            local_batch_size = config.get("batch_size", batch_size)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} starting training for round {current_round}")
            print(f"  Using config: LR={local_lr}, momentum={local_momentum}, epochs={local_epochs}, batch_size={local_batch_size}")
            
            set_weights(self.net, parameters)
            
            # Pass local configuration to train_model
            training_loss = self.train_model(
                self.net, 
                self.trainset, 
                epochs=local_epochs,
                learning_rate=local_lr, 
                momentum=local_momentum,
                batch_size=local_batch_size
            )
            
            # Evaluate after training to get validation metrics
            val_loss, val_accuracy = evaluate_model(self.net, self.testset)
            
            # Store metrics for this client
            if current_round not in client_metrics:
                client_metrics[current_round] = {"clients": []}
            elif "clients" not in client_metrics[current_round]:
                client_metrics[current_round]["clients"] = []
            
            # Check if this client is already in the metrics
            client_found = False
            for idx, client in enumerate(client_metrics[current_round]["clients"]):
                if client.get("id") == self.client_id:
                    # Update existing client entry
                    client_metrics[current_round]["clients"][idx] = {
                        "id": self.client_id,
                        "training_loss": training_loss,
                        "validation_loss": val_loss,
                        "validation_accuracy": val_accuracy,
                        "dataset_size": len(self.trainset),
                        "config": {
                            "learning_rate": local_lr,
                            "momentum": local_momentum,
                            "epochs": local_epochs,
                            "batch_size": local_batch_size
                        }
                    }
                    client_found = True
                    break
                
            # Add client if not found
            if not client_found:
                client_metrics[current_round]["clients"].append({
                    "id": self.client_id,
                    "training_loss": training_loss,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "dataset_size": len(self.trainset),
                    "config": {
                        "learning_rate": local_lr,
                        "momentum": local_momentum,
                        "epochs": local_epochs,
                        "batch_size": local_batch_size
                    }
                })
            
            train_time = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} completed round {current_round} in {train_time:.2f}s")
            print(f"  Training Loss: {training_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Save metrics after each client completes training to ensure we don't lose data
            save_round_metrics(current_round)
            
            return get_weights(self.net), len(self.trainset), {"training_loss": training_loss}
        
        except Exception as e:
            # Log client training failure
            error_msg = f"ERROR in client {self.client_id} fit method: {str(e)}"
            print(error_msg)
            
            # Write error to log file
            with open(f"{log_dir}/error_log.txt", "a") as f:
                f.write(f"{error_msg}\n")
            
            # Return original parameters to prevent model corruption
            # This allows the simulation to continue even if this client fails
            return parameters, 0, {"error": str(e)}
    
    def train_model(self, model, train_set, epochs=10, learning_rate=0.01, momentum=0.9, batch_size=128):
        # Debug output to identify if we're entering the training loop
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} training started with {len(train_set)} samples")
        
        try:
            # Create smaller batches to reduce memory usage and potential hanging
            effective_batch_size = min(batch_size, 32)  # Reduce batch size even further
            train_loader = DataLoader(train_set, batch_size=effective_batch_size, shuffle=True, num_workers=0, pin_memory=False)
            print(f"  Using batch size: {effective_batch_size}, total batches: {len(train_loader)}")

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

            model.train()
            total_loss = 0.0
            batches = 0
            
            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_loss = 0.0
                print(f"  Starting epoch {epoch+1}/{epochs}")
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    # Print progress more frequently for first few batches to verify it's working
                    if batch_idx < 5 or batch_idx % 10 == 0:
                        # Save training progress to JSON
                        progress_data = {
                            "client_id": self.client_id,
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "training_loss": 0.0  # Will be updated if loss computation succeeds
                        }
                        
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        batches += 1
                        
                        # Update progress data with actual loss if available
                        if batch_idx < 5 or batch_idx % 10 == 0:
                            progress_data["training_loss"] = loss.item()
                            
                    except Exception as e:
                        error_msg = f"\n  ERROR in batch {batch_idx}: {str(e)}"
                        print(error_msg)
                        
                        # Log error to file
                        with open(f"{log_dir}/error_log.txt", "a") as f:
                            f.write(f"Client {self.client_id}, Epoch {epoch+1}, Batch {batch_idx}: {str(e)}\n")
                        
                        # Continue with next batch instead of failing
                        continue
                    
                    # Free up memory
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Clear references to reduce memory pressure
                    del inputs, labels, outputs, loss
                    
                    # Save training progress to JSON if appropriate
                    if batch_idx < 5 or batch_idx % 10 == 0:
                        try:
                            with open(f"{log_dir}/training_progress_epoch{epoch+1}_client{self.client_id}.json", "a") as f:
                                json.dump(progress_data, f)
                                f.write("\n")
                        except Exception as e:
                            print(f"  Error saving progress data: {str(e)}")
                
                # Skip division if no batches were processed
                if batches > 0:
                    epoch_avg_loss = epoch_loss / max(len(train_loader), 1)  # Avoid division by zero
                    total_loss += epoch_avg_loss
                else:
                    epoch_avg_loss = 0.0
                    print("  WARNING: No batches were successfully processed in this epoch")
                    
                epoch_time = time.time() - epoch_start
                
                # Print epoch summary
                epoch_summary = {
                    "client_id": self.client_id,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "epoch_time": epoch_time,
                    "avg_loss": epoch_avg_loss
                }
                
                # Save to epoch summary file
                try:
                    with open(f"{log_dir}/epoch_summary_client{self.client_id}.json", "a") as f:
                        json.dump(epoch_summary, f)
                        f.write("\n")
                except Exception as e:
                    print(f"  Error saving epoch summary: {str(e)}")
                    
                print(f"\n  Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - Avg Loss: {epoch_avg_loss:.4f}")
            
            # Return average loss across all epochs
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} training finished")
            
            # Return 0 if no batches were processed to avoid division by zero
            if batches == 0:
                return 0.0
            
            return total_loss / max(epochs, 1)  # Avoid division by zero
        
        except Exception as e:
            # Log critical error
            error_msg = f"CRITICAL ERROR in client {self.client_id} training: {str(e)}"
            print(error_msg)
            with open(f"{log_dir}/error_log.txt", "a") as f:
                f.write(f"{error_msg}\n")
            
            # Return a default value to allow the simulation to continue
            return 0.0

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        try:
            set_weights(self.net, parameters)
            
            # Get evaluation configuration
            val_steps = config.get("val_steps", None)  # Number of batches to evaluate on
            current_round = config.get("server_round", 0)
            
            # Log evaluation start
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} evaluating in round {current_round}")
            
            # Perform evaluation with optional steps limit
            loss, accuracy = evaluate_model(self.net, self.testset, val_steps=val_steps)
            
            # Store evaluation metrics
            if current_round not in client_metrics:
                client_metrics[current_round] = {"clients": []}
            
            # Update or add client evaluation metrics
            client_data = {
                "id": self.client_id,
                "evaluation_loss": loss,
                "evaluation_accuracy": accuracy
            }
            
            # Check if this client already has metrics
            client_updated = False
            if "clients" in client_metrics[current_round]:
                for idx, client in enumerate(client_metrics[current_round]["clients"]):
                    if client.get("id") == self.client_id:
                        # Update existing client entry with evaluation metrics
                        client_metrics[current_round]["clients"][idx].update(client_data)
                        client_updated = True
                        break
            
            # Add new client entry if not found
            if not client_updated:
                if "clients" not in client_metrics[current_round]:
                    client_metrics[current_round]["clients"] = []
                client_metrics[current_round]["clients"].append(client_data)
            
            # Save updated metrics
            save_round_metrics(current_round)
            
            print(f"  Client {self.client_id} evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
            
            return loss, len(self.testset), {"loss": loss, "accuracy": accuracy}
        
        except Exception as e:
            # Log evaluation failure
            error_msg = f"ERROR in client {self.client_id} evaluate method: {str(e)}"
            print(error_msg)
            
            # Write error to log file
            with open(f"{log_dir}/error_log.txt", "a") as f:
                f.write(f"{error_msg}\n")
            
            # Return default values to allow simulation to continue
            return 0.0, 0, {"error": str(e)}
    
def save_round_metrics(round_num):
    """Save metrics for the current round to a JSON file"""
    if round_num not in client_metrics:
        return
    
    round_data = client_metrics[round_num]
    
    # Initialize global metrics dict if it doesn't exist
    if "global" not in round_data:
        round_data["global"] = {}
    
    # Ensure clients data is preserved in the output JSON
    if "clients" not in round_data:
        round_data["clients"] = []
    
    # Calculate global metrics (averages across all clients)
    if "clients" in round_data and round_data["clients"]:
        train_losses = [client["training_loss"] for client in round_data["clients"]]
        val_losses = [client["validation_loss"] for client in round_data["clients"]]
        val_accuracies = [client["validation_accuracy"] for client in round_data["clients"]]
        
        round_data["global"].update({
            "training_loss": sum(train_losses) / len(train_losses),
            "validation_loss": sum(val_losses) / len(val_losses),
            "validation_accuracy": sum(val_accuracies) / len(val_accuracies)
        })
    
    # Add test subset accuracies if round data includes them
    if "test_subset_metrics" in round_data:
        round_data["global"].update(round_data["test_subset_metrics"])
    
    # Create a clean copy of the metrics to save (this ensures the entire structure is saved)
    output_data = {
        "global": round_data.get("global", {}),
        "clients": round_data.get("clients", []),
        "test_subset_metrics": round_data.get("test_subset_metrics", {})
    }
    
    # Add special round information
    if "initial_round" in round_data:
        output_data["initial_round"] = round_data["initial_round"]
    if "message" in round_data:
        output_data["message"] = round_data["message"]
    
    # Save to file
    filename = f"{log_dir}/round_{round_num:03d}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved metrics for round {round_num} to {filename}")
    log(INFO, f"Saved metrics for round {round_num} to {filename}")

# Client function
def client_fn(context: Context) -> Client:
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating client with partition ID: {context.node_config['partition-id']}")
    net = ResNet20().to(device)
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = testset
    print(f"  Client {partition_id} dataset size: {len(client_train)}")
    return FlowerClient(net, client_train, client_test, partition_id).to_client()

client = ClientApp(client_fn)

def evaluate(server_round, parameters, config):
    # print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Round {server_round} - Server Evaluation Starting")
    start_time = time.time()
    
    net = ResNet20().to(device)
    set_weights(net, parameters)

    test_loss, accuracy = evaluate_model(net, testset)
    
    # Store test subset metrics
    test_subset_metrics = {}
    
    # Evaluate on all test subsets
    for name, subset in test_subsets.items():
        subset_loss, subset_accuracy = evaluate_model(net, subset)
        test_subset_metrics[f"{name}_accuracy"] = subset_accuracy
        test_subset_metrics[f"{name}_loss"] = subset_loss
        # print(f"  Test accuracy on {name}: {subset_accuracy:.4f}")
        log(INFO, f"test accuracy on {name}: %.4f", subset_accuracy)
    
    # print(f"  Test accuracy on all classes: {accuracy:.4f}")
    log(INFO, "test accuracy on all classes: %.4f", accuracy)

    # Store test subset metrics in the round data
    if server_round not in client_metrics:
        client_metrics[server_round] = {}
    
    # For initial round (0), create placeholder client data
    if server_round == 0 and "clients" not in client_metrics[server_round]:
        client_metrics[server_round]["clients"] = []
        # Add explanation for empty clients in round 0
        client_metrics[server_round]["initial_round"] = True
        client_metrics[server_round]["message"] = "This is the initial evaluation round before any client training"
    
    client_metrics[server_round]["test_subset_metrics"] = test_subset_metrics
    client_metrics[server_round]["global_accuracy"] = accuracy
    client_metrics[server_round]["global_loss"] = test_loss
    
    # Save metrics for this round
    save_round_metrics(server_round)
    
    eval_time = time.time() - start_time
    # print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {server_round} - Server Evaluation completed in {eval_time:.2f}s")

    if server_round == num_rounds:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed! Generating confusion matrix...")
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing global model...")
net = ResNet20().to(device)
params = ndarrays_to_parameters(get_weights(net))

# Define configuration functions to provide to clients during training
def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round.
    
    This allows for dynamic round-specific configuration.
    """
    # Optionally decrease learning rate as training progresses
    lr_decay = 1.0
    if server_round > 50:
        lr_decay = 0.5
    elif server_round > 30:
        lr_decay = 0.8
    
    config = {
        "server_round": server_round,  # Inform client of the current round
        "learning_rate": learning_rate * lr_decay,  # Adjust learning rate based on round
        "momentum": momentum,
        "batch_size": batch_size,
        "epochs": epochs,
        "local_epochs": epochs,  # Alias for epochs
    }
    return config

# Define configuration for evaluation
def evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """Return evaluation configuration dict for each round."""
    return {
        "server_round": server_round,
        "val_steps": 10,  # Number of batches to evaluate on
    }

# Aggregation function for evaluation metrics
def aggregate_evaluate_metrics(eval_metrics):
    """Aggregate evaluation metrics from multiple clients."""
    if not eval_metrics:
        return {}
    
    # Extract accuracies and losses
    accuracies = [
        metrics["accuracy"] for _, metrics in eval_metrics if "accuracy" in metrics
    ]
    losses = [
        metrics["loss"] for _, metrics in eval_metrics if "loss" in metrics
    ]
    
    metrics = {}
    if accuracies:
        metrics["accuracy"] = float(np.mean(accuracies))
    if losses:
        metrics["loss"] = float(np.mean(losses))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregated evaluation metrics: {metrics}")
    return metrics

def server_fn(context: Context):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Server initialized with {num_rounds} rounds")
    
    # Comprehensive FedAvg strategy with all configuration options
    strategy = FedAvg(
        # Sampling configuration
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=0.5,  # Use half of available clients for evaluation
        
        # Minimum client thresholds
        min_fit_clients=1,  # Allow training to proceed even if only one client succeeds
        min_evaluate_clients=1,  # Allow evaluation with at least one client
        min_available_clients=num_clients,  # Require all clients to be available
        
        # Handling failures
        accept_failures=True,  # Continue despite client failures
        min_completion_rate_fit=0.1,  # Allow a lower completion rate
        
        # Initial model
        initial_parameters=params,
        
        # Evaluation function
        evaluate_fn=evaluate,
        
        # Configuration functions
        on_fit_config_fn=fit_config,  # Pass configuration to clients during training
        on_evaluate_config_fn=evaluate_config,  # Pass configuration during evaluation
        
        # Metrics aggregation
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        
        # Optimization
        inplace=True  # Enable in-place aggregation for better memory efficiency
    )
    config=ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )

def aggregate_fit_metrics(fit_metrics):
    """Aggregate fit metrics from clients."""
    if not fit_metrics:
        return {}
    
    # Extract metrics
    training_losses = [
        metrics["training_loss"] for _, metrics in fit_metrics if "training_loss" in metrics
    ]
    
    if training_losses:
        avg_loss = float(np.mean(training_losses))
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregated training loss: {avg_loss:.4f}")
        return {
            "training_loss": avg_loss
        }
    return {}

server = ServerApp(server_fn=server_fn)

# Update backend_setup to explicitly specify a different backend
backend_setup = {
    "backend_name": "simulation",  # Use the built-in simulation backend instead of Ray
    "init_args": {
        "logging_level": ERROR, 
        "log_to_driver": False
    }
}

# Initiate the simulation passing the server and client apps
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting federated learning simulation...")
print(f"  Number of clients: {num_clients}")
print(f"  Number of rounds: {num_rounds}")
print(f"  Epochs per round: {epochs}")
print(f"  Device: {device}")
print(f"=========================================\n")

simulation_start = time.time()
# Specify the number of super nodes that will be selected on every round
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=num_clients,
    backend_config=backend_setup,
)

simulation_time = time.time() - simulation_start
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Simulation completed in {simulation_time:.2f} seconds")

# Save summary information
summary = {
    "timestamp": timestamp,
    "total_runtime_seconds": simulation_time,
    "config": {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "data_fraction": data_fraction,
        "device": str(device)
    },
    "final_metrics": client_metrics.get(num_rounds, {}).get("global", {})
}

# Add information about the last round if available
if num_rounds in client_metrics and "global_accuracy" in client_metrics[num_rounds]:
    summary["final_accuracy"] = client_metrics[num_rounds]["global_accuracy"]

# Save summary to the log directory
with open(f"{log_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {log_dir}/summary.json")

