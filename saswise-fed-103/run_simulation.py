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

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading configuration...")
# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

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
        start_time = time.time()
        current_round = config.get("server_round", 0)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} starting training for round {current_round}")
        
        set_weights(self.net, parameters)
        # Customize the training with parameters from config
        training_loss = self.train_model(self.net, self.trainset)
        # Evaluate after training to get validation metrics
        val_loss, val_accuracy = evaluate_model(self.net, self.testset)
        
        # Store metrics for this client
        if current_round not in client_metrics:
            client_metrics[current_round] = {"clients": []}
        
        client_metrics[current_round]["clients"].append({
            "id": self.client_id,
            "training_loss": training_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy
        })
        
        train_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} completed round {current_round} in {train_time:.2f}s")
        print(f"  Training Loss: {training_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return get_weights(self.net), len(self.trainset), {"training_loss": training_loss}
    
    def train_model(self, model, train_set):
        # Debug output to identify if we're entering the training loop
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} training started with {len(train_set)} samples")
        
        # Create smaller batches to reduce memory usage and potential hanging
        effective_batch_size = min(batch_size, 64)  # Limit batch size if it's too large
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
                    print(f"\r  Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Client {self.client_id}", end="")
                    # Force flush output to make sure progress is shown immediately
                    import sys
                    sys.stdout.flush()
                
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1
                except Exception as e:
                    print(f"\n  ERROR in batch {batch_idx}: {str(e)}")
                    # Continue with next batch instead of failing
                    continue
                
                # Free up memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Clear references to reduce memory pressure
                del inputs, labels, outputs, loss
            
            epoch_avg_loss = epoch_loss / max(len(train_loader), 1)  # Avoid division by zero
            total_loss += epoch_avg_loss
            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - Avg Loss: {epoch_avg_loss:.4f}")
        
        # Return average loss across all epochs
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Client {self.client_id} training finished")
        return total_loss / epochs

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}
    
def save_round_metrics(round_num):
    """Save metrics for the current round to a JSON file"""
    if round_num not in client_metrics:
        return
    
    round_data = client_metrics[round_num]
    
    # Initialize global metrics dict if it doesn't exist
    if "global" not in round_data:
        round_data["global"] = {}
    
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
    
    # Save to file
    filename = f"logs/round_{round_num:03d}.json"
    with open(filename, "w") as f:
        json.dump(round_data, f, indent=2)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved metrics for round {round_num} to {filename}")
    log(INFO, f"Saved metrics for round {round_num} to {filename}")

# Client function
def client_fn(context: Context) -> Client:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating client with partition ID: {context.node_config['partition-id']}")
    net = ResNet20().to(device)
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = testset
    print(f"  Client {partition_id} dataset size: {len(client_train)}")
    return FlowerClient(net, client_train, client_test, partition_id).to_client()

client = ClientApp(client_fn)

def evaluate(server_round, parameters, config):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Round {server_round} - Server Evaluation Starting")
    start_time = time.time()
    
    net = ResNet20().to(device)
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    
    # Store test subset metrics
    test_subset_metrics = {}
    
    # Evaluate on all test subsets
    for name, subset in test_subsets.items():
        subset_loss, subset_accuracy = evaluate_model(net, subset)
        test_subset_metrics[f"{name}_accuracy"] = subset_accuracy
        test_subset_metrics[f"{name}_loss"] = subset_loss
        print(f"  Test accuracy on {name}: {subset_accuracy:.4f}")
        log(INFO, f"test accuracy on {name}: %.4f", subset_accuracy)
    
    print(f"  Test accuracy on all classes: {accuracy:.4f}")
    log(INFO, "test accuracy on all classes: %.4f", accuracy)

    # Store test subset metrics in the round data
    if server_round not in client_metrics:
        client_metrics[server_round] = {}
    client_metrics[server_round]["test_subset_metrics"] = test_subset_metrics
    client_metrics[server_round]["global_accuracy"] = accuracy
    
    # Save metrics for this round
    save_round_metrics(server_round)
    
    eval_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {server_round} - Server Evaluation completed in {eval_time:.2f}s")

    if server_round == num_rounds:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed! Generating confusion matrix...")
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing global model...")
net = ResNet20().to(device)
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Server initialized with {num_rounds} rounds")
    
    # Set a shorter timeout to avoid hanging
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        min_fit_clients=num_clients,  # Make sure all clients participate
        min_available_clients=num_clients  # Ensure we have enough clients
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Aggregated training loss: {avg_loss:.4f}")
        return {
            "training_loss": avg_loss
        }
    return {}

server = ServerApp(server_fn=server_fn)

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

