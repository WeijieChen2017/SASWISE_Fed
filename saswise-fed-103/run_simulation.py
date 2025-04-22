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

from utils2 import *

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

trainset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=True, transform=transform
)

# Calculate correct split sizes that sum to the total dataset length
total_length = len(trainset)
split_sizes = []
remaining_samples = total_length
for i in range(num_clients - 1):
    # Use equal splits (or we could implement percentage-based splits)
    size = total_length // num_clients
    split_sizes.append(size)
    remaining_samples -= size
# Add the remaining samples to the last split
split_sizes.append(remaining_samples)

print(f"Total dataset size: {total_length}")
print(f"Split sizes: {split_sizes}, sum: {sum(split_sizes)}")

torch.manual_seed(42)
train_sets = random_split(trainset, split_sizes)

# Helper function to get labels from nested datasets
def get_label_from_subset(dataset, idx):
    """Extract label from potentially nested Subset datasets"""
    if isinstance(idx, int):
        # Direct index into dataset
        return dataset[idx][1]
    else:
        # For Subset objects, we need to navigate through the dataset hierarchy
        current_dataset = dataset
        current_idx = idx
        while hasattr(current_dataset, 'dataset'):
            if hasattr(current_idx, 'item'):
                current_idx = current_idx.item()  # Convert tensor to int if needed
            if isinstance(current_dataset, Subset):
                if isinstance(current_idx, int):
                    current_idx = current_dataset.indices[current_idx]
                current_dataset = current_dataset.dataset
            else:
                break
        return current_dataset[current_idx][1]

# Apply exclusions based on config and report client datasets
for i, client_config in enumerate(config["clients"]):
    original_size = len(train_sets[i])
    
    # Apply the data fraction if needed (e.g., 80% of data)
    if data_fraction < 1.0:
        subset_size = int(len(train_sets[i]) * data_fraction)
        indices = torch.randperm(len(train_sets[i]))[:subset_size]
        train_sets[i] = Subset(train_sets[i], indices)
        print(f"Client {i}: Using {data_fraction*100}% of data. Original size: {original_size}, New size: {len(train_sets[i])}")
    
    # Count classes before exclusion
    labels_before = [get_label_from_subset(trainset, train_sets[i][j]) for j in range(len(train_sets[i]))]
    class_counts_before = Counter(labels_before)
    
    # Apply exclusions
    train_sets[i] = exclude_classes(train_sets[i], excluded_classes=client_config["excluded_classes"])
    
    # Count classes after exclusion
    labels_after = [get_label_from_subset(trainset, train_sets[i][j]) for j in range(len(train_sets[i]))]
    class_counts_after = Counter(labels_after)
    
    print(f"\nClient {i} dataset report:")
    print(f"  Original samples: {original_size}")
    print(f"  After exclusion: {len(train_sets[i])}")
    print(f"  Excluded classes: {client_config['excluded_classes']}")
    print(f"  Class distribution before exclusion: {dict(class_counts_before)}")
    print(f"  Class distribution after exclusion: {dict(class_counts_after)}")
    print(f"  Classes present: {sorted(class_counts_after.keys())}")

testset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=False, transform=transform
)
print("\nNumber of examples in `testset`:", len(testset))

# Create test subsets based on config
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

class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        self.net = net
        self.trainset = trainset
        self.testset = testset

    # Train the model
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        # Customize the training with parameters from config
        self.train_model(self.net, self.trainset)
        return get_weights(self.net), len(self.trainset), {}
    
    def train_model(self, model, train_set):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}
    
# Client function
def client_fn(context: Context) -> Client:
    net = ResNet20().to(device)
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = testset
    return FlowerClient(net, client_train, client_test).to_client()

client = ClientApp(client_fn)

def evaluate(server_round, parameters, config):
    net = ResNet20().to(device)
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    
    # Evaluate on all test subsets
    for name, subset in test_subsets.items():
        _, subset_accuracy = evaluate_model(net, subset)
        log(INFO, f"test accuracy on {name}: %.4f", subset_accuracy)
    
    log(INFO, "test accuracy on all digits: %.4f", accuracy)

    if server_round == num_rounds:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

net = ResNet20().to(device)
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params,
        evaluate_fn=evaluate,
    )
    config=ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )

server = ServerApp(server_fn=server_fn)

# Initiate the simulation passing the server and client apps
# Specify the number of super nodes that will be selected on every round
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=num_clients,
    backend_config=backend_setup,
)

