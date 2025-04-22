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

print(f"Total dataset size: {len(trainset)}")

# Prepare client datasets - each client starts with the full dataset
train_sets = []
for i, client_config in enumerate(config["clients"]):
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
        print(f"Client {i}: Using {data_fraction*100}% of data. Size after fraction: {len(client_dataset)}")
    
    # Apply exclusions based on config
    client_dataset = exclude_classes(client_dataset, excluded_classes=client_config["excluded_classes"])
    print(f"Client {i}: Size after excluding classes {client_config['excluded_classes']}: {len(client_dataset)}")
    
    # Analyze class distribution
    labels = []
    for j in range(len(client_dataset)):
        # Get the label directly from the dataset item
        label = client_dataset[j][1]
        labels.append(label)
    
    class_counts = Counter(labels)
    
    print(f"\nClient {i} dataset report:")
    print(f"  Final dataset size: {len(client_dataset)}")
    print(f"  Excluded classes: {client_config['excluded_classes']}")
    print(f"  Class distribution: {dict(class_counts)}")
    print(f"  Classes present: {sorted(class_counts.keys())}")
    
    train_sets.append(client_dataset)

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

