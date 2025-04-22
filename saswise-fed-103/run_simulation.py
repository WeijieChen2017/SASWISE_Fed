from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
import json
import torch

from utils2 import *

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
with open("saswise-fed-103/config.json", "r") as f:
    config = json.load(f)

# Training parameters
num_clients = config["training"]["num_clients"]
num_rounds = config["training"]["num_rounds"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
momentum = config["training"]["momentum"]

trainset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=True, transform=transform
)

total_length = len(trainset)
split_size = total_length // num_clients
torch.manual_seed(42)
train_sets = random_split(trainset, [split_size] * num_clients)

# Apply exclusions based on config
for i, client_config in enumerate(config["clients"]):
    train_sets[i] = exclude_classes(train_sets[i], excluded_classes=client_config["excluded_classes"])

testset = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=False, transform=transform
)
print("Number of examples in `testset`:", len(testset))

# Create test subsets based on config
test_subsets = {}
for subset in config["evaluation"]["test_subsets"]:
    test_subsets[subset["name"]] = include_classes(testset, subset["classes"])

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

