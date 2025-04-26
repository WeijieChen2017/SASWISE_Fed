from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

# from utils2 import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from flwr.common.logger import log
from logging import INFO, ERROR
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": False}}


import torchvision.transforms as transforms
import torch
import ray

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 channel statistics (computed over the 50K training images)
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

train_transform_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # pad then random‐crop back to 32×32
    transforms.RandomHorizontalFlip(),       # 50% chance flip
    transforms.ToTensor(),                   # to [0,1] tensor
    transforms.Normalize(mean, std),         # zero‐mean, unit‐variance
])

test_transform_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_set_CIFAR10 = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=True, transform=train_transform_CIFAR10
)

total_length = len(train_set_CIFAR10)
split_size = total_length // 5
torch.manual_seed(42)
part1, part2, part3, part4, part5 = random_split(train_set_CIFAR10, [split_size] * 5)

def include_classes(dataset, included_classes):
    """
    Create a subset of CIFAR-10 dataset including only specified classes.
    
    Args:
        dataset: The CIFAR-10 dataset
        included_classes: List of class indices to include
        
    Returns:
        Subset of the dataset with only the included classes
    """
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] in included_classes
    ]
    return torch.utils.data.Subset(dataset, including_indices)

def exclude_classes(dataset, excluded_classes):
    """
    Create a subset of CIFAR-10 dataset excluding specified classes.
    
    Args:
        dataset: The CIFAR-10 dataset
        excluded_classes: List of class indices to exclude
        
    Returns:
        Subset of the dataset without the excluded classes
    """
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_classes
    ]
    return torch.utils.data.Subset(dataset, including_indices)

part1 = exclude_classes(part1, excluded_classes=[1, 3, 7])
part2 = exclude_classes(part2, excluded_classes=[2, 5, 8])
part3 = exclude_classes(part3, excluded_classes=[4, 6, 9])
part4 = exclude_classes(part4, excluded_classes=[3, 5, 9])
part5 = exclude_classes(part5, excluded_classes=[1, 4, 9])

train_sets_CIFAR10 = [part1, part2, part3, part4, part5]

testset_CIFAR10 = datasets.CIFAR10(
    "./CIFAR10_data/", download=True, train=False, transform=test_transform_CIFAR10
)
testset_137 = include_classes(testset_CIFAR10, [1, 3, 7])
testset_258 = include_classes(testset_CIFAR10, [2, 5, 8])
testset_469 = include_classes(testset_CIFAR10, [4, 6, 9])
testset_359 = include_classes(testset_CIFAR10, [3, 5, 9])
testset_149 = include_classes(testset_CIFAR10, [1, 4, 9])

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

def evaluate_model(model, test_set, val_steps=None, device=None):
    # If device is not provided, get it from the model
    if device is None:
        device = next(model.parameters()).device  # Get device from model
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # If val_steps is provided, only evaluate on that many batches
            if val_steps is not None and batch_idx >= val_steps:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # Adjust loss calculation if we're using val_steps
    if val_steps is not None and batch_idx < len(test_loader):
        num_batches = min(val_steps, batch_idx + 1)
    else:
        num_batches = len(test_loader)
        
    accuracy = correct / total
    average_loss = total_loss / num_batches
    # print(f"Test Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}")
    return average_loss, accuracy

def train_model(model, train_set, device=None):
    batch_size = 64
    num_epochs = 10

    # If device is not provided, get it from the model
    if device is None:
        device = next(model.parameters()).device

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        self.net = net.to(device)
        self.trainset = trainset
        self.testset = testset

    # Train the model
    def fit(self, parameters, config):
        try:
            set_weights(self.net, parameters)
            train_model(self.net, self.trainset, device=device)
            return get_weights(self.net), len(self.trainset), {}
        except Exception as e:
            print(f"Client training error: {e}")
            raise

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset, device=device)
        return loss, len(self.testset), {"accuracy": accuracy}
    
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet-20 has 3 stages, each with 3 residual blocks (3*6+2=20 layers)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def client_fn_CIFAR10(context: Context) -> Client:
    net = ResNet20().to(device)
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets_CIFAR10[int(partition_id)]
    client_test = testset_CIFAR10
    return FlowerClient(net, client_train, client_test).to_client()
client_app_CIFAR10 = ClientApp(client_fn_CIFAR10)

def evaluate_CIFAR10(server_round, parameters, config):
    net = ResNet20().to(device)
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset_CIFAR10, device=device)
    _, accuracy137 = evaluate_model(net, testset_137, device=device)
    _, accuracy258 = evaluate_model(net, testset_258, device=device)
    _, accuracy469 = evaluate_model(net, testset_469, device=device)
    _, accuracy359 = evaluate_model(net, testset_359, device=device)
    _, accuracy149 = evaluate_model(net, testset_149, device=device)

    log(INFO, "test accuracy on all digits: %.4f", accuracy)
    log(INFO, "test accuracy on [1,3,7]: %.4f", accuracy137)
    log(INFO, "test accuracy on [2,5,8]: %.4f", accuracy258)
    log(INFO, "test accuracy on [4,6,9]: %.4f", accuracy469)
    log(INFO, "test accuracy on [3,5,9]: %.4f", accuracy359)
    log(INFO, "test accuracy on [1,4,9]: %.4f", accuracy149)

    if server_round == 5:
        cm = compute_confusion_matrix(net, testset_CIFAR10, device=device)
        plot_confusion_matrix(cm, "Final Global Model")

net_CIFAR10 = ResNet20().to(device)
params_CIFAR10 = ndarrays_to_parameters(get_weights(net_CIFAR10))

def server_fn_CIFAR10(context: Context):
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        initial_parameters=params_CIFAR10,
        evaluate_fn=evaluate_CIFAR10,
    )
    config=ServerConfig(num_rounds=5)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )
server_CIFAR10 = ServerApp(server_fn=server_fn_CIFAR10)

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def compute_confusion_matrix(model, testset, device=None):
    # If device is not provided, get it from the model
    if device is None:
        device = next(model.parameters()).device
    
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over the test set to get predictions
    for image, label in testset:
        # Forward pass through the model to get predictions
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        output = model(image)
        _, predicted = torch.max(output, 1)

        # Append true and predicted labels to lists
        true_labels.append(label)
        predicted_labels.append(predicted.item())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    return cm

for i, dataset in enumerate(train_sets_CIFAR10):
    print(f"Client {i} dataset size: {len(dataset)}")

# Add this before run_simulation
if not ray.is_initialized():
    ray.init(num_gpus=1, ignore_reinit_error=True)

# Then modify your run_simulation call
run_simulation(
    server_app=server_CIFAR10,
    client_app=client_app_CIFAR10,
    num_supernodes=5,
    backend_config={
        **backend_setup,
        "ray_init_args": {"num_gpus": 1, "resources": {"GPU": 1}}
    },
)