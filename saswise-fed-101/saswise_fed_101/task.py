"""saswise-fed-101: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision
import numpy as np


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


# Wrapper around torchvision CIFAR-10 to match interface for flwr_datasets
class CIFAR10Wrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return {"img": img, "label": label}
    
    def __len__(self):
        return len(self.dataset)


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Load partition CIFAR10 data using torchvision to avoid SSL issues."""
    # Define transforms
    transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load the dataset from torchvision (will be cached automatically)
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train=True, 
            download=True,
            transform=transforms
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train=False, 
            download=True,
            transform=transforms
        )
    except Exception as e:
        print(f"Error downloading CIFAR-10 with SSL verification: {e}")
        # Try again with SSL verification disabled as a fallback
        import ssl
        old_https_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", 
                train=True, 
                download=True, 
                transform=transforms
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root="./data", 
                train=False, 
                download=True, 
                transform=transforms
            )
        finally:
            # Restore the original SSL context
            ssl._create_default_https_context = old_https_context
    
    # Create partitions for federated learning
    n_train = len(train_dataset)
    samples_per_partition = n_train // num_partitions
    
    # Determine the range for this partition
    start_idx = partition_id * samples_per_partition
    end_idx = min((partition_id + 1) * samples_per_partition, n_train)
    
    # Create subset for this partition
    partition_train = Subset(train_dataset, range(start_idx, end_idx))
    
    # Further split into train, validation, and test (70/15/15)
    partition_size = len(partition_train)
    train_size = int(0.7 * partition_size)
    val_size = int(0.15 * partition_size)
    test_size = partition_size - train_size - val_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, partition_size))
    
    train_subset = Subset(partition_train, train_indices)
    val_subset = Subset(partition_train, val_indices)
    test_subset = Subset(partition_train, test_indices)
    
    # Wrap in our CIFAR10Wrapper to match the expected interface
    wrapped_train = CIFAR10Wrapper(train_subset)
    wrapped_val = CIFAR10Wrapper(val_subset)
    wrapped_test = CIFAR10Wrapper(test_subset)
    
    # Create data loaders with specified batch size
    trainloader = DataLoader(wrapped_train, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(wrapped_val, batch_size=batch_size)
    testloader = DataLoader(wrapped_test, batch_size=batch_size)
    
    return trainloader, valloader, testloader


# Original function using flwr_datasets kept for reference
def load_data_original(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
