"""
SASWISE Fed-102 ML task implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import numpy as np


# Define the neural network model
class Net(nn.Module):
    """Simple CNN model for image classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_weights(model):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def load_data(partition_id=0, num_partitions=10):
    """Load CIFAR-10 training and validation data for a specific partition."""
    # Define data transformations
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training data
    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    
    # Partition dataset based on partition_id
    n_samples = len(trainset)
    partition_size = n_samples // num_partitions
    partition_indices = list(range(partition_id * partition_size, (partition_id + 1) * partition_size))
    
    # Create data loader for this partition
    trainloader = DataLoader(
        trainset, 
        batch_size=32, 
        sampler=torch.utils.data.SubsetRandomSampler(partition_indices)
    )
    
    # Load validation data (use the same for all clients)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)
    valloader = DataLoader(testset, batch_size=32)
    
    return trainloader, valloader


def train(net, trainloader, epochs, device="cuda:0"):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    running_loss = 0.0
    net.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Compute average loss for this epoch
        avg_loss = epoch_loss / len(trainloader)
        running_loss = avg_loss if epoch == 0 else (0.9 * running_loss + 0.1 * avg_loss)
        
    return running_loss


def test(net, testloader, device="cuda:0"):
    """Validate the model on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    
    with torch.no_grad():
        net.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return {"accuracy": accuracy, "loss": avg_loss}
