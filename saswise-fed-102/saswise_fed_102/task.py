"""
SASWISE Fed-102 ML task implementation for medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import nibabel as nib
import numpy as np


# Define a custom dataset for NIFTI medical images
class SIIMDataset(Dataset):
    """Dataset for SIIM medical images in NIFTI format."""
    
    def __init__(self, data_dir, labels_dir=None, transform=None):
        """
        Args:
            data_dir (str): Directory with all the NIFTI images
            labels_dir (str, optional): Directory with label files
            transform (callable, optional): Optional transform to be applied on samples
        """
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # Get all NIFTI files
        self.file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii')])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # Load NIFTI image
        img_name = self.file_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load image using nibabel
        nifti_img = nib.load(img_path)
        img_data = nifti_img.get_fdata()
        
        # Normalize image to 0-1 range
        if img_data.max() > 0:
            img_data = img_data / img_data.max()
        
        # Convert to appropriate tensor format - assuming 3D images, take middle slice
        if len(img_data.shape) == 3:
            middle_slice = img_data.shape[2] // 2
            img_data = img_data[:, :, middle_slice]
        
        # Resize to fixed dimensions if needed (e.g., 224x224)
        img_tensor = torch.from_numpy(img_data).float()
        
        # Expand dimensions to create channels (1 channel for grayscale)
        img_tensor = img_tensor.unsqueeze(0)
        
        # Apply transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Load label if available
        if self.labels_dir:
            label_path = os.path.join(self.labels_dir, img_name.replace('.nii', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = int(f.read().strip())
            else:
                # Default label if not found
                label = 0
        else:
            # Default label if no labels directory
            label = 0
            
        return img_tensor, label


# Define a 3D CNN model for medical images
class Net(nn.Module):
    """CNN model for medical image classification."""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # Input: 1 channel (grayscale medical image)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Assuming input size of 224x224, after 3 pooling layers: 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_weights(model):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def load_data(site_id=1, num_sites=6):
    """Load SIIM medical imaging data for a specific site."""
    # Base data directory path - adjust this to point to the actual path
    base_dir = "/shares/mimrtl/SharedDatasets/SIIM_FedLearn24/SIIM_Fed_Learning_Phase1Data"
    
    # Path for the specific site
    site_dir = os.path.join(base_dir, f"Site_{site_id}_Test")
    
    # Training data paths
    train_data_dir = os.path.join(site_dir, "For_FedTraining", "data")
    train_labels_dir = os.path.join(site_dir, "For_FedTraining", "labels")
    
    # If base_dir doesn't exist, use a relative path for testing
    if not os.path.exists(base_dir):
        print(f"Warning: {base_dir} not found. Using local directory for testing.")
        # Create a fallback directory structure for testing
        base_dir = "./data/SIIM_Fed_Learning_Phase1Data"
        site_dir = os.path.join(base_dir, f"Site_{site_id}_Test")
        train_data_dir = os.path.join(site_dir, "For_FedTraining", "data")
        train_labels_dir = os.path.join(site_dir, "For_FedTraining", "labels")
        
        # Create the directories if they don't exist
        os.makedirs(train_data_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
    
    # Create dataset for the site
    train_dataset = SIIMDataset(
        data_dir=train_data_dir,
        labels_dir=train_labels_dir
    )
    
    # Split into train and validation sets (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=4,  # Smaller batch size for 3D medical images
        shuffle=True
    )
    
    valloader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )
    
    return trainloader, valloader


def train(net, trainloader, epochs, device="cuda:0"):
    """Train the model on the training set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
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
        avg_loss = epoch_loss / len(trainloader) if len(trainloader) > 0 else 0
        running_loss = avg_loss if epoch == 0 else (0.9 * running_loss + 0.1 * avg_loss)
        
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')
        
    return running_loss


def test(net, testloader, device="cuda:0"):
    """Validate the model on the test set."""
    net.to(device)
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
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else 0
    return {"accuracy": accuracy, "loss": avg_loss}
