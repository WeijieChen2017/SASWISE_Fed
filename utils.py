import os
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import medmnist
from medmnist import INFO

# Function to create a dataset partition for a single client
def create_client_partition(
    train_dataset, 
    test_dataset,
    indices, 
    batch_size,
    seed=42
):
    """
    Create train, validation, and test dataloaders for a client.
    
    Args:
        train_dataset: The full training dataset
        test_dataset: The full test dataset
        indices: List of indices to use from the training dataset
        batch_size: Batch size for dataloaders
        seed: Random seed
    """
    # Create a subset of the training dataset
    train_subset = Subset(train_dataset, indices)
    
    # Split training subset into train and validation
    val_size = max(1, int(0.1 * len(train_subset)))  # 10% for validation
    train_size = len(train_subset) - val_size
    
    train_subset, val_subset = data.random_split(
        train_subset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create a client-specific test loader
    # In a real-world scenario, each client would have its own test set
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_train_examples": len(train_subset),
        "num_val_examples": len(val_subset),
        "num_test_examples": len(test_dataset)
    }

def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    """
    Partition dataset among clients using Dirichlet distribution.
    This creates a non-uniform partition of the data based on labels.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients to create partitions for
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        # Get labels by iterating through dataset
        # Note: This is inefficient but necessary if dataset doesn't have .targets attribute
        labels = []
        for _, target in dataset:
            labels.append(target)
        labels = torch.tensor(labels)
    
    # Ensure labels are 1D
    if len(labels.shape) > 1:
        labels = labels.squeeze()
    
    num_samples = len(dataset)
    num_classes = len(torch.unique(labels))
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute examples among clients
    for class_idx in range(num_classes):
        # Get indices of examples for this class
        class_indices = torch.where(labels == class_idx)[0].tolist()
        # Distribute these examples using Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to actual sample counts
        proportions = np.array([p*(len(class_indices)) for p in proportions])
        proportions = np.round(proportions).astype(int)
        # Make sure we don't use more samples than available
        if sum(proportions) > len(class_indices):
            proportions[-1] = len(class_indices) - sum(proportions[:-1])
        
        # Distribute samples
        class_indices_idx = 0
        for client_idx, count in enumerate(proportions):
            if count > 0:
                client_indices[client_idx].extend(
                    class_indices[class_indices_idx:class_indices_idx+count]
                )
                class_indices_idx += count
    
    # Ensure every client has at least one example
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            # Choose a random example from the largest client
            largest_client = np.argmax([len(indices) for indices in client_indices])
            example_to_move = random.choice(client_indices[largest_client])
            client_indices[i].append(example_to_move)
            client_indices[largest_client].remove(example_to_move)
    
    return client_indices

def uniform_partition(dataset, num_clients, seed=42):
    """
    Create uniform partitions of dataset for each client.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Get all indices
    indices = list(range(len(dataset)))
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    # Split indices evenly
    partition_size = len(indices) // num_clients
    client_indices = []
    
    for i in range(num_clients):
        if i < num_clients - 1:
            client_indices.append(indices[i*partition_size:(i+1)*partition_size])
        else:
            # Last client gets the remainder
            client_indices.append(indices[i*partition_size:])
    
    return client_indices

def create_client_data_partitions(
    train_dataset, 
    test_dataset, 
    num_clients=5, 
    batch_size=64, 
    partition_type="uniform", 
    alpha=0.5,
    seed=42
):
    """
    Create data partitions for multiple clients.
    
    Args:
        train_dataset: The full training dataset
        test_dataset: The full test dataset
        num_clients: Number of clients to create
        batch_size: Batch size for dataloaders
        partition_type: Type of partitioning ("uniform" or "dirichlet")
        alpha: Dirichlet concentration parameter (only used if partition_type="dirichlet")
        seed: Random seed
    """
    # Create indices for each client based on partition type
    if partition_type == "uniform":
        client_indices = uniform_partition(train_dataset, num_clients, seed)
    else:  # dirichlet
        client_indices = dirichlet_partition(train_dataset, num_clients, alpha, seed)
    
    # Create partitions for each client
    client_partitions = {}
    
    for client_id in range(num_clients):
        client_partitions[client_id] = create_client_partition(
            train_dataset,
            test_dataset,
            client_indices[client_id],
            batch_size,
            seed
        )
    
    # Print partition statistics
    print(f"\nCreated {num_clients} client partitions using {partition_type} partitioning:")
    for client_id, partition in client_partitions.items():
        print(f"Client {client_id}: {partition['num_train_examples']} training examples")
    
    # Calculate class distribution for heterogeneous (dirichlet) split
    if partition_type == "dirichlet":
        print("\nClass distribution across clients:")
        # TODO: Add code to calculate and print class distribution
    
    return client_partitions

def load_datasets(data_flag, img_size=64):
    """
    Load MedMNIST datasets.
    
    Args:
        data_flag: Dataset flag 
        img_size: Image size to use
    
    Returns:
        train_dataset, test_dataset
    """
    # Define data transforms
    info = INFO[data_flag]
    
    # 3D datasets handle their own transforms internally
    if data_flag in ['organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 
                      'fracturemnist3d', 'vesselmnist3d', 'synapsemnist3d']:
        data_transform = None
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    
    # Get the dataset class
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=True, size=img_size)
    test_dataset = DataClass(split='test', transform=data_transform, download=True, size=img_size)
    
    print(f"Dataset: {data_flag}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, test_dataset 