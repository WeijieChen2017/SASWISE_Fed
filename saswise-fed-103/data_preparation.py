import torch
import time
from torchvision import datasets
from torch.utils.data import Subset
from collections import Counter
from datetime import datetime

# Assuming these functions are available in utils2.py
# If not, their definitions need to be included here or in utils2.py
try:
    from utils2 import transform, exclude_classes, include_classes
except ImportError:
    # Provide dummy implementations or raise a clearer error if utils2.py is missing/incomplete
    print("Warning: utils2.py not found or incomplete. Using dummy data functions.")
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    def exclude_classes(dataset, excluded_classes):
        if not excluded_classes:
            return dataset
        indices = [i for i, (_, label) in enumerate(dataset) if label not in excluded_classes]
        return Subset(dataset, indices)

    def include_classes(dataset, included_classes):
        if not included_classes:
            return dataset
        indices = [i for i, (_, label) in enumerate(dataset) if label in included_classes]
        return Subset(dataset, indices)


def load_datasets(data_path="./CIFAR10_data/"):
    """Loads the CIFAR-10 train and test datasets."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading CIFAR10 dataset...")
    start_time = time.time()
    trainset = datasets.CIFAR10(
        data_path, download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR10(
        data_path, download=True, train=False, transform=transform
    )
    load_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset loaded in {load_time:.2f} seconds")
    print(f"Total training set size: {len(trainset)}")
    print(f"Total test set size: {len(testset)}")
    return trainset, testset

def prepare_client_datasets(trainset, config):
    """Prepares partitioned datasets for each client based on the config."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing client datasets...")
    train_sets = []
    data_fraction = config.get("training", {}).get("data_fraction", 1.0) # Default to 100%

    if "clients" not in config:
        raise ValueError("Configuration must include a 'clients' section.")

    for i, client_config in enumerate(config["clients"]):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Setting up Client {i}...")
        client_dataset = trainset # Start with the full dataset

        # Apply data fraction if needed
        if data_fraction < 1.0:
            full_size = len(client_dataset)
            subset_size = int(full_size * data_fraction)
            torch.manual_seed(42 + i) # Use a different seed per client for reproducibility
            indices = torch.randperm(full_size)[:subset_size]
            # Check if original dataset is already a Subset
            if isinstance(client_dataset, Subset):
                 # Map indices relative to the subset's indices
                original_indices = [client_dataset.indices[j] for j in indices]
                client_dataset = Subset(client_dataset.dataset, original_indices)
            else:
                client_dataset = Subset(client_dataset, indices.tolist())

            print(f"  Using {data_fraction*100:.1f}% of data. Size after fraction: {len(client_dataset)}")

        # Apply exclusions based on config
        excluded = client_config.get("excluded_classes", [])
        if excluded:
            client_dataset = exclude_classes(client_dataset, excluded_classes=excluded)
            print(f"  Size after excluding classes {excluded}: {len(client_dataset)}")
        else:
            print(f"  No classes excluded.")

        # Analyze class distribution
        labels = []
        if isinstance(client_dataset, Subset):
             # Access underlying dataset correctly for Subset
            for idx in client_dataset.indices:
                 label = client_dataset.dataset[idx][1]
                 labels.append(label)
        else:
             # Access labels directly for full dataset
             # This part might need adjustment depending on how labels are stored if not a standard torchvision dataset
             try:
                 # Attempt standard access assuming dataset items are (data, label)
                 labels = [client_dataset[j][1] for j in range(len(client_dataset))]
             except (TypeError, IndexError):
                 print(f"  Warning: Could not automatically extract labels for class distribution analysis for Client {i}.")
                 labels = [] # Fallback

        class_counts = Counter(labels)
        # Ensure all classes 0-9 are represented, defaulting to 0 if not present
        full_class_counts = {cls: class_counts.get(cls, 0) for cls in range(10)}

        print(f"  Final dataset size: {len(client_dataset)}")
        print(f"  Excluded classes: {excluded}")
        print(f"  Class distribution: {dict(sorted(full_class_counts.items()))}")
        print(f"  Classes present: {sorted(class_counts.keys())}")

        train_sets.append(client_dataset)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Client datasets prepared.")
    return train_sets


def prepare_test_subsets(testset, config):
    """Creates named test subsets based on the config."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating test subsets...")
    test_subsets = {}
    if "evaluation" in config and "test_subsets" in config["evaluation"]:
        for subset_config in config["evaluation"]["test_subsets"]:
            name = subset_config["name"]
            classes = subset_config["classes"]
            subset_data = include_classes(testset, classes)
            test_subsets[name] = subset_data
            print(f"  Test subset '{name}': {len(subset_data)} samples, classes: {classes}")
    else:
        print("  No specific test subsets defined in configuration.")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Test subsets created.")
    return test_subsets 