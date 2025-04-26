import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.utils import make_grid
from data_preparation import load_datasets, prepare_client_datasets, prepare_test_subsets

def show_images(dataset, num_images=20, classes=None, title="Dataset Images"):
    """Display a sample of images from the dataset."""
    # Create a figure for visualization
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(title, fontsize=16)
    
    # Get total dataset size
    dataset_size = len(dataset)
    indices = np.random.choice(dataset_size, min(num_images, dataset_size), replace=False)
    
    images = []
    labels = []
    for idx in indices:
        # Handle both regular dataset and Subset
        if hasattr(dataset, 'indices'):
            img, label = dataset.dataset[dataset.indices[idx]]
        else:
            img, label = dataset[idx]
        images.append(img)
        labels.append(label)
    
    # Make grid of images
    img_grid = make_grid(images, nrow=5, normalize=True)
    
    # Convert to numpy for matplotlib
    img_grid = img_grid.numpy().transpose((1, 2, 0))
    
    # Show the image grid
    plt.imshow(img_grid)
    plt.axis('off')
    
    # Print the labels
    label_str = ', '.join([str(label) for label in labels])
    print(f"Labels: {label_str}")
    
    # Save the visualization
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def analyze_class_distribution(dataset, title="Class Distribution"):
    """Analyze and print class distribution in the dataset."""
    # Extract labels
    labels = []
    if hasattr(dataset, 'indices'):
        for idx in dataset.indices:
            label = dataset.dataset[idx][1]
            labels.append(label)
    else:
        for i in range(len(dataset)):
            labels.append(dataset[i][1])
    
    # Count classes
    class_counts = Counter(labels)
    
    print(f"\n{title}:")
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"Classes present: {sorted(class_counts.keys())}")
    
    return class_counts

def main():
    # Load the configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("Testing data preparation module...")
    
    # 1. Test loading datasets
    print("\n=== Testing load_datasets ===")
    trainset, testset = load_datasets(config['dataset']['name'])
    print(f"Trainset size: {len(trainset)}")
    print(f"Testset size: {len(testset)}")
    
    # Visualize a few training images
    show_images(trainset, num_images=20, title="Original Training Dataset")
    
    # 2. Test preparing client datasets
    print("\n=== Testing prepare_client_datasets ===")
    client_datasets = prepare_client_datasets(trainset, config)
    
    # Verify the number of clients
    assert len(client_datasets) == config['training']['num_clients'], \
        f"Expected {config['training']['num_clients']} client datasets, got {len(client_datasets)}"
    
    # Analyze and visualize each client's dataset
    for i, client_dataset in enumerate(client_datasets):
        print(f"\nClient {i} dataset:")
        print(f"Size: {len(client_dataset)}")
        
        # Check that excluded classes are actually excluded
        client_config = config['clients'][i]
        excluded_classes = client_config.get('excluded_classes', [])
        
        # Analyze class distribution
        class_counts = analyze_class_distribution(client_dataset, f"Client {i} Class Distribution")
        
        # Verify excluded classes
        for excluded_class in excluded_classes:
            assert excluded_class not in class_counts, \
                f"Client {i} contains excluded class {excluded_class}"
        
        # Visualize client dataset
        show_images(client_dataset, num_images=20, title=f"Client {i} Dataset")
    
    # 3. Test preparing test subsets
    print("\n=== Testing prepare_test_subsets ===")
    test_subsets = prepare_test_subsets(testset, config)
    
    # Verify test subsets
    expected_subsets = {subset['name']: subset for subset in config['evaluation']['test_subsets']}
    assert len(test_subsets) == len(expected_subsets), \
        f"Expected {len(expected_subsets)} test subsets, got {len(test_subsets)}"
    
    for name, subset in test_subsets.items():
        print(f"\nTest subset '{name}':")
        print(f"Size: {len(subset)}")
        
        # Check that only specified classes are included
        expected_classes = set(expected_subsets[name]['classes'])
        
        # Analyze class distribution
        class_counts = analyze_class_distribution(subset, f"Test Subset '{name}' Class Distribution")
        
        # Verify included classes
        for class_idx in class_counts.keys():
            assert class_idx in expected_classes, \
                f"Test subset '{name}' contains unexpected class {class_idx}"
        
        # Visualize test subset
        show_images(subset, num_images=20, title=f"Test Subset {name}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 