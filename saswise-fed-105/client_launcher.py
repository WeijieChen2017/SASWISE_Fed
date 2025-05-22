import argparse
import os
import torch
import random
import numpy as np
import flwr as fl

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Flower client launcher")
    parser.add_argument(
        "--server_address", 
        type=str, 
        default="127.0.0.1:8080", 
        help="Server address"
    )
    parser.add_argument(
        "--client_id", 
        type=int, 
        required=True, 
        help="Client ID"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="MedMNIST dataset to use for federated learning"
    )
    parser.add_argument(
        "--num_clients", 
        type=int, 
        default=5, 
        help="Total number of clients in the simulation"
    )
    parser.add_argument(
        "--partition_type", 
        type=str,
        choices=["uniform", "dirichlet"], 
        default="uniform",
        help="Type of data partitioning"
    )
    parser.add_argument(
        "--dirichlet_alpha", 
        type=float, 
        default=0.5,
        help="Dirichlet alpha parameter for non-uniform partitioning"
    )
    args = parser.parse_args()
    
    # Import required modules here to avoid circular imports
    from utils import load_datasets, create_client_data_partitions
    from client import FlowerClient
    
    # Define MedMNIST datasets
    RESNET_DATASETS = [
        'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
        'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
        'organcmnist', 'organsmnist'
    ]
    CNN_DATASETS = [
        'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 
        'vesselmnist3d', 'synapsemnist3d'
    ]
    DATASETS = RESNET_DATASETS + CNN_DATASETS
    
    # Check dataset exists
    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset {args.dataset} not recognized")
    
    # Set parameters
    data_flag = args.dataset
    client_id = args.client_id
    batch_size = 64
    img_size = 64  # Fixed image size
    
    # Check client ID is valid
    if client_id < 0 or client_id >= args.num_clients:
        raise ValueError(f"Client ID must be between 0 and {args.num_clients-1}")
    
    # Configure GPU memory usage if specified via environment variable
    if device.type == 'cuda' and 'MEMORY_FRACTION' in os.environ:
        memory_fraction = float(os.environ['MEMORY_FRACTION'])
        print(f"Limiting GPU memory usage to {memory_fraction*100:.1f}% for client {client_id}")
        
        # Configure PyTorch to limit memory usage
        # This prevents clients from using all GPU memory
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable memory growth to avoid allocating all memory at once
        torch.cuda.empty_cache()
    
    print(f"Starting client {client_id} for dataset {data_flag}")
    
    # Load datasets
    print(f"Loading {data_flag} dataset...")
    train_dataset, test_dataset = load_datasets(data_flag, img_size)
    
    # Create client partitions
    client_partitions = create_client_data_partitions(
        train_dataset, 
        test_dataset,
        num_clients=args.num_clients,
        batch_size=batch_size,
        partition_type=args.partition_type,
        alpha=args.dirichlet_alpha,
        seed=SEED
    )
    
    # Get this client's partition
    client_partition = client_partitions[client_id]
    
    # Create the client
    client = FlowerClient(
        client_id=client_id, 
        data_flag=data_flag,
        train_loader=client_partition["train_loader"],
        val_loader=client_partition["val_loader"],
        test_loader=client_partition["test_loader"],
        device=device,
        num_local_epochs=5,
        is_3d=data_flag in CNN_DATASETS
    )
    
    # Create directory for client output
    output_dir = f"client_{client_id}_{data_flag}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

if __name__ == "__main__":
    main() 