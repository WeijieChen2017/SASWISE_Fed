import os
import json
import time
import random
import argparse
import numpy as np
import torch
import flwr as fl
from typing import List, Dict, Tuple

from client import FlowerClient
from utils import load_datasets, create_client_data_partitions
from train_all_datasets import evaluate, create_resnet50
from resnet3d import resnet3d50
from medmnist import INFO, Evaluator

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seed set to {SEED}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# Training parameters
NUM_EPOCHS_LOCAL = 5  # Number of local epochs per round
NUM_ROUNDS = 10       # Number of federated rounds for simulation
BATCH_SIZE = 64       # Smaller batch size for clients
IMG_SIZE = 64         # Image size

def get_evaluate_fn(test_loader, data_flag, is_3d=False):
    """Return evaluation function for centralized evaluation."""
    def evaluate_fn(weights):
        # Get dataset info
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        # Create model
        if data_flag in RESNET_DATASETS:
            model = create_resnet50(in_channels=n_channels, num_classes=n_classes)
        else:
            model = resnet3d50(num_classes=n_classes, in_channels=n_channels)
        
        model.to(device)
        
        # Load weights
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        test_true, test_score = evaluate(model, test_loader, task, device, is_3d=is_3d)
        
        # Calculate metrics
        evaluator = Evaluator(data_flag, 'test', size=IMG_SIZE)
        test_metrics = evaluator.evaluate(test_score)
        test_auc, test_acc = test_metrics
        
        return float(test_acc), {"auc": float(test_auc)}
    
    return evaluate_fn

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation results from multiple clients."""
    # Sum of all evaluation metrics
    auc_sum = 0.0
    acc_sum = 0.0
    examples_sum = 0
    
    # Sum up metrics from all clients
    for num_examples, metrics in metrics:
        auc_sum += metrics["auc"] * num_examples
        acc_sum += metrics["acc"] * num_examples
        examples_sum += num_examples
    
    # Compute weighted average
    return {
        "auc": auc_sum / examples_sum,
        "acc": acc_sum / examples_sum,
    }

def fit_config(server_round: int) -> Dict[str, str]:
    """Return training configuration dict for each round."""
    config = {
        "round": server_round,
        "batch_size": BATCH_SIZE,
        "local_epochs": NUM_EPOCHS_LOCAL,
    }
    return config

def run_simulation(args):
    """Run a simulated federated learning experiment."""
    # Set data flag from args
    data_flag = args.dataset
    
    # Load datasets
    print(f"Loading {data_flag} dataset...")
    train_dataset, test_dataset = load_datasets(data_flag, IMG_SIZE)
    
    # Create client partitions
    client_partitions = create_client_data_partitions(
        train_dataset, 
        test_dataset,
        num_clients=args.num_clients,
        batch_size=BATCH_SIZE,
        partition_type=args.partition_type,
        alpha=args.dirichlet_alpha,
        seed=SEED
    )
    
    # Create a list of clients
    client_fn = get_client_fn(data_flag, client_partitions)
    
    # Create central test set for global evaluation
    from torch.utils.data import DataLoader
    central_test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2*BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create directory for results
    output_dir = f"federated_{data_flag}_{args.num_clients}clients"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save experiment config
    config = {
        "dataset": data_flag,
        "num_clients": args.num_clients,
        "partition_type": args.partition_type,
        "dirichlet_alpha": args.dirichlet_alpha,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": NUM_EPOCHS_LOCAL,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "seed": SEED
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_evaluate_fn(central_test_loader, data_flag, is_3d=data_flag in CNN_DATASETS),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start simulation
    print(f"\nStarting federated learning simulation with {args.num_clients} clients for {NUM_ROUNDS} rounds")
    
    # Setup history callback to save results
    history = []
    
    def __get_metrics_aggregation_fn(round_idx):
        def __fn(metrics_list):
            # Calculate aggregated metrics
            aggregated_metrics = weighted_average(metrics_list)
            
            # Save history
            history.append({
                "round": round_idx,
                "accuracy": aggregated_metrics["acc"],
                "auc": aggregated_metrics["auc"]
            })
            
            # Save history after each round
            with open(os.path.join(output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=4)
            
            # Print progress
            print(f"Round {round_idx}: accuracy={aggregated_metrics['acc']:.4f}, AUC={aggregated_metrics['auc']:.4f}")
            
            return aggregated_metrics
        return __fn
    
    # Update strategy to track metrics
    for i in range(NUM_ROUNDS):
        strategy.evaluate_metrics_aggregation_fn = __get_metrics_aggregation_fn(i+1)
    
    # Run simulation
    start_time = time.time()
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Save final results
    results = {
        "config": config,
        "history": history,
        "duration_seconds": duration,
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSimulation completed in {duration:.2f} seconds")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")
    print(f"Results saved to {output_dir}/results.json")

def get_client_fn(data_flag, client_partitions):
    """Return a function that creates a client with the given ID."""
    def client_fn(cid: str) -> fl.client.Client:
        """Create a Flower client representing a single organization."""
        # Parse client ID string
        client_id = int(cid)
        
        # Get this client's data partition
        partition = client_partitions[client_id]
        
        # Create and return client
        return FlowerClient(
            client_id=client_id, 
            data_flag=data_flag,
            train_loader=partition["train_loader"],
            val_loader=partition["val_loader"],
            test_loader=partition["test_loader"],
            device=device,
            num_local_epochs=NUM_EPOCHS_LOCAL,
            is_3d=data_flag in CNN_DATASETS
        )
    
    return client_fn

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Simulate federated learning with Flower")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=DATASETS, 
        default="pathmnist",
        help="MedMNIST dataset to use for federated learning"
    )
    parser.add_argument(
        "--num_clients", 
        type=int, 
        default=5, 
        help="Number of clients/agents to simulate"
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
        help="Dirichlet alpha parameter for non-uniform partitioning (lower is more heterogeneous)"
    )
    
    args = parser.parse_args()
    
    # Run the simulation
    run_simulation(args) 