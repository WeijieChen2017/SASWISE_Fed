import os
import argparse
import subprocess
import time
import threading
import torch

def run_server(args):
    """Run the federated learning server"""
    # Prepare environment variables for server
    env = os.environ.copy()
    
    # Server can use a small portion of GPU memory as it mainly aggregates models
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        env["MEMORY_FRACTION"] = "0.1"  # Allocate 10% for server
    
    cmd = [
        "python", "train_federated.py",
        "--dataset", args.dataset,
        "--num_clients", str(args.num_clients),
        "--partition_type", args.partition_type
    ]
    
    if args.partition_type == "dirichlet":
        cmd.extend(["--dirichlet_alpha", str(args.dirichlet_alpha)])
    
    # Run server
    print("Starting federated learning server...")
    server_process = subprocess.Popen(cmd, env=env)
    
    # Give server time to start up before launching clients
    print("Waiting for server to start...")
    time.sleep(5)
    
    return server_process

def run_clients(args):
    """Run multiple federated learning clients using the parallel client launcher"""
    # Prepare environment variables for clients
    env = os.environ.copy()
    
    cmd = [
        "python", "run_parallel_clients.py",
        "--dataset", args.dataset,
        "--num_clients", str(args.num_clients),
        "--gpu_id", str(args.gpu_id),
        "--partition_type", args.partition_type,
        "--server_address", "127.0.0.1:8080"
    ]
    
    if args.partition_type == "dirichlet":
        cmd.extend(["--dirichlet_alpha", str(args.dirichlet_alpha)])
    
    if args.parallel_clients:
        cmd.append("--parallel")
    
    # Run clients
    print("Starting federated learning clients...")
    client_process = subprocess.Popen(cmd, env=env)
    
    return client_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entire federated learning pipeline on a single GPU")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="pathmnist",
        help="MedMNIST dataset to use for federated learning"
    )
    parser.add_argument(
        "--num_clients", 
        type=int, 
        default=5, 
        help="Number of clients to run"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0,
        help="GPU ID to use"
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
    parser.add_argument(
        "--parallel_clients", 
        action="store_true",
        help="Run clients in parallel (True) or sequentially (False)"
    )
    
    args = parser.parse_args()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("Warning: CUDA is not available, will run on CPU")
    
    start_time = time.time()
    
    # Start server
    server_process = run_server(args)
    
    try:
        # Start clients
        client_process = run_clients(args)
        
        # Wait for clients to finish
        client_process.wait()
        
        # After clients finish, stop the server
        print("Clients completed. Stopping server...")
        server_process.terminate()
        server_process.wait()
        
        end_time = time.time()
        print(f"Federated learning completed in {end_time - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping all processes...")
        # Clean shutdown
        if 'client_process' in locals():
            client_process.terminate()
        server_process.terminate()
        
        # Wait for processes to terminate
        if 'client_process' in locals():
            client_process.wait()
        server_process.wait()
        print("All processes terminated.") 