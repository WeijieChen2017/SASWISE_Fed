import os
import argparse
import multiprocessing
import subprocess
import torch
import time

def run_client(client_id, args, gpu_id, memory_fraction):
    """Run a single client process with the specified GPU settings"""
    # Prepare environment variables to control GPU memory usage
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Add variable for memory fraction if using PyTorch
    env["MEMORY_FRACTION"] = str(memory_fraction)
    
    # Construct the client command
    cmd = [
        "python", "client_launcher.py",
        "--client_id", str(client_id),
        "--dataset", args.dataset,
        "--num_clients", str(args.num_clients),
        "--partition_type", args.partition_type,
        "--server_address", args.server_address
    ]
    
    # Add alpha if using dirichlet partitioning
    if args.partition_type == "dirichlet":
        cmd.extend(["--dirichlet_alpha", str(args.dirichlet_alpha)])
    
    # Run the client process
    print(f"Starting client {client_id} with GPU {gpu_id} (memory fraction: {memory_fraction:.2f})")
    
    # Run the process and wait for it to complete
    process = subprocess.Popen(cmd, env=env)
    process.wait()
    return process.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple Flower clients on the same GPU")
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
        "--server_address", 
        type=str, 
        default="127.0.0.1:8080",
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run clients in parallel (True) or sequentially (False)"
    )
    
    args = parser.parse_args()
    
    # Check that GPU is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available, clients will run on CPU")
        gpu_id = -1
    else:
        gpu_id = args.gpu_id
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Determine memory fraction for each client (leave some memory for server)
    # Adjust this value based on your model size and available GPU memory
    memory_per_client = 0.8 / args.num_clients  # Using 80% of GPU memory divided among clients
    
    # Create and run processes
    processes = []
    start_time = time.time()
    
    if args.parallel:
        # Run all clients in parallel
        for client_id in range(args.num_clients):
            p = multiprocessing.Process(
                target=run_client,
                args=(client_id, args, gpu_id, memory_per_client)
            )
            processes.append(p)
            p.start()
            
        # Wait for all processes to complete
        for p in processes:
            p.join()
    else:
        # Run clients sequentially
        for client_id in range(args.num_clients):
            run_client(client_id, args, gpu_id, memory_per_client)
    
    end_time = time.time()
    print(f"All clients completed in {end_time - start_time:.2f} seconds") 