# remember to install ray and flwr
# pip install "numpy<2"

import time
import json
import os
import torch
from datetime import datetime
from logging import INFO, ERROR

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.common import ndarrays_to_parameters
from flwr.simulation import run_simulation

# Import refactored components
from config_loader import load_config, setup_logging, get_device
from data_preparation import load_datasets, prepare_client_datasets, prepare_test_subsets
from models import ResNet20, get_weights # Assuming other model utils like set_weights, evaluate_model are used internally by client/server logic
from client_logic import client_fn_factory
from server_logic import server_fn_factory, evaluate_fn_factory

# Assuming log is available from utils2 or defined elsewhere (e.g., basic print)
try:
    from utils2 import log
except ImportError:
    # Simple print logger if utils2.log is not available
    def log(level, msg, *args):
        if level >= INFO: # Basic filtering
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level}: {msg % args}")

def main():
    # --- Configuration and Setup ---
    config_path = "config.json"
    config = load_config(config_path)
    log_dir, timestamp = setup_logging(config, config_path)
    device = get_device()

    # Extract training parameters from config
    training_params = config["training"]
    num_clients = training_params["num_clients"]
    num_rounds = training_params["num_rounds"]
    epochs = training_params["epochs"] # Needed by client factory
    batch_size = training_params["batch_size"] # Needed by client factory
    learning_rate = training_params["learning_rate"] # Needed by client factory
    momentum = training_params["momentum"] # Needed by client factory
    data_fraction = training_params.get("data_fraction", 1.0) # Keep track if needed

    log(INFO, f"Starting main script execution...")
    log(INFO, f"Config loaded: Clients={num_clients}, Rounds={num_rounds}, Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, Device={device}")

    # --- Data Preparation ---
    trainset, testset = load_datasets()
    train_sets = prepare_client_datasets(trainset, config) # Pass full config for client specifics
    test_subsets = prepare_test_subsets(testset, config) # Pass full config for evaluation specifics

    # --- Shared State ---
    # Central dictionary to store metrics from clients and server evaluation
    # Passed by reference to client/server functions that need to update/read it
    client_metrics = {}

    # --- Model Initialization ---
    log(INFO, "Initializing global model (ResNet20)...")
    net = ResNet20().to(device)
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    log(INFO, "Global model initialized.")

    # --- Create Flower Simulation Components ---

    # 1. Server-side Evaluation Function
    evaluate_fn = evaluate_fn_factory(
        testset=testset,
        test_subsets=test_subsets,
        device=device,
        num_rounds=num_rounds,
        log_dir=log_dir,
        client_metrics_ref=client_metrics # Pass the reference
    )

    # 2. Server Function (defines strategy and server app components)
    server_fn = server_fn_factory(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn, # Pass the created evaluation function
        num_clients=num_clients,
        num_rounds=num_rounds
        # Add strategy factors from config if needed, e.g.:
        # fraction_fit=config.get("strategy", {}).get("fraction_fit", 1.0),
        # min_fit_clients_factor=config.get("strategy", {}).get("min_fit_clients_factor", 1.0),
        # min_available_clients_factor=config.get("strategy", {}).get("min_available_clients_factor", 1.0)
    )
    server_app = ServerApp(server_fn=server_fn)

    # 3. Client Function (instantiates clients)
    # Pass necessary parameters for the client instances
    client_fn = client_fn_factory(
        train_sets=train_sets,
        testset=testset, # Clients use the overall testset for validation
        device=device,
        training_params=training_params, # Pass dict with epochs, lr, etc.
        log_dir=log_dir,
        client_metrics_ref=client_metrics # Pass the reference
    )
    client_app = ClientApp(client_fn=client_fn)


    # --- Simulation Backend Configuration ---
    # Using the built-in simulation backend
    backend_setup = {
        "backend_name": "simulation",
        "init_args": {
            "logging_level": ERROR, # Flower's internal simulation logging level
            "log_to_driver": False
        }
    }
    log(INFO, f"Using simulation backend: {backend_setup['backend_name']}")

    # --- Run Simulation ---
    log(INFO, f"=========================================")
    log(INFO, f"Starting Flower Simulation:")
    log(INFO, f"  Num Clients: {num_clients}")
    log(INFO, f"  Num Rounds: {num_rounds}")
    log(INFO, f"=========================================")

    simulation_start = time.time()
    # History object is not explicitly used here, but could be captured if needed
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients, # Matches number of clients for simulation
        backend_config=backend_setup,
    )
    simulation_time = time.time() - simulation_start

    log(INFO, f"[{datetime.now().strftime('%H:%M:%S')}] Simulation completed in {simulation_time:.2f} seconds")

    # --- Save Summary ---
    # Retrieve final metrics from the shared dictionary
    # Look for server evaluation results in the last round
    final_metrics = {}
    if num_rounds in client_metrics and "server_evaluation" in client_metrics[num_rounds]:
         final_metrics = client_metrics[num_rounds]["server_evaluation"]
    elif num_rounds in client_metrics and "global" in client_metrics[num_rounds]: # Fallback if server eval structure missing
         final_metrics = client_metrics[num_rounds]["global"]


    summary = {
        "run_timestamp": timestamp,
        "total_runtime_seconds": round(simulation_time, 2),
        "configuration": config, # Save the original config used
        "final_global_metrics": final_metrics # Include metrics from server eval
    }

    summary_path = os.path.join(log_dir, "summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        log(INFO, f"Summary saved to {summary_path}")
    except Exception as e:
        log(ERROR, f"Error saving summary to {summary_path}: {e}")

if __name__ == "__main__":
    main()

