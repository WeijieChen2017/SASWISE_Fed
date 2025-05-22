# SASWISE‑Fed

**SASWISE‑Fed** brings the SASWISE sub‑model ensemble framework into a federated learning setting, enabling privacy‑preserving, distributed training of medical imaging models with built‑in uncertainty estimation and interpretability.

## 🚀 Features

- **Federated orchestration** via [Flower](https://flower.dev/)
- **SASWISE ensemble**: generate diverse sub‑models from a single checkpoint
- **Uncertainty mapping** for reliability assessment at each client

# MedMNIST Federated Learning

This project implements federated learning for MedMNIST datasets using the Flower framework.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

There are two ways to run the federated learning setup:

### 1. Simulation Mode (Easiest)

Run a simulated federated learning experiment with multiple clients in a single process:

```bash
python simulate_federated.py --dataset pathmnist --num_clients 5 --partition_type uniform
```

Parameters:
- `--dataset`: MedMNIST dataset to use (e.g., pathmnist, chestmnist, etc.)
- `--num_clients`: Number of federated clients to simulate
- `--partition_type`: Data partitioning strategy (uniform or dirichlet)
- `--dirichlet_alpha`: Controls data heterogeneity when using dirichlet partitioning (lower is more heterogeneous)

### 2. Distributed Mode

For a more realistic setup, you can run the server and clients separately.

1. Start the server:
   ```bash
   python train_federated.py --dataset pathmnist --num_clients 5 --partition_type uniform
   ```

2. Start each client in separate terminals:
   ```bash
   python client_launcher.py --dataset pathmnist --client_id 0 --num_clients 5 --partition_type uniform
   python client_launcher.py --dataset pathmnist --client_id 1 --num_clients 5 --partition_type uniform
   # ... repeat for all clients
   ```

## Supported Datasets

### 2D Datasets (Using ResNet-50)
- pathmnist
- chestmnist
- dermamnist
- octmnist
- pneumoniamnist
- retinamnist
- breastmnist
- bloodmnist
- tissuemnist
- organamnist
- organcmnist
- organsmnist

### 3D Datasets (Using 3D ResNet-50)
- organmnist3d
- nodulemnist3d
- adrenalmnist3d
- fracturemnist3d
- vesselmnist3d
- synapsemnist3d

## Results

Results are saved in the `federated_{dataset}_{num_clients}clients` directory, including:
- Configuration information
- Training history
- Final model metrics

## Example

To run a federated learning simulation with 8 clients on the pathmnist dataset:

```bash
python simulate_federated.py --dataset pathmnist --num_clients 8 --partition_type uniform
```

For non-uniform data distribution (to simulate real-world heterogeneity):

```bash
python simulate_federated.py --dataset pathmnist --num_clients 8 --partition_type dirichlet --dirichlet_alpha 0.5
```

Lower alpha values (e.g., 0.1) create more heterogeneous data partitions, simulating more realistic federated learning scenarios where each participant has different data distributions.
