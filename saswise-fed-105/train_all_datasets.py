import os
import json
import time
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import medmnist
from medmnist import INFO, Evaluator
import matplotlib.pyplot as plt

# Import our custom 3D ResNet
from resnet3d import resnet3d50

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

# Training parameters
NUM_EPOCHS_2D = 100
NUM_EPOCHS_3D = 100  # 100 epochs for 3D models
BATCH_SIZE = 256
LR = 0.001
DOWNLOAD = True
IMG_SIZE = 64  # Set image size to 64

# List of 2D datasets for ResNet-50
RESNET_DATASETS = [
    'pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
    'organcmnist', 'organsmnist'
]

# List of 3D datasets for custom CNN
CNN_DATASETS = [
    'organmnist3d', 'nodulemnist3d', 'adrenalmnist3d', 'fracturemnist3d', 
    'vesselmnist3d', 'synapsemnist3d'
]

# Combine all datasets
# DATASETS = RESNET_DATASETS + CNN_DATASETS
DATASETS = RESNET_DATASETS + CNN_DATASETS

# Function to create a ResNet-50 model
def create_resnet50(in_channels, num_classes):
    model = models.resnet50(weights=None)
    
    # Modify the first layer to accept the correct number of input channels
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the last layer for the correct number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def evaluate(model, data_loader, task, device, is_3d=False):
    model.eval()
    y_true = torch.tensor([], device=device)
    y_score = torch.tensor([], device=device)
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move data to device and ensure float32 type
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device)
            
            # For 3D datasets, manually normalize if no transform was used
            if is_3d:
                inputs = (inputs - 0.5) / 0.5
            
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        # Move to CPU for numpy conversion
        y_true = y_true.cpu().numpy()
        y_score = y_score.cpu().detach().numpy()
        
        return y_true, y_score

def save_dataset_preview(dataset, data_flag, save_dir):
    """
    Save a montage preview of the dataset using direct save_folder parameter
    """
    try:
        # Create preview directory
        preview_dir = os.path.join(save_dir, 'previews')
        os.makedirs(preview_dir, exist_ok=True)
        
        # Use the montage method directly - should work for both 2D and 3D
        # Print sample dimensions only if needed for debugging
        # sample_data = dataset[0][0]  # Get first image from dataset
        # print(f"Sample data dimensions for {data_flag}: {sample_data.shape}")
        frames = dataset.montage(length=8, save_folder=preview_dir)
        
        print(f"Saved dataset preview for {data_flag} to {preview_dir}")
        return True
    except Exception as e:
        print(f"Error creating montage for {data_flag}: {str(e)}")
        return False

def train_and_evaluate(data_flag):
    # Create directory for this dataset
    dataset_dir = os.path.join(data_flag)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(dataset_dir, 'training_log.txt')
    
    # Get dataset info
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    is_3d_dataset = data_flag in CNN_DATASETS
    
    # Set number of epochs based on dataset type
    num_epochs = NUM_EPOCHS_3D if is_3d_dataset else NUM_EPOCHS_2D
    
    # Log basic info
    with open(log_file, 'w') as f:
        f.write(f"Dataset: {data_flag}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Channels: {n_channels}\n")
        f.write(f"Classes: {n_classes}\n")
        f.write(f"Is 3D dataset: {is_3d_dataset}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Set worker init function for reproducibility in data loading
    def worker_init_fn(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Get the dataset class
    DataClass = getattr(medmnist, info['python_class'])
    
    # Set different transforms for 2D and 3D datasets
    if is_3d_dataset:
        # For 3D datasets, no need for transforms since data is already in tensor format
        data_transform = None
    else:
        # Standard preprocessing for 2D datasets
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    
    # Load the data with specified image size
    train_dataset = DataClass(split='train', transform=data_transform, download=DOWNLOAD, size=IMG_SIZE)
    test_dataset = DataClass(split='test', transform=data_transform, download=DOWNLOAD, size=IMG_SIZE)
    
    # Save dataset preview (with unprocessed dataset)
    pil_dataset = DataClass(split='train', download=DOWNLOAD, size=IMG_SIZE)
    save_dataset_preview(pil_dataset, data_flag, dataset_dir)
    
    # DataLoaders with worker_init_fn for reproducibility
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader_at_eval = data.DataLoader(
        dataset=train_dataset, 
        batch_size=2*BATCH_SIZE, 
        shuffle=False,
        worker_init_fn=worker_init_fn
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, 
        batch_size=2*BATCH_SIZE, 
        shuffle=False,
        worker_init_fn=worker_init_fn
    )
    
    # Create model based on dataset type - each model handles its own input dimensions
    if data_flag in RESNET_DATASETS:
        torch.manual_seed(SEED)  # Reset seed before model init
        model = create_resnet50(in_channels=n_channels, num_classes=n_classes)
        model_type = "ResNet-50-2D"
    else:
        # Use 3D ResNet for 3D datasets
        torch.manual_seed(SEED)  # Reset seed before model init
        model = resnet3d50(num_classes=n_classes, in_channels=n_channels)
        model_type = "ResNet-50-3D"
    
    model = model.to(device)
    
    # Log model type
    with open(log_file, 'a') as f:
        f.write(f"Model architecture: {model_type}\n\n")
    
    # Loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer with deterministic behavior
    torch.manual_seed(SEED)  # Reset seed before optimizer init
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    
    # Training metrics storage
    best_test_acc = 0.0
    epoch_data = []
    
    # Training and evaluation loop
    print(f'==> Training on {data_flag} with {model_type} for {num_epochs} epochs...')
    for epoch in range(num_epochs):
        train_loss = 0.0
        num_batches = 0
        
        # Training
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training'):
            # Move data to device and ensure float32 type
            inputs = inputs.to(device, dtype=torch.float32)
            targets = targets.to(device)
            
            # For 3D datasets, manually normalize if no transform was used
            if is_3d_dataset:
                inputs = (inputs - 0.5) / 0.5
            
            # Forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            # Track loss
            train_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            optimizer.step()
        
        # Calculate average loss for this epoch
        avg_loss = train_loss / num_batches
        
        # Evaluate on training and test sets
        train_true, train_score = evaluate(model, train_loader_at_eval, task, device, is_3d=is_3d_dataset)
        test_true, test_score = evaluate(model, test_loader, task, device, is_3d=is_3d_dataset)
        
        # Calculate metrics
        train_evaluator = Evaluator(data_flag, 'train', size=IMG_SIZE)
        test_evaluator = Evaluator(data_flag, 'test', size=IMG_SIZE)
        
        train_metrics = train_evaluator.evaluate(train_score)
        test_metrics = test_evaluator.evaluate(test_score)
        
        train_auc, train_acc = train_metrics
        test_auc, test_acc = test_metrics
        
        # Save epoch data
        epoch_info = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'train_auc': float(train_auc),
            'train_acc': float(train_acc),
            'test_auc': float(test_auc),
            'test_acc': float(test_acc)
        }
        epoch_data.append(epoch_info)
        
        # Log results
        log_message = f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
        log_message += f'Train AUC: {train_auc:.3f}, Train ACC: {train_acc:.3f}, '
        log_message += f'Test AUC: {test_auc:.3f}, Test ACC: {test_acc:.3f}'
        
        print(log_message)
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(dataset_dir, 'best_model.pth'))
            with open(log_file, 'a') as f:
                f.write(f'New best model saved with test accuracy: {test_acc:.4f}\n')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(dataset_dir, 'final_model.pth'))
    
    # Save model architecture information
    model_info = {
        'model_type': model_type,
        'in_channels': n_channels,
        'num_classes': n_classes
    }
    with open(os.path.join(dataset_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Save epoch data as JSON
    with open(os.path.join(dataset_dir, 'training_history.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)
    
    # Return summary information
    return {
        'dataset': data_flag,
        'model_type': model_type,
        'task': task,
        'n_channels': n_channels,
        'n_classes': n_classes,
        'best_test_acc': float(best_test_acc),
        'final_train_acc': float(train_acc),
        'final_test_acc': float(test_acc)
    }

# Main execution
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train models on MedMNIST datasets')
    parser.add_argument('--dataset_type', type=str, choices=['2d', '3d', 'both'], default='both',
                      help='Type of datasets to process: 2d, 3d, or both')
    parser.add_argument('--model_index', type=int, choices=[0, 1, 2, 3], required=True,
                      help='Model index (0-3) to select subset of datasets')
    args = parser.parse_args()
    
    model_index = args.model_index
    
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
    
    # Select datasets based on input flag
    if args.dataset_type == '2d':
        selected_datasets = RESNET_DATASETS
    elif args.dataset_type == '3d':
        selected_datasets = CNN_DATASETS
    else:  # 'both'
        selected_datasets = DATASETS
    
    # Select datasets based on model index
    selected_datasets = selected_datasets[model_index::4]
    
    # Storage for all results
    all_results = []
    
    # Train on all datasets
    for data_flag in selected_datasets:
        try:
            print(f"\n{'='*50}\nProcessing dataset: {data_flag}\n{'='*50}")
            result = train_and_evaluate(data_flag)
            all_results.append(result)
            
            # Save intermediate results
            with open(f'all_datasets_results_{model_index}.json', 'w') as f:
                json.dump(all_results, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {data_flag}: {str(e)}")
            with open(f'errors_{model_index}.log', 'a') as f:
                f.write(f"Error processing {data_flag}: {str(e)}\n")
    
    print(f"\nAll datasets processed. Results saved to all_datasets_results_{model_index}.json")