#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    SpatialCropd,
    CenterSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    ToTensord,
    EnsureTyped,
    SaveImaged,
)
from monai.data import PersistentDataset, partition_dataset
from monai.inferers import SlidingWindowInferer
import nibabel as nib
from tqdm import tqdm
import logging
from datetime import datetime
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import shutil
import copy

# Set up logging
def setup_logger(log_dir, experiment_dir=None):
    os.makedirs(log_dir, exist_ok=True)
    
    if experiment_dir:
        # If experiment directory is provided, log there too
        log_file = os.path.join(experiment_dir, f"training.log")
    else:
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# Create experiment directory structure
def create_experiment_dir(base_dir="experiments"):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(experiment_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    logs_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    return experiment_dir, models_dir, checkpoints_dir, logs_dir

# Save training configuration
def save_config(args, experiment_dir):
    config_path = os.path.join(experiment_dir, "config.json")
    config = vars(copy.deepcopy(args))
    
    # Convert non-serializable objects to strings
    for key, value in config.items():
        if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
            config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom dataset for loading medical images
class SIIMSegmentationDataset(Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_path = data['image']
        label_path = data['label']
        
        # Load image and label
        item = {
            'image': image_path,
            'label': label_path
        }
        
        if self.transforms:
            item = self.transforms(item)
            
        return item

# Model definition - based on HNTS-MRG24-UWLAIR
class SegmentationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=128):
        super(SegmentationModel, self).__init__()
        
        # Use MONAI's UNet with correct parameters and increased features
        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(features, features*2, features*4, features*8, features*16),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            dropout=0.2
        )
        
    def forward(self, x):
        return self.model(x)

def train_epoch(model, dataloader, optimizer, loss_function, device, logger):
    model.train()
    epoch_loss = 0
    step = 0
    
    logger.info("Starting data iteration loop")
    batch_start_time = time.time()
    
    for batch_data in tqdm(dataloader, desc="Training"):
        step += 1
        
        # Explicitly move data to device
        inputs = batch_data["image"].to(device, non_blocking=True)
        labels = batch_data["label"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use amp for mixed precision training
        with autocast(device_type='cuda', enabled=args.use_amp):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        
        # Only log basic progress every 10 steps
        if step % 10 == 0:
            logger.info(f"Training step {step}, current loss: {loss.item():.4f}")
    
    return epoch_loss / step

def validate_epoch(model, dataloader, loss_function, dice_metric, device, logger, sliding_window_inferer=None):
    model.eval()
    epoch_loss = 0
    step = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation"):
            step += 1
            # Explicitly move data to device
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=args.use_amp):
                # Use sliding window inference if provided
                if sliding_window_inferer is not None:
                    outputs = sliding_window_inferer(inputs, model)
                else:
                    outputs = model(inputs)
                loss = loss_function(outputs, labels)
                
            epoch_loss += loss.item()
            
            # Apply sigmoid and threshold to convert logits to binary predictions
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            
            # Calculate Dice metric
            dice_metric(y_pred=outputs, y=labels)
            
    # Aggregate Dice metric for all validation data
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return epoch_loss / step, dice_score

def filter_invalid_samples(data_list, min_z_dim=32, logger=None):
    """Filter out samples with z-dimension less than min_z_dim"""
    valid_samples = []
    invalid_samples = []
    
    for sample in data_list:
        # Load image header to check dimensions without loading entire image
        # print(f"Loading image: {sample['image']}")
        nii_file = nib.load(sample['image'])
        # print(f"Here are key params in its header: ")
        # print(f"affine: {nii_file.affine}")
        # print(f"pixel_dims: {nii_file.header.get_zooms()}")
        # print(f"data_type: {nii_file.header.get_data_dtype()}")
        # print(f"data_shape: {nii_file.header.get_data_shape()}")
        img = nii_file.get_fdata()
        z_dim = img.shape[2]
        
        if z_dim >= min_z_dim:
            valid_samples.append(sample)
        else:
            invalid_samples.append(sample)
    
    return valid_samples, invalid_samples

def main(args):
    # Create experiment directory structure
    experiment_dir, models_dir, checkpoints_dir, logs_dir = create_experiment_dir(args.experiment_dir)
    
    # Set up logging
    logger = setup_logger(args.log_dir, experiment_dir)
    logger.info(f"Starting training with arguments: {args}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Save configuration
    save_config(args, experiment_dir)
    logger.info(f"Saved configuration to {os.path.join(experiment_dir, 'config.json')}")
    
    # Initialize training metrics log
    training_metrics = []
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed: {args.seed}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Print CUDA information
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        # Make sure CUDA is initialized
        torch.cuda.init()
        torch.cuda.empty_cache()
    
    # Initialize scaler for mixed precision training
    global scaler
    scaler = GradScaler('cuda', enabled=args.use_amp)
    logger.info(f"Using automatic mixed precision: {args.use_amp}")
    
    # Determine which dataset to use based on resolution
    if args.resolution == "low":
        fold_file = "balanced_siim_4fold_128_mor.json" if not args.fold_file else args.fold_file
        # Default ROI size for low resolution (128x128)
        if args.roi_size == [256, 256, 32]:  # If using the default value
            args.roi_size = [128, 128, 32]
        logger.info(f"Using low resolution dataset (128x128xZ) with ROI size: {args.roi_size}")
    else:  # high or original resolution
        fold_file = "balanced_siim_4fold_mor.json" if not args.fold_file else args.fold_file
        logger.info(f"Using original resolution dataset (512x512xZ) with ROI size: {args.roi_size}")
    
    # Load fold data
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    logger.info(f"Using fold file: {fold_file}")
    
    # Get training and validation data for the specified fold
    train_folds = [f"fold_{(args.fold+1) % 4}", f"fold_{(args.fold+2) % 4}"]
    val_fold = f"fold_{(args.fold+3) % 4}"
    test_fold = f"fold_{(args.fold+4) % 4}"  # This is equivalent to fold_{args.fold}
    
    logger.info(f"Training folds: {train_folds}")
    logger.info(f"Validation fold: {val_fold}")
    logger.info(f"Test fold: {test_fold}")
    
    train_data = []
    for fold in train_folds:
        if fold in fold_data:  # Check if fold exists in fold_data
            train_data.extend(fold_data[fold])
    
    val_data = fold_data.get(val_fold, [])
    test_data = fold_data.get(test_fold, [])
    
    # log memory before loading data
    logger.info(f"GPU memory before data loading: {torch.cuda.memory_allocated() / (1024**2):.2f} MB / {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    
    # Filter out samples with z-dimension < min_z_dim
    train_data, invalid_train = filter_invalid_samples(train_data, min_z_dim=args.min_z_dim, logger=logger)
    val_data, invalid_val = filter_invalid_samples(val_data, min_z_dim=args.min_z_dim, logger=logger)
    
    logger.info(f"Training samples: {len(train_data)} (filtered out {len(invalid_train)} samples with z < {args.min_z_dim})")
    logger.info(f"Validation samples: {len(val_data)} (filtered out {len(invalid_val)} samples with z < {args.min_z_dim})")
    
    # Define min-max normalization parameters
    min_intensity = -1024
    max_intensity = 1976
    
    # Split transforms into load and training transforms for better caching
    # Load transforms will be cached, training transforms will be applied dynamically
    load_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Min-Max normalization with clipping
        ScaleIntensityd(keys=["image"], minv=min_intensity, maxv=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
    ])
    
    train_aug_transforms = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], prob=0.5, offsets=0.1),
        RandScaleIntensityd(keys=["image"], prob=0.5, factors=0.1),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Min-Max normalization with clipping
        ScaleIntensityd(keys=["image"], minv=min_intensity, maxv=max_intensity, clip=True),
        CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Make sure cache directory exists
    if args.cache_type in ["disk", "both"]:
        os.makedirs(args.cache_dir, exist_ok=True)
        logger.info(f"Using disk cache directory: {args.cache_dir}")
    
    # Create datasets with appropriate caching strategy
    logger.info(f"Creating datasets with cache_type={args.cache_type}...")
    
    if args.cache_type == "none":
        # No caching - direct dataset access
        train_dataset = SIIMSegmentationDataset(train_data, transforms=Compose([load_transforms, train_aug_transforms]))
        val_dataset = SIIMSegmentationDataset(val_data, transforms=val_transforms)
        
    elif args.cache_type == "disk":
        # Persistent disk cache
        train_dataset = PersistentDataset(
            data=train_data,
            transform=train_aug_transforms,
            cache_dir=os.path.join(args.cache_dir, "train"),
            pre_transform=load_transforms
        )
        val_dataset = PersistentDataset(
            data=val_data,
            transform=val_transforms,
            cache_dir=os.path.join(args.cache_dir, "val")
        )
        logger.info("Using persistent disk cache - first run will be slow, subsequent runs will be fast")
        
    elif args.cache_type == "memory":
        # In-memory cache
        from monai.data import CacheDataset
        train_dataset = CacheDataset(
            data=train_data,
            transform=Compose([load_transforms, train_aug_transforms]),
            cache_rate=args.cache_rate,
            num_workers=args.num_workers
        )
        val_dataset = CacheDataset(
            data=val_data,
            transform=val_transforms,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers
        )
        logger.info(f"Using memory cache with {args.cache_rate} cache rate")
        
    else:  # "both"
        # Both disk and memory cache - create PersistentDataset first, then wrap with CacheDataset
        from monai.data import CacheDataset
        
        # Create persistent datasets for initial loading
        train_persistent = PersistentDataset(
            data=train_data,
            transform=load_transforms,
            cache_dir=os.path.join(args.cache_dir, "train_pre")
        )
        
        val_persistent = PersistentDataset(
            data=val_data,
            transform=Compose([
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"], minv=min_intensity, maxv=max_intensity, clip=True),
                CenterSpatialCropd(keys=["image", "label"], roi_size=args.roi_size)
            ]),
            cache_dir=os.path.join(args.cache_dir, "val_pre")
        )
        
        # Then create cache datasets that use the outputs from persistent datasets
        train_dataset = CacheDataset(
            data=train_persistent,
            transform=train_aug_transforms,
            cache_rate=args.cache_rate,
            num_workers=args.num_workers
        )
        
        val_dataset = CacheDataset(
            data=val_persistent,
            transform=Compose([ToTensord(keys=["image", "label"]), EnsureTyped(keys=["image", "label"])]),
            cache_rate=args.cache_rate,
            num_workers=args.num_workers
        )
        
        logger.info("Using both persistent disk cache and memory cache")
        
    # Preload data if requested (makes startup slower but training much faster)
    if args.preload_data:
        logger.info("Preloading and transforming data (may take a while)...")
        start_time = time.time()
        
        # Load first few samples to prime cache
        preload_count = min(10, len(train_dataset))
        for i in tqdm(range(preload_count), desc="Preloading data"):
            _ = train_dataset[i]
            
        load_time = time.time() - start_time
        logger.info(f"Preloaded {preload_count} samples in {load_time:.2f} seconds")
    
    # Configure DataLoader for optimal performance
    logger.info(f"Creating DataLoaders with batch_size={args.batch_size}, num_workers={args.num_workers}")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model and explicitly move to GPU
    logger.info("Creating model and transferring to GPU...")
    model = SegmentationModel(in_channels=1, out_channels=1).to(device)
    if torch.cuda.is_available():
        # Ensure the model is on GPU
        if next(model.parameters()).device != device:
            model = model.to(device)
        torch.cuda.synchronize()
        logger.info(f"Model transferred to {next(model.parameters()).device}")
    
    # Display model architecture and parameter summary
    logger.info(f"\nModel Architecture:\n{model}")
    
    # Calculate and display total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print parameter summary
    logger.info(f"\nModel Parameters:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Print shapes of each parameter
    logger.info("\nLayer shapes:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.shape}")
        
    # Create a dummy input to trace the output shapes
    dummy_input = torch.zeros(1, 1, *args.roi_size).to(device)
    logger.info(f"\nInput shape: {dummy_input.shape}")
    
    # Get output shape
    with torch.no_grad():
        dummy_output = model(dummy_input)
        logger.info(f"Output shape: {dummy_output.shape}\n")
    
    # Define sliding window inferer for validation
    sliding_window_inferer = SlidingWindowInferer(
        roi_size=args.roi_size,
        sw_batch_size=4,
        overlap=0.5,
        mode="gaussian",
        padding_mode="constant",
        cache_roi_weight_map=True
    )
    
    # Define loss function, optimizer, scheduler, and metrics
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Cosine annealing scheduler to gradually decrease learning rate from initial to minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.min_learning_rate
    )
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Use CUDA events for synchronization and timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.empty_cache()
        # Prefetch to GPU
        logger.info("Warming up CUDA cache...")
        dummy_input = torch.randn(2, 1, *args.roi_size, device=device)
        with autocast(device_type='cuda', enabled=args.use_amp):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        logger.info("CUDA cache warmed up")

    # Training loop
    best_dice = -1
    best_model_path = os.path.join(models_dir, f"best_model_fold{args.fold}.pth")
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Log memory before training
        logger.info(f"GPU memory before training epoch: {torch.cuda.memory_allocated() / (1024**2):.2f} MB / {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        
        if epoch == 0:
            logger.info("Testing data loading speed...")
            try:
                # Get dataloader iterator
                start_time = time.time()
                train_iter = iter(train_dataloader)
                iter_time = time.time() - start_time
                logger.info(f"Creating dataloader iterator took {iter_time:.4f} seconds")
                
                # Time how long it takes to get first batch
                start_time = time.time()
                logger.info("Loading first batch from iterator... (this might take time)")
                first_batch = next(train_iter)
                load_time = time.time() - start_time
                logger.info(f"First batch loaded in {load_time:.4f} seconds")
                logger.info(f"First batch shapes: Image={first_batch['image'].shape}, Label={first_batch['label'].shape}")
                
                # Time how long it takes to load to GPU
                start_time = time.time()
                logger.info("Transferring first batch to GPU...")
                first_batch_gpu = {
                    'image': first_batch['image'].to(device, non_blocking=True),
                    'label': first_batch['label'].to(device, non_blocking=True)
                }
                gpu_time = time.time() - start_time
                logger.info(f"GPU transfer took {gpu_time:.4f} seconds")
                
                # Free memory
                del first_batch, first_batch_gpu, train_iter
                torch.cuda.empty_cache()
                
                logger.info("Data loading test complete, proceeding with normal training")
            except Exception as e:
                logger.error(f"Error during data loading test: {str(e)}")
        
        if torch.cuda.is_available():
            start_event.record()
            
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_function, device, logger)
        
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            logger.info(f"Training epoch time: {start_event.elapsed_time(end_event) / 1000:.2f} seconds")
        
        # Log memory after training
        logger.info(f"GPU memory after training epoch: {torch.cuda.memory_allocated() / (1024**2):.2f} MB / {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_dataloader, loss_function, dice_metric, device, logger, sliding_window_inferer)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, LR: {current_lr:.6f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save metrics to training log
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "learning_rate": current_lr,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        training_metrics.append(epoch_metrics)
        
        # Save training metrics as JSON
        metrics_path = os.path.join(logs_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=4)
        
        # Save checkpoint every N epochs (as specified by args.checkpoint_freq)
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': current_lr,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved new best model with Dice: {best_dice:.4f}")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"final_model_fold{args.fold}.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed! Best Dice: {best_dice:.4f}")
    
    # Log final GPU memory stats
    logger.info(f"Final GPU memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB / {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    
    # Copy the best model to the experiment root directory for easy access
    shutil.copy(best_model_path, os.path.join(experiment_dir, f"best_model.pth"))
    logger.info(f"Copied best model to {os.path.join(experiment_dir, 'best_model.pth')}")
    
    return best_dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIIM Segmentation Training")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to use for validation (0-3)")
    parser.add_argument("--fold_file", type=str, default="", help="Path to the fold file")
    parser.add_argument("--output_dir", type=str, default="training", help="Directory to save model outputs")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save training logs")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to save persistent cache files")
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Base directory for experiment outputs")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Save model checkpoints every N epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloader workers")
    parser.add_argument("--cache_rate", type=float, default=1.0, help="Proportion of data to cache in memory")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Minimum learning rate at the end of training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--min_z_dim", type=int, default=32, help="Minimum required z dimension for samples")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[256, 256, 32], help="ROI size for training [x, y, z]")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Cache dataset in memory for faster loading")
    parser.add_argument("--cache_type", choices=["memory", "disk", "both", "none"], default="both", 
                        help="Caching strategy: memory, disk, both, or none")
    parser.add_argument("--preload_data", action="store_true", default=True, 
                        help="Pre-load and transform data before training begins (slower startup, faster epochs)")
    parser.add_argument("--resolution", choices=["high", "low"], default="high",
                        help="Dataset resolution: high (512x512) or low (128x128)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    main(args) 