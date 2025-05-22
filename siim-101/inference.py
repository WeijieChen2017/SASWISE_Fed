#!/usr/bin/env python3

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    NormalizeIntensityd,
    ToTensord,
    EnsureTyped,
)
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
)
from tqdm import tqdm

# Import our model from training script
from train_segmentation import SegmentationModel, SIIMSegmentationDataset, filter_invalid_samples

def inference(model, dataloader, device, output_dir):
    model.eval()
    
    # Post-processing transforms
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Inference"):
            inputs = batch_data["image"].to(device)
            
            # Get file paths for saving outputs
            image_paths = batch_data["image_meta_dict"]["filename_or_obj"]
            
            # Model inference
            outputs = model(inputs)
            
            # Post-processing
            outputs = [post_trans(i) for i in decollate_batch(outputs)]
            
            # Save the outputs as NIfTI files
            for i, output in enumerate(outputs):
                # Get the output path by modifying the input path
                input_path = image_paths[i]
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, f"pred_{filename}")
                
                # Convert to numpy and save
                output_np = output.detach().cpu().numpy().astype(np.float32)
                
                # Load original image to get orientation and affine
                original_img = nib.load(input_path)
                
                # Save prediction with same orientation and affine as original
                pred_img = nib.Nifti1Image(output_np[0], original_img.affine, original_img.header)
                nib.save(pred_img, output_path)
                
                print(f"Saved prediction to {output_path}")

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load fold data
    fold_file = args.fold_file
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    # Get test data for the specified fold
    test_fold = f"fold_{args.fold}"
    test_data = fold_data[test_fold]
    
    # Filter out samples with z-dimension < min_z_dim
    test_data, invalid_test = filter_invalid_samples(test_data, min_z_dim=args.min_z_dim)
    print(f"Test fold: {test_fold}, Valid samples: {len(test_data)} (filtered out {len(invalid_test)} samples with z < {args.min_z_dim})")
    
    # Define z-score normalization parameters from dataset analysis (same as training)
    mean_intensity = -718.3605
    std_intensity = 804.9248
    
    # Define transforms
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),  # Keep metadata
        AddChanneld(keys=["image", "label"]),
        # Z-Score normalization based on dataset statistics
        NormalizeIntensityd(keys=["image"], subtrahend=mean_intensity, divisor=std_intensity),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Create test dataset and dataloader
    test_dataset = SIIMSegmentationDataset(test_data, transforms=test_transforms)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Load model
    model = SegmentationModel(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    inference(model, test_dataloader, device, args.output_dir)
    print(f"Inference completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIIM Segmentation Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to use for testing (0-3)")
    parser.add_argument("--fold_file", type=str, default="balanced_siim_4fold.json", help="Path to the fold file")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--min_z_dim", type=int, default=32, help="Minimum required z dimension for samples")
    
    args = parser.parse_args()
    
    main(args) 