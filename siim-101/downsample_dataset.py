#!/usr/bin/env python3

import os
import json
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

def downsample_image(img_data, target_shape, order=3):
    """
    Downsample image data to target shape
    order=3 for cubic interpolation (for images)
    order=0 for nearest neighbor (for labels)
    """
    current_shape = img_data.shape
    # Calculate zoom factors for each dimension
    # We don't want to resample the channel dimension (if it exists)
    if len(current_shape) > 3:  # Has channel dimension
        zoom_factors = [1]  # Keep channel dimension unchanged
        zoom_factors.extend([target_shape[i] / current_shape[i+1] for i in range(3)])
    else:  # No channel dimension
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
    
    # Perform resampling
    resampled_data = zoom(img_data, zoom_factors, order=order)
    return resampled_data

def process_file(input_file, output_file, is_label=False):
    """Process a single file and save the downsampled version"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load image
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Get original shape and determine target shape
    original_shape = data.shape
    target_shape = [128, 128, original_shape[2]]  # Keep z dimension the same
    
    # Resample
    if is_label:
        # Use nearest neighbor for labels
        resampled_data = downsample_image(data, target_shape, order=0)
    else:
        # Use cubic interpolation for images
        resampled_data = downsample_image(data, target_shape, order=3)
    
    # Create new image with same header
    new_img = nib.Nifti1Image(resampled_data, img.affine, img.header)
    
    # Update header to reflect new dimensions
    pixdim = list(new_img.header['pixdim'])
    pixdim[1] = pixdim[1] * (original_shape[0] / target_shape[0])  # Update x spacing
    pixdim[2] = pixdim[2] * (original_shape[1] / target_shape[1])  # Update y spacing
    new_img.header['pixdim'] = pixdim
    
    # Save downsampled image
    nib.save(new_img, output_file)
    
    return output_file

def create_downsampled_dataset(json_path, output_json_path, suffix="_128"):
    """Create downsampled version of the entire dataset"""
    # Load the dataset configuration
    with open(json_path, 'r') as f:
        dataset_config = json.load(f)
    
    # Create a copy of the configuration for the downsampled dataset
    downsampled_config = {}
    
    # Process each fold
    for fold_key, fold_data in tqdm(dataset_config.items(), desc="Processing folds"):
        # Skip metadata
        if fold_key == "metadata":
            downsampled_config["metadata"] = fold_data
            continue
        
        downsampled_config[fold_key] = []
        
        # Process each sample in the fold
        for sample in tqdm(fold_data, desc=f"Processing {fold_key}"):
            image_path = sample["image"]
            label_path = sample["label"]
            
            # Create output paths with suffix
            image_output_path = add_suffix_to_path(image_path, suffix)
            label_output_path = add_suffix_to_path(label_path, suffix)
            
            # Process files
            try:
                process_file(image_path, image_output_path, is_label=False)
                process_file(label_path, label_output_path, is_label=True)
                
                # Add downsampled sample to configuration
                downsampled_config[fold_key].append({
                    "image": image_output_path,
                    "label": label_output_path
                })
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    
    # Save downsampled configuration
    with open(output_json_path, 'w') as f:
        json.dump(downsampled_config, f, indent=2)
    
    print(f"Downsampled dataset created and saved to {output_json_path}")

def add_suffix_to_path(file_path, suffix):
    """Add suffix to file path before extension"""
    path = Path(file_path)
    stem = path.stem  # Get filename without extension
    
    # Handle .nii.gz files (two extensions)
    if stem.endswith('.nii'):
        stem = stem[:-4]
        return str(path.with_name(f"{stem}{suffix}.nii.gz"))
    
    return str(path.with_stem(f"{path.stem}{suffix}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample SIIM dataset from 512x512xZ to 128x128xZ")
    parser.add_argument("--input_json", type=str, default="balanced_siim_4fold.json", 
                        help="Path to input dataset JSON file")
    parser.add_argument("--output_json", type=str, default="balanced_siim_4fold_128.json",
                        help="Path to output JSON file for downsampled dataset")
    parser.add_argument("--suffix", type=str, default="_128",
                        help="Suffix to add to downsampled files")
    
    args = parser.parse_args()
    
    create_downsampled_dataset(args.input_json, args.output_json, args.suffix) 