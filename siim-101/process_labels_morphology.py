#!/usr/bin/env python3

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
from scipy import ndimage

def create_circular_kernel(size=5):
    """
    Create a circular kernel for morphological operations
    
    Args:
        size: Size of the kernel (default: 5)
        
    Returns:
        Circular kernel as numpy array
    """
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    cv2.circle(kernel, (center, center), center, 1, -1)
    return kernel

def process_label_with_morphology(label_path, suffix="_mor", save=True):
    """
    Load a label, apply morphological operations (dilate and erode), binarize, and save
    
    Args:
        label_path: Path to the label file
        suffix: Suffix to add to the output filename (default: "_mor")
        save: Whether to save the processed label (default: True)
        
    Returns:
        Path to the processed label if saved, otherwise None
    """
    # Load the label
    img = nib.load(label_path)
    data = img.get_fdata()
    
    # Create circular kernel for morphological operations
    kernel = create_circular_kernel(5)
    
    # Process each slice in each orientation
    processed_data = np.zeros_like(data)
    
    # Axial slices (z-axis)
    for z in range(data.shape[2]):
        slice_data = data[:, :, z]
        # Apply dilate and erode
        dilated = ndimage.binary_dilation(slice_data, structure=kernel)
        processed = ndimage.binary_erosion(dilated, structure=kernel)
        # Update processed data
        processed_data[:, :, z] = processed
    
    # Coronal slices (y-axis)
    for y in range(data.shape[1]):
        slice_data = data[:, y, :]
        # Apply dilate and erode
        dilated = ndimage.binary_dilation(slice_data, structure=kernel)
        processed = ndimage.binary_erosion(dilated, structure=kernel)
        # Update processed data (averaging with existing processed data)
        processed_data[:, y, :] = np.maximum(processed_data[:, y, :], processed)
    
    # Sagittal slices (x-axis)
    for x in range(data.shape[0]):
        slice_data = data[x, :, :]
        # Apply dilate and erode
        dilated = ndimage.binary_dilation(slice_data, structure=kernel)
        processed = ndimage.binary_erosion(dilated, structure=kernel)
        # Update processed data (averaging with existing processed data)
        processed_data[x, :, :] = np.maximum(processed_data[x, :, :], processed)
    
    # Binarize using threshold
    binary_data = (processed_data >= 0.5).astype(np.uint8)
    
    # Create output path
    if save:
        path = Path(label_path)
        # Handle .nii.gz files
        stem = path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
            output_path = str(path.with_name(f"{stem}{suffix}.nii.gz"))
        else:
            output_path = str(path.with_stem(f"{path.stem}{suffix}"))
        
        # Save the binary label
        binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
        nib.save(binary_img, output_path)
        print(f"Processed and saved to {output_path}")
        
        return output_path
    
    return None

def process_json_file(json_path, suffix="_mor", save=True):
    """
    Process all labels in a JSON file
    
    Args:
        json_path: Path to the JSON file
        suffix: Suffix to add to the output filenames (default: "_mor")
        save: Whether to save the processed labels (default: True)
        
    Returns:
        Dictionary mapping original labels to processed labels
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract all label paths
    label_paths = []
    for fold_key, fold_data in data.items():
        if fold_key == "metadata":
            continue
        
        for item in fold_data:
            if "label" in item:
                label_paths.append(item["label"])
    
    # Remove duplicates
    label_paths = list(set(label_paths))
    print(f"Found {len(label_paths)} unique labels in {json_path}")
    
    # Process each label
    results = {}
    for label_path in tqdm(label_paths, desc="Processing labels"):
        output_path = process_label_with_morphology(label_path, suffix, save)
        if output_path:
            results[label_path] = output_path
    
    print(f"Processed {len(results)} labels")
    return results

def update_json_file(json_path, mapping, output_path=None):
    """
    Update the JSON file with new label paths
    
    Args:
        json_path: Path to the original JSON file
        mapping: Dictionary mapping original labels to processed labels
        output_path: Path to save the updated JSON file (default: None, which adds a suffix to the original path)
    
    Returns:
        Path to the updated JSON file
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update label paths
    for fold_key, fold_data in data.items():
        if fold_key == "metadata":
            continue
        
        for item in fold_data:
            if "label" in item and item["label"] in mapping:
                item["label"] = mapping[item["label"]]
    
    # Create output path
    if not output_path:
        path = Path(json_path)
        output_path = str(path.with_stem(f"{path.stem}_mor"))
    
    # Save the updated JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated JSON file saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply morphological operations and binarize label files")
    parser.add_argument("--json_path", type=str, default="balanced_siim_4fold.json",
                        help="Path to the JSON file containing label paths")
    parser.add_argument("--suffix", type=str, default="_mor",
                        help="Suffix to add to the output filenames (default: '_mor')")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save the processed labels")
    parser.add_argument("--update_json", action="store_true", default=False,
                        help="Update the JSON file with new label paths")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save the updated JSON file (default: adds '_mor' suffix to original path)")
    
    args = parser.parse_args()
    
    # Process the labels
    mapping = process_json_file(args.json_path, args.suffix, args.save)
    
    # Update the JSON file if requested
    if args.update_json:
        update_json_file(args.json_path, mapping, args.output_json) 