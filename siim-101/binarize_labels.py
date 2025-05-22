#!/usr/bin/env python3

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def binarize_label(label_path, threshold=0.5, suffix="_binary", save=True):
    """
    Load a label, apply threshold, and save the binary version
    
    Args:
        label_path: Path to the label file
        threshold: Threshold value for binarization (default: 0.5)
        suffix: Suffix to add to the output filename (default: "_binary")
        save: Whether to save the binarized label (default: True)
        
    Returns:
        Path to the binarized label if saved, otherwise None
    """
    # Load the label
    img = nib.load(label_path)
    data = img.get_fdata()
    
    # Apply threshold
    binary_data = (data >= threshold).astype(np.float32)
    
    # Check if any changes were made
    if np.array_equal(data, binary_data):
        print(f"No changes needed for {label_path}")
        return None
    
    # Create output path
    if save:
        path = Path(label_path)
        if suffix:
            # Handle .nii.gz files
            stem = path.stem
            if stem.endswith('.nii'):
                stem = stem[:-4]
                output_path = str(path.with_name(f"{stem}{suffix}.nii.gz"))
            else:
                output_path = str(path.with_stem(f"{path.stem}{suffix}"))
        else:
            # Overwrite the original file
            output_path = label_path
        
        # Save the binary label
        binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
        nib.save(binary_img, output_path)
        
        return output_path
    
    return None

def process_json_file(json_path, threshold=0.5, suffix="_binary", save=True):
    """
    Process all labels in a JSON file
    
    Args:
        json_path: Path to the JSON file
        threshold: Threshold value for binarization (default: 0.5)
        suffix: Suffix to add to the output filename (default: "_binary")
        save: Whether to save the binarized labels (default: True)
        
    Returns:
        Dictionary mapping original labels to binarized labels
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
        output_path = binarize_label(label_path, threshold, suffix, save)
        if output_path:
            results[label_path] = output_path
    
    print(f"Processed {len(results)} labels, {len(label_paths) - len(results)} were already binary")
    return results

def update_json_file(json_path, mapping, output_path=None):
    """
    Update the JSON file with new label paths
    
    Args:
        json_path: Path to the original JSON file
        mapping: Dictionary mapping original labels to binarized labels
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
        output_path = str(path.with_stem(f"{path.stem}_binary"))
    
    # Save the updated JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated JSON file saved to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize label files using a threshold")
    parser.add_argument("--json_path", type=str, default="balanced_siim_4fold_128.json",
                        help="Path to the JSON file containing label paths")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold value for binarization (default: 0.5)")
    parser.add_argument("--suffix", type=str, default="_binary",
                        help="Suffix to add to the output filenames (default: '_binary')")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save the binarized labels")
    parser.add_argument("--update_json", action="store_true", default=False,
                        help="Update the JSON file with new label paths")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save the updated JSON file (default: adds '_binary' suffix to original path)")
    
    args = parser.parse_args()
    
    # Process the labels
    mapping = process_json_file(args.json_path, args.threshold, args.suffix, args.save)
    
    # Update the JSON file if requested
    if args.update_json:
        update_json_file(args.json_path, mapping, args.output_json) 