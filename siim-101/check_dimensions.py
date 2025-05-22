#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from collections import defaultdict

def check_nifti_file(file_path):
    """Load a NIfTI file and extract its dimensions, min, and max values."""
    try:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        
        stats = {
            "shape": img.shape,
            "dimensions": len(img.shape),
            "min_value": float(np.min(img_data)),
            "max_value": float(np.max(img_data)),
            "mean_value": float(np.mean(img_data)),
            "std_value": float(np.std(img_data)),
            "data_type": str(img.get_data_dtype()),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "is_valid": True
        }
        return stats
    except Exception as e:
        return {
            "error": str(e),
            "is_valid": False
        }

def main(args):
    # Load fold data
    print(f"Loading fold data from {args.fold_file}...")
    with open(args.fold_file, 'r') as f:
        fold_data = json.load(f)
    
    # Initialize statistics
    stats = []
    shape_counts = defaultdict(int)
    
    # Process each pair across all folds
    all_pairs = []
    for fold_name, fold_pairs in fold_data.items():
        if fold_name != "metadata":  # Skip metadata
            all_pairs.extend(fold_pairs)
    
    print(f"Processing {len(all_pairs)} image-label pairs...")
    for idx, pair in enumerate(tqdm(all_pairs)):
        image_path = pair["image"]
        label_path = pair["label"]
        
        # Check image and label files
        image_stats = check_nifti_file(image_path)
        label_stats = check_nifti_file(label_path)
        
        # Record shape count for statistics
        if image_stats["is_valid"]:
            shape_str = "x".join(str(dim) for dim in image_stats["shape"])
            shape_counts[shape_str] += 1
        
        # Store statistics
        pair_stats = {
            "id": idx,
            "image_path": image_path,
            "label_path": label_path,
            "image_valid": image_stats["is_valid"],
            "label_valid": label_stats["is_valid"],
        }
        
        # Add image stats if valid
        if image_stats["is_valid"]:
            pair_stats.update({
                "image_shape": image_stats["shape"],
                "image_dimensions": image_stats["dimensions"],
                "image_min": image_stats["min_value"],
                "image_max": image_stats["max_value"],
                "image_mean": image_stats["mean_value"],
                "image_std": image_stats["std_value"],
                "image_data_type": image_stats["data_type"],
                "image_file_size_mb": image_stats["file_size_mb"],
            })
        else:
            pair_stats["image_error"] = image_stats.get("error", "Unknown error")
        
        # Add label stats if valid
        if label_stats["is_valid"]:
            pair_stats.update({
                "label_shape": label_stats["shape"],
                "label_dimensions": label_stats["dimensions"],
                "label_min": label_stats["min_value"],
                "label_max": label_stats["max_value"],
                "label_mean": label_stats["mean_value"],
                "label_std": label_stats["std_value"],
                "label_data_type": label_stats["data_type"],
                "label_file_size_mb": label_stats["file_size_mb"],
            })
        else:
            pair_stats["label_error"] = label_stats.get("error", "Unknown error")
        
        stats.append(pair_stats)
    
    # Calculate aggregate statistics
    total_valid_pairs = sum(1 for s in stats if s["image_valid"] and s["label_valid"])
    shape_mismatch_count = sum(1 for s in stats if s["image_valid"] and s["label_valid"] and s["image_shape"] != s["label_shape"])
    
    # Add summary
    summary = {
        "total_pairs": len(stats),
        "valid_pairs": total_valid_pairs,
        "invalid_pairs": len(stats) - total_valid_pairs,
        "shape_mismatches": shape_mismatch_count,
        "shapes_distribution": {k: v for k, v in shape_counts.items()},
    }
    
    # Create output data
    output_data = {
        "summary": summary,
        "pairs": stats
    }
    
    # Save as JSON
    json_path = os.path.join(args.output_dir, args.output_prefix + "_stats.json")
    print(f"Saving detailed statistics to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)  # Use default=str to handle non-serializable objects
    
    # Save as CSV (flattening the data structure)
    csv_data = []
    for pair in stats:
        if pair["image_valid"] and pair["label_valid"]:
            csv_row = {
                "id": pair["id"],
                "image_path": pair["image_path"],
                "label_path": pair["label_path"],
                "image_shape_x": pair["image_shape"][0],
                "image_shape_y": pair["image_shape"][1],
                "image_shape_z": pair["image_shape"][2] if len(pair["image_shape"]) > 2 else None,
                "label_shape_x": pair["label_shape"][0],
                "label_shape_y": pair["label_shape"][1],
                "label_shape_z": pair["label_shape"][2] if len(pair["label_shape"]) > 2 else None,
                "image_min": pair["image_min"],
                "image_max": pair["image_max"],
                "image_mean": pair["image_mean"],
                "image_std": pair["image_std"],
                "label_min": pair["label_min"],
                "label_max": pair["label_max"],
                "label_mean": pair["label_mean"],
                "label_std": pair["label_std"],
                "shape_match": pair["image_shape"] == pair["label_shape"]
            }
            csv_data.append(csv_row)
    
    # Create DataFrame and save as CSV
    csv_path = os.path.join(args.output_dir, args.output_prefix + "_stats.csv")
    print(f"Saving CSV summary to {csv_path}...")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\nData Statistics Summary:")
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"Valid pairs: {summary['valid_pairs']}")
    print(f"Invalid pairs: {summary['invalid_pairs']}")
    print(f"Pairs with shape mismatches: {summary['shape_mismatches']}")
    print("\nImage shapes distribution:")
    for shape_str, count in summary["shapes_distribution"].items():
        print(f"  {shape_str}: {count} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dimensions, min, and max values of SIIM dataset files")
    parser.add_argument("--fold_file", type=str, default="balanced_siim_4fold.json", help="Path to the fold file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    parser.add_argument("--output_prefix", type=str, default="dataset", help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 