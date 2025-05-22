#!/usr/bin/env python3

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_pairs(stats_file, min_z_dim=32):
    """
    Analyze pairs from the stats file and mark entries with z-dimension < min_z_dim as invalid
    """
    # Load the statistics file
    print(f"Loading data from {stats_file}...")
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    pairs = data['pairs']
    
    # Initialize counters and collections
    total_pairs = len(pairs)
    invalid_z_count = 0
    already_invalid_count = 0
    shapes_count = defaultdict(int)
    intensity_ranges = []
    
    # Lists for collecting data about normalization
    image_min_values = []
    image_max_values = []
    image_mean_values = []
    image_std_values = []
    
    # Track invalid IDs
    invalid_ids = []
    
    # Analyze each pair
    for pair in pairs:
        # Skip already invalid pairs
        if not pair['image_valid'] or not pair['label_valid']:
            already_invalid_count += 1
            continue
        
        # Extract shape information
        img_shape = pair['image_shape']
        shape_str = f"{img_shape[0]}x{img_shape[1]}x{img_shape[2]}"
        shapes_count[shape_str] += 1
        
        # Check z dimension
        z_dim = img_shape[2]
        if z_dim < min_z_dim:
            invalid_z_count += 1
            invalid_ids.append({
                'id': pair['id'],
                'image_path': pair['image_path'],
                'label_path': pair['label_path'],
                'shape': shape_str
            })
        
        # Collect normalization info
        image_min_values.append(pair['image_min'])
        image_max_values.append(pair['image_max'])
        image_mean_values.append(pair['image_mean'])
        image_std_values.append(pair['image_std'])
        
        # Collect intensity range info
        intensity_ranges.append({
            'min': pair['image_min'],
            'max': pair['image_max'],
            'mean': pair['image_mean'],
            'std': pair['image_std']
        })
    
    # Sort shapes by frequency
    sorted_shapes = sorted(shapes_count.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate statistics for normalization advice
    min_value = np.min(image_min_values)
    max_value = np.max(image_max_values)
    mean_of_means = np.mean(image_mean_values)
    mean_of_stds = np.mean(image_std_values)
    
    # Generate report
    report = {
        'total_pairs': total_pairs,
        'already_invalid': already_invalid_count,
        'invalid_z_dimension': invalid_z_count,
        'valid_for_128x128x32': total_pairs - already_invalid_count - invalid_z_count,
        'shape_distribution': dict(sorted_shapes),
        'normalization_statistics': {
            'min_value': min_value,
            'max_value': max_value,
            'mean_of_means': mean_of_means,
            'mean_of_stds': mean_of_stds
        },
        'recommended_normalization_methods': [
            {
                'name': 'Min-Max Normalization',
                'description': 'Scale data to range [0, 1]',
                'implementation': 'ScaleIntensityd(keys=["image"], minv=0, maxv=1)',
            },
            {
                'name': 'Z-Score Normalization',
                'description': 'Standardize data to have zero mean and unit variance',
                'implementation': f'NormalizeIntensityd(keys=["image"], subtrahend={mean_of_means:.4f}, divisor={mean_of_stds:.4f})',
            },
            {
                'name': 'Custom Range Normalization',
                'description': 'Scale data to specific range, e.g., [-1, 1]',
                'implementation': 'ScaleIntensityd(keys=["image"], minv=-1, maxv=1)',
            }
        ],
        'invalid_pairs': invalid_ids
    }
    
    return report

def main(args):
    # Analyze pairs
    report = analyze_pairs(args.stats_file, args.min_z_dim)
    
    # Print console summary
    print("\n=== Dataset Analysis Summary ===")
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Already invalid pairs: {report['already_invalid']}")
    print(f"Pairs with z-dimension < {args.min_z_dim}: {report['invalid_z_dimension']}")
    print(f"Valid pairs for {args.patch_size} patches: {report['valid_for_128x128x32']}")
    
    print("\n=== Shape Distribution ===")
    shapes = report['shape_distribution']
    for shape, count in list(shapes.items())[:10]:  # Show top 10
        print(f"  {shape}: {count} pairs")
    if len(shapes) > 10:
        print(f"  ... and {len(shapes) - 10} more shapes")
    
    print("\n=== Normalization Statistics ===")
    norm_stats = report['normalization_statistics']
    print(f"Image value range: [{norm_stats['min_value']:.4f}, {norm_stats['max_value']:.4f}]")
    print(f"Mean intensity: {norm_stats['mean_of_means']:.4f}")
    print(f"Mean standard deviation: {norm_stats['mean_of_stds']:.4f}")
    
    print("\n=== Recommended Normalization Methods ===")
    for i, method in enumerate(report['recommended_normalization_methods']):
        print(f"{i+1}. {method['name']} - {method['description']}")
        print(f"   Implementation: {method['implementation']}")
    
    print(f"\n=== Invalid Pairs (z < {args.min_z_dim}) ===")
    invalid_pairs = report['invalid_pairs']
    for i, pair in enumerate(invalid_pairs[:10]):  # Show at most 10
        print(f"  #{pair['id']}: {pair['shape']} - {os.path.basename(pair['image_path'])}")
    if len(invalid_pairs) > 10:
        print(f"  ... and {len(invalid_pairs) - 10} more invalid pairs")
    
    # Save the report
    output_path = os.path.join(args.output_dir, "z_dimension_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {output_path}")
    
    # Update training script with recommended patch size
    if args.update_training_script:
        print("\n=== Updating Training Script with 128x128x32 Patch Size ===")
        update_training_script(args.training_script, args.patch_size)
    
    return 0

def update_training_script(script_path, patch_size):
    """Update the training script to use the specified patch size."""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Extract dimensions from patch_size
        x, y, z = patch_size.split('x')
        
        # Replace patch size in RandCropd transforms
        content = content.replace(
            "RandCropd(keys=[\"image\", \"label\"], roi_size=[96, 96, 32], random_size=False)",
            f"RandCropd(keys=[\"image\", \"label\"], roi_size=[{x}, {y}, {z}], random_size=False)"
        )
        
        # Write updated content
        with open(script_path, 'w') as f:
            f.write(content)
        
        print(f"Successfully updated {script_path} with patch size {patch_size}")
        return True
    except Exception as e:
        print(f"Error updating training script: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset for compatibility with specified patch size")
    parser.add_argument("--stats_file", type=str, default="analysis/dataset_stats.json", help="Path to the statistics JSON file")
    parser.add_argument("--min_z_dim", type=int, default=32, help="Minimum required z dimension")
    parser.add_argument("--patch_size", type=str, default="128x128x32", help="Target patch size (XxYxZ)")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save output files")
    parser.add_argument("--update_training_script", action="store_true", help="Update the training script with the new patch size")
    parser.add_argument("--training_script", type=str, default="train_segmentation.py", help="Path to the training script")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    exit(main(args)) 