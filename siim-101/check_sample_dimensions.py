#!/usr/bin/env python3

import os
import json
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import argparse


def check_sample_dimensions(json_file_path, dataset_name):
    """
    Check dimensions and pixel dimensions for all samples in a JSON fold file.
    
    Args:
        json_file_path (str): Path to the JSON fold file
        dataset_name (str): Name identifier for the dataset (e.g., "128_mor", "original_mor")
    
    Returns:
        list: List of dictionaries containing sample information
    """
    print(f"Processing {dataset_name} dataset from {json_file_path}")
    
    # Load JSON file
    with open(json_file_path, 'r') as f:
        fold_data = json.load(f)
    
    results = []
    
    # Process each fold
    for fold_name, samples in fold_data.items():
        if fold_name == "metadata":  # Skip metadata section
            continue
            
        print(f"  Processing {fold_name} with {len(samples)} samples...")
        
        for sample_idx, sample in enumerate(tqdm(samples, desc=f"  {fold_name}")):
            sample_info = {
                'dataset': dataset_name,
                'fold': fold_name,
                'sample_idx': sample_idx,
                'image_path': sample['image'],
                'label_path': sample['label']
            }
            
            # Process image file
            try:
                if os.path.exists(sample['image']):
                    nii_file = nib.load(sample['image'])
                    img_shape = nii_file.header.get_data_shape()
                    img_zooms = nii_file.header.get_zooms()
                    img_dtype = nii_file.header.get_data_dtype()
                    
                    sample_info.update({
                        'image_exists': True,
                        'image_shape_x': img_shape[0] if len(img_shape) > 0 else None,
                        'image_shape_y': img_shape[1] if len(img_shape) > 1 else None,
                        'image_shape_z': img_shape[2] if len(img_shape) > 2 else -1,  # Set to -1 for 2D cases
                        'image_shape_t': img_shape[3] if len(img_shape) > 3 else None,
                        'image_pixel_dim_x': img_zooms[0] if len(img_zooms) > 0 else None,
                        'image_pixel_dim_y': img_zooms[1] if len(img_zooms) > 1 else None,
                        'image_pixel_dim_z': img_zooms[2] if len(img_zooms) > 2 else -1,  # Set to -1 for 2D cases
                        'image_dtype': str(img_dtype),
                        'image_error': None,
                        'image_is_2d': len(img_shape) < 3  # Flag for 2D images
                    })
                else:
                    sample_info.update({
                        'image_exists': False,
                        'image_shape_x': None, 'image_shape_y': None, 'image_shape_z': None, 'image_shape_t': None,
                        'image_pixel_dim_x': None, 'image_pixel_dim_y': None, 'image_pixel_dim_z': None,
                        'image_dtype': None,
                        'image_error': 'File not found',
                        'image_is_2d': None
                    })
            except Exception as e:
                sample_info.update({
                    'image_exists': False,
                    'image_shape_x': None, 'image_shape_y': None, 'image_shape_z': None, 'image_shape_t': None,
                    'image_pixel_dim_x': None, 'image_pixel_dim_y': None, 'image_pixel_dim_z': None,
                    'image_dtype': None,
                    'image_error': str(e),
                    'image_is_2d': None
                })
            
            # Process label file
            try:
                if os.path.exists(sample['label']):
                    nii_file = nib.load(sample['label'])
                    label_shape = nii_file.header.get_data_shape()
                    label_zooms = nii_file.header.get_zooms()
                    label_dtype = nii_file.header.get_data_dtype()
                    
                    sample_info.update({
                        'label_exists': True,
                        'label_shape_x': label_shape[0] if len(label_shape) > 0 else None,
                        'label_shape_y': label_shape[1] if len(label_shape) > 1 else None,
                        'label_shape_z': label_shape[2] if len(label_shape) > 2 else -1,  # Set to -1 for 2D cases
                        'label_shape_t': label_shape[3] if len(label_shape) > 3 else None,
                        'label_pixel_dim_x': label_zooms[0] if len(label_zooms) > 0 else None,
                        'label_pixel_dim_y': label_zooms[1] if len(label_zooms) > 1 else None,
                        'label_pixel_dim_z': label_zooms[2] if len(label_zooms) > 2 else -1,  # Set to -1 for 2D cases
                        'label_dtype': str(label_dtype),
                        'label_error': None,
                        'label_is_2d': len(label_shape) < 3  # Flag for 2D labels
                    })
                else:
                    sample_info.update({
                        'label_exists': False,
                        'label_shape_x': None, 'label_shape_y': None, 'label_shape_z': None, 'label_shape_t': None,
                        'label_pixel_dim_x': None, 'label_pixel_dim_y': None, 'label_pixel_dim_z': None,
                        'label_dtype': None,
                        'label_error': 'File not found',
                        'label_is_2d': None
                    })
            except Exception as e:
                sample_info.update({
                    'label_exists': False,
                    'label_shape_x': None, 'label_shape_y': None, 'label_shape_z': None, 'label_shape_t': None,
                    'label_pixel_dim_x': None, 'label_pixel_dim_y': None, 'label_pixel_dim_z': None,
                    'label_dtype': None,
                    'label_error': str(e),
                    'label_is_2d': None
                })
            
            # Check if shapes match (considering -1 as valid for 2D cases)
            img_shape_3d = (sample_info.get('image_shape_x'), sample_info.get('image_shape_y'), sample_info.get('image_shape_z'))
            label_shape_3d = (sample_info.get('label_shape_x'), sample_info.get('label_shape_y'), sample_info.get('label_shape_z'))
            sample_info['shapes_match'] = img_shape_3d == label_shape_3d and None not in img_shape_3d
            
            results.append(sample_info)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check dimensions and pixel dimensions of samples in JSON fold files")
    parser.add_argument("--json1", type=str, default="balanced_siim_4fold_128_mor.json", 
                        help="Path to first JSON file (default: balanced_siim_4fold_128_mor.json)")
    parser.add_argument("--json2", type=str, default="balanced_siim_4fold_mor.json", 
                        help="Path to second JSON file (default: balanced_siim_4fold_mor.json)")
    parser.add_argument("--output", type=str, default="sample_dimensions_analysis.csv", 
                        help="Output CSV file path (default: sample_dimensions_analysis.csv)")
    
    args = parser.parse_args()
    
    all_results = []
    
    # Process first JSON file (128 resolution)
    if os.path.exists(args.json1):
        results1 = check_sample_dimensions(args.json1, "128_mor")
        all_results.extend(results1)
    else:
        print(f"Warning: {args.json1} not found, skipping...")
    
    # Process second JSON file (original resolution)
    if os.path.exists(args.json2):
        results2 = check_sample_dimensions(args.json2, "original_mor")
        all_results.extend(results2)
    else:
        print(f"Warning: {args.json2} not found, skipping...")
    
    if not all_results:
        print("No data to process. Exiting.")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        'dataset', 'fold', 'sample_idx',
        'image_path', 'image_exists', 'image_is_2d',
        'image_shape_x', 'image_shape_y', 'image_shape_z', 'image_shape_t',
        'image_pixel_dim_x', 'image_pixel_dim_y', 'image_pixel_dim_z',
        'image_dtype', 'image_error',
        'label_path', 'label_exists', 'label_is_2d',
        'label_shape_x', 'label_shape_y', 'label_shape_z', 'label_shape_t',
        'label_pixel_dim_x', 'label_pixel_dim_y', 'label_pixel_dim_z',
        'label_dtype', 'label_error',
        'shapes_match'
    ]
    
    df = df[column_order]
    df.to_csv(args.output, index=False)
    
    print(f"\nAnalysis complete! Results saved to {args.output}")
    print(f"Total samples processed: {len(df)}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        print(f"\nDataset: {dataset}")
        print(f"  Total samples: {len(dataset_df)}")
        print(f"  Images found: {dataset_df['image_exists'].sum()}")
        print(f"  Labels found: {dataset_df['label_exists'].sum()}")
        print(f"  Shape matches: {dataset_df['shapes_match'].sum()}")
        
        if dataset_df['image_exists'].any():
            print(f"  Image dimensions (X,Y,Z):")
            for _, row in dataset_df[dataset_df['image_exists']].groupby(['image_shape_x', 'image_shape_y', 'image_shape_z']).size().reset_index(name='count').iterrows():
                z_dim_str = "2D" if row['image_shape_z'] == -1 else str(int(row['image_shape_z']))
                print(f"    ({int(row['image_shape_x'])}, {int(row['image_shape_y'])}, {z_dim_str}): {row['count']} samples")
            
            print(f"  Pixel dimensions (X,Y,Z) - unique values:")
            pixel_dims = dataset_df[dataset_df['image_exists']].groupby(['image_pixel_dim_x', 'image_pixel_dim_y', 'image_pixel_dim_z']).size().reset_index(name='count')
            for _, row in pixel_dims.iterrows():
                z_pdim_str = "2D" if row['image_pixel_dim_z'] == -1 else f"{row['image_pixel_dim_z']:.4f}"
                print(f"    ({row['image_pixel_dim_x']:.4f}, {row['image_pixel_dim_y']:.4f}, {z_pdim_str}): {row['count']} samples")
    
    # Check for any errors
    error_df = df[(df['image_error'].notna()) | (df['label_error'].notna())]
    if len(error_df) > 0:
        print(f"\n=== ERRORS FOUND ===")
        print(f"Samples with errors: {len(error_df)}")
        print("First few errors:")
        print(error_df[['dataset', 'fold', 'image_path', 'image_error', 'label_error']].head())
    
    # Summary of 2D cases
    print(f"\n=== 2D CASES SUMMARY ===")
    image_2d_df = df[(df['image_is_2d'] == True) & (df['image_exists'] == True)]
    label_2d_df = df[(df['label_is_2d'] == True) & (df['label_exists'] == True)]
    
    if len(image_2d_df) > 0:
        print(f"Found {len(image_2d_df)} images with 2D dimensions:")
        for dataset in image_2d_df['dataset'].unique():
            dataset_2d = image_2d_df[image_2d_df['dataset'] == dataset]
            print(f"\n  Dataset {dataset}: {len(dataset_2d)} 2D images")
            for fold in dataset_2d['fold'].unique():
                fold_2d = dataset_2d[dataset_2d['fold'] == fold]
                print(f"    {fold}: {len(fold_2d)} samples")
                # Show first few examples
                for _, row in fold_2d.head(3).iterrows():
                    print(f"      - {row['image_path']} (shape: {int(row['image_shape_x'])}x{int(row['image_shape_y'])})")
                if len(fold_2d) > 3:
                    print(f"      ... and {len(fold_2d) - 3} more")
    else:
        print("No 2D images found.")
    
    if len(label_2d_df) > 0:
        print(f"\nFound {len(label_2d_df)} labels with 2D dimensions:")
        for dataset in label_2d_df['dataset'].unique():
            dataset_2d = label_2d_df[label_2d_df['dataset'] == dataset]
            print(f"\n  Dataset {dataset}: {len(dataset_2d)} 2D labels")
            for fold in dataset_2d['fold'].unique():
                fold_2d = dataset_2d[dataset_2d['fold'] == fold]
                print(f"    {fold}: {len(fold_2d)} samples")
                # Show first few examples
                for _, row in fold_2d.head(3).iterrows():
                    print(f"      - {row['label_path']} (shape: {int(row['label_shape_x'])}x{int(row['label_shape_y'])})")
                if len(fold_2d) > 3:
                    print(f"      ... and {len(fold_2d) - 3} more")
    else:
        print("No 2D labels found.")


if __name__ == "__main__":
    main() 