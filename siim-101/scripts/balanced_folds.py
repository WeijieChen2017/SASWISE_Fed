#!/usr/bin/env python3

import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Base path - use relative path from the script location
base_path = "../SIIM_Fed_Learning_Phase1Data"

# Site folders
site_folders = [
    "Site_1_Test",
    "Site_2_Test",
    "Site_3_Test",
    "Site_4_Test",
    "Site_5_Test",
    "Site_6_Test"
]

# Function to get all image and label files for a site
def get_site_data(site_folder):
    site_path = os.path.join(base_path, site_folder, "For_FedTraining")
    
    # Get all image files (.nii.gz files in data directory)
    image_files = sorted(glob.glob(os.path.join(site_path, "data", "*.nii*")))
    
    # Get all label files (.nii.gz files in labels directory)
    label_files = sorted(glob.glob(os.path.join(site_path, "labels", "*.nii*")))
    
    # Create image-label pairs
    data_pairs = []
    for img_file in image_files:
        # Extract the file number (e.g., "10.nii.gz" -> "10")
        file_name = os.path.basename(img_file)
        file_num = os.path.splitext(file_name)[0]
        if file_num.endswith('.nii'):  # Handle both .nii and .nii.gz
            file_num = os.path.splitext(file_num)[0]
        
        # Construct the corresponding label file path
        label_file = os.path.join(site_path, "labels", os.path.basename(img_file))
        
        # Check if label file exists
        if os.path.exists(label_file):
            data_pairs.append({
                "image": img_file,
                "label": label_file,
                "site": site_folder
            })
    
    return data_pairs

# Collect all data from all sites
all_site_data = {}
for site in site_folders:
    site_data = get_site_data(site)
    if site_data:  # Only add if there's data
        all_site_data[site] = site_data
    else:
        print(f"Warning: No data found for {site}")

# Initialize the 4 folds
folds = {f"fold_{i}": [] for i in range(4)}

# For each site, split its data into 4 roughly equal parts
for site, data in all_site_data.items():
    # Shuffle the data to ensure random distribution
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate the number of samples per fold for this site
    samples_per_fold = len(shuffled_data) // 4
    remainder = len(shuffled_data) % 4
    
    # Distribute samples to folds
    start_idx = 0
    for i in range(4):
        # Calculate end index for this fold
        # Add one extra sample for the first 'remainder' folds to handle non-divisible cases
        end_idx = start_idx + samples_per_fold + (1 if i < remainder else 0)
        
        # Add this slice of data to the corresponding fold
        folds[f"fold_{i}"].extend(shuffled_data[start_idx:end_idx])
        
        # Update start index for next fold
        start_idx = end_idx

# Create a matrix to show the distribution of samples across folds and sites
matrix_data = np.zeros((4, len(site_folders)), dtype=int)

# Fill the matrix
for fold_idx in range(4):
    fold_data = folds[f"fold_{fold_idx}"]
    for item in fold_data:
        site = item["site"]
        site_idx = site_folders.index(site)
        matrix_data[fold_idx, site_idx] += 1

# Create DataFrame for better visualization
df = pd.DataFrame(matrix_data, index=[f"fold_{i}" for i in range(4)], columns=site_folders)

# Add row and column totals
df['Total'] = df.sum(axis=1)
df.loc['Total'] = df.sum(axis=0)

# Print the matrix
print("\nMatrix of Folds vs Sites (values are sample counts):")
print(df)

# Save the matrix to a CSV file
matrix_path = "../balanced_fold_site_matrix.csv"
df.to_csv(matrix_path)
print(f"\nMatrix saved to {matrix_path}")

# Format data for JSON output
output = {f"fold_{i}": [{"image": item["image"], "label": item["label"]} for item in folds[f"fold_{i}"]] for i in range(4)}

# Add metadata
output["metadata"] = {
    "fold_distribution": "balanced",
    "total_samples": sum(len(fold_data) for fold_data in folds.values()),
    "site_count_per_fold": df.drop('Total').to_dict(),
    "fold_sizes": {k: len(v) for k, v in folds.items()}
}

# Save to JSON file
output_path = "../balanced_siim_4fold.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nBalanced fold data saved to {output_path}")

# Additional statistics
print("\nFold Distribution Summary:")
for fold in folds.keys():
    fold_data = folds[fold]
    sites_in_fold = set(item["site"] for item in fold_data)
    site_counts = {site: sum(1 for item in fold_data if item["site"] == site) for site in sites_in_fold}
    print(f"{fold}: Contains {len(fold_data)} total samples from {len(sites_in_fold)} sites")
    for site, count in site_counts.items():
        site_total = len(all_site_data[site])
        print(f"  {site}: {count}/{site_total} samples ({(count/site_total)*100:.1f}%)") 