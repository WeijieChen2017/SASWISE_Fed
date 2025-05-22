#!/usr/bin/env python3

import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np

# Base path
base_path = "siim-101/SIIM_Fed_Learning_Phase1Data"

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
                "label": label_file
            })
    
    return data_pairs

# Collect all data from all sites with counts
site_counts = {}
for site in site_folders:
    site_data = get_site_data(site)
    site_counts[site] = len(site_data)

# Define fold distribution
site_to_fold = {
    "Site_1_Test": 0,
    "Site_2_Test": 0,
    "Site_3_Test": 1,
    "Site_4_Test": 1,
    "Site_5_Test": 2,
    "Site_6_Test": 3
}

# Create matrix of fold vs site
fold_names = [f"fold_{i}" for i in range(4)]
matrix_data = np.zeros((len(fold_names), len(site_folders)), dtype=int)

# Fill matrix with sample counts
for i, site in enumerate(site_folders):
    fold_idx = site_to_fold[site]
    matrix_data[fold_idx, i] = site_counts[site]

# Create DataFrame for better visualization
df = pd.DataFrame(matrix_data, index=fold_names, columns=site_folders)

# Add row and column totals
df['Total'] = df.sum(axis=1)
df.loc['Total'] = df.sum(axis=0)

# Print the matrix
print("\nMatrix of Folds vs Sites (values are sample counts):")
print(df)

# Save the matrix to a CSV file
matrix_path = "siim-101/fold_site_matrix.csv"
df.to_csv(matrix_path)
print(f"\nMatrix saved to {matrix_path}")

# Additional statistics
print("\nFold Distribution Summary:")
for fold in fold_names:
    sites_in_fold = [site for site, fold_idx in site_to_fold.items() if f"fold_{fold_idx}" == fold]
    print(f"{fold}: Contains {sites_in_fold} with {df.loc[fold, 'Total']} total samples")

print("\nSite Distribution Summary:")
for site in site_folders:
    fold = f"fold_{site_to_fold[site]}"
    print(f"{site}: Assigned to {fold} with {site_counts[site]} samples") 