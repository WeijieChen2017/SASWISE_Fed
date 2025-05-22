#!/usr/bin/env python3

import os
import json
import glob
from collections import defaultdict

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
                "label": label_file
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

# Create 4 folds by distributing sites evenly
# We'll put 1-2 sites in each fold
folds = defaultdict(list)

# Simple distribution: site 1,2 -> fold 0, site 3,4 -> fold 1, site 5 -> fold 2, site 6 -> fold 3
site_to_fold = {
    "Site_1_Test": 0,
    "Site_2_Test": 0,
    "Site_3_Test": 1,
    "Site_4_Test": 1,
    "Site_5_Test": 2,
    "Site_6_Test": 3
}

# Populate folds
for site, data in all_site_data.items():
    fold_idx = site_to_fold[site]
    folds[f"fold_{fold_idx}"].extend(data)

# Save folds to JSON file
output = {
    f"fold_{i}": folds[f"fold_{i}"] for i in range(4)
}

# Add metadata
output["metadata"] = {
    "fold_distribution": site_to_fold,
    "total_samples": sum(len(fold) for fold in folds.values())
}

# Output file path inside siim-101 folder
output_path = "../siim_4fold.json"

# Save to JSON file
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Created 4 folds with distribution:")
for fold, data in folds.items():
    print(f"{fold}: {len(data)} samples")
print(f"Total samples: {output['metadata']['total_samples']}")
print(f"Data saved to {output_path}") 