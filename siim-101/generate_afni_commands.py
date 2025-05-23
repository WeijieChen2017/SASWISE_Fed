#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import glob
import nibabel as nib

def delete_existing_files(file_path, suffixes_to_delete=["_128", "_binary"]):
    """
    Delete existing files with the specified suffixes
    
    Args:
        file_path: Original file path to use as a base
        suffixes_to_delete: List of suffixes to identify files for deletion
    """
    path = Path(file_path)
    base_name = path.stem
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    
    # Get the directory and the base file name without suffix
    base_dir = path.parent
    
    # Check for each suffix and delete if found
    for suffix in suffixes_to_delete:
        # Create the pattern to match
        pattern = str(base_dir / f"{base_name}{suffix}.*")
        matching_files = glob.glob(pattern)
        
        # Delete each matching file
        for match in matching_files:
            try:
                os.remove(match)
                print(f"Deleted existing file: {match}")
            except (OSError, PermissionError) as e:
                print(f"Error deleting {match}: {e}")

def get_image_dimensions(img_path):
    """
    Get the original pixel spacing/resolution of an image
    
    Args:
        img_path: Path to the NIFTI image
    
    Returns:
        Tuple of (x_dim, y_dim, z_dim) pixel spacing
    """
    try:
        img = nib.load(img_path)
        header = img.header
        pixel_dimensions = header.get_zooms()
        return pixel_dimensions
    except Exception as e:
        print(f"Error reading dimensions from {img_path}: {e}")
        # Default values if we can't read the file
        return (1.0, 1.0, 1.0)

def generate_resample_commands(json_path, output_suffix="_128", output_script="run_afni_resample.sh", 
                               delete_existing=True, suffixes_to_delete=["_128", "_binary"]):
    """
    Generate AFNI 3dresample commands for all images and labels in a JSON file
    
    Args:
        json_path: Path to the JSON file with image and label paths
        output_suffix: Suffix to add to output files (default: "_128")
        output_script: Path to save the shell script with all commands (default: "run_afni_resample.sh")
        delete_existing: Whether to delete existing files with the specified suffixes (default: True)
        suffixes_to_delete: List of suffixes to identify files for deletion (default: ["_128", "_binary"])
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract all image and label paths
    image_paths = []
    label_paths = []
    for fold_key, fold_data in data.items():
        if fold_key == "metadata":
            continue
        
        for item in fold_data:
            if "image" in item and "label" in item:
                image_paths.append(item["image"])
                label_paths.append(item["label"])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_image_paths = [x for x in image_paths if not (x in seen or seen.add(x))]
    seen = set()
    unique_label_paths = [x for x in label_paths if not (x in seen or seen.add(x))]
    
    print(f"Found {len(unique_image_paths)} unique images and {len(unique_label_paths)} unique labels")
    
    # Delete existing files if requested
    if delete_existing:
        print("Deleting existing files with specified suffixes...")
        for img_path in tqdm(unique_image_paths, desc="Cleaning image files"):
            delete_existing_files(img_path, suffixes_to_delete)
        
        for lbl_path in tqdm(unique_label_paths, desc="Cleaning label files"):
            delete_existing_files(lbl_path, suffixes_to_delete)
    
    # Generate commands
    commands = []
    image_commands = []
    label_commands = []
    
    # Process images (cubic interpolation)
    for img_path in tqdm(unique_image_paths, desc="Processing images"):
        # Create output path
        path = Path(img_path)
        stem = path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        output_path = str(path.with_name(f"{stem}{output_suffix}.nii.gz"))
        
        # Get original dimensions
        x_dim, y_dim, z_dim = get_image_dimensions(img_path)
        
        # Calculate new dimensions (4x larger for x and y)
        new_x_dim = x_dim * 4
        new_y_dim = y_dim * 4
        new_z_dim = z_dim  # Keep z dimension the same
        
        # Generate command
        cmd = f"3dresample -dxyz {new_x_dim} {new_y_dim} {new_z_dim} -rmode Cu -prefix {output_path} -input {img_path}"
        image_commands.append(cmd)
    
    # Process labels (nearest neighbor interpolation)
    for lbl_path in tqdm(unique_label_paths, desc="Processing labels"):
        # Create output path
        path = Path(lbl_path)
        stem = path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]
        output_path = str(path.with_name(f"{stem}{output_suffix}.nii.gz"))
        
        # Get original dimensions from corresponding image (if available)
        # If not available, get from the label itself
        found_image = False
        for img_path in unique_image_paths:
            img_stem = Path(img_path).stem
            if img_stem.endswith('.nii'):
                img_stem = img_stem[:-4]
            
            lbl_stem = stem
            # Check if image stem is in label stem (common naming convention)
            if img_stem in lbl_stem:
                x_dim, y_dim, z_dim = get_image_dimensions(img_path)
                found_image = True
                break
        
        if not found_image:
            x_dim, y_dim, z_dim = get_image_dimensions(lbl_path)
        
        # Calculate new dimensions (4x larger for x and y)
        new_x_dim = x_dim * 4
        new_y_dim = y_dim * 4
        new_z_dim = z_dim  # Keep z dimension the same
        
        # Generate command
        cmd = f"3dresample -dxyz {new_x_dim} {new_y_dim} {new_z_dim} -rmode NN -prefix {output_path} -input {lbl_path}"
        label_commands.append(cmd)
    
    # Combine all commands
    commands = image_commands + label_commands
    
    # Write commands to a shell script
    with open(output_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# AFNI 3dresample commands for images (cubic interpolation)\n")
        for cmd in image_commands:
            f.write(f"{cmd}\n")
        
        f.write("\n# AFNI 3dresample commands for labels (nearest neighbor interpolation)\n")
        for cmd in label_commands:
            f.write(f"{cmd}\n")
    
    # Make the script executable
    os.chmod(output_script, 0o755)
    
    print(f"Generated {len(commands)} commands and saved to {output_script}")
    print(f"Run with: bash {output_script}")
    
    # Create a new JSON file with updated paths
    if output_suffix:
        new_data = data.copy()
        # Update paths in the JSON
        for fold_key, fold_data in new_data.items():
            if fold_key == "metadata":
                continue
            
            for item in fold_data:
                if "image" in item:
                    path = Path(item["image"])
                    stem = path.stem
                    if stem.endswith('.nii'):
                        stem = stem[:-4]
                    item["image"] = str(path.with_name(f"{stem}{output_suffix}.nii.gz"))
                
                if "label" in item:
                    path = Path(item["label"])
                    stem = path.stem
                    if stem.endswith('.nii'):
                        stem = stem[:-4]
                    item["label"] = str(path.with_name(f"{stem}{output_suffix}.nii.gz"))
        
        # Create output JSON path
        json_path_obj = Path(json_path)
        new_json_path = str(json_path_obj.with_stem(f"{json_path_obj.stem}{output_suffix}"))
        
        # Save the updated JSON
        with open(new_json_path, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"Created updated JSON file: {new_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate AFNI 3dresample commands for medical image resampling")
    parser.add_argument("--json_path", type=str, default="balanced_siim_4fold.json",
                        help="Path to the JSON file with image and label paths")
    parser.add_argument("--output_suffix", type=str, default="_128",
                        help="Suffix to add to output files (default: '_128')")
    parser.add_argument("--output_script", type=str, default="run_afni_resample.sh",
                        help="Path to save the shell script with all commands")
    parser.add_argument("--delete_existing", action="store_true", default=True,
                        help="Delete existing files with specified suffixes")
    parser.add_argument("--suffixes_to_delete", nargs="+", default=["_128", "_binary"],
                        help="List of suffixes to identify files for deletion")
    
    args = parser.parse_args()
    
    generate_resample_commands(args.json_path, args.output_suffix, args.output_script, 
                              args.delete_existing, args.suffixes_to_delete)

if __name__ == "__main__":
    main() 