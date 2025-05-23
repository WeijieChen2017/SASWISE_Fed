#!/usr/bin/env python3

import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import subprocess
import nibabel as nib

def find_mor_files(search_dir, pattern="*_mor*.nii.gz"):
    """
    Find all files with '_mor' in their names
    
    Args:
        search_dir: Directory to search for files
        pattern: Glob pattern to match files (default: '*_mor*.nii.gz')
        
    Returns:
        List of file paths matching the pattern
    """
    # Ensure path is absolute
    abs_dir = os.path.abspath(search_dir)
    
    # Create search pattern
    search_pattern = os.path.join(abs_dir, "**", pattern)
    
    # Find all matching files recursively
    mor_files = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(mor_files)} files matching pattern '{pattern}'")
    
    return sorted(mor_files)

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

def delete_existing_file(file_path):
    """
    Delete a file if it exists
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        True if file was deleted, False otherwise
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted existing file: {file_path}")
            return True
        except (OSError, PermissionError) as e:
            print(f"Error deleting {file_path}: {e}")
    return False

def generate_output_path(file_path, output_suffix="_128"):
    """
    Generate output path for a file with the given suffix
    
    Args:
        file_path: Original file path
        output_suffix: Suffix to add to the output filename
        
    Returns:
        Output file path
    """
    path = Path(file_path)
    stem = path.stem
    if stem.endswith('.nii'):
        stem = stem[:-4]
    return str(path.with_name(f"{stem}{output_suffix}.nii.gz"))

def downsample_mor_files(search_dir, output_suffix="_128", output_script="run_downsample_mor.sh", 
                        delete_existing=True, run_commands=False):
    """
    Generate and optionally run AFNI 3dresample commands for all _mor files
    
    Args:
        search_dir: Directory to search for _mor files
        output_suffix: Suffix to add to output files (default: "_128")
        output_script: Path to save the shell script with all commands
        delete_existing: Whether to delete existing output files before generating new ones
        run_commands: Whether to run the commands immediately
    """
    # Find all _mor files
    mor_files = find_mor_files(search_dir)
    
    if not mor_files:
        print("No files found matching the pattern. Exiting.")
        return
    
    # Generate commands
    commands = []
    
    for file_path in tqdm(mor_files, desc="Processing _mor files"):
        # Generate output path
        output_path = generate_output_path(file_path, output_suffix)
        
        # Delete existing output file if requested
        if delete_existing:
            delete_existing_file(output_path)
        
        # Get original dimensions
        x_dim, y_dim, z_dim = get_image_dimensions(file_path)
        
        # Calculate new dimensions (4x larger for x and y)
        new_x_dim = x_dim * 4
        new_y_dim = y_dim * 4
        new_z_dim = z_dim  # Keep z dimension the same
        
        # Generate command for Nearest Neighbor interpolation with dynamic dimensions
        cmd = f"3dresample -dxyz {new_x_dim} {new_y_dim} {new_z_dim} -rmode NN -prefix {output_path} -input {file_path}"
        commands.append(cmd)
    
    # Write commands to a shell script
    with open(output_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# AFNI 3dresample commands for {len(commands)} _mor files using Nearest Neighbor interpolation\n")
        f.write("# Dynamic resolution: 4x larger for x and y dimensions, z dimension preserved\n")
        for cmd in commands:
            f.write(f"{cmd}\n")
    
    # Make the script executable
    os.chmod(output_script, 0o755)
    
    print(f"Generated {len(commands)} commands and saved to {output_script}")
    print(f"Run with: bash {output_script}")
    
    # Run commands immediately if requested
    if run_commands and commands:
        print(f"\nRunning {len(commands)} downsample commands...")
        for cmd in tqdm(commands, desc="Downsampling"):
            try:
                result = subprocess.run(cmd, shell=True, check=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    print(f"Error running command: {cmd}")
                    print(f"Error: {result.stderr.decode('utf-8')}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {cmd}")
                print(f"Error: {str(e)}")
        
        print("All commands completed")

def main():
    parser = argparse.ArgumentParser(description="Downsample _mor label files using Nearest Neighbor interpolation")
    parser.add_argument("--search_dir", type=str, default="SIIM_Fed_Learning_Phase1Data",
                        help="Directory to search for _mor files")
    parser.add_argument("--output_suffix", type=str, default="_128",
                        help="Suffix to add to output files (default: '_128')")
    parser.add_argument("--output_script", type=str, default="run_downsample_mor.sh",
                        help="Path to save the shell script with all commands")
    parser.add_argument("--delete_existing", action="store_true", default=True,
                        help="Delete existing output files before generating new ones")
    parser.add_argument("--run", action="store_true", default=False,
                        help="Run the commands immediately")
    
    args = parser.parse_args()
    
    downsample_mor_files(args.search_dir, args.output_suffix, args.output_script, 
                        args.delete_existing, args.run)

if __name__ == "__main__":
    main() 