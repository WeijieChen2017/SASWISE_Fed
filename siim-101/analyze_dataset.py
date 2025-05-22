#!/usr/bin/env python3

import os
import argparse
import subprocess
import time

def run_command(command):
    print(f"Running: {' '.join(command)}")
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Stream output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Get any errors
    stderr = process.communicate()[1]
    if stderr:
        print(f"Error: {stderr}")
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    
    return process.returncode

def main(args):
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Step 1: Run the dimension checking script
    print("\n=== Step 1: Checking dataset dimensions and statistics ===\n")
    cmd_check = [
        "python3", "check_dimensions.py",
        "--fold_file", args.fold_file,
        "--output_dir", args.output_dir,
        "--output_prefix", args.prefix
    ]
    
    result = run_command(cmd_check)
    if result != 0:
        print("Error: Dimension checking failed. Exiting.")
        return result
    
    # Step 2: Run the visualization script
    print("\n=== Step 2: Creating data visualizations ===\n")
    cmd_vis = [
        "python3", "visualize_statistics.py",
        "--stats_dir", args.output_dir,
        "--stats_prefix", args.prefix,
        "--output_dir", vis_dir,
        "--output_prefix", args.prefix
    ]
    
    result = run_command(cmd_vis)
    if result != 0:
        print("Error: Visualization failed. Exiting.")
        return result
    
    print("\n=== Analysis Complete ===")
    print(f"Output files are in {args.output_dir}")
    print(f"Visualizations are in {vis_dir}")
    print(f"HTML Report: {os.path.join(vis_dir, args.prefix + '_report.html')}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SIIM dataset: check dimensions and visualize statistics")
    parser.add_argument("--fold_file", type=str, default="balanced_siim_4fold.json", help="Path to the fold file")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save analysis results")
    parser.add_argument("--prefix", type=str, default="dataset", help="Prefix for output files")
    
    args = parser.parse_args()
    
    exit(main(args)) 