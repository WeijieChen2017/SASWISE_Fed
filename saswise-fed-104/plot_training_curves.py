#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob
import numpy as np

def plot_training_curves(dataset_dir):
    """
    Load training history for a dataset and plot training curves
    
    Args:
        dataset_dir: Path to the dataset directory
    
    Returns:
        True if successful, False otherwise
    """
    history_path = os.path.join(dataset_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"No training history found for {dataset_dir}")
        return False
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if not history:
            print(f"No training data found for {dataset_dir}")
            return False
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(history)
        
        # Create a figure with 2 subplots (for accuracy and AUC)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot accuracy
        axes[0].plot(df['epoch'], df['train_acc'], 'b-', label='Training Accuracy')
        axes[0].plot(df['epoch'], df['test_acc'], 'r-', label='Testing Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Training and Testing Accuracy\n{os.path.basename(dataset_dir)}')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot AUC
        axes[1].plot(df['epoch'], df['train_auc'], 'b-', label='Training AUC')
        axes[1].plot(df['epoch'], df['test_auc'], 'r-', label='Testing AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title(f'Training and Testing AUC\n{os.path.basename(dataset_dir)}')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Add loss as a separate plot at the bottom if desired
        # plt.tight_layout()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure in the dataset directory
        output_path = os.path.join(dataset_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves for {dataset_dir} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {dataset_dir}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot training curves for each dataset')
    parser.add_argument('--datasets', '-d', nargs='+', 
                        help='Specific dataset directories to process (if not provided, will scan all directories)')
    args = parser.parse_args()
    
    # Get list of datasets to process
    if args.datasets:
        dataset_dirs = args.datasets
    else:
        # Find all directories that might be dataset directories
        dataset_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and 
                       os.path.exists(os.path.join(d, 'training_history.json'))]
    
    if not dataset_dirs:
        print("No dataset directories found")
        return
    
    print(f"Found {len(dataset_dirs)} dataset directories to process")
    
    # Process each dataset
    successful = 0
    for dataset_dir in dataset_dirs:
        if plot_training_curves(dataset_dir):
            successful += 1
    
    print(f"\nSummary: Successfully processed {successful} out of {len(dataset_dirs)} datasets")

if __name__ == "__main__":
    main() 