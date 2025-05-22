#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import argparse

def extract_epoch_metrics(dataset_dirs=None, epoch_num=100):
    """
    Extract metrics from training_history.json files for all datasets at a specific epoch.
    
    Args:
        dataset_dirs: List of dataset directories to check. If None, scan current directory.
        epoch_num: Which epoch to extract (default: 100)
    
    Returns:
        DataFrame containing metrics for all datasets
    """
    results = []
    
    # If no dataset directories provided, find all subdirectories
    if dataset_dirs is None:
        dataset_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    for dataset_dir in dataset_dirs:
        history_path = os.path.join(dataset_dir, 'training_history.json')
        
        if not os.path.exists(history_path):
            print(f"No training history found for {dataset_dir}")
            continue
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Find the epoch we want (or last epoch if fewer epochs)
            target_epoch = min(epoch_num, len(history))
            if target_epoch == 0:
                print(f"No training data found for {dataset_dir}")
                continue
                
            # Extract the metrics (epoch is 1-indexed in the data, but 0-indexed in the list)
            epoch_data = history[target_epoch - 1]
            
            # Add dataset name and metrics to results
            results.append({
                'dataset': dataset_dir,
                'epoch': epoch_data['epoch'],
                'train_auc': epoch_data['train_auc'],
                'train_acc': epoch_data['train_acc'],
                'test_auc': epoch_data['test_auc'],
                'test_acc': epoch_data['test_acc'],
                'loss': epoch_data['loss']
            })
            
            print(f"Extracted metrics for {dataset_dir} at epoch {target_epoch}")
            
        except Exception as e:
            print(f"Error processing {dataset_dir}: {str(e)}")
    
    # Convert to DataFrame for easy manipulation
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        print("No results found")
        return pd.DataFrame()

def plot_metrics(df, output_prefix="metrics", epoch_num=100):
    """
    Create a bar plot visualization of all metrics for all datasets.
    
    Args:
        df: DataFrame containing the metrics
        output_prefix: Prefix for the output file names
        epoch_num: Epoch number for the title
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Set the figure size based on number of datasets
    fig_width = max(12, len(df) * 1.2)
    fig_height = 10
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set up positions for the bars
    datasets = df['dataset']
    x = np.arange(len(datasets))
    width = 0.2  # Width of the bars
    
    # Create four bars for each dataset
    plt.bar(x - 1.5*width, df['train_auc'], width, label='Train AUC', color='royalblue')
    plt.bar(x - 0.5*width, df['test_auc'], width, label='Test AUC', color='lightblue')
    plt.bar(x + 0.5*width, df['train_acc'], width, label='Train Acc', color='firebrick')
    plt.bar(x + 1.5*width, df['test_acc'], width, label='Test Acc', color='lightcoral')
    
    # Add value labels on top of bars
    for i, metric in enumerate([df['train_auc'], df['test_auc'], df['train_acc'], df['test_acc']]):
        offset = (i - 1.5) * width
        for j, v in enumerate(metric):
            plt.text(j + offset, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Add labels, title and custom x-axis tick labels
    plt.xlabel('Dataset', fontweight='bold')
    plt.ylabel('Metric Value', fontweight='bold')
    plt.title(f'Training and Testing Metrics at Epoch {epoch_num}', fontweight='bold')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4)
    
    # Add a grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and set y-axis limit
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{output_prefix}_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to {plot_filename}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract model training metrics at a specific epoch.')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Epoch number to extract metrics from (default: 100)')
    args = parser.parse_args()
    
    # Extract metrics from specified epoch
    metrics_df = extract_epoch_metrics(epoch_num=args.epoch)
    
    if not metrics_df.empty:
        # Save results to CSV
        output_file = f'metrics_summary_epoch_{args.epoch}.csv'
        metrics_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print summary to console
        print(f"\nMetrics Summary (Epoch {args.epoch}):")
        print("=" * 80)
        print(metrics_df.to_string())
        print("=" * 80)
        
        # Create bar plots of metrics
        plot_metrics(metrics_df, output_prefix=f'metrics_epoch_{args.epoch}', epoch_num=args.epoch)

if __name__ == "__main__":
    main() 