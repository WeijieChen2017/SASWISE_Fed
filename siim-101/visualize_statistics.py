#!/usr/bin/env python3

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_shape_distribution(stats_data, output_dir):
    """Plot the distribution of image shapes."""
    shapes_dist = stats_data["summary"]["shapes_distribution"]
    shapes = list(shapes_dist.keys())
    counts = list(shapes_dist.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(shapes, counts, color='skyblue')
    plt.xlabel('Shape')
    plt.ylabel('Count')
    plt.title('Distribution of Image Shapes')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shape_distribution.png'), dpi=300)
    plt.close()

def plot_value_distributions(df, output_dir):
    """Plot distributions of min, max, mean values for images and labels."""
    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot image value distributions
    sns.histplot(df['image_min'], bins=30, kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Image Minimum Value Distribution')
    
    sns.histplot(df['image_max'], bins=30, kde=True, ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('Image Maximum Value Distribution')
    
    sns.histplot(df['image_mean'], bins=30, kde=True, ax=axes[0, 2], color='blue')
    axes[0, 2].set_title('Image Mean Value Distribution')
    
    # Plot label value distributions
    sns.histplot(df['label_min'], bins=30, kde=True, ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Label Minimum Value Distribution')
    
    sns.histplot(df['label_max'], bins=30, kde=True, ax=axes[1, 1], color='green')
    axes[1, 1].set_title('Label Maximum Value Distribution')
    
    sns.histplot(df['label_mean'], bins=30, kde=True, ax=axes[1, 2], color='green')
    axes[1, 2].set_title('Label Mean Value Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_distributions.png'), dpi=300)
    plt.close()

def plot_intensity_correlations(df, output_dir):
    """Plot correlations between image and label statistics."""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix for numeric columns
    numeric_cols = ['image_min', 'image_max', 'image_mean', 'image_std', 
                   'label_min', 'label_max', 'label_mean', 'label_std']
    corr = df[numeric_cols].corr()
    
    # Plot correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Image and Label Statistics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlations.png'), dpi=300)
    plt.close()

def plot_dimension_scatterplots(df, output_dir):
    """Create scatterplots comparing image and label dimensions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # X dimension
    axes[0].scatter(df['image_shape_x'], df['label_shape_x'], alpha=0.6)
    axes[0].set_xlabel('Image X Dimension')
    axes[0].set_ylabel('Label X Dimension')
    axes[0].set_title('X Dimension: Image vs Label')
    axes[0].plot([df['image_shape_x'].min(), df['image_shape_x'].max()], 
                [df['image_shape_x'].min(), df['image_shape_x'].max()], 
                'r--', alpha=0.5)
    
    # Y dimension
    axes[1].scatter(df['image_shape_y'], df['label_shape_y'], alpha=0.6)
    axes[1].set_xlabel('Image Y Dimension')
    axes[1].set_ylabel('Label Y Dimension')
    axes[1].set_title('Y Dimension: Image vs Label')
    axes[1].plot([df['image_shape_y'].min(), df['image_shape_y'].max()], 
                [df['image_shape_y'].min(), df['image_shape_y'].max()], 
                'r--', alpha=0.5)
    
    # Z dimension
    axes[2].scatter(df['image_shape_z'], df['label_shape_z'], alpha=0.6)
    axes[2].set_xlabel('Image Z Dimension')
    axes[2].set_ylabel('Label Z Dimension')
    axes[2].set_title('Z Dimension: Image vs Label')
    axes[2].plot([df['image_shape_z'].min(), df['image_shape_z'].max()], 
                [df['image_shape_z'].min(), df['image_shape_z'].max()], 
                'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_scatterplots.png'), dpi=300)
    plt.close()

def generate_html_report(stats_data, output_dir, prefix):
    """Generate an HTML report summarizing the dataset statistics."""
    summary = stats_data["summary"]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SIIM Dataset Statistics Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .warning {{ color: #e74c3c; }}
            .good {{ color: #27ae60; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>SIIM Dataset Statistics Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr><th>Total Pairs</th><td>{summary["total_pairs"]}</td></tr>
                <tr><th>Valid Pairs</th><td>{summary["valid_pairs"]}</td></tr>
                <tr><th>Invalid Pairs</th><td>{summary["invalid_pairs"]}</td></tr>
                <tr><th>Shape Mismatches</th><td>
                    {summary["shape_mismatches"]} 
                    {f'<span class="warning">⚠️ Some images and labels have different shapes!</span>' if summary["shape_mismatches"] > 0 else '<span class="good">✓ All images and labels have matching shapes</span>'}
                </td></tr>
            </table>
            
            <h3>Image Shapes Distribution</h3>
            <table>
                <tr><th>Shape</th><th>Count</th></tr>
                {"".join(f"<tr><td>{shape}</td><td>{count}</td></tr>" for shape, count in summary["shapes_distribution"].items())}
            </table>
        </div>
        
        <h2>Visualizations</h2>
        
        <h3>Shape Distribution</h3>
        <img src="shape_distribution.png" alt="Shape Distribution">
        
        <h3>Value Distributions</h3>
        <img src="value_distributions.png" alt="Value Distributions">
        
        <h3>Correlations</h3>
        <img src="correlations.png" alt="Correlations">
        
        <h3>Dimension Comparisons</h3>
        <img src="dimension_scatterplots.png" alt="Dimension Scatterplots">
        
        <h2>Data Issues</h2>
        <p>
    """
    
    # Add information about problematic files
    invalid_count = 0
    mismatch_count = 0
    
    for pair in stats_data["pairs"]:
        issues = []
        
        # Check for invalid files
        if not pair["image_valid"]:
            issues.append(f"Invalid image: {pair.get('image_error', 'Unknown error')}")
            invalid_count += 1
        if not pair["label_valid"]:
            issues.append(f"Invalid label: {pair.get('label_error', 'Unknown error')}")
            invalid_count += 1
        
        # Check for shape mismatches
        if pair["image_valid"] and pair["label_valid"]:
            if pair["image_shape"] != pair["label_shape"]:
                issues.append(f"Shape mismatch: Image {pair['image_shape']} vs Label {pair['label_shape']}")
                mismatch_count += 1
        
        if issues:
            html_content += f"""
            <div class="issue">
                <strong>ID {pair['id']}:</strong>
                <ul>
                    <li>Image: {pair['image_path']}</li>
                    <li>Label: {pair['label_path']}</li>
                    <li class="warning">{"</li><li class='warning'>".join(issues)}</li>
                </ul>
            </div>
            """
    
    if invalid_count == 0 and mismatch_count == 0:
        html_content += '<p class="good">✓ No data issues detected.</p>'
    
    html_content += """
        </p>
        
        <footer>
            <p>Generated on """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(os.path.join(output_dir, f"{prefix}_report.html"), "w") as f:
        f.write(html_content)

def main(args):
    # Load statistics data
    json_path = os.path.join(args.stats_dir, args.stats_prefix + "_stats.json")
    csv_path = os.path.join(args.stats_dir, args.stats_prefix + "_stats.csv")
    
    print(f"Loading statistics from {json_path} and {csv_path}...")
    
    with open(json_path, 'r') as f:
        stats_data = json.load(f)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded data for {len(df)} valid pairs")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    plot_shape_distribution(stats_data, args.output_dir)
    plot_value_distributions(df, args.output_dir)
    plot_intensity_correlations(df, args.output_dir)
    plot_dimension_scatterplots(df, args.output_dir)
    
    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(stats_data, args.output_dir, args.output_prefix)
    
    print(f"Visualization complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SIIM dataset statistics")
    parser.add_argument("--stats_dir", type=str, default=".", help="Directory containing statistics files")
    parser.add_argument("--stats_prefix", type=str, default="dataset", help="Prefix for statistics files")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--output_prefix", type=str, default="dataset", help="Prefix for output files")
    
    args = parser.parse_args()
    
    main(args) 