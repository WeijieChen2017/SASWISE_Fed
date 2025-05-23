#!/usr/bin/env python3

import json
import os
from pathlib import Path

def create_mor_json(input_json_path, output_json_path, mor_suffix="_mor"):
    """
    Create a new JSON file with label paths replaced to use _mor files
    
    Args:
        input_json_path: Path to the input JSON file
        output_json_path: Path to save the output JSON file
        mor_suffix: Suffix to add before the file extension for _mor files
    """
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a copy for modification
    new_data = data.copy()
    
    # Process each fold
    for fold_key, fold_data in new_data.items():
        if fold_key == "metadata":
            continue
        
        for item in fold_data:
            if "label" in item:
                # Get the original label path
                label_path = Path(item["label"])
                
                # Extract components
                directory = label_path.parent
                filename = label_path.name
                stem = label_path.stem
                
                # Handle .nii.gz files (double extension)
                if stem.endswith('.nii'):
                    base_stem = stem[:-4]  # Remove .nii part
                    new_filename = f"{base_stem}{mor_suffix}.nii.gz"
                else:
                    # For other files, just add suffix before extension
                    new_filename = f"{stem}{mor_suffix}{label_path.suffix}"
                
                # Create new path
                new_label_path = directory / new_filename
                item["label"] = str(new_label_path)
    
    # Save the new JSON
    with open(output_json_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Created {output_json_path} with _mor labels")
    
    # Count total samples processed
    total_samples = 0
    for fold_key, fold_data in new_data.items():
        if fold_key != "metadata":
            total_samples += len(fold_data)
    
    print(f"Processed {total_samples} samples across all folds")

def main():
    # Define input and output files
    json_files = [
        {
            "input": "balanced_siim_4fold.json",
            "output": "balanced_siim_4fold_mor.json"
        },
        {
            "input": "balanced_siim_4fold_128.json", 
            "output": "balanced_siim_4fold_128_mor.json"
        }
    ]
    
    # Process each JSON file
    for json_file in json_files:
        input_path = json_file["input"]
        output_path = json_file["output"]
        
        if os.path.exists(input_path):
            print(f"\nProcessing {input_path}...")
            create_mor_json(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found, skipping...")
    
    print("\nAll JSON files processed successfully!")

if __name__ == "__main__":
    main() 