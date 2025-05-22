#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def run_command(cmd):
    """Run a command and return the output"""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error: {stderr.decode('utf-8')}")
        return False, stderr.decode('utf-8')
    
    return True, stdout.decode('utf-8')

def train_fold(args, fold):
    """Train model on specified fold"""
    model_dir = os.path.join(args.output_dir, f"fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)
    
    cmd = [
        "python3", "train_segmentation.py",
        "--fold", str(fold),
        "--fold_file", args.fold_file,
        "--output_dir", model_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--num_epochs", str(args.num_epochs),
        "--learning_rate", str(args.learning_rate),
        "--min_z_dim", str(args.min_z_dim),
        "--seed", str(args.seed)
    ]
    
    success, output = run_command(cmd)
    return success, model_dir

def run_inference(args, fold, model_path):
    """Run inference on test fold using trained model"""
    pred_dir = os.path.join(args.output_dir, f"fold_{fold}", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    cmd = [
        "python3", "inference.py",
        "--model_path", model_path,
        "--fold", str(fold),
        "--fold_file", args.fold_file,
        "--output_dir", pred_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--min_z_dim", str(args.min_z_dim)
    ]
    
    success, output = run_command(cmd)
    return success, pred_dir

def evaluate_results(results_dir):
    """Evaluate model performance across all folds"""
    # Placeholder for evaluation function 
    # This would calculate performance metrics across all folds
    # For example: calculating mean Dice scores, etc.
    pass

def main(args):
    start_time = time.time()
    
    # Track results for each fold
    fold_results = {}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each fold
    for fold in range(4):
        if fold in args.skip_folds:
            print(f"Skipping fold {fold} as requested")
            continue
            
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLD {fold}")
        print(f"{'='*80}\n")
        
        fold_start = time.time()
        
        if not args.inference_only:
            print(f"Training model on fold {fold}...")
            success, model_dir = train_fold(args, fold)
            
            if not success:
                print(f"Error training fold {fold}, skipping to next fold")
                continue
                
            # Use the best model for inference
            model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
        else:
            # When inference only, use the specified model or look for the best model
            if args.model_path:
                model_path = args.model_path
            else:
                model_dir = os.path.join(args.output_dir, f"fold_{fold}")
                model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")
                
                if not os.path.exists(model_path):
                    print(f"No model found at {model_path}, skipping fold {fold}")
                    continue
                    
        print(f"Running inference on fold {fold} using model: {model_path}")
        success, pred_dir = run_inference(args, fold, model_path)
        
        if not success:
            print(f"Error running inference on fold {fold}")
            continue
            
        fold_time = time.time() - fold_start
        print(f"Completed fold {fold} in {fold_time:.2f} seconds")
        
        # Store results for this fold
        fold_results[fold] = {
            "model_path": model_path,
            "pred_dir": pred_dir,
            "time": fold_time
        }
    
    # Evaluate overall results
    if args.evaluate and fold_results:
        print("\nEvaluating results across all processed folds...")
        evaluate_results(fold_results)
    
    total_time = time.time() - start_time
    print(f"\nCompleted processing in {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIIM Segmentation Pipeline Runner")
    parser.add_argument("--fold_file", type=str, default="balanced_siim_4fold.json", help="Path to the fold file")
    parser.add_argument("--output_dir", type=str, default="segmentation_results", help="Directory to save all outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training/inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_z_dim", type=int, default=32, help="Minimum required z dimension for samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip_folds", type=int, nargs="+", default=[], help="Fold numbers to skip")
    parser.add_argument("--inference_only", action="store_true", help="Only run inference without training")
    parser.add_argument("--model_path", type=str, help="Path to model for inference (used with --inference_only)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate results after processing")
    
    args = parser.parse_args()
    
    main(args) 