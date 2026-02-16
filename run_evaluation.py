#!/usr/bin/env python3
"""
Script to calculate rPPG evaluation metrics and save results to CSV.
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import evaluate_algorithms, save_results_to_csv

def generate_synthetic_data(duration_sec=30, fps=30):
    """Generate synthetic ground truth and algorithm signals for testing"""
    t = np.linspace(0, duration_sec, int(duration_sec * fps))
    # Base heart rate signal (1.1 Hz = 66 bpm)
    ground_truth = np.sin(2 * np.pi * 1.1 * t)
    
    # Simulate various algorithms with different noise levels and phase shifts
    estimated_signals = {
        'POS': ground_truth + 0.1 * np.random.randn(len(t)),
        'CHROM': ground_truth + 0.2 * np.random.randn(len(t)),
        'ICA': ground_truth * 0.8 + 0.3 * np.random.randn(len(t)),
        'SSR': np.sin(2 * np.pi * 1.1 * t + 0.5) + 0.2 * np.random.randn(len(t)),
        'GREEN': ground_truth + 0.6 * np.random.randn(len(t)),
    }
    
    return ground_truth, estimated_signals, fps

def main():
    print("="*60)
    print("rPPG Evaluation Metrics Calculator")
    print("="*60)
    
    # 1. Generate or load data
    print("\n[1/3] Generating synthetic rPPG data...")
    ground_truth, estimated_signals, fps = generate_synthetic_data()
    
    # 2. Calculate metrics
    print(f"[2/3] Calculating metrics for {len(estimated_signals)} algorithms...")
    results = evaluate_algorithms(ground_truth, estimated_signals, fps=fps)
    
    # 3. Save to CSV
    output_file = 'rppg_evaluation_results.csv'
    print(f"[3/3] Saving results to {output_file}...")
    df = save_results_to_csv(results, output_file)
    
    print("\n" + "-"*60)
    print("Preview of Results:")
    print("-"*60)
    # Display a subset of columns for readability in terminal
    display_cols = ['Algorithm', 'MSE', 'RMSE', 'MAE', 'PCC', 'SNR', 'ZCR', 'rBS']
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "="*60)
    print(f"✓ Success! Full results saved to: {os.path.abspath(output_file)}")
    print("="*60)

if __name__ == "__main__":
    main()
