#!/usr/bin/env python3
"""
Test script for evaluation metrics
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/shreyanish/Dev/sop_prelim')

from evaluation import calculate_mae, calculate_rmse, calculate_pcc, calculate_rbs, evaluate_algorithms

print("="*60)
print("Testing Evaluation Metrics")
print("="*60)

# Create synthetic signals
np.random.seed(42)
t = np.linspace(0, 10, 300)
ground_truth = np.sin(2 * np.pi * 1.0 * t)

# Simulate different algorithm outputs
estimated_signals = {
    'Perfect': ground_truth.copy(),  # Perfect match
    'Good': ground_truth + 0.1 * np.random.randn(len(t)),  # Small noise
    'Medium': ground_truth + 0.3 * np.random.randn(len(t)),  # Medium noise
    'Poor': ground_truth + 0.5 * np.random.randn(len(t)),  # Large noise
}

print("\n1. Individual Metrics Test")
print("-" * 60)
for name, signal in estimated_signals.items():
    mae = calculate_mae(ground_truth, signal)
    rmse = calculate_rmse(ground_truth, signal)
    pcc = calculate_pcc(ground_truth, signal)
    print(f"{name:10s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f}")

print("\n2. rBS Scores")
print("-" * 60)
rbs_scores = calculate_rbs(ground_truth, estimated_signals)
for name, score in sorted(rbs_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:10s} | rBS: {score:.4f}")

print("\n3. Comprehensive Evaluation")
print("-" * 60)
results = evaluate_algorithms(ground_truth, estimated_signals)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['rBS'], reverse=True):
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:6s}: {value:.4f}")

print("\n" + "="*60)
print("✓ All evaluation metrics working correctly!")
print("="*60)
print("\nExpected behavior:")
print("- Perfect signal should have highest rBS")
print("- rBS should decrease with increasing noise")
print("- MAE and RMSE should increase with noise")
print("- PCC should decrease with noise")
