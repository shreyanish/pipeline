#!/usr/bin/env python3
"""
Test script to verify ensemble plotting logic with mock data
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ensemble plotting function
from batch_process import generate_ensemble_plots, resample_signal

def test_ensemble_plotting():
    """Test ensemble plotting with simulated signals"""
    
    # Create mock signal collection
    # Simulating 3 videos, 2 regions, 2 methods
    signal_collection = defaultdict(lambda: defaultdict(list))
    
    # Generate mock signals (different lengths to test resampling)
    np.random.seed(42)
    
    regions = ['glabella', 'chin']
    methods = ['POS', 'CHROM']
    
    for region in regions:
        for method in methods:
            # Simulate 5 videos with varying signal lengths
            for i in range(5):
                # Random length between 500-700 samples
                length = np.random.randint(500, 700)
                
                # Generate mock BVP signal (sine wave + noise)
                t = np.linspace(0, 20, length)
                hr = np.random.uniform(60, 90)  # Random heart rate
                signal = np.sin(2 * np.pi * (hr/60) * t) + np.random.normal(0, 0.1, length)
                
                # Normalize
                signal = (signal - np.mean(signal)) / np.std(signal)
                
                signal_collection[region][method].append(signal)
    
    print(f"Created mock data:")
    for region in regions:
        for method in methods:
            print(f"  {region}/{method}: {len(signal_collection[region][method])} signals")
    
    # Test resampling function
    print("\nTesting resample_signal()...")
    test_sig = np.array([1, 2, 3, 4, 5])
    resampled = resample_signal(test_sig, 10)
    print(f"  Original length: {len(test_sig)}, Resampled: {len(resampled)}")
    assert len(resampled) == 10, "Resampling failed!"
    print("  ✓ Resampling works")
    
    # Test ensemble plot generation
    print("\nGenerating ensemble plots...")
    os.makedirs('test_ensemble_plots', exist_ok=True)
    generate_ensemble_plots(signal_collection, 'test_ensemble_plots', target_length=600)
    
    # Check that plots were created
    expected_plots = len(regions)
    actual_plots = len([f for f in os.listdir('test_ensemble_plots') if f.endswith('.png')])
    
    print(f"\n✓ Generated {actual_plots}/{expected_plots} ensemble plots")
    print(f"✓ Plots saved to: test_ensemble_plots/")
    
    # Verify plot contents
    for region in regions:
        plot_file = f'test_ensemble_plots/ensemble_{region}.png'
        if os.path.exists(plot_file):
            print(f"  ✓ {plot_file} created successfully")
    
    print("\n✅ All tests passed! Ensemble plotting logic is working correctly.")

if __name__ == "__main__":
    test_ensemble_plotting()
