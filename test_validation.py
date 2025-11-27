#!/usr/bin/env python3
"""
Validation tests for the expanded rPPG pipeline
Tests CHROM, ICA algorithms and feature extraction
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/shreyanish/Dev/sop_prelim')

from pipeline import process_signal_pos, process_signal_chrom, process_signal_ica, extract_spo2_features

print("="*60)
print("Test 1: Algorithm Output Validation")
print("="*60)

# Create synthetic RGB signal (30 fps, 10 seconds)
fs = 30.0
t = np.linspace(0, 10, int(fs * 10))
# Simulate 1 Hz pulse (60 BPM)
pulse = np.sin(2 * np.pi * 1.0 * t)
raw_signal = np.column_stack([
    128 + 5 * pulse,  # Red
    128 + 3 * pulse,  # Green
    128 + 2 * pulse   # Blue
])

print(f"\nCreated synthetic signal: {raw_signal.shape[0]} frames at {fs} fps")

# Test each algorithm
print("\nTesting POS algorithm...")
bvp_pos = process_signal_pos(raw_signal, fs)
assert len(bvp_pos) > 0, 'POS failed'
print(f"✓ POS produced {len(bvp_pos)} samples")

print("\nTesting CHROM algorithm...")
bvp_chrom = process_signal_chrom(raw_signal, fs)
assert len(bvp_chrom) > 0, 'CHROM failed'
print(f"✓ CHROM produced {len(bvp_chrom)} samples")

print("\nTesting ICA algorithm...")
bvp_ica = process_signal_ica(raw_signal, fs)
assert len(bvp_ica) > 0, 'ICA failed'
print(f"✓ ICA produced {len(bvp_ica)} samples")

print("\n" + "="*60)
print("Test 2: Feature Extraction Validation")
print("="*60)

# Test feature extraction with POS output
features = extract_spo2_features(raw_signal, bvp_pos, fs)

print("\nExtracted features:")
for key, value in features.items():
    print(f"  {key}: {value:.4f}")

# Validate required features exist
assert 'R_Green_Blue' in features, 'Missing Green/Blue ratio'
assert 'AC_Green' in features, 'Missing AC_Green'
assert 'DC_Green' in features, 'Missing DC_Green'
assert 'R_Ratio' in features, 'Missing R_Ratio'
assert not np.isnan(features['R_Green_Blue']), 'Green/Blue ratio is NaN'
assert not np.isnan(features['R_Ratio']), 'R_Ratio is NaN'

print("\n✓ All required features present and valid")

print("\n" + "="*60)
print("Test 3: Region Configuration Validation")
print("="*60)

from pipeline import ALL_REGIONS, RPPG_METHODS, SELECTED_REGIONS

print(f"\nTotal regions defined: {len(ALL_REGIONS)}")
print(f"rPPG methods configured: {', '.join(RPPG_METHODS)}")
print(f"Selected regions: {SELECTED_REGIONS}")

# Validate region definitions
for region_name, indices in ALL_REGIONS.items():
    assert len(indices) >= 3, f"Region {region_name} has too few landmarks"
    assert all(isinstance(i, int) for i in indices), f"Region {region_name} has non-integer indices"
    assert all(0 <= i < 478 for i in indices), f"Region {region_name} has invalid landmark indices"

print(f"✓ All {len(ALL_REGIONS)} regions have valid landmark definitions")

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
