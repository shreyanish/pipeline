#!/usr/bin/env python3
"""
Quick test of the pipeline with a subset of regions
This will test 3 regions × 3 methods = 9 combinations per video
"""

import sys
sys.path.insert(0, '/Users/shreyanish/Dev/sop_prelim')

# Temporarily override SELECTED_REGIONS for quick test
import pipeline
pipeline.SELECTED_REGIONS = ['forehead', 'left_cheek', 'top5_disjoint']
pipeline.OUTPUT_FILE = 'spo2_dataset_test.csv'

# Run the pipeline
if __name__ == "__main__":
    pipeline.run_pipeline(pipeline.VIDEO_FOLDER, pipeline.GROUND_TRUTH_FILE)
