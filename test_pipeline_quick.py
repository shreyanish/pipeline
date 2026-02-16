#!/usr/bin/env python3
"""
Quick test of the pipeline with a subset of regions
This will test 3 regions × 4 methods = 12 combinations per video
"""

import sys
# Removed hardcoded path for portability

# Temporarily override SELECTED_REGIONS for quick test
import pipeline
pipeline.SELECTED_REGIONS = ['glabella', 'chin', 'philtrum']

# Run the pipeline
if __name__ == "__main__":
    pipeline.run_pipeline(pipeline.VIDEO_FOLDER, max_frames=600)
