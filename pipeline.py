import os
import pandas as pd
import numpy as np

from config import VIDEO_FOLDER, GROUND_TRUTH_FILE, OUTPUT_FILE, RPPG_METHODS, SELECTED_REGIONS
from roi_definitions import ALL_REGIONS
from signal_extraction import get_raw_rgb_signal
from rppg_algorithms import process_signal_pos, process_signal_chrom, process_signal_ica
from feature_extraction import extract_spo2_features

def run_pipeline(video_folder, gt_file):
    print("=== Starting Multi-Region Multi-Algorithm rPPG SpO2 Pipeline ===")
    
    # 1. Data Curation (Load and map ground truth)
    try:
        gt_df = pd.read_csv(gt_file)
        gt_map = gt_df.set_index("Video File Name")["Oxygen Level"].to_dict()
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file}.")
        return
    except KeyError:
        print("Error: Ground truth file must contain 'Video File Name' and 'Oxygen Level' columns.")
        return

    # Determine which regions to test
    if SELECTED_REGIONS == 'ALL':
        regions_to_test = ALL_REGIONS
    else:
        regions_to_test = {k: ALL_REGIONS[k] for k in SELECTED_REGIONS if k in ALL_REGIONS}
    
    print(f"\nConfiguration:")
    print(f"  - Testing {len(regions_to_test)} regions")
    print(f"  - Testing {len(RPPG_METHODS)} rPPG methods: {', '.join(RPPG_METHODS)}")
    print(f"  - Total combinations per video: {len(regions_to_test) * len(RPPG_METHODS)}\n")

    final_dataset = []
    
    # Iterate through all video files in the specified folder
    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_folder, video_file)

            if video_id in gt_map:
                print(f"\n{'='*60}")
                print(f"Processing video: {video_file}")
                print(f"{'='*60}")
                
                # Iterate through all regions
                for region_name, region_indices in regions_to_test.items():
                    print(f"\n  Region: {region_name}")
                    
                    # Extract raw RGB signal for this region
                    raw_signal, fps = get_raw_rgb_signal(video_path, region_indices)
                    
                    if raw_signal is None or raw_signal.shape[0] < 100:
                        print(f"    ⚠ Skipping region {region_name}: Failed to extract signal or too few frames")
                        continue
                    
                    # Test each rPPG method on this signal
                    for method in RPPG_METHODS:
                        # Apply the appropriate rPPG algorithm
                        if method == 'POS':
                            filtered_bvp = process_signal_pos(raw_signal, fps)
                        elif method == 'CHROM':
                            filtered_bvp = process_signal_chrom(raw_signal, fps)
                        elif method == 'ICA':
                            filtered_bvp = process_signal_ica(raw_signal, fps)
                        else:
                            print(f"    ⚠ Unknown method: {method}")
                            continue
                        
                        # Extract SpO2 features
                        features = extract_spo2_features(raw_signal, filtered_bvp, fps)
                        
                        if not features or (features.get('R_Ratio') is None and features.get('R_Green_Blue') is None):
                            print(f"    ⚠ {method}: Failed to extract features")
                            continue
                        
                        # Append the results to the final dataset
                        row = {
                            'Video_ID': video_id,
                            'Region_Name': region_name,
                            'rPPG_Method': method,
                            'SpO2_Ground_Truth': gt_map[video_id],
                            **features  # Unpack the calculated features
                        }
                        final_dataset.append(row)
                        
                        # Print summary
                        r_ratio = features.get('R_Ratio', np.nan)
                        r_gb = features.get('R_Green_Blue', np.nan)
                        print(f"    ✓ {method}: R_Ratio={r_ratio:.3f}, R_Green/Blue={r_gb:.3f}")
                
                print(f"\n  → Completed {video_id}: {len([r for r in final_dataset if r['Video_ID'] == video_id])} region-method combinations")
            else:
                print(f"\n⚠ Skipping {video_file}: No matching SpO2 label found in ground truth.")

    # Save the final ML-ready dataset
    if final_dataset:
        output_df = pd.DataFrame(final_dataset)
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'='*60}")
        print("=== Pipeline Complete ===")
        print(f"{'='*60}")
        print(f"Successfully processed {len(output_df['Video_ID'].unique())} videos")
        print(f"Total rows in dataset: {len(output_df)}")
        print(f"Regions tested: {len(output_df['Region_Name'].unique())}")
        print(f"Methods tested: {', '.join(output_df['rPPG_Method'].unique())}")
        print(f"Output saved to: {OUTPUT_FILE}")
    else:
        print("\n⚠ No data was processed. Please check your video files and ground truth.")
    
    
if __name__ == "__main__":
    if os.path.exists(VIDEO_FOLDER) and os.path.exists(GROUND_TRUTH_FILE):
        run_pipeline(VIDEO_FOLDER, GROUND_TRUTH_FILE)
    else:
        print("Please ensure 'data' folder and 'ground_truth.csv' exist.")