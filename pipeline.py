import os
import matplotlib.pyplot as plt
import numpy as np

from config import VIDEO_FOLDER, RPPG_METHODS, SELECTED_REGIONS
from roi_definitions import ALL_REGIONS
from signal_extraction import get_raw_rgb_signal
from rppg_algorithms import (process_signal_pos, process_signal_chrom, process_signal_ica, process_signal_ssr,
                             process_signal_green, process_signal_pca, process_signal_pbv, process_signal_lgi,
                             process_signal_omit, process_signal_samc, process_signal_2sr)
from evaluation import evaluate_algorithms

def run_pipeline(video_folder, max_frames=None, ground_truth_signal=None):
    print("=== Starting Multi-Region Multi-Algorithm rPPG Pipeline ===")
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Determine which regions to test
    if SELECTED_REGIONS == 'ALL':
        regions_to_test = ALL_REGIONS
    else:
        regions_to_test = {k: ALL_REGIONS[k] for k in SELECTED_REGIONS if k in ALL_REGIONS}
    
    print(f"\nConfiguration:")
    print(f"  - Testing {len(regions_to_test)} regions")
    print(f"  - Testing {len(RPPG_METHODS)} rPPG methods: {', '.join(RPPG_METHODS)}")
    print(f"  - Total combinations per video: {len(regions_to_test) * len(RPPG_METHODS)}\n")

    # Iterate through all video files in the specified folder
    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_folder, video_file)

            print(f"\n{'='*60}")
            print(f"Processing video: {video_file}")
            print(f"{'='*60}")
            
            # Iterate through all regions
            for region_name, region_indices in regions_to_test.items():
                print(f"\n  Region: {region_name}")
                
                extracted_signals = {}  # Store signals for plotting
                
                # Extract raw RGB signal for this region
                raw_signal, fps = get_raw_rgb_signal(video_path, region_indices, max_frames=max_frames)
                
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
                    elif method == 'SSR':
                        filtered_bvp = process_signal_ssr(raw_signal, fps)
                    elif method == 'GREEN':
                        filtered_bvp = process_signal_green(raw_signal, fps)
                    elif method == 'PCA':
                        filtered_bvp = process_signal_pca(raw_signal, fps)
                    elif method == 'PBV':
                        filtered_bvp = process_signal_pbv(raw_signal, fps)
                    elif method == 'LGI':
                        filtered_bvp = process_signal_lgi(raw_signal, fps)
                    elif method == 'OMIT':
                        filtered_bvp = process_signal_omit(raw_signal, fps)
                    elif method == 'SAMC':
                        filtered_bvp = process_signal_samc(raw_signal, fps)
                    elif method == '2SR':
                        filtered_bvp = process_signal_2sr(raw_signal, fps)
                    else:
                        print(f"    ⚠ Unknown method: {method}")
                        continue

                    # Store signal for plotting
                    if len(filtered_bvp) > 0:
                        extracted_signals[method] = filtered_bvp
                        print(f"    ✓ {method}: Signal extracted ({len(filtered_bvp)} samples)")
                    else:
                        print(f"    ⚠ {method}: Failed to extract signal")
            
                # Evaluate algorithms if ground truth is provided
                if ground_truth_signal is not None and len(extracted_signals) > 0:
                    print(f"\n  Evaluation Metrics (rBS):")
                    eval_results = evaluate_algorithms(ground_truth_signal, extracted_signals)
                    # Sort by rBS (higher is better)
                    sorted_results = sorted(eval_results.items(), key=lambda x: x[1]['rBS'], reverse=True)
                    for method, metrics in sorted_results:
                        print(f"    {method:6s} | rBS: {metrics['rBS']:7.3f} | PCC: {metrics['PCC']:6.3f} | MAE: {metrics['MAE']:6.3f} | RMSE: {metrics['RMSE']:6.3f}")
            
                # Plot signals for this region
                if extracted_signals:
                    plt.figure(figsize=(12, 6))
                    for method, signal in extracted_signals.items():
                        # Normalize signal for better comparison in plot
                        if np.std(signal) != 0:
                            norm_signal = (signal - np.mean(signal)) / np.std(signal)
                            plt.plot(norm_signal, label=method, alpha=0.7)
                    
                    plt.title(f"rPPG Signals - {video_id} - {region_name}")
                    plt.xlabel("Frame")
                    plt.ylabel("Normalized Amplitude")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plot_path = os.path.join(plots_dir, f"{video_id}_{region_name}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"    📊 Plot saved: {plot_path}")
            
            print(f"\n  → Completed {video_id}")

    print(f"\n{'='*60}")
    print("=== Pipeline Complete ===")
    print(f"{'='*60}")
    print(f"Plots saved to: {plots_dir}/")
    
    
if __name__ == "__main__":
    if os.path.exists(VIDEO_FOLDER):
        run_pipeline(VIDEO_FOLDER)
    else:
        print(f"Please ensure '{VIDEO_FOLDER}' folder exists.")