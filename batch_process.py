#!/usr/bin/env python3
"""
Batch processing script for server deployment
Processes videos with rPPG algorithms and generates ensemble average plots

Usage:
    python3 batch_process.py --video-dir /path/to/videos --max-frames 1800
    python3 batch_process.py --video-dir /path/to/videos --max-frames 1800 --regions all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
from multiprocessing import Pool, Manager

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RPPG_METHODS, SELECTED_REGIONS
from roi_definitions import ALL_REGIONS
from signal_extraction import get_raw_rgb_signal, get_all_regions_signals
from rppg_algorithms import (process_signal_pos, process_signal_chrom, process_signal_ica, process_signal_ssr,
                             process_signal_green, process_signal_pca, process_signal_pbv, process_signal_lgi,
                             process_signal_omit, process_signal_samc, process_signal_2sr)
from evaluation import evaluate_blind_metrics, save_results_to_csv
import matplotlib.pyplot as plt
import numpy as np

# Try importing torch
try:
    import torch
    import rppg_pytorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def resample_signal(signal, target_length):
    """Resample signal to target length using linear interpolation"""
    if len(signal) == target_length:
        return signal
    
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    interpolator = interp1d(x_old, signal, kind='linear')
    return interpolator(x_new)


def process_single_video(args_tuple):
    """Wrapper for multiprocessing - unpacks arguments"""
    video_path, regions_to_test, max_frames, signal_collection, metrics_collection, use_gpu = args_tuple
    return _process_single_video(video_path, regions_to_test, max_frames, signal_collection, metrics_collection, use_gpu)


def _process_single_video(video_path, regions_to_test, max_frames, signal_collection, metrics_collection, use_gpu=False):
    """
    Process a single video and collect signals for ensemble averaging
    
    Args:
        video_path: Path to video file
        regions_to_test: Dict of region names to landmark indices
        max_frames: Maximum frames to process
        signal_collection: Dict to accumulate signals
    
    Returns:
        (success: bool, num_signals: int, error_msg: str)
    """
    video_file = os.path.basename(video_path)
    video_path_obj = Path(video_path)
    video_id = f"{video_path_obj.parent.name}_{video_path_obj.stem}"
    
    try:
        signals_extracted = 0
        
        # Optimize: Single pass extraction for ALL regions
        # This prevents reading the video 31 times
        all_signals, fps = get_all_regions_signals(
            video_path, 
            regions_to_test, 
            max_frames=max_frames
        )
        
        # Determine device for GPU processing
        device = 'cpu'
        if use_gpu and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
        
        for region_name, raw_signal in all_signals.items():
            if raw_signal is None or raw_signal.shape[0] < 100:
                continue
            
            # Apply all rPPG algorithms
            for method in RPPG_METHODS:
                filtered_bvp = []
                
                if use_gpu and TORCH_AVAILABLE and device != 'cpu':
                    # GPU Processing
                    try:
                        if method == 'POS':
                            bvp_tensor = rppg_pytorch.process_signal_pos_torch(raw_signal, fps, device)
                        elif method == 'CHROM':
                            bvp_tensor = rppg_pytorch.process_signal_chrom_torch(raw_signal, fps, device)
                        elif method == 'ICA':
                            bvp_tensor = rppg_pytorch.process_signal_ica_torch(raw_signal, fps, device)
                        elif method == 'SSR':
                            bvp_tensor = rppg_pytorch.process_signal_ssr_torch(raw_signal, fps, device)
                        elif method == 'GREEN':
                            bvp_tensor = rppg_pytorch.process_signal_green_torch(raw_signal, fps, device)
                        elif method == 'PCA':
                            bvp_tensor = rppg_pytorch.process_signal_pca_torch(raw_signal, fps, device)
                        elif method == 'PBV':
                            bvp_tensor = rppg_pytorch.process_signal_pbv_torch(raw_signal, fps, device)
                        elif method == 'LGI':
                            bvp_tensor = rppg_pytorch.process_signal_lgi_torch(raw_signal, fps, device)
                        elif method == 'OMIT':
                            bvp_tensor = rppg_pytorch.process_signal_omit_torch(raw_signal, fps, device)
                        elif method == 'SAMC':
                            bvp_tensor = rppg_pytorch.process_signal_samc_torch(raw_signal, fps, device)
                        elif method == '2SR':
                            bvp_tensor = rppg_pytorch.process_signal_2sr_torch(raw_signal, fps, device)
                        else:
                            continue
                            
                        if bvp_tensor is not None and len(bvp_tensor) > 0:
                            filtered_bvp = bvp_tensor.cpu().numpy()
                    except Exception as e:
                        logger.warning(f"GPU processing failed for {method}, falling back to CPU: {e}")
                        # Fallback to CPU if GPU fails
                        pass

                # CPU Processing (fallback or default)
                if len(filtered_bvp) == 0:
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
                        continue
                
                if len(filtered_bvp) > 0:
                    # Calculate blind metrics (reference-free)
                    try:
                        metrics = evaluate_blind_metrics(filtered_bvp, fps)
                        metrics_collection[region_name][method].append(metrics)
                    except Exception as e:
                        logger.warning(f"Failed to calculate metrics for {method} in {region_name}: {e}")

                    # Normalize signal before storing
                    if np.std(filtered_bvp) != 0:
                        norm_signal = (filtered_bvp - np.mean(filtered_bvp)) / np.std(filtered_bvp)
                        signal_collection[region_name][method].append(norm_signal)
                        signals_extracted += 1
        
        return True, signals_extracted, None
        
    except Exception as e:
        error_msg = f"Error processing {video_id}: {str(e)}"
        logger.error(error_msg)
        return False, 0, error_msg


def generate_ensemble_plots(signal_collection, plots_dir, target_length=1800):
    """
    Generate a single global ensemble average plot across all regions
    
    Args:
        signal_collection: Dict[region][method] = List[signals]
        plots_dir: Output directory
        target_length: Common length to resample all signals to
    """
    logger.info("Generating Global Ensemble Average plot...")
    
    plt.figure(figsize=(16, 9))
    
    # Get all unique methods across all regions
    methods = set()
    for region_data in signal_collection.values():
        methods.update(region_data.keys())
    
    methods = sorted(list(methods))
    
    for method in methods:
        all_method_signals = []
        
        # Collect signals for this method from ALL regions
        for region_name, methods_data in signal_collection.items():
            if method in methods_data:
                signals = methods_data[method]
                for sig in signals:
                    try:
                        resampled_sig = resample_signal(sig, target_length)
                        all_method_signals.append(resampled_sig)
                    except:
                        continue
        
        if len(all_method_signals) == 0:
            continue
        
        # Convert to array
        signal_array = np.array(all_method_signals)
        
        # Compute global ensemble statistics
        mean_signal = np.mean(signal_array, axis=0)
        std_signal = np.std(signal_array, axis=0)
        
        # Time axis (assuming 30 fps)
        time_axis = np.arange(target_length) / 30.0
        
        # Plot mean with transparency based on number of samples
        plt.plot(time_axis, mean_signal, label=f'{method} (total n={len(all_method_signals)})', linewidth=2.5)
        plt.fill_between(time_axis, 
                       mean_signal - (0.5 * std_signal), # Slightly tighter band for readability
                       mean_signal + (0.5 * std_signal), 
                       alpha=0.1)
    
    plt.title("Global Ensemble Average rPPG Signals (All Regions Combined)", fontsize=16, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.legend(loc='upper right', fontsize=11, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "global_ensemble_summary.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    logger.info(f"  ✓ Saved global summary plot to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch process videos with rPPG algorithms')
    parser.add_argument('--video-dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--max-frames', type=int, default=1800, help='Max frames to process (default: 1800 = 60 sec at 30fps)')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum number of videos to process (default: all)')
    parser.add_argument('--regions', type=str, default='all', help='Regions to process: "all" or comma-separated list')
    parser.add_argument('--plots-dir', type=str, default='ensemble_plots', help='Output directory for plots')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1, recommended: 8-16)')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration if available')
    
    args = parser.parse_args()
    
    # Setup
    video_dir = Path(args.video_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    # Determine regions
    if args.regions.lower() == 'all':
        regions_to_test = ALL_REGIONS
    else:
        region_names = [r.strip() for r in args.regions.split(',')]
        regions_to_test = {k: ALL_REGIONS[k] for k in region_names if k in ALL_REGIONS}
    
    # Get video files recursively
    video_extensions = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(video_dir.rglob(f"*{ext}")))
    
    video_files.sort()
    
    # Limit number of videos if specified
    if args.max_videos is not None:
        video_files = video_files[:args.max_videos]
    
    logger.info(f"Found {len(video_files)} videos in {video_dir}")
    if args.max_videos:
        logger.info(f"Limited to first {args.max_videos} videos")
    logger.info(f"Processing {len(regions_to_test)} regions with {len(RPPG_METHODS)} algorithms")
    logger.info(f"Max frames per video: {args.max_frames}")
    
    # Initialize signal collection structure (shared for multiprocessing)
    # signal_collection[region_name][method_name] = [list of signals]
    if args.workers > 1:
        manager = Manager()
        signal_collection = manager.dict()
        metrics_collection = manager.dict()
        # Pre-initialize structure for thread-safe access
        for region_name in regions_to_test.keys():
            signal_collection[region_name] = manager.dict()
            metrics_collection[region_name] = manager.dict()
            for method in RPPG_METHODS:
                signal_collection[region_name][method] = manager.list()
                metrics_collection[region_name][method] = manager.list()
    else:
        signal_collection = defaultdict(lambda: defaultdict(list))
        metrics_collection = defaultdict(lambda: defaultdict(list))
    
    # Process videos
    stats = {'success': 0, 'failed': 0, 'total_signals': 0}
    
    if args.workers > 1:
        # Parallel processing
        logger.info(f"Using {args.workers} parallel workers")
        
        # Prepare arguments for each video
        video_args = [(str(vp), regions_to_test, args.max_frames, signal_collection, metrics_collection, args.use_gpu) 
                      for vp in video_files]
        
        # For GPU usage with multiprocessing, we need 'spawn' method
        # But 'spawn' is slower to start. With CPU bottleneck in FaceMesh, 
        # standard fork might work IF we don't initialize CUDA in parent.
        # But we imported torch above? Importing torch might initialize context.
        # Safest is to rely on standard pool, knowing GPU context is per-process.
        # However, too many contexts = OOM.
        
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_video, video_args),
                total=len(video_files),
                desc="Processing videos"
            ))
        
        # Aggregate results
        for success, num_signals, error in results:
            if success:
                stats['success'] += 1
                stats['total_signals'] += num_signals
            else:
                stats['failed'] += 1
    else:
        # Sequential processing
        for video_path in tqdm(video_files, desc="Processing videos"):
            success, num_signals, error = _process_single_video(
                str(video_path), 
                regions_to_test, 
                args.max_frames, 
                signal_collection,
                metrics_collection,
                args.use_gpu
            )
            
            if success:
                stats['success'] += 1
                stats['total_signals'] += num_signals
            else:
                stats['failed'] += 1
    
    # Generate ensemble plots
    logger.info("")
    logger.info("="*60)
    logger.info("Processing Complete - Generating Ensemble Plots")
    logger.info("="*60)
    
    generate_ensemble_plots(signal_collection, str(plots_dir), target_length=args.max_frames)
    
    # Save evaluation metrics
    logger.info("Aggregating evaluation metrics...")
    final_metrics = {}
    for region_name, methods_data in metrics_collection.items():
        for method, metrics_list in methods_data.items():
            if len(metrics_list) == 0:
                continue
            
            # Combine region and method for the CSV row
            name = f"{region_name}_{method}"
            
            # Calculate mean of all collected metrics for this region/method
            avg_metrics = {}
            for key in metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])
            
            final_metrics[name] = avg_metrics
            
    if final_metrics:
        metrics_csv = plots_dir / "batch_evaluation_results.csv"
        save_results_to_csv(final_metrics, str(metrics_csv))
        logger.info(f"Summary metrics saved to: {metrics_csv}")
    
    # Save raw signal data as .npy for Deep Learning
    logger.info("Saving signal dataset to .npy file...")
    dataset_to_save = {}
    for region_name, methods_data in signal_collection.items():
        dataset_to_save[region_name] = {}
        for method, signals in methods_data.items():
            if len(signals) > 0:
                # Convert list of signals to a single numpy array
                # Note: signals are already normalized in _process_single_video
                # We resample them here to ensure consistent shape for DL
                resampled_batch = []
                for sig in signals:
                    try:
                        resampled_batch.append(resample_signal(sig, args.max_frames))
                    except:
                        continue
                if resampled_batch:
                    dataset_to_save[region_name][method] = np.array(resampled_batch)
    
    if dataset_to_save:
        npy_path = plots_dir / "rppg_signals_dataset.npy"
        np.save(npy_path, dataset_to_save)
        logger.info(f"Signal dataset saved to: {npy_path}")
        logger.info(f"Dataset structure: {len(dataset_to_save)} regions, {len(RPPG_METHODS)} potential algorithms")
    
    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("Batch Processing Summary")
    logger.info("="*60)
    logger.info(f"Successfully processed: {stats['success']} videos")
    logger.info(f"Failed: {stats['failed']} videos")
    logger.info(f"Total signals collected: {stats['total_signals']}")
    logger.info(f"Ensemble plots saved to: {plots_dir}")
    logger.info(f"Total plots generated: {len(signal_collection)} (one per region)")


if __name__ == "__main__":
    main()
