#!/usr/bin/env python3
"""
spo2/run_spo2_pipeline.py
─────────────────────────────────────────────────────────────────────────────
OPTIMIZED VERSION: High-Speed (Multiprocessing) + Auto-Resume + Spectral Math

What this script does
─────────────────────
1. Loads all ground-truth CSVs (smart-matching video files to participant IDs).
2. Parallel Video Processing: Uses all CPU cores to run MediaPipe (the bottleneck).
3. Auto-Resume: Skips already-processed videos found in existing results CSV.
4. Spectral SpO2: Uses FFT peaks at the detected heart rate to estimate SpO2,
   making results algorithm-dependent.
5. Incremental Saving: Saves each video result immediately (crash-proof).
"""

import os
import sys
import argparse
import logging
import time
import multiprocessing
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch

# ── project root on path ──────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from signal_extraction import get_all_regions_signals
from roi_definitions import ALL_REGIONS, SKIN_REGIONS
import rppg_pytorch as rppg

# ── optional torch ────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler("spo2_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Algorithm Map ─────────────────────────────────────────────────────────────
ALGO_MAP = {
    "POS":   rppg.process_signal_pos_torch,
    "CHROM": rppg.process_signal_chrom_torch,
    "ICA":   rppg.process_signal_ica_torch,
    "SSR":   rppg.process_signal_ssr_torch,
    "GREEN": rppg.process_signal_green_torch,
    "PCA":   rppg.process_signal_pca_torch,
    "PBV":   rppg.process_signal_pbv_torch,
    "LGI":   rppg.process_signal_lgi_torch,
    "SAMC":  rppg.process_signal_samc_torch,
    "2SR":   rppg.process_signal_2sr_torch,
    "OMIT":  rppg.process_signal_omit_torch,
}
ALGORITHMS = list(ALGO_MAP.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loader
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(gt_dir: str) -> dict:
    """Read all CSVs and build a lookup: { lowercase_id : {spo2, hr, raw_name} }"""
    gt_dir = Path(gt_dir)
    gt = {}
    if not gt_dir.exists():
        logger.warning(f"GT directory {gt_dir} not found.")
        return gt

    for csv_path in sorted(gt_dir.glob("*.csv")):
        if csv_path.name.startswith("._"): continue
        try:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
            except:
                df = pd.read_csv(csv_path, encoding='latin-1')

            df.columns = [c.strip() for c in df.columns]
            
            # Column mapping
            col_map = {
                "Oxygen Level": ["Oxygen Level", "Oxygen", "SpO2"],
                "Pulse ": ["Pulse ", "Pulse", "Heart Rate", "HR"],
                "Video File Name": ["Video File Name", "Video Name", "File Name"]
            }
            for standard, variations in col_map.items():
                for v in variations:
                    if v in df.columns and standard not in df.columns:
                        df = df.rename(columns={v: standard})

            if not {"Video File Name", "Oxygen Level", "Pulse "}.issubset(set(df.columns)):
                continue

            for _, row in df.iterrows():
                vid_name = str(row["Video File Name"]).strip()
                if not vid_name or vid_name == "nan": continue
                
                try:
                    spo2_gt = float(str(row["Oxygen Level"]).strip())
                    hr_gt   = float(str(row["Pulse "]).strip())
                except:
                    spo2_gt, hr_gt = float("nan"), float("nan")

                gt[vid_name.lower()] = {"spo2": spo2_gt, "hr": hr_gt, "raw_name": vid_name}
        except Exception as e:
            logger.debug(f"Could not parse {csv_path.name}: {e}")

    logger.info(f"Loaded ground truth for {len(gt)} records.")
    return gt

def find_gt_record(video_stem: str, gt_lookup: dict) -> dict:
    """Smart matcher for filenames (handles dates, year-prefixes, etc.)"""
    name = video_stem.lower()
    # 1. Direct match
    if name in gt_lookup: return gt_lookup[name]
    
    # 2. Match by participant ID (e.g. contains 'P001')
    import re
    p_match = re.search(r'p\d{3}', name)
    if p_match:
        p_id = p_match.group()
        for key, rec in gt_lookup.items():
            if p_id in key and ("2024" in key or "2024" in name): # year check
                return rec
                
    # 3. Last 8 chars match (common for Wave/Participant IDs)
    if len(name) >= 8:
        suffix = name[-8:]
        for key, rec in gt_lookup.items():
            if key.endswith(suffix): return rec
            
    return None

# ─────────────────────────────────────────────────────────────────────────────
# SpO2 Math (Spectral)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spo2_spectral(bvp_signal: np.ndarray,
                          raw_rgb: np.ndarray,
                          fs: float) -> float:
    """Algorithm-dependent SpO2 using FFT amplitudes at the detected HR frequency."""
    if len(bvp_signal) < 64: return float("nan")
    
    # 1. Find the HR frequency from the algorithm's BVP
    f, pxx = welch(bvp_signal, fs, nperseg=min(len(bvp_signal), 256))
    mask = (f >= 0.7) & (f <= 3.0)
    if not any(mask): return float("nan")
    hr_freq = f[mask][np.argmax(pxx[mask])]
    
    # 2. Extract AC of Red and Green at this HR frequency
    # We detrend channels first
    red   = raw_rgb[:len(bvp_signal), 0]
    green = raw_rgb[:len(bvp_signal), 1]
    
    def get_ac(signal, freq, fs_in):
        s_detrend = signal - np.mean(signal)
        n = len(s_detrend)
        f_axis = np.fft.rfftfreq(n, 1/fs_in)
        fft_vals = np.abs(np.fft.rfft(s_detrend))
        idx = np.argmin(np.abs(f_axis - freq))
        # Use sum of power around peak for robustness
        return np.sum(fft_vals[max(0, idx-1):idx+2])

    ac_r = get_ac(red, hr_freq, fs)
    ac_g = get_ac(green, hr_freq, fs)
    dc_r = np.mean(red)
    dc_g = np.mean(green)
    
    if dc_r < 1e-6 or dc_g < 1e-6 or ac_g < 1e-9: return float("nan")
    
    R = (ac_r / dc_r) / (ac_g / dc_g)
    spo2 = 110.0 - (25.0 * R)
    return float(np.clip(spo2, 70.0, 100.0))

# ─────────────────────────────────────────────────────────────────────────────
# Worker Function
# ─────────────────────────────────────────────────────────────────────────────

def process_video_worker(vp_item):
    """Worker task for multiprocessing pool"""
    try:
        vp, gt_record, device, max_frames = vp_item
        stem = Path(vp).stem
        
        # 1. Extract signals (CPU — MediaPipe)
        all_signals, fps = get_all_regions_signals(str(vp), ALL_REGIONS, max_frames=max_frames)
        if not all_signals: return []

        # 2. Average SKIN_REGIONS
        skin_sigs = [all_signals[k] for k in SKIN_REGIONS if k in all_signals]
        if not skin_sigs: return []
        min_len = min(s.shape[0] for s in skin_sigs)
        avg_rgb = np.mean(np.stack([s[:min_len] for s in skin_sigs], axis=0), axis=0) # (N,3)

        results = []
        spo2_gt = gt_record.get("spo2", np.nan) if gt_record else np.nan
        hr_gt   = gt_record.get("hr",   np.nan) if gt_record else np.nan

        # 3. Iterative Algos
        for name, fn in ALGO_MAP.items():
            t_start = time.time()
            bvp_np = None
            try:
                # GPU attempt
                if TORCH_AVAILABLE and device != "cpu":
                    bvp_tensor = fn(avg_rgb, fps, device)
                    if bvp_tensor is not None: bvp_np = bvp_tensor.cpu().numpy()
                # CPU fallback
                if bvp_np is None:
                    bvp_tensor = fn(avg_rgb, fps, "cpu")
                    if bvp_tensor is not None: bvp_np = bvp_tensor.numpy()
            except: pass
            
            algo_sec = time.time() - t_start
            
            if bvp_np is not None and len(bvp_np) > 30:
                spo2_est = compute_spo2_spectral(bvp_np, avg_rgb, fps)
                err = abs(spo2_est - spo2_gt) if not np.isnan(spo2_gt) and not np.isnan(spo2_est) else np.nan
                
                results.append({
                    "video": stem, "algorithm": name, "spo2_estimated": round(spo2_est, 3),
                    "spo2_gt": spo2_gt, "spo2_error": round(err, 3), "hr_gt": hr_gt,
                    "proc_time_s": round(algo_sec, 3), "status": "ok"
                })
            else:
                results.append({"video": stem, "algorithm": name, "status": "failed"})

        return results
    except Exception as e:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="High-Speed SpO2 Pipeline")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--gt-dir", default="./ground-truth")
    parser.add_argument("--output-dir", default="spo2/results")
    parser.add_argument("--max-videos", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = Path(args.output_dir) / "spo2_results_per_video.csv"
    
    # 1. Handle Auto-Resume
    done_videos = set()
    if out_csv.exists():
        try:
            df_old = pd.read_csv(out_csv)
            done_videos = set(df_old["video"].unique())
            logger.info(f"Auto-Resume: {len(done_videos)} videos already processed. Skipping.")
        except: pass

    # 2. Load GT and Videos
    gt_lookup = load_ground_truth(args.gt_dir)
    video_dir = Path(args.video_dir)
    video_paths = []
    for ext in ("*.MOV", "*.mov", "*.mp4", "*.avi"):
        video_paths.extend(list(video_dir.rglob(ext)))
    
    # Filter and map to GT
    video_paths = sorted(list(set(video_paths)))
    tasks = []
    for vp in video_paths:
        if vp.stem in done_videos: continue
        if len(tasks) >= args.max_videos: break
        
        gt_rec = find_gt_record(vp.stem, gt_lookup)
        tasks.append((vp, gt_rec, args.device, args.max_frames))

    if not tasks:
        logger.info("No new videos to process.")
        return

    logger.info(f"Starting pipeline: {len(tasks)} videos, using {args.device}")
    
    # 3. Multiprocessing Pool
    num_workers = args.workers or max(1, multiprocessing.cpu_count() - 2)
    logger.info(f"Using {num_workers} worker processes.")

    # Write header if new file
    if not out_csv.exists():
        with open(out_csv, "w") as f:
            f.write("video,algorithm,spo2_estimated,spo2_gt,spo2_error,hr_gt,proc_time_s,status\n")

    # 4. Process with progress bar
    with multiprocessing.Pool(num_workers) as pool:
        for vid_results in tqdm(pool.imap_unordered(process_video_worker, tasks), total=len(tasks)):
            if not vid_results: continue
            
            # Incremental save
            df_vid = pd.DataFrame(vid_results)
            df_vid.to_csv(out_csv, mode='a', header=False, index=False)

    # 5. Final Summary
    logger.info(f"Done! Results updated at {out_csv}")
    try:
        final_df = pd.read_csv(out_csv)
        final_df = final_df[final_df["status"] == "ok"]
        summary = final_df.groupby("algorithm")["spo2_error"].agg(["mean", "std", "count"]).rename(columns={"mean": "MAE", "std": "STD"})
        summary = summary.sort_values("MAE")
        
        summary_path = Path(args.output_dir) / "spo2_results_summary.csv"
        summary.to_csv(summary_path)
        logger.info("\nAlgorithm Summary (MAE):\n" + summary.to_string())
    except:
        logger.warning("Could not generate summary.")

if __name__ == "__main__":
    main()
