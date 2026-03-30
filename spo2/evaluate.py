try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np
import os
import cv2
import time
import logging
from pathlib import Path
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from signal_extraction import get_all_regions_signals
from roi_definitions import ALL_REGIONS, SKIN_REGIONS, FACE_REGIONS, SKIN_FACE_REGIONS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_gpu_usage():
    if not TORCH_AVAILABLE:
        logger.info("Torch not available - skipping GPU logging")
        return
        
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    elif torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def save_landmark_quality_images(video_path, regions_signal_quality, output_dir):
    """
    Save top 5 and bottom 5 landmark regions in separate images.
    """
    sorted_regions = sorted(regions_signal_quality.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_regions[:5]
    bottom_5 = sorted_regions[-5:]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret: 
        cap.release()
        return
    
    video_id = Path(video_path).stem
    
    # Initialize MediaPipe to get landmark coordinates for the first frame
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            cap.release()
            return
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        def draw_and_save(regions, filename_suffix):
            canvas = frame.copy()
            overlay = frame.copy()
            
            for name, qual in regions:
                indices = ALL_REGIONS[name]
                points = []
                for idx in indices:
                    lx = min(int(landmarks[idx].x * w), w - 1)
                    ly = min(int(landmarks[idx].y * h), h - 1)
                    points.append((lx, ly))
                
                if len(points) >= 3:
                    points_arr = np.array(points, dtype=np.int32)
                    # Draw a semi-transparent polygon
                    color = (0, 255, 0) if "Top" in filename_suffix else (0, 0, 255)
                    cv2.fillConvexPoly(overlay, points_arr, color)
                    # Draw text label
                    center_x = int(np.mean([p[0] for p in points]))
                    center_y = int(np.mean([p[1] for p in points]))
                    cv2.putText(canvas, name.split('_')[0], (center_x, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Blend overlay
            cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
            out_path = os.path.join(output_dir, f"{video_id}_{filename_suffix}.jpg")
            cv2.imwrite(out_path, canvas)
            logger.info(f"Saved {out_path}")

        draw_and_save(top_5, "top_5_quality")
        draw_and_save(bottom_5, "bottom_5_quality")
    
    cap.release()

def evaluate_spo2_pipeline(video_dir, max_videos=100, max_frames=None):
    output_dir = "spo2/results"
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = list(Path(video_dir).rglob("*.mp4"))
    video_paths += list(Path(video_dir).rglob("*.avi"))
    video_paths += list(Path(video_dir).rglob("*.MOV"))
    video_paths = sorted(video_paths)[:max_videos]
    
    results_log = []
    
    for vp in video_paths:
        logger.info(f"Processing {vp.name} (Max frames: {max_frames})...")
        log_gpu_usage()
        
        # 1. Extract signals for ALL regions
        try:
            all_signals, fps = get_all_regions_signals(str(vp), ALL_REGIONS, max_frames=max_frames)
        except Exception as e:
            logger.error(f"Failed to extract signals for {vp.name}: {e}")
            continue
            
        if not all_signals: continue
        
        # 2. Comparison Logic: Average Variance (as proxy for signal cleanliness)
        skin_vars = [np.var(all_signals[k]) for k in SKIN_REGIONS if k in all_signals]
        face_vars = [np.var(all_signals[k]) for k in FACE_REGIONS if k in all_signals]
        face_skin_vars = [np.var(all_signals[k]) for k in SKIN_FACE_REGIONS if k in all_signals]
        
        avg_skin_var = np.mean(skin_vars) if skin_vars else 0
        avg_face_var = np.mean(face_vars) if face_vars else 0
        avg_face_skin_var = np.mean(face_skin_vars) if face_skin_vars else 0
        
        logger.info(f"  Avg Variance - Face: {avg_face_var:.5f} | Skin: {avg_skin_var:.5f} | Face+Skin: {avg_face_skin_var:.5f}")
        
        # 3. Quality metric for individual regions
        quality = {name: np.var(sig) for name, sig in all_signals.items()}
        
        # 4. Save visualization
        save_landmark_quality_images(str(vp), quality, output_dir)
        
        results_log.append({
            'video': vp.name,
            'face_var': avg_face_var,
            'skin_var': avg_skin_var,
            'face_skin_var': avg_face_skin_var
        })

    # Save summary results
    import pandas as pd
    df = pd.DataFrame(results_log)
    summary_path = os.path.join(output_dir, "spo2_comparison_summary.csv")
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary results saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="./data")
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=300) # Default 10 seconds at 30fps
    args = parser.parse_args()
        
    evaluate_spo2_pipeline(args.video_dir, max_videos=args.max_videos, max_frames=args.max_frames)
