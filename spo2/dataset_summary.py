import os
import cv2
import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dataset_summary(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_paths = []
    
    for ext in ['*.mp4', '*.avi', '*.MOV', '*.mov']:
        video_paths.extend(list(Path(video_dir).rglob(ext)))
        
    if not video_paths:
        logger.warning(f"No videos found in {video_dir}")
        return
        
    records = []
    
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            logger.warning(f"Could not open {vp.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        duration = frame_count / fps if fps > 0 else 0
        
        records.append({
            'Filename': vp.name,
            'Resolution': f"{width}x{height}",
            'Width': width,
            'Height': height,
            'FPS': fps,
            'Total_Frames': frame_count,
            'Duration_Secs': duration
        })
        
    df = pd.DataFrame(records)
    
    # Save individual stats
    output_csv = os.path.join(output_dir, "dataset_individual_stats.csv")
    df.to_csv(output_csv, index=False)
    
    # Generate aggregated summary
    summary_records = [
        {"Metric": "Total Videos", "Value": str(len(df))},
        {"Metric": "Total Duration (mins)", "Value": f"{df['Duration_Secs'].sum() / 60:.2f}"},
        {"Metric": "Average Duration (secs)", "Value": f"{df['Duration_Secs'].mean():.2f}"},
        {"Metric": "Min Duration (secs)", "Value": f"{df['Duration_Secs'].min():.2f}"},
        {"Metric": "Max Duration (secs)", "Value": f"{df['Duration_Secs'].max():.2f}"},
        {"Metric": "Average FPS", "Value": f"{df['FPS'].mean():.2f}"},
        {"Metric": "Unique Resolutions", "Value": ", ".join(df['Resolution'].unique())}
    ]
    
    summary_df = pd.DataFrame(summary_records)
    summary_out = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    
    logger.info("Dataset statistics summary:")
    for r in summary_records:
        logger.info(f"  {r['Metric']}: {r['Value']}")
        
    logger.info(f"Summary files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="./data")
    parser.add_argument("--output-dir", default="./spo2/results")
    args = parser.parse_args()
    
    generate_dataset_summary(args.video_dir, args.output_dir)
