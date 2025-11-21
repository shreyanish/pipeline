import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
from scipy.signal import detrend, butter, filtfilt


VIDEO_FOLDER = "./data"
GROUND_TRUTH_FILE = "ground_truth.csv"
OUTPUT_FILE = "spo2_dataset.csv"

FS_MIN = 0.7  # 42 BPM
FS_MAX = 3.0  # 180 BPM
BVP_WINDOW_SEC = 180

ROI_REGIONS = {
    'forehead_glabella': [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 171, 68], # Approx 
    'left_cheek': [454, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152], # Left Malar area
    'right_cheek': [234, 127, 162, 21, 54, 103, 67, 109, 10], # Right Malar area (using symmetric landmarks)
    'left_malar': [116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 203, 206, 216], # Approx Right side on image
    'right_malar': [345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 423, 426, 436] # Approx Left side on image
}

FINAL_ROI_INDICES = [
    # Forehead
    [9, 107, 66, 105, 104, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336],
    # Right Cheek
    [118, 119, 100, 126, 209, 49, 129, 203, 205, 50],
    # Left Cheek
    [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]
]

mp_face_mesh = mp.solutions.face_mesh

def get_raw_rgb_signal(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    
    # Pre-allocate array (Optimization)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_signal = np.zeros((total_frames, 3))
    
    frame_idx = 0
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                for region_indices in FINAL_ROI_INDICES:
                    points = []
                    for idx in region_indices:
                        lx = min(int(landmarks[idx].x * w), w - 1)
                        ly = min(int(landmarks[idx].y * h), h - 1)
                        points.append((lx, ly))
                    
                    points_arr = np.array(points, dtype=np.int32)
                    cv2.fillConvexPoly(combined_mask, points_arr, 255)
                
                means = cv2.mean(frame, mask=combined_mask)
                raw_signal[frame_idx] = [means[2], means[1], means[0]]
            
            frame_idx += 1

    cap.release()
    return raw_signal[:frame_idx], fps

def process_signal_pos(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    if len(raw_signal) < 30: return np.array([])
    
    # 1. Detrending
    signal = detrend(raw_signal, axis=0)

    # 2. POS Algorithm [cite: 319, 394]
    # Standard POS projection matrix
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    S = np.dot(signal, P.T)
    BVP = S[:, 0] + (np.std(S[:, 0]) / np.std(S[:, 1])) * S[:, 1]

    # 3. Bandpass Filtering (Zero-Phase)
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP) 

    return filtered_bvp

def extract_spo2_features(raw_signal: np.ndarray, filtered_bvp: np.ndarray, fs: float) -> dict:
    if filtered_bvp.size == 0: return {}

    nyquist = 0.5 * fs
    
    # Filters for feature extraction
    b_dc, a_dc = butter(2, 0.5 / nyquist, btype='low') 
    b_ac, a_ac = butter(2, 0.7 / nyquist, btype='high') 

    # Extract components (Zero-phase)
    R_raw = raw_signal[:, 0]
    B_raw = raw_signal[:, 2] # Red is 0, Blue is 2 (based on previous function)
    
    # DC = Baseline (Low freq)
    R_DC = np.mean(filtfilt(b_dc, a_dc, R_raw))
    B_DC = np.mean(filtfilt(b_dc, a_dc, B_raw))
    
    # AC = Pulsatile (High freq)
    R_AC = np.std(filtfilt(b_ac, a_ac, R_raw))
    B_AC = np.std(filtfilt(b_ac, a_ac, B_raw))
    
    if R_DC == 0 or B_DC == 0: return {'R_Ratio': np.nan}

    # "Ratio of Ratios"
    R_Ratio = (R_AC / R_DC) / (B_AC / B_DC)
    
    return {
        'R_Ratio': R_Ratio,
        'AC_Red': R_AC,
        'DC_Red': R_DC,
        'AC_Blue': B_AC,
        'DC_Blue': B_DC
    }

def run_pipeline(video_folder, gt_file):
    print("--- Starting rPPG SpO2 Pipeline ---")
    
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

    final_dataset = []
    
    # Iterate through all video files in the specified folder
    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_folder, video_file)

            if video_id in gt_map:
                print(f"Processing video: {video_file}...")
                
                # Step 2 & 3: Signal Extraction and POS Filtering
                raw_signal, fps = get_raw_rgb_signal(video_path)
                
                if raw_signal is None or raw_signal.shape[0] < 100:
                    print(f"Skipping {video_file}: Failed to process or too few frames extracted.")
                    continue
                    
                filtered_bvp = process_signal_pos(raw_signal, fps)
                
                # Step 4: SpO2 Feature Calculation
                features = extract_spo2_features(raw_signal, filtered_bvp, fps)
                
                # Append the results to the final dataset
                row = {
                    'Video_ID': video_id,
                    'SpO2_Ground_Truth': gt_map[video_id],
                    **features # Unpack the calculated features
                }
                final_dataset.append(row)
                print(f"-> Completed {video_id}. SpO2 R_Ratio: {features['R_Ratio']:.3f}. GT: {row['SpO2_Ground_Truth']}")
            else:
                print(f"Skipping {video_file}: No matching SpO2 label found in ground truth.")

    # Save the final ML-ready dataset
    output_df = pd.DataFrame(final_dataset)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print("\n--- Pipeline Complete ---")
    print(f"Successfully processed {len(final_dataset)} videos.")
    print(f"Output saved to {OUTPUT_FILE}")
    
if __name__ == "__main__":
    if os.path.exists(VIDEO_FOLDER) and os.path.exists(GROUND_TRUTH_FILE):
        run_pipeline(VIDEO_FOLDER, GROUND_TRUTH_FILE)
    else:
        print("Please ensure 'data' folder and 'ground_truth.csv' exist.")