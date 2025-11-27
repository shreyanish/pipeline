import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
from scipy.signal import detrend, butter, filtfilt
from sklearn.decomposition import FastICA


VIDEO_FOLDER = "./data"
GROUND_TRUTH_FILE = "ground_truth.csv"
OUTPUT_FILE = "spo2_dataset.csv"

FS_MIN = 0.7  # 42 BPM
FS_MAX = 3.0  # 180 BPM
BVP_WINDOW_SEC = 180

# Configuration: Select which rPPG methods to test
RPPG_METHODS = ['POS', 'CHROM', 'ICA']

# Configuration: Select which regions to test ('ALL' or list of region names)
SELECTED_REGIONS = 'ALL'  # Can be changed to specific list like ['forehead', 'left_cheek']

# Comprehensive 31+ Facial Regions using MediaPipe 468 landmarks
ALL_REGIONS = {
    # === FOREHEAD REGIONS (5) ===
    'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    'forehead_upper': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378],
    'forehead_lower': [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400],
    'forehead_left': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
    'forehead_right': [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58],
    
    # === CHEEK REGIONS (6) ===
    'left_cheek': [454, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150],
    'right_cheek': [234, 127, 162, 21, 54, 103, 67, 109, 10, 151, 9, 8, 168, 6, 197],
    'left_cheek_upper': [454, 356, 323, 361, 288, 397, 365, 379],
    'right_cheek_upper': [234, 127, 162, 21, 54, 103, 67, 109],
    'left_cheek_lower': [378, 400, 377, 152, 148, 176, 149, 150, 136, 172],
    'right_cheek_lower': [10, 151, 9, 8, 168, 6, 197, 195, 5, 4],
    
    # === MALAR (CHEEKBONE) REGIONS (4) ===
    'left_malar': [345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 423, 426, 436],
    'right_malar': [116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 203, 206, 216],
    'left_malar_extended': [454, 356, 323, 345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 423, 426, 436, 434, 432],
    'right_malar_extended': [234, 127, 162, 116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 203, 206, 216, 214, 212],
    
    # === NOSE REGIONS (3) ===
    'nose_bridge': [6, 168, 197, 195, 5, 4, 1, 19, 94, 2],
    'nose_tip': [1, 2, 98, 97, 327, 326],
    'nose_full': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 327, 326, 2, 326, 327],
    
    # === PERIORAL/MOUTH REGIONS (4) ===
    'upper_lip': [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78],
    'lower_lip': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    'perioral': [185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40],
    'chin': [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338],
    
    # === PERIORBITAL/EYE REGIONS (4) ===
    'left_periorbital': [246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33, 246],
    'right_periorbital': [466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466],
    'left_temple': [356, 454, 323, 361, 288, 397, 365, 379, 378, 400],
    'right_temple': [127, 234, 93, 132, 58, 172, 136, 150, 149, 176],
    
    # === COMBINED REGIONS (9+) ===
    'forehead_cheeks': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    'forehead_left_cheek': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
    'forehead_right_cheek': [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
    'both_cheeks': [454, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 234, 127, 162, 21, 54, 103, 67, 109],
    'both_malars': [345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 423, 426, 436, 116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 203, 206, 216],
    'top5_disjoint': [9, 107, 66, 105, 104, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 118, 119, 100, 126, 209, 49, 129, 203, 205, 50, 347, 348, 329, 355, 429, 279, 358, 423, 425, 280],  # Original TOP-5 from paper
    'full_face_no_eyes': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    'glabella': [9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94],
    'left_side_face': [454, 356, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 345, 346, 347, 348, 349, 350, 277, 355],
    'right_side_face': [234, 127, 162, 21, 54, 103, 67, 109, 10, 151, 9, 116, 117, 118, 119, 120, 121, 47, 126, 142],
}

mp_face_mesh = mp.solutions.face_mesh

def get_raw_rgb_signal(video_path: str, region_indices: list):
    """Extract raw RGB signal from specified facial region"""
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
                
                # Create mask for the specified region
                mask = np.zeros((h, w), dtype=np.uint8)
                points = []
                
                for idx in region_indices:
                    lx = min(int(landmarks[idx].x * w), w - 1)
                    ly = min(int(landmarks[idx].y * h), h - 1)
                    points.append((lx, ly))
                
                if len(points) >= 3:  # Need at least 3 points for a polygon
                    points_arr = np.array(points, dtype=np.int32)
                    cv2.fillConvexPoly(mask, points_arr, 255)
                    
                    means = cv2.mean(frame, mask=mask)
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

def process_signal_chrom(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """CHROM (Chrominance-based) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # 1. Normalize RGB channels
    R = raw_signal[:, 0]
    G = raw_signal[:, 1]
    B = raw_signal[:, 2]
    
    # Avoid division by zero
    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)
    
    if R_mean == 0 or G_mean == 0 or B_mean == 0:
        return np.array([])
    
    Xn = R / R_mean
    Yn = G / G_mean
    Zn = B / B_mean
    
    # 2. Chrominance projection
    Xs = 3 * Xn - 2 * Yn
    Ys = 1.5 * Xn + Yn - 1.5 * Zn
    
    # 3. Calculate BVP signal
    std_Xs = np.std(Xs)
    std_Ys = np.std(Ys)
    
    if std_Ys == 0:
        return np.array([])
    
    BVP = Xs - (std_Xs / std_Ys) * Ys
    
    # 4. Bandpass Filtering (Zero-Phase)
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_ica(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """ICA (Independent Component Analysis) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # 1. Detrend RGB signals
    signal = detrend(raw_signal, axis=0)
    
    # 2. Apply FastICA
    try:
        ica = FastICA(n_components=3, random_state=0, max_iter=500)
        components = ica.fit_transform(signal)
    except:
        return np.array([])
    
    # 3. Select component with strongest periodicity in HR range
    # Use FFT to find component with peak in physiological range
    nyquist = 0.5 * fs
    best_component = 0
    max_power = 0
    
    for i in range(3):
        # Compute FFT
        fft_vals = np.fft.rfft(components[:, i])
        fft_freqs = np.fft.rfftfreq(len(components[:, i]), 1.0 / fs)
        
        # Find power in physiological range (0.7-3.0 Hz)
        mask = (fft_freqs >= FS_MIN) & (fft_freqs <= FS_MAX)
        power = np.sum(np.abs(fft_vals[mask])**2)
        
        if power > max_power:
            max_power = power
            best_component = i
    
    BVP = components[:, best_component]
    
    # 4. Bandpass Filtering (Zero-Phase)
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
    G_raw = raw_signal[:, 1]
    B_raw = raw_signal[:, 2]
    
    # DC = Baseline (Low freq)
    R_DC = np.mean(filtfilt(b_dc, a_dc, R_raw))
    G_DC = np.mean(filtfilt(b_dc, a_dc, G_raw))
    B_DC = np.mean(filtfilt(b_dc, a_dc, B_raw))
    
    # AC = Pulsatile (High freq)
    R_AC = np.std(filtfilt(b_ac, a_ac, R_raw))
    G_AC = np.std(filtfilt(b_ac, a_ac, G_raw))
    B_AC = np.std(filtfilt(b_ac, a_ac, B_raw))
    
    # Calculate ratios (with safety checks)
    R_Ratio = np.nan
    R_Green_Blue = np.nan
    
    if R_DC != 0 and B_DC != 0:
        R_Ratio = (R_AC / R_DC) / (B_AC / B_DC)
    
    if G_DC != 0 and B_DC != 0:
        R_Green_Blue = (G_AC / G_DC) / (B_AC / B_DC)
    
    return {
        'R_Ratio': R_Ratio,
        'R_Green_Blue': R_Green_Blue,
        'AC_Red': R_AC,
        'DC_Red': R_DC,
        'AC_Green': G_AC,
        'DC_Green': G_DC,
        'AC_Blue': B_AC,
        'DC_Blue': B_DC
    }

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