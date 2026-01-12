import cv2
import numpy as np
import mediapipe as mp

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
