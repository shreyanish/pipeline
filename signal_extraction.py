import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_raw_rgb_signal(video_path: str, region_indices: list, max_frames: int = None, face_mesh=None):
    """
    Extract raw RGB signal from specified facial region
    
    Args:
        video_path: Path to video file
        region_indices: List of landmark indices defining the ROI
        max_frames: Maximum number of frames to process
        face_mesh: Optional pre-initialized FaceMesh instance for reuse
    
    Returns:
        (raw_signal, fps): RGB signal array and frame rate
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    
    # Pre-allocate array (Optimization)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_signal = np.zeros((total_frames, 3))
    
    frame_idx = 0
    
    # Use provided face_mesh or create a new one
    should_close = False
    if face_mesh is None:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        should_close = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if max_frames is not None and frame_idx >= max_frames:
                break
            
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
    finally:
        cap.release()
        if should_close:
            face_mesh.close()
    
    return raw_signal[:frame_idx], fps


def get_all_regions_signals(video_path: str, regions_dict: dict, max_frames: int = None):
    """
    Extract raw RGB signals for ALL regions in a single pass.
    
    Args:
        video_path: Path to video file
        regions_dict: Dict of {region_name: region_indices}
        max_frames: Maximum number of frames to process
    
    Returns:
        (signals_dict, fps): Dict {region_name: signal_array} and frame rate
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Pre-allocate arrays for all regions
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        
    signals_dict = {name: np.zeros((total_frames, 3)) for name in regions_dict.keys()}
    
    # Initialize FaceMesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if max_frames is not None and frame_idx >= max_frames:
                break
            
            # Process FaceMesh ONCE per frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                
                # Extract signal for EACH region
                for region_name, indices in regions_dict.items():
                    # Create mask for this region
                    mask = np.zeros((h, w), dtype=np.uint8)
                    points = []
                    
                    for idx in indices:
                        lx = min(int(landmarks[idx].x * w), w - 1)
                        ly = min(int(landmarks[idx].y * h), h - 1)
                        points.append((lx, ly))
                    
                    if len(points) >= 3:
                        points_arr = np.array(points, dtype=np.int32)
                        cv2.fillConvexPoly(mask, points_arr, 255)
                        
                        # Calculate mean RGB
                        means = cv2.mean(frame, mask=mask)
                        signals_dict[region_name][frame_idx] = [means[2], means[1], means[0]]
            
            frame_idx += 1
            
            # Break if we've reached pre-allocated size (safety)
            if frame_idx >= total_frames:
                break
            
    finally:
        cap.release()
        face_mesh.close()
    
    # Trim signals to actual number of processed frames
    for name in signals_dict:
        signals_dict[name] = signals_dict[name][:frame_idx]
        
    return signals_dict, fps
