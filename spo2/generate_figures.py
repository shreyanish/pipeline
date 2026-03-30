import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from roi_definitions import ALL_REGIONS

mp_face_mesh = mp.solutions.face_mesh

def generate_methodology_figures(video_path, output_dir):
    """
    Generate the 4 exact pictorial panels for the Methodology figure:
    1. Dataset (Raw Frame)
    2. Generate Face Mesh (Red dots on landmarks)
    3. Generate ROI Candidates (White polygons overlay)
    4. Select ROI (Top regions in yellow)
    """
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return
    
    # 1. Dataset Image (Raw)
    cv2.imwrite(os.path.join(output_dir, "fig_1_dataset.jpg"), frame)
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("No face detected")
            return
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        
        # 2. Face Mesh (Red dots)
        mesh_img = frame.copy()
        for idx, lm in enumerate(landmarks):
            lx = min(int(lm.x * w), w - 1)
            ly = min(int(lm.y * h), h - 1)
            cv2.circle(mesh_img, (lx, ly), 1, (0, 0, 255), -1) # tiny red dots
            
        cv2.imwrite(os.path.join(output_dir, "fig_2_facemesh.jpg"), mesh_img)
        
        # 3. ROI Candidates (White transparent polygons)
        roi_img = mesh_img.copy() # Start with the red dots
        overlay_all = mesh_img.copy()
        
        for name, region_indices in ALL_REGIONS.items():
            points = []
            for l_idx in region_indices:
                lx = min(int(landmarks[l_idx].x * w), w - 1)
                ly = min(int(landmarks[l_idx].y * h), h - 1)
                points.append((lx, ly))
                
            if len(points) >= 3:
                points_arr = np.array(points, dtype=np.int32)
                # Fill with white semi-transparent
                cv2.fillConvexPoly(overlay_all, points_arr, (255, 255, 255))
                # Draw dark red/black border for candidate split
                cv2.polylines(overlay_all, [points_arr], isClosed=True, color=(0, 0, 100), thickness=1)

        cv2.addWeighted(overlay_all, 0.4, roi_img, 0.6, 0, roi_img)
        cv2.imwrite(os.path.join(output_dir, "fig_3_roi_candidates.jpg"), roi_img)
        
        # 4. Select ROI (Specific top regions in solid Yellow, rest are just face mesh)
        select_img = mesh_img.copy()
        
        # Regions matching the reference diagram's visual
        top_regions = [
            'upper_medial_forehead', 
            'lower_medial_forehead', 
            'right_malar', 
            'left_malar'
        ]
        
        for name in top_regions:
            if name in ALL_REGIONS:
                region_indices = ALL_REGIONS[name]
                points = []
                for l_idx in region_indices:
                    lx = min(int(landmarks[l_idx].x * w), w - 1)
                    ly = min(int(landmarks[l_idx].y * h), h - 1)
                    points.append((lx, ly))
                    
                if len(points) >= 3:
                    points_arr = np.array(points, dtype=np.int32)
                    # Solid yellow fill
                    cv2.fillConvexPoly(select_img, points_arr, (0, 255, 255)) # BGR for Yellow
        
        cv2.imwrite(os.path.join(output_dir, "fig_4_select_roi.jpg"), select_img)
        
        # 5. All 31 Landmarks Numbered (Original Request)
        canvas_31 = frame.copy()
        overlay_31 = frame.copy()
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        idx = 0
        for name, region_indices in ALL_REGIONS.items():
            points = []
            for l_idx in region_indices:
                lx = min(int(landmarks[l_idx].x * w), w - 1)
                ly = min(int(landmarks[l_idx].y * h), h - 1)
                points.append((lx, ly))
                
            if len(points) >= 3:
                points_arr = np.array(points, dtype=np.int32)
                color = colors[idx % len(colors)]
                cv2.fillConvexPoly(overlay_31, points_arr, color)
                
                cx = int(np.mean([p[0] for p in points]))
                cy = int(np.mean([p[1] for p in points]))
                cv2.circle(canvas_31, (cx, cy), 3, (255, 255, 255), -1)
                cv2.putText(canvas_31, str(idx+1), (cx+5, cy+5), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            idx += 1
            
        cv2.addWeighted(overlay_31, 0.5, canvas_31, 0.5, 0, canvas_31)
        cv2.imwrite(os.path.join(output_dir, "fig_5_all_31_landmarks.jpg"), canvas_31)
        
        # 6. Create Overall Methodology Composite (Pictorial + Logic Flow)
        create_composite_methodology(output_dir)
        
        print(f"Successfully generated all 6 methodology panels in {output_dir}")

def create_composite_methodology(output_dir):
    """
    Stitches the generated panels into a single high-quality diagram
    matching the user's reference structure.
    """
    panels = []
    for i in range(1, 5):
        p = cv2.imread(os.path.join(output_dir, f"fig_{i}.jpg" if i < 5 else f"fig_{i}.jpg")) # Helper handles the name mapping
        # Map back to our specific naming
        name_map = {1: "dataset", 2: "facemesh", 3: "roi_candidates", 4: "select_roi"}
        p = cv2.imread(os.path.join(output_dir, f"fig_{i}_{name_map[i]}.jpg"))
        if p is not None:
            panels.append(p)
    
    if len(panels) < 4:
        print("Missing panels for composite")
        return

    # Constants for the layout
    padding = 40
    img_h, img_w, _ = panels[0].shape
    # Scale down panels for a reasonable output size
    scale = 0.5
    s_w, s_h = int(img_w * scale), int(img_h * scale)
    
    # Canvas size: 4 images wide + padding, and enough height for boxes below
    canvas_w = (s_w * 4) + (padding * 5)
    canvas_h = (s_h * 2) + (padding * 4)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    titles = ["Dataset", "Generate Face Mesh", "Generate ROI Candidates", "Select ROI"]
    
    for i, panel in enumerate(panels):
        # Resize
        resized = cv2.resize(panel, (s_w, s_h))
        x_offset = padding + i * (s_w + padding)
        y_offset = padding + 40 # space for title
        
        # Draw Image
        canvas[y_offset : y_offset + s_h, x_offset : x_offset + s_w] = resized
        
        # Draw Title Box
        cv2.rectangle(canvas, (x_offset, y_offset - 40), (x_offset + s_w, y_offset), (0, 0, 0), 2)
        cv2.putText(canvas, titles[i], (x_offset + 10, y_offset - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        
        # Arrows between top panels
        if i < 3:
            arrow_start = (x_offset + s_w, y_offset + s_h // 2)
            arrow_end = (x_offset + s_w + padding, y_offset + s_h // 2)
            cv2.arrowedLine(canvas, arrow_start, arrow_end, (0, 0, 0), 2, tipLength=0.3)

    # Draw Bottom Logic Boxes (Mimicking the reference diagram)
    # Start under Select ROI
    start_x = padding + 3 * (s_w + padding)
    start_y = padding + 40 + s_h
    
    # Vertical line down from Select ROI
    cv2.line(canvas, (start_x + s_w // 2, start_y), (start_x + s_w // 2, start_y + 100), (0, 0, 0), 2)
    # Line left towards Method
    cv2.line(canvas, (start_x + s_w // 2, start_y + 100), (canvas_w // 2 + 100, start_y + 100), (0, 0, 0), 2)
    # Arrow into Method
    cv2.arrowedLine(canvas, (canvas_w // 2 + 100, start_y + 100), (canvas_w // 2 + 100, start_y + 150), (0, 0, 0), 2)

    # Method Box
    m_x, m_y = canvas_w // 2 - 50, start_y + 150
    m_w, m_h = 150, 60
    cv2.rectangle(canvas, (m_x, m_y), (m_x + m_w, m_y + m_h), (0, 0, 0), 2)
    cv2.putText(canvas, "Method", (m_x + 30, m_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    
    # Arrow to Evaluation
    cv2.arrowedLine(canvas, (m_x + m_w, m_y + m_h // 2), (m_x + m_w + 50, m_y + m_h // 2), (0, 0, 0), 2)
    
    # Evaluation Box
    e_x, e_y = m_x + m_w + 50, m_y
    e_w, e_h = 150, 60
    cv2.rectangle(canvas, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 0, 0), 2)
    cv2.putText(canvas, "Evaluation", (e_x + 15, e_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    # Sub-boxes for Method (DeepPhys, etc - simplified like reference)
    methods = ["POS", "CHROM", "ICA", "SSR"]
    metrics = ["MAE", "RMSE", "PCC", "SNR"]
    
    for i, m in enumerate(methods):
        sub_x = m_x
        sub_y = m_y + m_h + 20 + i * 40
        cv2.rectangle(canvas, (sub_x + 20, sub_y), (sub_x + m_w, sub_y + 30), (0, 0, 0), 1)
        cv2.putText(canvas, m, (sub_x + 40, sub_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.line(canvas, (m_x + 10, m_y + m_h), (m_x + 10, sub_y + 15), (0, 0, 0), 1)
        cv2.line(canvas, (m_x + 10, sub_y + 15), (sub_x + 20, sub_y + 15), (0, 0, 0), 1)

    for i, met in enumerate(metrics):
        sub_x = e_x
        sub_y = e_y + e_h + 20 + i * 40
        cv2.rectangle(canvas, (sub_x + 20, sub_y), (sub_x + e_w, sub_y + 30), (0, 0, 0), 1)
        cv2.putText(canvas, met, (sub_x + 40, sub_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.line(canvas, (e_x + 10, e_y + e_h), (e_x + 10, sub_y + 15), (0, 0, 0), 1)
        cv2.line(canvas, (e_x + 10, sub_y + 15), (sub_x + 20, sub_y + 15), (0, 0, 0), 1)

    output_path = os.path.join(output_dir, "methodology_overall_pictorial.png")
    cv2.imwrite(output_path, canvas)
    print(f"Generated composite methodology figure at {output_path}")

if __name__ == "__main__":
    test_video = "./data/2024W0101P001.MOV"
    output_dir = "./spo2/results"
    generate_methodology_figures(test_video, output_dir)
