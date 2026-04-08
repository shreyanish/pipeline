import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from roi_definitions import ALL_REGIONS

mp_face_mesh = mp.solutions.face_mesh

# ──────────────────────────────────────────────────────────────────────────────
# Multi-video landmark averaging
# ──────────────────────────────────────────────────────────────────────────────

def collect_averaged_landmarks(video_paths):
    """
    Run MediaPipe on the first detectable frame of each video and return
    the per-landmark centroid averaged across all successful detections.

    Args:
        video_paths: list of str paths to video files

    Returns:
        (landmarks_avg, best_frame, h, w)
        landmarks_avg : list of (x_pixel, y_pixel) tuples, indexed by landmark idx
        best_frame    : BGR frame from the video that had the cleanest detection
        h, w          : frame dimensions
    """
    all_landmark_coords = []   # list of dicts: {lm_idx: (x_norm, y_norm)}
    best_frame = None
    best_conf = -1.0

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            # Try up to 5 frames to find a good detection
            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    coords = {i: (lm.x, lm.y) for i, lm in enumerate(lms)}
                    all_landmark_coords.append(coords)

                    # Use this frame as 'best' if it has higher avg landmark visibility
                    vis = np.mean([lm.visibility for lm in lms
                                   if hasattr(lm, 'visibility')])
                    if vis > best_conf:
                        best_conf = vis
                        best_frame = frame.copy()
                    break
            cap.release()

    if not all_landmark_coords:
        return None, None, None, None

    # Average normalised (x, y) across all videos
    n_landmarks = 478  # MediaPipe refine = 478 landmarks
    avg_x = np.zeros(n_landmarks)
    avg_y = np.zeros(n_landmarks)
    count = np.zeros(n_landmarks)

    for coords in all_landmark_coords:
        for idx, (x, y) in coords.items():
            avg_x[idx] += x
            avg_y[idx] += y
            count[idx] += 1

    count[count == 0] = 1  # avoid div-by-zero for unused indices
    avg_x /= count
    avg_y /= count

    h, w, _ = best_frame.shape

    # Convert to pixel coords list-of-tuples (indexed by lm idx)
    landmarks_avg = [(min(int(avg_x[i] * w), w - 1),
                      min(int(avg_y[i] * h), h - 1))
                     for i in range(n_landmarks)]

    return landmarks_avg, best_frame, h, w


# ──────────────────────────────────────────────────────────────────────────────
# Panel generation helpers (use pre-averaged landmarks)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_facemesh_panel(frame, landmarks_avg):
    """Panel 2: tiny red dots at every averaged landmark."""
    img = frame.copy()
    for (lx, ly) in landmarks_avg:
        cv2.circle(img, (lx, ly), 1, (0, 0, 255), -1)
    return img


def _draw_roi_candidates_panel(mesh_img, landmarks_avg):
    """Panel 3: semi-transparent white polygons for all 31 ROI candidates."""
    roi_img = mesh_img.copy()
    overlay = mesh_img.copy()
    for name, region_indices in ALL_REGIONS.items():
        points = [landmarks_avg[i] for i in region_indices
                  if i < len(landmarks_avg)]
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(overlay, pts, (255, 255, 255))
            cv2.polylines(overlay, [pts], isClosed=True,
                          color=(0, 0, 100), thickness=1)
    cv2.addWeighted(overlay, 0.4, roi_img, 0.6, 0, roi_img)
    return roi_img


def _draw_select_roi_panel(mesh_img, landmarks_avg):
    """Panel 4: selected top ROI regions in solid yellow."""
    select_img = mesh_img.copy()
    top_regions = [
        'upper_medial_forehead',
        'lower_medial_forehead',
        'right_malar',
        'left_malar',
    ]
    for name in top_regions:
        if name not in ALL_REGIONS:
            continue
        points = [landmarks_avg[i] for i in ALL_REGIONS[name]
                  if i < len(landmarks_avg)]
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.fillConvexPoly(select_img, pts, (0, 255, 255))  # BGR yellow
    return select_img


def _draw_all31_panel(frame, landmarks_avg):
    """Panel 5: all 31 ROI polygons with colour coding and index labels."""
    canvas = frame.copy()
    overlay = frame.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    for idx, (name, region_indices) in enumerate(ALL_REGIONS.items()):
        points = [landmarks_avg[i] for i in region_indices
                  if i < len(landmarks_avg)]
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            color = colors[idx % len(colors)]
            cv2.fillConvexPoly(overlay, pts, color)
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)
            cv2.putText(canvas, str(idx + 1), (cx + 5, cy + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main figure generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_methodology_figures(video_dir, output_dir, n_videos=10):
    """
    Generate the 5 methodology panels by averaging landmarks across N videos.

    Args:
        video_dir  : directory containing .MOV / .mp4 / .avi files
        output_dir : where to save the generated figures
        n_videos   : how many videos to use for landmark averaging (default 10)
    """
    video_dir = Path(video_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Collect video paths
    exts = ('*.MOV', '*.mov', '*.mp4', '*.MP4', '*.avi', '*.AVI')
    video_paths = []
    for ext in exts:
        video_paths.extend(sorted(video_dir.rglob(ext)))
    video_paths = video_paths[:n_videos]

    if not video_paths:
        print(f"No videos found in {video_dir}")
        return

    print(f"Averaging landmarks across {len(video_paths)} video(s)…")
    landmarks_avg, best_frame, h, w = collect_averaged_landmarks(video_paths)

    if landmarks_avg is None:
        print("No face detected in any video.")
        return

    print(f"  Frame size: {w}×{h}, landmarks from {len(video_paths)} video(s)")

    # Panel 1 — Raw dataset frame
    cv2.imwrite(os.path.join(output_dir, "fig_1_dataset.jpg"), best_frame)
    print("  ✓ fig_1_dataset.jpg")

    # Panel 2 — Face mesh (averaged landmarks)
    mesh_img = _draw_facemesh_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_2_facemesh.jpg"), mesh_img)
    print("  ✓ fig_2_facemesh.jpg")

    # Panel 3 — All ROI candidates
    roi_img = _draw_roi_candidates_panel(mesh_img, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_3_roi_candidates.jpg"), roi_img)
    print("  ✓ fig_3_roi_candidates.jpg")

    # Panel 4 — Selected ROI (yellow)
    select_img = _draw_select_roi_panel(mesh_img, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_4_select_roi.jpg"), select_img)
    print("  ✓ fig_4_select_roi.jpg")

    # Panel 5 — All 31 numbered landmarks
    all31_img = _draw_all31_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_5_all_31_landmarks.jpg"), all31_img)
    print("  ✓ fig_5_all_31_landmarks.jpg")

    # Composite methodology figure
    create_composite_methodology(output_dir)

    print(f"\nAll figures saved to: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Composite methodology figure
# ──────────────────────────────────────────────────────────────────────────────

def create_composite_methodology(output_dir):
    """
    Stitch the 4 pipeline panels into a single diagram matching the reference:

        [Dataset] → [Face Mesh] → [ROI Candidates] → [Select ROI]
                                                            ↓
                               [Dataset Sources]   [Method] → [Evaluation]
                                  UBFC                POS        MAE
                                  LGI-PPGI           CHROM       RMSE
                                                      ICA          PCC
                                                      SSR          SNR
                                                      ...
    """
    name_map = {
        1: "fig_1_dataset.jpg",
        2: "fig_2_facemesh.jpg",
        3: "fig_3_roi_candidates.jpg",
        4: "fig_4_select_roi.jpg",
    }

    panels = []
    for i in range(1, 5):
        p = cv2.imread(os.path.join(output_dir, name_map[i]))
        if p is not None:
            panels.append(p)

    if len(panels) < 4:
        print(f"Missing panels for composite ({len(panels)}/4 found). "
              "Run generate_methodology_figures first.")
        return

    padding = 40
    img_h, img_w = panels[0].shape[:2]
    scale = 0.5
    s_w, s_h = int(img_w * scale), int(img_h * scale)

    # Canvas: top row (4 images) + bottom row (logic boxes)
    bottom_h = 260
    canvas_w = (s_w * 4) + (padding * 5)
    canvas_h = s_h + padding * 3 + 40 + bottom_h  # 40 for title bar
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    titles = ["Dataset", "Generate\nFace Mesh",
              "Generate ROI\nCandidates", "Select ROI"]
    title_h = 40  # pixels reserved above each image for the title box

    # ── Top Row: 4 panels with arrows ──────────────────────────────────────
    for i, panel in enumerate(panels):
        resized = cv2.resize(panel, (s_w, s_h))
        x = padding + i * (s_w + padding)
        y = padding + title_h

        # Title box
        cv2.rectangle(canvas, (x, padding), (x + s_w, padding + title_h),
                      (0, 0, 0), 2)
        # Split multi-line title
        lines = titles[i].split('\n')
        line_y = padding + 14
        for line in lines:
            cv2.putText(canvas, line, (x + 8, line_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
            line_y += 18

        # Image
        canvas[y:y + s_h, x:x + s_w] = resized

        # Arrow to next panel
        if i < 3:
            ax_start = (x + s_w, y + s_h // 2)
            ax_end = (x + s_w + padding, y + s_h // 2)
            cv2.arrowedLine(canvas, ax_start, ax_end, (0, 0, 0), 2,
                            tipLength=0.3)

    # ── Bottom Section: logic flow ──────────────────────────────────────────
    top_img_bot = padding + title_h + s_h   # y where images end
    flow_y = top_img_bot + padding          # top of bottom logic area

    # Vertical line down from "Select ROI" centre
    select_cx = padding + 3 * (s_w + padding) + s_w // 2
    v_line_bot = flow_y + 40
    cv2.line(canvas, (select_cx, top_img_bot), (select_cx, v_line_bot),
             (0, 0, 0), 2)

    # ── Dataset sub-box (bottom-left) ──
    ds_x, ds_y = padding, flow_y
    ds_w, ds_h = 140, 60
    cv2.rectangle(canvas, (ds_x, ds_y), (ds_x + ds_w, ds_y + ds_h),
                  (0, 0, 0), 2)
    cv2.putText(canvas, "Dataset", (ds_x + 30, ds_y + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
    datasets = ["UBFC", "LGI-PPGI"]
    for j, d in enumerate(datasets):
        bx, by = ds_x + 20, ds_y + ds_h + 10 + j * 35
        cv2.rectangle(canvas, (bx, by), (bx + ds_w - 20, by + 28),
                      (0, 0, 0), 1)
        cv2.putText(canvas, d, (bx + 10, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.line(canvas, (ds_x + 10, ds_y + ds_h),
                 (ds_x + 10, by + 14), (0, 0, 0), 1)
        cv2.line(canvas, (ds_x + 10, by + 14), (bx, by + 14), (0, 0, 0), 1)

    # Horizontal line from Select ROI down to Method box centre
    method_cx = canvas_w // 2 - 60
    cv2.line(canvas, (select_cx, v_line_bot),
             (method_cx + 75, v_line_bot), (0, 0, 0), 2)
    cv2.arrowedLine(canvas, (method_cx + 75, v_line_bot),
                    (method_cx + 75, v_line_bot + 20),
                    (0, 0, 0), 2, tipLength=0.4)

    # ── Method box ──
    m_x = method_cx
    m_y = v_line_bot + 20
    m_w, m_h = 150, 50
    cv2.rectangle(canvas, (m_x, m_y), (m_x + m_w, m_y + m_h),
                  (0, 0, 0), 2)
    cv2.putText(canvas, "Method", (m_x + 30, m_y + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    # Arrow Method → Evaluation
    cv2.arrowedLine(canvas,
                    (m_x + m_w, m_y + m_h // 2),
                    (m_x + m_w + 50, m_y + m_h // 2),
                    (0, 0, 0), 2, tipLength=0.3)

    # ── Evaluation box ──
    e_x = m_x + m_w + 50
    e_y = m_y
    e_w, e_h = 150, 50
    cv2.rectangle(canvas, (e_x, e_y), (e_x + e_w, e_y + e_h),
                  (0, 0, 0), 2)
    cv2.putText(canvas, "Evaluation", (e_x + 10, e_y + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

    # ── Method sub-boxes ──
    methods = ["POS", "CHROM", "ICA", "SSR", "PBV", "LGI", "SAMC", "2SR", "OMIT"]
    for j, method in enumerate(methods):
        bx = m_x + 20
        by = m_y + m_h + 10 + j * 28
        if by + 22 > canvas_h:
            break
        cv2.rectangle(canvas, (bx, by), (bx + m_w - 20, by + 22),
                      (0, 0, 0), 1)
        cv2.putText(canvas, method, (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.line(canvas, (m_x + 10, m_y + m_h),
                 (m_x + 10, by + 11), (0, 0, 0), 1)
        cv2.line(canvas, (m_x + 10, by + 11), (bx, by + 11), (0, 0, 0), 1)

    # ── Evaluation sub-boxes ──
    metrics = ["MAE", "RMSE", "PCC", "SNR", "NSQI", "rBS"]
    for j, met in enumerate(metrics):
        bx = e_x + 20
        by = e_y + e_h + 10 + j * 28
        if by + 22 > canvas_h:
            break
        cv2.rectangle(canvas, (bx, by), (bx + e_w - 20, by + 22),
                      (0, 0, 0), 1)
        cv2.putText(canvas, met, (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.line(canvas, (e_x + 10, e_y + e_h),
                 (e_x + 10, by + 11), (0, 0, 0), 1)
        cv2.line(canvas, (e_x + 10, by + 11), (bx, by + 11), (0, 0, 0), 1)

    output_path = os.path.join(output_dir, "methodology_overall_pictorial.png")
    cv2.imwrite(output_path, canvas)
    print(f"  ✓ Composite methodology figure → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate methodology figures for the rPPG / SpO2 paper.")
    parser.add_argument(
        "--video-dir", default="./data",
        help="Directory containing video files (default: ./data)")
    parser.add_argument(
        "--output-dir", default="./spo2/results",
        help="Output directory for figures (default: ./spo2/results)")
    parser.add_argument(
        "--n-videos", type=int, default=10,
        help="Number of videos to average landmarks over (default: 10)")
    args = parser.parse_args()

    generate_methodology_figures(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        n_videos=args.n_videos,
    )
