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
# Multi-video landmark averaging  (window-aware, multi-frame per video)
# ──────────────────────────────────────────────────────────────────────────────

def collect_averaged_landmarks(
    video_paths,
    window_start: float = 30.0,
    window_duration: float = 60.0,
    frames_per_video: int = 8,
):
    """
    For each video, sample `frames_per_video` frames spread evenly across the
    analysis window [window_start, window_start+window_duration].  Run MediaPipe
    on every sampled frame and return the per-landmark centroid averaged across
    all successful detections.

    Returns:
        landmarks_avg : list of (x_pixel, y_pixel) indexed by landmark index
        best_frame    : BGR frame with the highest median landmark visibility
        h, w          : frame dimensions
    """
    all_landmark_coords = []
    best_frame = None
    best_conf = -1.0

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Define the window in frame indices
            start_f = int(window_start * fps)
            end_f   = min(start_f + int(window_duration * fps), total_frames - 1)

            # If video is too short, fall back to first 30 frames
            if end_f <= start_f or start_f >= total_frames:
                start_f = 0
                end_f   = min(int(10 * fps), total_frames - 1)

            # Sample frames evenly across the window
            sample_indices = np.linspace(start_f, end_f,
                                         num=frames_per_video, dtype=int)

            for fi in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue

                lms = results.multi_face_landmarks[0].landmark
                coords = {i: (lm.x, lm.y) for i, lm in enumerate(lms)}
                all_landmark_coords.append(coords)

                vis = np.median([lm.visibility for lm in lms
                                 if hasattr(lm, 'visibility')])
                if vis > best_conf:
                    best_conf = vis
                    best_frame = frame.copy()

            cap.release()

    if not all_landmark_coords:
        return None, None, None, None

    n_landmarks = 478  # MediaPipe refine = 478 landmarks
    avg_x = np.zeros(n_landmarks)
    avg_y = np.zeros(n_landmarks)
    count = np.zeros(n_landmarks)

    for coords in all_landmark_coords:
        for idx, (x, y) in coords.items():
            avg_x[idx] += x
            avg_y[idx] += y
            count[idx] += 1

    count[count == 0] = 1
    avg_x /= count
    avg_y /= count

    h, w, _ = best_frame.shape
    landmarks_avg = [(min(int(avg_x[i] * w), w - 1),
                      min(int(avg_y[i] * h), h - 1))
                     for i in range(n_landmarks)]

    return landmarks_avg, best_frame, h, w


# ──────────────────────────────────────────────────────────────────────────────
# Panel helpers  (use pre-averaged landmarks)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_facemesh_panel(frame, landmarks_avg):
    """Panel 2: tiny red dots at every averaged landmark."""
    img = frame.copy()
    for (lx, ly) in landmarks_avg:
        cv2.circle(img, (lx, ly), 2, (0, 0, 220), -1)
    return img


def _draw_roi_candidates_panel(frame, landmarks_avg):
    """Panel 3: semi-transparent white polygons for all 31 ROI candidates."""
    img = frame.copy()
    overlay = img.copy()
    for name, region_indices in ALL_REGIONS.items():
        pts = [landmarks_avg[i] for i in region_indices if i < len(landmarks_avg)]
        if len(pts) >= 3:
            ordered = [_order_pts_clockwise(pts)]
            cv2.fillPoly(overlay, ordered, (230, 230, 230))
            cv2.polylines(overlay, ordered, isClosed=True,
                          color=(40, 40, 200), thickness=1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    return img


def _draw_select_roi_panel(frame, landmarks_avg):
    """Panel 4: selected top ROI regions in solid yellow with outline."""
    img = frame.copy()
    overlay = img.copy()
    top_regions = [
        'upper_medial_forehead',
        'lower_medial_forehead',
        'right_malar',
        'left_malar',
    ]
    for name in top_regions:
        if name not in ALL_REGIONS:
            continue
        pts = [landmarks_avg[i] for i in ALL_REGIONS[name] if i < len(landmarks_avg)]
        if len(pts) >= 3:
            ordered = [_order_pts_clockwise(pts)]
            cv2.fillPoly(overlay, ordered, (0, 215, 255))
            cv2.polylines(overlay, ordered, isClosed=True,
                          color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    return img


def _draw_all31_panel(frame, landmarks_avg):
    """Panel 5: all 31 ROI polygons colour-coded with region index labels."""
    canvas = frame.copy()
    overlay = frame.copy()
    palette = [
        (220, 50,  50), (50, 180,  50), (50,  50, 220), (200, 180,  0),
        (180,  0, 180), ( 0, 180, 180), (160,  80,   0), ( 0, 140,  80),
        ( 80,   0, 160), (140, 140,   0), (140,   0, 100), ( 0, 100, 140),
    ]
    for idx, (name, region_indices) in enumerate(ALL_REGIONS.items()):
        pts = [landmarks_avg[i] for i in region_indices if i < len(landmarks_avg)]
        if len(pts) >= 3:
            ordered = [_order_pts_clockwise(pts)]
            color = palette[idx % len(palette)]
            cv2.fillPoly(overlay, ordered, color)
            cv2.polylines(overlay, ordered, isClosed=True,
                          color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(canvas, str(idx + 1), (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main figure generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_methodology_figures(
    video_dir,
    output_dir,
    participant_id: str = "",
    n_videos: int = 10,
    window_start: float = 30.0,
    window_duration: float = 60.0,
):
    """
    Generate the 5 methodology panels by averaging landmarks across N videos.

    Args:
        video_dir       : directory containing .MOV / .mp4 / .avi files
        output_dir      : where to save the generated figures
        participant_id  : if non-empty, only use videos whose path contains this string
        n_videos        : max number of videos to use for landmark averaging
        window_start    : analysis window start in seconds (default 30)
        window_duration : analysis window length in seconds (default 60)
    """
    video_dir = Path(video_dir)
    os.makedirs(output_dir, exist_ok=True)

    exts = ('*.MOV', '*.mov', '*.mp4', '*.MP4', '*.avi', '*.AVI')
    video_paths = []
    for ext in exts:
        video_paths.extend(sorted(video_dir.rglob(ext)))

    if participant_id:
        video_paths = [p for p in video_paths if participant_id in p.name]
        print(f"Filtered to {len(video_paths)} video(s) for participant '{participant_id}'")

    video_paths = video_paths[:n_videos]

    if not video_paths:
        print(f"No videos found in {video_dir}" +
              (f" matching '{participant_id}'" if participant_id else ""))
        return

    print(f"Averaging landmarks across {len(video_paths)} video(s) "
          f"(window {window_start:.0f}–{window_start+window_duration:.0f}s)…")
    landmarks_avg, best_frame, h, w = collect_averaged_landmarks(
        video_paths,
        window_start=window_start,
        window_duration=window_duration,
    )

    if landmarks_avg is None:
        print("No face detected in any video.")
        return

    print(f"  Frame size: {w}×{h}, landmarks averaged from {len(video_paths)} video(s)")

    # Panel 1 — Raw dataset frame
    cv2.imwrite(os.path.join(output_dir, "fig_1_dataset.png"), best_frame)
    print("  ✓ fig_1_dataset.png")

    # Panel 2 — Face mesh (averaged landmarks)
    mesh_img = _draw_facemesh_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_2_facemesh.png"), mesh_img)
    print("  ✓ fig_2_facemesh.png")

    # Panel 3 — All ROI candidates
    roi_img = _draw_roi_candidates_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_3_roi_candidates.png"), roi_img)
    print("  ✓ fig_3_roi_candidates.png")

    # Panel 4 — Selected ROI (yellow)
    select_img = _draw_select_roi_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_4_select_roi.png"), select_img)
    print("  ✓ fig_4_select_roi.png")

    # Panel 5 — All 31 numbered landmarks
    all31_img = _draw_all31_panel(best_frame, landmarks_avg)
    cv2.imwrite(os.path.join(output_dir, "fig_5_all_31_landmarks.png"), all31_img)
    print("  ✓ fig_5_all_31_landmarks.png")

    create_composite_methodology(output_dir)
    print(f"\nAll figures saved to: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Composite methodology figure
# ──────────────────────────────────────────────────────────────────────────────

def create_composite_methodology(output_dir):
    """
    Stitch the 4 pipeline panels into a single diagram:

        [Dataset] → [Face Mesh] → [ROI Candidates] → [Select ROI]
                                                            ↓
                           [Dataset Sources]   [Method] → [Evaluation]
                             ToHealth           POS          MAE
                             Repeat_Set        CHROM         RMSE
                                                ICA           PCC
                                                ...           SNR
    """
    name_map = {
        1: "fig_1_dataset.png",
        2: "fig_2_facemesh.png",
        3: "fig_3_roi_candidates.png",
        4: "fig_4_select_roi.png",
    }

    panels = []
    for i in range(1, 5):
        p = cv2.imread(os.path.join(output_dir, name_map[i]))
        if p is not None:
            panels.append(p)

    if len(panels) < 4:
        print(f"Missing panels for composite ({len(panels)}/4 found).")
        return

    padding = 40
    img_h, img_w = panels[0].shape[:2]
    scale = 0.5
    s_w, s_h = int(img_w * scale), int(img_h * scale)

    bottom_h = 280
    title_h = 40
    canvas_w = (s_w * 4) + (padding * 5)
    canvas_h = s_h + padding * 3 + title_h + bottom_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    titles = ["Dataset", "Generate\nFace Mesh",
              "Generate ROI\nCandidates", "Select ROI"]

    # ── Top Row: 4 panels with arrows ──────────────────────────────────────────
    for i, panel in enumerate(panels):
        resized = cv2.resize(panel, (s_w, s_h))
        x = padding + i * (s_w + padding)
        y = padding + title_h

        # Title box
        cv2.rectangle(canvas, (x, padding), (x + s_w, padding + title_h),
                      (30, 30, 30), 2)
        for li, line in enumerate(titles[i].split('\n')):
            cv2.putText(canvas, line, (x + 8, padding + 15 + li * 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (30, 30, 30), 1,
                        cv2.LINE_AA)

        canvas[y:y + s_h, x:x + s_w] = resized

        # Arrow to next panel
        if i < 3:
            mid_y = y + s_h // 2
            cv2.arrowedLine(canvas,
                            (x + s_w + 4, mid_y),
                            (x + s_w + padding - 4, mid_y),
                            (30, 30, 30), 2, tipLength=0.35, line_type=cv2.LINE_AA)

    # ── Bottom Section ─────────────────────────────────────────────────────────
    top_img_bot = padding + title_h + s_h
    flow_y = top_img_bot + padding

    # Vertical arrow down from "Select ROI" centre
    select_cx = padding + 3 * (s_w + padding) + s_w // 2
    v_line_bot = flow_y + 40
    cv2.arrowedLine(canvas, (select_cx, top_img_bot + 4),
                    (select_cx, v_line_bot), (30, 30, 30), 2,
                    tipLength=0.2, line_type=cv2.LINE_AA)

    # ── Dataset sub-box (bottom-left) ──────────────────────────────────────────
    ds_x, ds_y = padding, flow_y
    ds_w, ds_h = 150, 55
    cv2.rectangle(canvas, (ds_x, ds_y), (ds_x + ds_w, ds_y + ds_h),
                  (30, 30, 30), 2)
    cv2.putText(canvas, "Dataset", (ds_x + 28, ds_y + 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    datasets = ["ToHealth", "Repeat_Set"]
    for j, d in enumerate(datasets):
        bx, by = ds_x + 15, ds_y + ds_h + 12 + j * 36
        cv2.rectangle(canvas, (bx, by), (bx + ds_w - 15, by + 28),
                      (80, 80, 80), 1)
        cv2.putText(canvas, d, (bx + 10, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        mid_bx = ds_x + 12
        cv2.line(canvas, (mid_bx, ds_y + ds_h), (mid_bx, by + 14),
                 (80, 80, 80), 1)
        cv2.line(canvas, (mid_bx, by + 14), (bx, by + 14), (80, 80, 80), 1)

    # Horizontal arrow from Select ROI column to Method box
    method_cx = canvas_w // 2 - 70
    cv2.line(canvas, (select_cx, v_line_bot),
             (method_cx + 75, v_line_bot), (30, 30, 30), 2)
    cv2.arrowedLine(canvas, (method_cx + 75, v_line_bot),
                    (method_cx + 75, v_line_bot + 20),
                    (30, 30, 30), 2, tipLength=0.4, line_type=cv2.LINE_AA)

    # ── Method box ────────────────────────────────────────────────────────────
    m_x = method_cx
    m_y = v_line_bot + 20
    m_w, m_h = 150, 50
    cv2.rectangle(canvas, (m_x, m_y), (m_x + m_w, m_y + m_h),
                  (30, 30, 30), 2)
    cv2.putText(canvas, "Method", (m_x + 30, m_y + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    # Arrow Method → Evaluation
    cv2.arrowedLine(canvas,
                    (m_x + m_w + 4, m_y + m_h // 2),
                    (m_x + m_w + 50, m_y + m_h // 2),
                    (30, 30, 30), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # ── Evaluation box ────────────────────────────────────────────────────────
    e_x = m_x + m_w + 50
    e_y = m_y
    e_w, e_h = 150, 50
    cv2.rectangle(canvas, (e_x, e_y), (e_x + e_w, e_y + e_h),
                  (30, 30, 30), 2)
    cv2.putText(canvas, "Evaluation", (e_x + 10, e_y + 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)

    # ── Method sub-boxes ─────────────────────────────────────────────────────
    methods = ["GREEN", "POS", "CHROM", "ICA", "SSR", "PBV",
               "LGI", "SAMC", "2SR", "OMIT", "PCA"]
    for j, method in enumerate(methods):
        bx = m_x + 15
        by = m_y + m_h + 10 + j * 26
        if by + 22 > canvas_h:
            break
        cv2.rectangle(canvas, (bx, by), (bx + m_w - 15, by + 22),
                      (80, 80, 80), 1)
        cv2.putText(canvas, method, (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
        mid_mx = m_x + 12
        cv2.line(canvas, (mid_mx, m_y + m_h), (mid_mx, by + 11), (80, 80, 80), 1)
        cv2.line(canvas, (mid_mx, by + 11), (bx, by + 11), (80, 80, 80), 1)

    # ── Evaluation sub-boxes ─────────────────────────────────────────────────
    metrics = ["MAE", "RMSE", "PCC", "SNR", "NSQI", "rBS"]
    for j, met in enumerate(metrics):
        bx = e_x + 15
        by = e_y + e_h + 10 + j * 28
        if by + 22 > canvas_h:
            break
        cv2.rectangle(canvas, (bx, by), (bx + e_w - 15, by + 22),
                      (80, 80, 80), 1)
        cv2.putText(canvas, met, (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
        mid_ex = e_x + 12
        cv2.line(canvas, (mid_ex, e_y + e_h), (mid_ex, by + 11), (80, 80, 80), 1)
        cv2.line(canvas, (mid_ex, by + 11), (bx, by + 11), (80, 80, 80), 1)

    out_path = os.path.join(output_dir, "methodology_overall_pictorial.png")
    cv2.imwrite(out_path, canvas)
    print(f"  ✓ Composite methodology figure → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Landmark figures: all-31 and top/bottom-5 highlight
# ──────────────────────────────────────────────────────────────────────────────

# Top-5 and Bottom-5 regions by mean SNR from full 1300-video dataset
TOP5_REGIONS = [
    'right_malar',
    'left_malar',
    'right_lower_cheek',
    'left_lower_cheek',
    'lower_medial_forehead',
]
BOT5_REGIONS = [
    'right_marionette_fold',
    'left_marionette_fold',
    'nasal_tip',
    'right_eye',
    'left_eye',
]

# Distinct color per region for the all-31 map
_PALETTE_31 = [
    (52,  152, 219), (46,  204, 113), (231,  76,  60), (241, 196,  15),
    (155,  89, 182), ( 26, 188, 156), (230, 126,  34), ( 52,  73,  94),
    (189, 195, 199), ( 39, 174,  96), (192,  57,  43), (243, 156,  18),
    (142,  68, 173), ( 22, 160, 133), (211,  84,   0), ( 44,  62,  80),
    (127, 140, 141), ( 41, 128, 185), ( 39, 174,  96), (192,  57,  43),
    (243, 156,  18), (142,  68, 173), ( 22, 160, 133), (211,  84,   0),
    (230, 126,  34), (155,  89, 182), ( 26, 188, 156), (52,  152, 219),
    (46,  204, 113), (231,  76,  60), (241, 196,  15),
]


def _get_landmarks_and_frame(video_paths, window_start=30.0, window_duration=60.0):
    """Shared helper: collect averaged landmarks + best frame from video list."""
    return collect_averaged_landmarks(
        video_paths,
        window_start=window_start,
        window_duration=window_duration,
        frames_per_video=8,
    )


def _order_pts_clockwise(pts):
    """
    Sort a set of 2-D points by angle from their centroid so fillPoly gets a
    proper (non-self-intersecting) winding order.  Works for any convex or
    mildly non-convex polygon defined as an unordered point set.
    """
    pts = np.array(pts, dtype=np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order  = np.argsort(angles)
    return pts[order].astype(np.int32)


def _crop_to_face(frame, landmarks_avg, pad_frac=0.18):
    """
    Crop the frame to a bounding box around all landmark points + padding.
    Returns (cropped_frame, x_offset, y_offset).
    """
    h, w = frame.shape[:2]
    xs = [p[0] for p in landmarks_avg]
    ys = [p[1] for p in landmarks_avg]
    pad_x = int((max(xs) - min(xs)) * pad_frac)
    pad_y = int((max(ys) - min(ys)) * pad_frac)
    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(w, max(xs) + pad_x)
    y2 = min(h, max(ys) + pad_y)
    return frame[y1:y2, x1:x2], x1, y1


def _shift_landmarks(landmarks_avg, x_off, y_off):
    """Shift pixel landmark coords after a crop."""
    return [(x - x_off, y - y_off) for x, y in landmarks_avg]


def generate_all31_figure(video_paths, output_path,
                          window_start=30.0, window_duration=60.0):
    """
    High-quality figure: all 31 ROIs colour-coded with region index labels.
    Crops tightly to the face, uses fillPoly for accurate polygon shapes.
    """
    landmarks_avg, best_frame, h, w = _get_landmarks_and_frame(
        video_paths, window_start, window_duration)
    if landmarks_avg is None:
        print("No face detected.")
        return

    # Crop tightly to face then upscale for quality
    face_crop, x_off, y_off = _crop_to_face(best_frame, landmarks_avg)
    lm_crop = _shift_landmarks(landmarks_avg, x_off, y_off)

    # Upscale the crop to at least 1200px on the short edge
    ch, cw = face_crop.shape[:2]
    short_edge = min(ch, cw)
    if short_edge < 1200:
        scale = 1200 / short_edge
        face_crop = cv2.resize(face_crop, (int(cw * scale), int(ch * scale)),
                               interpolation=cv2.INTER_LANCZOS4)
        lm_crop = [(int(x * scale), int(y * scale)) for x, y in lm_crop]

    fh, fw = face_crop.shape[:2]
    canvas = face_crop.copy()
    overlay = face_crop.copy()

    outline_t = max(2, fh // 300)

    for idx, (name, region_indices) in enumerate(ALL_REGIONS.items()):
        pts = [lm_crop[i] for i in region_indices if i < len(lm_crop)]
        if len(pts) < 3:
            continue
        ordered = _order_pts_clockwise(pts)
        pts_arr = [ordered]
        color = _PALETTE_31[idx % len(_PALETTE_31)]
        cv2.fillPoly(overlay, pts_arr, color)
        cv2.polylines(overlay, pts_arr, isClosed=True,
                      color=(255, 255, 255), thickness=outline_t,
                      lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    # Region number labels after blend
    font_scale = max(0.6, fh / 1000)
    thickness  = max(1, fh // 500)
    for idx, (name, region_indices) in enumerate(ALL_REGIONS.items()):
        pts = [lm_crop[i] for i in region_indices if i < len(lm_crop)]
        if len(pts) < 3:
            continue
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        label = str(idx + 1)
        cv2.putText(canvas, label, (cx + 2, cy + 2),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"  ✓ {Path(output_path).name}  ({fw}×{fh})")


def generate_top_bot5_figure(video_paths, output_path,
                             window_start=30.0, window_duration=60.0):
    """
    Two-panel figure side by side (face-cropped):
      Left  — Top-5 regions (green / optimal)
      Right — Bottom-5 regions (red / noisy)
    """
    landmarks_avg, best_frame, h, w = _get_landmarks_and_frame(
        video_paths, window_start, window_duration)
    if landmarks_avg is None:
        print("No face detected.")
        return

    face_crop, x_off, y_off = _crop_to_face(best_frame, landmarks_avg)
    lm_crop = _shift_landmarks(landmarks_avg, x_off, y_off)

    ch, cw = face_crop.shape[:2]
    short_edge = min(ch, cw)
    if short_edge < 1200:
        scale = 1200 / short_edge
        face_crop = cv2.resize(face_crop, (int(cw * scale), int(ch * scale)),
                               interpolation=cv2.INTER_LANCZOS4)
        lm_crop = [(int(x * scale), int(y * scale)) for x, y in lm_crop]

    fh, fw = face_crop.shape[:2]

    TOP_COLORS = [
        (34, 197, 80), (22, 163, 60), (74, 222, 128),
        (16, 185, 129), (52, 211, 100),
    ]
    BOT_COLORS = [
        (239, 68,  68), (220, 38,  38), (248, 113, 113),
        (185, 28,  28), (252, 165, 165),
    ]

    outline_t  = max(3, fh // 250)
    font_scale = max(0.8, fh / 900)
    font_t     = max(2, fh // 450)

    panels = []
    for group_regions, colors, label_text in [
        (TOP5_REGIONS, TOP_COLORS, "Top-5 Regions (Optimal SNR)"),
        (BOT5_REGIONS, BOT_COLORS, "Bottom-5 Regions (Noisy)"),
    ]:
        base = face_crop.copy()

        # Darken non-highlighted regions
        dark_mask = np.zeros((fh, fw), dtype=np.uint8)
        for name, region_indices in ALL_REGIONS.items():
            if name in group_regions:
                continue
            pts = [lm_crop[i] for i in region_indices if i < len(lm_crop)]
            if len(pts) >= 3:
                cv2.fillPoly(dark_mask,
                             [_order_pts_clockwise(pts)], 200)
        base[dark_mask > 0] = (base[dark_mask > 0] * 0.35).astype(np.uint8)

        overlay = base.copy()

        # Highlighted regions with fillPoly (angle-sorted for correct winding)
        for i, name in enumerate(group_regions):
            if name not in ALL_REGIONS:
                continue
            pts = [lm_crop[j] for j in ALL_REGIONS[name] if j < len(lm_crop)]
            if len(pts) < 3:
                continue
            ordered = _order_pts_clockwise(pts)
            pts_arr = [ordered]
            color = colors[i % len(colors)]
            cv2.fillPoly(overlay, pts_arr, color)
            cv2.polylines(overlay, pts_arr, isClosed=True,
                          color=(255, 255, 255), thickness=outline_t,
                          lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.65, base, 0.35, 0, base)

        # Numbered labels at region centroids
        for i, name in enumerate(group_regions):
            if name not in ALL_REGIONS:
                continue
            pts = [lm_crop[j] for j in ALL_REGIONS[name] if j < len(lm_crop)]
            if len(pts) < 3:
                continue
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            label = str(i + 1)
            cv2.putText(base, label, (cx + 2, cy + 2),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        (0, 0, 0), font_t + 2, cv2.LINE_AA)
            cv2.putText(base, label, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        (255, 255, 255), font_t, cv2.LINE_AA)

        # Title bar
        is_top = "Optimal" in label_text
        bar_h = max(70, fh // 15)
        bar = np.zeros((bar_h, fw, 3), dtype=np.uint8)
        bar[:] = (34, 139, 34) if is_top else (180, 30, 30)
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX,
                                 font_scale * 1.0, font_t + 1)[0]
        tx = (fw - t_size[0]) // 2
        ty = (bar_h + t_size[1]) // 2
        cv2.putText(bar, label_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale * 1.0,
                    (255, 255, 255), font_t + 1, cv2.LINE_AA)

        # Legend bar at bottom — one row per region
        row_h   = max(55, fh // 18)
        leg_h   = row_h * len(group_regions) + 20
        legend  = np.ones((leg_h, fw, 3), dtype=np.uint8) * 245
        swatch  = row_h - 14
        for i, name in enumerate(group_regions):
            color = colors[i % len(colors)]
            ly = 10 + i * row_h
            cv2.rectangle(legend, (14, ly), (14 + swatch, ly + swatch), color, -1)
            cv2.rectangle(legend, (14, ly), (14 + swatch, ly + swatch), (80, 80, 80), 1)
            label_str = f"{i+1}.  {name.replace('_', ' ').title()}"
            cv2.putText(legend, label_str,
                        (14 + swatch + 16, ly + swatch - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65,
                        (30, 30, 30), font_t, cv2.LINE_AA)

        panel = np.vstack([bar, base, legend])
        panels.append(panel)

    # Equalise heights
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        gap = max_h - p.shape[0]
        if gap > 0:
            pad = np.ones((gap, p.shape[1], 3), dtype=np.uint8) * 245
            p = np.vstack([p, pad])
        padded.append(p)

    divider = np.ones((max_h, 8, 3), dtype=np.uint8) * 180
    composite = np.hstack([padded[0], divider, padded[1]])
    cv2.imwrite(str(output_path), composite, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f"  ✓ {Path(output_path).name}  ({composite.shape[1]}×{composite.shape[0]})")


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
        "--participant-id", default="",
        help="Filter videos to files whose name contains this string.")
    parser.add_argument(
        "--n-videos", type=int, default=10,
        help="Max number of videos to use for landmark averaging (default: 10)")
    parser.add_argument(
        "--window-start", type=float, default=30.0,
        help="Analysis window start in seconds (default: 30)")
    parser.add_argument(
        "--window-duration", type=float, default=60.0,
        help="Analysis window duration in seconds (default: 60)")
    parser.add_argument(
        "--landmark-figures", action="store_true",
        help="Also generate the all-31 and top/bottom-5 landmark figures.")
    args = parser.parse_args()

    # Gather video paths
    vdir = Path(args.video_dir)
    exts = ('*.MOV', '*.mov', '*.mp4', '*.MP4', '*.avi', '*.AVI')
    vpaths = []
    for ext in exts:
        vpaths.extend(sorted(vdir.rglob(ext)))
    if args.participant_id:
        vpaths = [p for p in vpaths if args.participant_id in p.name]
    vpaths = vpaths[:args.n_videos]

    os.makedirs(args.output_dir, exist_ok=True)

    generate_methodology_figures(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        participant_id=args.participant_id,
        n_videos=args.n_videos,
        window_start=args.window_start,
        window_duration=args.window_duration,
    )

    if args.landmark_figures and vpaths:
        print("\nGenerating landmark figures…")
        generate_all31_figure(
            vpaths,
            os.path.join(args.output_dir, "fig_landmark_all31.png"),
            window_start=args.window_start,
            window_duration=args.window_duration,
        )
        generate_top_bot5_figure(
            vpaths,
            os.path.join(args.output_dir, "fig_landmark_top5_bot5.png"),
            window_start=args.window_start,
            window_duration=args.window_duration,
        )
