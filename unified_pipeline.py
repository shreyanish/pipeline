#!/usr/bin/env python3
"""
unified_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Single entry point for the full rPPG + SpO2 research pipeline.

Outputs (all in --output-dir):
  1. rppg_results.csv          — 1 row per (video × ROI × algorithm)
                                 blind quality metrics: SNR, NSQI, SE, Variance, ZCR
  2. spo2_classical_results.csv — 1 row per (video × ROI × algorithm)
                                 SpO2 estimate via spectral ratio-of-ratios + MAE/RMSE/MSE/R²
  3. spo2_dl_results.csv        — 1 row per (video × DL model)
                                 SpO2 estimate from trained DeepPhys / EfficientPhys / Contrastive

Architecture
────────────
  Step 1 [CPU workers] — MediaPipe signal extraction, all 31 ROIs, all videos in parallel
  Step 2 [GPU]         — 11 classical rPPG algorithms (rppg_pytorch)
  Step 3 [GPU]         — Spectral ratio-of-ratios SpO2 for all (ROI × algo) combos
  Step 4 [GPU]         — DL model training (70/15/15 participant split) + inference
  
Usage
─────
  # Run full pipeline (classical + DL) on GPU:
  python unified_pipeline.py \\
      --video-dir  /path/to/videos \\
      --gt-dir     ./ground-truth \\
      --output-dir ./results \\
      --max-frames 1800 \\
      --workers    16 \\
      --device     cuda \\
      --dl-epochs  30

  # Classical only (skip DL training):
  python unified_pipeline.py \\
      --video-dir /path/to/videos --gt-dir ./ground-truth \\
      --output-dir ./results --skip-dl

  # Resume an interrupted run:
  python unified_pipeline.py ... --resume
"""

import os
import sys
import re
import time
import logging
import argparse
import multiprocessing
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch

# ── project root on path ──────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from config import RPPG_METHODS, FS_MIN, FS_MAX
from signal_extraction import get_all_regions_signals
from roi_definitions import ALL_REGIONS, SKIN_REGIONS, FACE_REGIONS
import rppg_pytorch as rppg

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler("unified_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Algorithm map ─────────────────────────────────────────────────────────────
ALGO_FN_MAP = {
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

# =============================================================================
# STEP 1: Ground Truth Loader + Smart Matcher
# =============================================================================

def load_ground_truth(gt_dir: str) -> dict:
    """Load all GT CSVs → {video_stem_lower: {spo2, hr, participant_id}}"""
    gt_dir = Path(gt_dir)
    gt = {}
    if not gt_dir.exists():
        logger.warning(f"GT directory not found: {gt_dir}")
        return gt

    for csv_path in sorted(gt_dir.glob("*.csv")):
        if csv_path.name.startswith("._"):
            continue
        try:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(csv_path, encoding="latin-1")

            df.columns = [c.strip() for c in df.columns]
            col_map = {
                "Video File Name": ["Video File Name", "Video Name", "File Name"],
                "Oxygen Level":    ["Oxygen Level", "Oxygen", "SpO2"],
                "Pulse ":          ["Pulse ", "Pulse", "Heart Rate", "HR"],
            }
            for standard, aliases in col_map.items():
                for alias in aliases:
                    if alias in df.columns and standard not in df.columns:
                        df = df.rename(columns={alias: standard})

            required = {"Video File Name", "Oxygen Level", "Pulse "}
            if not required.issubset(df.columns):
                continue

            for _, row in df.iterrows():
                vid = str(row["Video File Name"]).strip()
                if not vid or vid == "nan":
                    continue
                try:
                    spo2 = float(str(row["Oxygen Level"]).strip())
                    hr   = float(str(row["Pulse "]).strip())
                except Exception:
                    spo2, hr = float("nan"), float("nan")

                # Extract participant ID (e.g. P001)
                p_match = re.search(r"P\d{3}", vid, re.IGNORECASE)
                pid = p_match.group().upper() if p_match else None

                gt[vid.lower()] = {"spo2": spo2, "hr": hr, "raw_name": vid, "participant_id": pid}
        except Exception as e:
            logger.debug(f"Could not parse {csv_path.name}: {e}")

    logger.info(f"Loaded GT for {len(gt)} records.")
    return gt


def find_gt(video_stem: str, gt_lookup: dict) -> Optional[dict]:
    """Smart matcher: direct → participant ID → suffix fallback."""
    name = video_stem.lower()
    if name in gt_lookup:
        return gt_lookup[name]
    p_match = re.search(r"p\d{3}", name)
    if p_match:
        p_id = p_match.group().upper()
        year = re.search(r"20\d{2}", name)
        year_str = year.group() if year else "2024"
        for key, rec in gt_lookup.items():
            if p_id.lower() in key and year_str in key:
                return rec
    if len(name) >= 8:
        for key, rec in gt_lookup.items():
            if key.endswith(name[-8:]):
                return rec
    return None


# =============================================================================
# STEP 2: Signal Extraction Worker (CPU, parallelized)
# =============================================================================

def extract_signals_worker(task: tuple) -> tuple:
    """
    Worker function for multiprocessing pool.
    Returns (video_stem, all_signals_dict, fps) or (stem, None, 0) on failure.
    """
    video_path, max_frames = task
    stem = Path(video_path).stem
    try:
        all_signals, fps = get_all_regions_signals(
            video_path, ALL_REGIONS, max_frames=max_frames
        )
        return stem, all_signals, fps
    except Exception as e:
        logger.debug(f"Extraction failed for {stem}: {e}")
        return stem, None, 0


# =============================================================================
# STEP 3: Metrics Helpers
# =============================================================================

def bvp_blind_metrics(sig: np.ndarray, fps: float) -> dict:
    """Blind signal quality metrics (no GT needed)."""
    if len(sig) < 32:
        return {"SNR": np.nan, "NSQI": np.nan, "SE": np.nan, "Variance": np.nan, "ZCR": np.nan}

    f, pxx = welch(sig, fps, nperseg=min(len(sig), 256))
    mask = (f >= FS_MIN) & (f <= FS_MAX)
    snr, nsqi, se = np.nan, np.nan, np.nan

    if any(mask):
        peak_idx = np.argmax(pxx[mask])
        peak_f   = f[mask][peak_idx]
        pm       = (f >= peak_f - 0.1) & (f <= peak_f + 0.1)
        sig_p    = float(np.sum(pxx[pm]))
        noise_p  = float(np.sum(pxx[mask])) - sig_p
        snr      = float(10 * np.log10(sig_p / max(noise_p, 1e-10)))
        nsqi     = sig_p / float(np.sum(pxx[mask])) if np.sum(pxx[mask]) > 0 else np.nan
        norm_pxx = pxx[mask] / (np.sum(pxx[mask]) + 1e-10)
        se       = float(-np.sum(norm_pxx * np.log(norm_pxx + 1e-10)))

    centered = sig - np.mean(sig)
    zcr = float(np.sum(centered[:-1] * centered[1:] < 0)) / len(sig)

    return {"SNR": snr, "NSQI": nsqi, "SE": se,
            "Variance": float(np.var(sig)), "ZCR": zcr}


def compute_spo2_spectral(bvp: np.ndarray, raw_rgb: np.ndarray, fps: float) -> float:
    """Algorithm-dependent SpO2 via spectral ratio-of-ratios at detected HR frequency."""
    if len(bvp) < 64:
        return np.nan
    f, pxx = welch(bvp, fps, nperseg=min(len(bvp), 256))
    mask = (f >= FS_MIN) & (f <= FS_MAX)
    if not any(mask):
        return np.nan
    hr_freq = float(f[mask][np.argmax(pxx[mask])])

    def ac_at_freq(channel, freq):
        s = channel - np.mean(channel)
        n = len(s)
        f_ax = np.fft.rfftfreq(n, 1.0 / fps)
        fft_v = np.abs(np.fft.rfft(s))
        idx = int(np.argmin(np.abs(f_ax - freq)))
        return float(np.sum(fft_v[max(0, idx - 1):idx + 2]))

    min_len = min(len(bvp), len(raw_rgb))
    red   = raw_rgb[:min_len, 0].astype(float)
    green = raw_rgb[:min_len, 1].astype(float)
    dc_r, dc_g = float(np.mean(red)), float(np.mean(green))
    if dc_r < 1e-6 or dc_g < 1e-6:
        return np.nan

    ac_r = ac_at_freq(red, hr_freq)
    ac_g = ac_at_freq(green, hr_freq)
    if ac_g < 1e-9:
        return np.nan

    R    = (ac_r / dc_r) / (ac_g / dc_g)
    spo2 = 110.0 - 25.0 * R
    return float(np.clip(spo2, 70.0, 100.0))


def summary_metrics(estimated: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute MAE, RMSE, MSE, R² between arrays (NaN-safe)."""
    mask = ~(np.isnan(estimated) | np.isnan(ground_truth))
    e, g = estimated[mask], ground_truth[mask]
    if len(e) < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "MSE": np.nan, "R2": np.nan}
    errors = np.abs(e - g)
    mse    = float(np.mean((e - g) ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    r2     = float(1 - mse * len(g) / (ss_tot + 1e-10))
    return {
        "MAE":  float(np.mean(errors)),
        "RMSE": float(np.sqrt(mse)),
        "MSE":  mse,
        "R2":   r2,
    }


DL_ALGOS = ["GREEN", "ICA", "POS"]


def load_dl_algo_cache(cache_dir: Optional[str]) -> Dict[str, dict]:
    """Load per-algorithm BVP cache files. Returns {algo: {stem: array}}."""
    if not cache_dir:
        return {algo: {} for algo in DL_ALGOS}
    result = {}
    for algo in DL_ALGOS:
        path = Path(cache_dir) / f"{algo}.npz"
        if path.exists():
            try:
                data = np.load(path, allow_pickle=False)
                result[algo] = {k: data[k] for k in data.files}
            except Exception:
                result[algo] = {}
        else:
            result[algo] = {}
    cached = sum(len(v) for v in result.values()) // len(DL_ALGOS)
    if cached:
        logger.info(f"DL algo cache loaded: {cached} video(s) cached across {DL_ALGOS}")
    return result


def save_dl_algo_cache(cache_dir: str, dl_cache: Dict[str, dict]) -> None:
    """Save per-algorithm BVP cache files. Merges with any existing entries."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for algo, signals in dl_cache.items():
        if signals:
            np.savez_compressed(str(Path(cache_dir) / f"{algo}.npz"), **signals)
    logger.info(f"DL algo cache saved → {cache_dir} "
                f"({len(next(iter(dl_cache.values())))} video(s))")


def apply_window(signals: dict, fps: float,
                 start_s: float, duration_s: Optional[float]) -> dict:
    """Slice all ROI signals to [start_s, start_s + duration_s)."""
    start = int(start_s * fps)
    end   = (start + int(duration_s * fps)) if duration_s is not None else None
    return {
        roi: (sig[start:end] if sig is not None else None)
        for roi, sig in signals.items()
    }


def run_algo_on_signal(raw_rgb: np.ndarray, fps: float, algo_name: str, device: str) -> Optional[np.ndarray]:
    """Run one classical rPPG algorithm on a raw_rgb signal. Returns BVP or None."""
    fn = ALGO_FN_MAP.get(algo_name)
    if fn is None:
        return None
    try:
        if TORCH_AVAILABLE and device != "cpu":
            tensor = fn(raw_rgb, fps, device)
            if tensor is not None and len(tensor) > 0:
                return tensor.cpu().numpy()
    except Exception:
        pass
    try:
        tensor = fn(raw_rgb, fps, "cpu")
        if tensor is not None and len(tensor) > 0:
            return tensor.numpy()
    except Exception:
        pass
    return None


# =============================================================================
# STEP 4: 70/15/15 Participant-Level Split
# =============================================================================

def make_data_split(gt_lookup: dict, video_paths: List[Path],
                    splits_dir: Path, seed: int = 42) -> dict:
    """
    Split participants 70/15/15. Returns dict with 'train'/'val'/'test' video stem lists.
    Loads from disk if already exists (idempotent).
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    files = {k: splits_dir / f"{k}.txt" for k in ("train", "val", "test")}

    # Load existing split if all three files exist
    if all(f.exists() for f in files.values()):
        split = {}
        for k, fp in files.items():
            split[k] = set(fp.read_text().splitlines())
        logger.info(f"Loaded existing split: "
                    f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
        return split

    # Build participant → video mapping
    participant_videos: Dict[str, List[str]] = defaultdict(list)
    unmatched = []
    for vp in video_paths:
        stem = vp.stem
        rec  = find_gt(stem, gt_lookup)
        if rec and rec.get("participant_id"):
            participant_videos[rec["participant_id"]].append(stem)
        else:
            unmatched.append(stem)

    participants = sorted(participant_videos.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)

    n = len(participants)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    train_p = participants[:n_train]
    val_p   = participants[n_train:n_train + n_val]
    test_p  = participants[n_train + n_val:]

    split = {"train": set(), "val": set(), "test": set()}
    for p in train_p:
        split["train"].update(participant_videos[p])
    for p in val_p:
        split["val"].update(participant_videos[p])
    for p in test_p:
        split["test"].update(participant_videos[p])

    # Videos with no GT go into train
    split["train"].update(unmatched)

    for k, stems in split.items():
        files[k].write_text("\n".join(sorted(stems)))

    logger.info(f"Created split: "
                f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    if unmatched:
        logger.warning(f"{len(unmatched)} videos had no GT participant ID → assigned to train")
    return split


# =============================================================================
# STEP 5: DL Dataset + Models (signal-based, not frame-based)
# =============================================================================

class BVPSpO2Dataset(Dataset):
    """
    Dataset for DL SpO2 regression.
    Input: averaged skin-ROI BVP signal (after one rPPG algorithm), shape (T,)
    Target: GT SpO2 value (scalar)
    """
    def __init__(self, records: List[dict], algo_key: str = "POS", seq_len: int = 1800):
        self.algo_key = algo_key
        try:
            self.records  = [r for r in records if not np.isnan(r["spo2_gt"])]
        except KeyError:
            self.records = []
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sig_key = f"signal_{self.algo_key}"
        sig = rec[sig_key].astype(np.float32) if sig_key in rec else rec["signal"].astype(np.float32)

        # Pad or truncate to fixed length
        if len(sig) >= self.seq_len:
            sig = sig[:self.seq_len]
        else:
            sig = np.pad(sig, (0, self.seq_len - len(sig)))

        # Normalise
        std = sig.std()
        if std > 1e-6:
            sig = (sig - sig.mean()) / std

        return torch.tensor(sig).unsqueeze(0), torch.tensor(float(rec["spo2_gt"]))


class SpO2RegressorCNN(nn.Module):
    """
    Lightweight 1D CNN to regress SpO2 from a BVP signal segment.
    Used for both DeepPhys-style and EfficientPhys-style variants.
    """
    def __init__(self, seq_len: int = 900, variant: str = "standard"):
        super().__init__()
        if variant == "efficient":
            channels = [1, 16, 32, 64]
            kernel   = 7
        else:  # deepphys-style: deeper
            channels = [1, 32, 64, 128]
            kernel   = 5

        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        self.encoder = nn.Sequential(*layers)
        # Compute flattened size
        dummy = torch.zeros(1, 1, seq_len)
        flat  = self.encoder(dummy).numel()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.regressor(self.encoder(x)).squeeze(-1)


class SpO2ContrastiveCNN(nn.Module):
    """
    Contrastive encoder (SimCLR-style) trained without GT labels.
    Two augmented windows from the same BVP are a positive pair.
    After pretraining, a linear head is fitted on labeled val data.
    """
    def __init__(self, seq_len: int = 900, proj_dim: int = 64):
        super().__init__()
        channels = [1, 32, 64, 128]
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], 7, padding=3),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        self.encoder = nn.Sequential(*layers)
        dummy = torch.zeros(1, 1, seq_len)
        flat  = self.encoder(dummy).numel()
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        self.linear_head = None  # fitted after pretraining
        self._flat = flat

    def forward(self, x):
        return self.projector(self.encoder(x))

    def predict(self, x):
        """After linear head is fitted, call this for SpO2 regression."""
        if self.linear_head is None:
            raise RuntimeError("linear_head not fitted yet — call fit_linear_head first")
        emb = self.encoder(x)
        emb = emb.view(emb.size(0), -1)
        self.linear_head = self.linear_head.to(x.device)
        return self.linear_head(emb).squeeze(-1)

    def fit_linear_head(self, embeddings: np.ndarray, targets: np.ndarray):
        """Fit a simple linear regression head on embeddings."""
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0)
        reg.fit(embeddings, targets)
        # Convert to torch layer
        device = next(self.parameters()).device
        self.linear_head = nn.Linear(self._flat, 1, bias=True).to(device)
        with torch.no_grad():
            self.linear_head.weight.copy_(torch.tensor(reg.coef_, dtype=torch.float32, device=device))
            self.linear_head.bias.copy_(torch.tensor([reg.intercept_], dtype=torch.float32, device=device))


class SpO2TSCAN1D(nn.Module):
    """
    1D adaptation of TS-CAN for SpO2 regression from BVP signals.
    Dual-branch: appearance (raw signal) + motion (frame-difference analog),
    with channel attention from appearance guiding the motion branch.
    """
    def __init__(self, seq_len: int = 900):
        super().__init__()
        self.app_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.mot_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.Sigmoid(),
        )
        dummy = torch.zeros(1, 1, seq_len)
        flat = self.mot_branch(dummy).numel()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        diff = torch.zeros_like(x)
        diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        app = self.app_branch(x)
        mot = self.mot_branch(diff)
        attn = self.channel_attn(app).unsqueeze(-1)
        mot = mot * attn
        return self.regressor(mot).squeeze(-1)


class SpO2PhysFormer1D(nn.Module):
    """
    1D adaptation of PhysFormer for SpO2 regression from BVP signals.
    Splits the signal into fixed-size patches, embeds them, and processes
    with a Transformer encoder before regressing to SpO2.
    """
    def __init__(self, seq_len: int = 900, patch_size: int = 30,
                 embed_dim: int = 64, num_heads: int = 4, depth: int = 4):
        super().__init__()
        self.patch_size = patch_size
        num_patches = seq_len // patch_size
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.patch_embed(x).transpose(1, 2)          # (B, N, D)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(-1)


def _make_supervised_model(model_name: str, seq_len: int) -> "nn.Module":
    if model_name == "deepphys":
        return SpO2RegressorCNN(seq_len=seq_len, variant="standard")
    if model_name == "efficientphys":
        return SpO2RegressorCNN(seq_len=seq_len, variant="efficient")
    if model_name == "tscan":
        return SpO2TSCAN1D(seq_len=seq_len)
    if model_name == "physformer":
        return SpO2PhysFormer1D(seq_len=seq_len)
    raise ValueError(f"Unknown supervised model: {model_name}")


def nt_xent_loss(z1: "torch.Tensor", z2: "torch.Tensor", temperature: float = 0.5) -> "torch.Tensor":
    """NT-Xent contrastive loss."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)               # (2B, D)
    sim = torch.matmul(z, z.T) / temperature       # (2B, 2B)
    n   = z1.shape[0]
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    mask   = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim    = sim.masked_fill(mask, -1e9)
    return nn.functional.cross_entropy(sim, labels)


def augment_signal(sig: "torch.Tensor", seq_len: int) -> "torch.Tensor":
    """Random temporal crop + noise for contrastive augmentation."""
    if sig.shape[-1] <= seq_len // 2:
        return sig
    start = torch.randint(0, sig.shape[-1] - seq_len // 2, (1,)).item()
    crop  = sig[..., start:start + seq_len // 2]
    # Pad back
    crop  = torch.nn.functional.pad(crop, (0, seq_len - crop.shape[-1]))
    noise = 0.01 * torch.randn_like(crop)
    return crop + noise


# =============================================================================
# STEP 6: DL Training Loop
# =============================================================================

def train_dl_models(
    signal_records: List[dict],
    split: dict,
    output_dir: Path,
    ckpt_dir: Path,
    device: str,
    epochs: int = 50,
    best_epochs: int = 100,
    seq_len: int = 1800,
    batch_size: int = 32,
) -> List[dict]:
    """
    Train DeepPhys-style, EfficientPhys-style, and Contrastive models across
    multiple signal algorithms (9-experiment matrix).

    Phase 1: Run all 9 combos for `epochs` (default 50).
    Phase 2: For each DL model, identify the best-performing algorithm (lowest
             test MAE), then re-run only that model–algo pair for `best_epochs`
             (default 100). Results are tagged rerun_100ep=True.

    Returns list of result dicts for spo2_dl_results.csv.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available — skipping DL training.")
        return []

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Partition records by split ────────────────────────────────────────────
    train_recs = [r for r in signal_records if r["video"] in split["train"]]
    val_recs   = [r for r in signal_records if r["video"] in split["val"]]
    test_recs  = [r for r in signal_records if r["video"] in split["test"]]

    logger.info(f"DL split — train: {len(train_recs)} | val: {len(val_recs)} | test: {len(test_recs)}")

    def make_loader(records, algo_key, shuffle):
        ds = BVPSpO2Dataset(records, algo_key=algo_key, seq_len=seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    all_results = []
    
    # Run the 15 distinct experiments loop (5 models × 3 algos)
    for algo in ["GREEN", "ICA", "POS"]:
        logger.info(f"\n{'='*50}\nRUNNING 5 DL MODELS FOR SIGNAL: {algo}\n{'='*50}")
        
        train_loader = make_loader(train_recs, algo, shuffle=True)
        val_loader   = make_loader(val_recs,   algo, shuffle=False)
        test_loader  = make_loader(test_recs,  algo, shuffle=False)

        # ── Helper: supervised training loop ──────────────────────────────────────
        def supervised_train(model_name: str, model: nn.Module):
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.MSELoss()

            best_val_mse = float("inf")
            ckpt_path    = ckpt_dir / f"{model_name}_{algo}_best.pt"

            for epoch in range(1, epochs + 1):
                model.train()
                train_losses = []
                for x, y in train_loader:
                    x, y = x.to(device), y.float().to(device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                scheduler.step()

                model.eval()
                val_preds, val_gt = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        pred = model(x.to(device)).cpu().numpy()
                        val_preds.extend(pred.tolist())
                        val_gt.extend(y.numpy().tolist())

                val_mse = float(np.mean((np.array(val_preds) - np.array(val_gt)) ** 2))
                if epoch % 10 == 0 or epoch == epochs:
                    logger.info(f"  [{model_name} | {algo}] Epoch {epoch}/{epochs} — "
                                f"train_loss: {np.mean(train_losses):.4f}  val_MSE: {val_mse:.4f}")

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    torch.save(model.state_dict(), ckpt_path)

            # Test evaluation
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            test_preds, test_gt = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    pred = model(x.to(device)).cpu().numpy()
                    test_preds.extend(pred.tolist())
                    test_gt.extend(y.numpy().tolist())

            sm = summary_metrics(np.array(test_preds), np.array(test_gt))
            logger.info(f"  [{model_name} | {algo}] TEST → MAE: {sm['MAE']:.3f}  RMSE: {sm['RMSE']:.3f}  "
                        f"MSE: {sm['MSE']:.3f}  R²: {sm['R2']:.4f}")

            # Tag each prediction back to video
            for rec, pred_val, gt_val in zip(
                    [r for r in test_recs if not np.isnan(r["spo2_gt"])],
                    test_preds,
                    test_gt):
                err = abs(pred_val - gt_val) if not np.isnan(gt_val) else np.nan
                all_results.append({
                    "video":          rec["video"],
                    "dl_model":       model_name,
                    "algorithm":      algo,
                    "spo2_estimated": round(pred_val, 3),
                    "spo2_gt":        gt_val,
                    "spo2_error":     round(err, 3) if not np.isnan(err) else np.nan,
                    "MAE_test":       round(sm["MAE"], 3),
                    "RMSE_test":      round(sm["RMSE"], 3),
                    "MSE_test":       round(sm["MSE"], 3),
                    "R2_test":        round(sm["R2"], 4),
                    "lt5pct":         float(sm["MAE"]) < 5.0,
                })

        # ── Train supervised models ───────────────────────────────────────────────
        for sup_model in ["deepphys", "efficientphys", "tscan", "physformer"]:
            logger.info(f"Training {sup_model} on {algo}...")
            supervised_train(sup_model, _make_supervised_model(sup_model, seq_len))

        # ── Contrastive Pretraining ───────────────────────────────────────────────
        logger.info(f"Contrastive pretraining on {algo}...")
        contrastive_model = SpO2ContrastiveCNN(seq_len=seq_len).to(device)
        con_optimizer = optim.Adam(contrastive_model.parameters(), lr=1e-3)

        # Use training set
        all_train_signals = [r[f"signal_{algo}"] for r in train_recs]
        contrastive_model.train()
        for epoch in range(1, epochs + 1):
            losses = []
            idx_list = np.random.permutation(len(all_train_signals))
            for i in range(0, len(idx_list) - batch_size, batch_size):
                batch_sigs = [all_train_signals[idx_list[j]] for j in range(i, i + batch_size)]
                x_base = []
                for sig in batch_sigs:
                    s = sig.astype(np.float32)
                    if len(s) < seq_len:
                        s = np.pad(s, (0, seq_len - len(s)))
                    else:
                        s = s[:seq_len]
                    std = s.std()
                    if std > 1e-6:
                        s = (s - s.mean()) / std
                    x_base.append(s)
                x_t = torch.tensor(np.array(x_base), dtype=torch.float32).unsqueeze(1).to(device)
                z1  = contrastive_model(x_t)
                z2  = contrastive_model(augment_signal(x_t, seq_len))
                loss = nt_xent_loss(z1, z2)
                con_optimizer.zero_grad()
                loss.backward()
                con_optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0 or epoch == epochs:
                logger.info(f"  [Contrastive | {algo}] Epoch {epoch}/{epochs} — loss: {np.mean(losses):.4f}")

        torch.save(contrastive_model.state_dict(), ckpt_dir / f"contrastive_{algo}_pretrained.pt")

        # Fit linear head on val set
        logger.info(f"Fitting linear regression head on val set for {algo}...")
        contrastive_model.eval()
        val_embeddings, val_targets = [], []
        for rec in val_recs:
            if np.isnan(rec.get("spo2_gt", np.nan)):
                continue
            s = rec[f"signal_{algo}"].astype(np.float32)
            if len(s) < seq_len:
                s = np.pad(s, (0, seq_len - len(s)))
            else:
                s = s[:seq_len]
            std = s.std()
            if std > 1e-6:
                s = (s - s.mean()) / std
            x_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = contrastive_model.encoder(x_t)
                emb = emb.view(emb.size(0), -1).cpu().numpy()
            val_embeddings.append(emb[0])
            val_targets.append(float(rec["spo2_gt"]))

        if val_embeddings:
            contrastive_model.fit_linear_head(np.array(val_embeddings), np.array(val_targets))
            torch.save(contrastive_model.state_dict(), ckpt_dir / f"contrastive_{algo}_best.pt")

        # Test contrastive
        logger.info(f"Evaluating Contrastive model on test set for {algo}...")
        test_preds_con, test_gt_con = [], []
        for rec in test_recs:
            if np.isnan(rec.get("spo2_gt", np.nan)):
                continue
            s = rec[f"signal_{algo}"].astype(np.float32)
            if len(s) < seq_len:
                s = np.pad(s, (0, seq_len - len(s)))
            else:
                s = s[:seq_len]
            std = s.std()
            if std > 1e-6:
                s = (s - s.mean()) / std
            x_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = contrastive_model.predict(x_t).cpu().item()
            test_preds_con.append(pred)
            test_gt_con.append(float(rec["spo2_gt"]))

        if test_preds_con:
            sm_c = summary_metrics(np.array(test_preds_con), np.array(test_gt_con))
            logger.info(f"  [Contrastive | {algo}] TEST → MAE: {sm_c['MAE']:.3f}  RMSE: {sm_c['RMSE']:.3f}  "
                        f"MSE: {sm_c['MSE']:.3f}  R²: {sm_c['R2']:.4f}")
            for rec, pred_val, gt_val in zip(
                    [r for r in test_recs if not np.isnan(r.get("spo2_gt", np.nan))],
                    test_preds_con, test_gt_con):
                err = abs(pred_val - gt_val)
                all_results.append({
                    "video":          rec["video"],
                    "dl_model":       "contrastive",
                    "algorithm":      algo,
                    "spo2_estimated": round(pred_val, 3),
                    "spo2_gt":        gt_val,
                    "spo2_error":     round(err, 3),
                    "MAE_test":       round(sm_c["MAE"], 3),
                    "RMSE_test":      round(sm_c["RMSE"], 3),
                    "MSE_test":       round(sm_c["MSE"], 3),
                    "R2_test":        round(sm_c["R2"], 4),
                    "lt5pct":         float(sm_c["MAE"]) < 5.0,
                })

    # ── PHASE 2: Best algo per model → 100-epoch rerun ───────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Identifying best algo per DL model and re-running for "
                f"{best_epochs} epochs...")
    logger.info("=" * 60)

    # Detect best algo per model from Phase 1 MAEs
    phase1_mae: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in all_results:
        phase1_mae[r["dl_model"]][r["algorithm"]] = r["MAE_test"]

    best_combos: List[Tuple[str, str]] = []
    for model_name in ["deepphys", "efficientphys", "contrastive", "tscan", "physformer"]:
        if model_name not in phase1_mae:
            continue
        best_algo = min(phase1_mae[model_name], key=lambda a: phase1_mae[model_name][a])
        best_combos.append((model_name, best_algo))
        logger.info(f"  {model_name} → best algo: {best_algo} "
                    f"(MAE {phase1_mae[model_name][best_algo]:.3f}) "
                    f"→ retraining for {best_epochs} epochs")

    # Re-run each best combo for best_epochs
    for model_name, algo in best_combos:
        logger.info(f"\nRetraining {model_name} + {algo} for {best_epochs} epochs...")

        train_loader = make_loader(train_recs, algo, shuffle=True)
        val_loader   = make_loader(val_recs,   algo, shuffle=False)
        test_loader  = make_loader(test_recs,  algo, shuffle=False)

        if model_name in ("deepphys", "efficientphys", "tscan", "physformer"):
            model_ep = _make_supervised_model(model_name, seq_len).to(device)
            optimizer_ep = optim.Adam(model_ep.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler_ep = optim.lr_scheduler.CosineAnnealingLR(optimizer_ep, T_max=best_epochs)
            criterion    = nn.MSELoss()
            ckpt_path    = ckpt_dir / f"{model_name}_{algo}_best100.pt"

            best_val_mse_ep = float("inf")
            for epoch in range(1, best_epochs + 1):
                model_ep.train()
                train_losses = []
                for x, y in train_loader:
                    x, y = x.to(device), y.float().to(device)
                    optimizer_ep.zero_grad()
                    loss = criterion(model_ep(x), y)
                    loss.backward()
                    optimizer_ep.step()
                    train_losses.append(loss.item())
                scheduler_ep.step()

                model_ep.eval()
                val_preds_ep, val_gt_ep = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        val_preds_ep.extend(model_ep(x.to(device)).cpu().numpy().tolist())
                        val_gt_ep.extend(y.numpy().tolist())
                val_mse_ep = float(np.mean((np.array(val_preds_ep) - np.array(val_gt_ep)) ** 2))
                if epoch % 10 == 0 or epoch == best_epochs:
                    logger.info(f"  [{model_name}|{algo}|100ep] Epoch {epoch}/{best_epochs} — "
                                f"train_loss: {np.mean(train_losses):.4f}  val_MSE: {val_mse_ep:.4f}")
                if val_mse_ep < best_val_mse_ep:
                    best_val_mse_ep = val_mse_ep
                    torch.save(model_ep.state_dict(), ckpt_path)

            model_ep.load_state_dict(torch.load(ckpt_path, map_location=device))
            model_ep.eval()
            test_preds_ep, test_gt_ep = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    test_preds_ep.extend(model_ep(x.to(device)).cpu().numpy().tolist())
                    test_gt_ep.extend(y.numpy().tolist())

            sm_ep = summary_metrics(np.array(test_preds_ep), np.array(test_gt_ep))
            logger.info(f"  [{model_name}|{algo}|100ep] TEST → MAE: {sm_ep['MAE']:.3f}  "
                        f"RMSE: {sm_ep['RMSE']:.3f}  R²: {sm_ep['R2']:.4f}")

            for rec, pred_val, gt_val in zip(
                    [r for r in test_recs if not np.isnan(r.get("spo2_gt", np.nan))],
                    test_preds_ep, test_gt_ep):
                err = abs(pred_val - gt_val) if not np.isnan(gt_val) else np.nan
                all_results.append({
                    "video":          rec["video"],
                    "dl_model":       model_name,
                    "algorithm":      algo,
                    "spo2_estimated": round(pred_val, 3),
                    "spo2_gt":        gt_val,
                    "spo2_error":     round(err, 3) if not np.isnan(err) else np.nan,
                    "MAE_test":       round(sm_ep["MAE"], 3),
                    "RMSE_test":      round(sm_ep["RMSE"], 3),
                    "MSE_test":       round(sm_ep["MSE"], 3),
                    "R2_test":        round(sm_ep["R2"], 4),
                    "lt5pct":         float(sm_ep["MAE"]) < 5.0,
                    "rerun_100ep":    True,
                })

        elif model_name == "contrastive":
            # Re-pretrain contrastive encoder for best_epochs then refit linear head
            con_model_ep = SpO2ContrastiveCNN(seq_len=seq_len).to(device)
            con_opt_ep   = optim.Adam(con_model_ep.parameters(), lr=1e-3)
            all_train_sigs_ep = [r[f"signal_{algo}"] for r in train_recs]

            con_model_ep.train()
            for epoch in range(1, best_epochs + 1):
                losses = []
                idx_list = np.random.permutation(len(all_train_sigs_ep))
                for i in range(0, len(idx_list) - batch_size, batch_size):
                    batch_sigs = [all_train_sigs_ep[idx_list[j]] for j in range(i, i + batch_size)]
                    x_base = []
                    for sig in batch_sigs:
                        s = sig.astype(np.float32)
                        s = s[:seq_len] if len(s) >= seq_len else np.pad(s, (0, seq_len - len(s)))
                        std = s.std()
                        if std > 1e-6: s = (s - s.mean()) / std
                        x_base.append(s)
                    x_t = torch.tensor(np.array(x_base), dtype=torch.float32).unsqueeze(1).to(device)
                    z1  = con_model_ep(x_t)
                    z2  = con_model_ep(augment_signal(x_t, seq_len))
                    loss = nt_xent_loss(z1, z2)
                    con_opt_ep.zero_grad(); loss.backward(); con_opt_ep.step()
                    losses.append(loss.item())
                if epoch % 10 == 0 or epoch == best_epochs:
                    logger.info(f"  [contrastive|{algo}|100ep] Epoch {epoch}/{best_epochs} — "
                                f"loss: {np.mean(losses):.4f}")

            # Fit linear head on val
            con_model_ep.eval()
            val_embs_ep, val_tgts_ep = [], []
            for rec in val_recs:
                if np.isnan(rec.get("spo2_gt", np.nan)): continue
                s = rec[f"signal_{algo}"].astype(np.float32)
                s = s[:seq_len] if len(s) >= seq_len else np.pad(s, (0, seq_len - len(s)))
                std = s.std()
                if std > 1e-6: s = (s - s.mean()) / std
                x_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = con_model_ep.encoder(x_t).view(1, -1).cpu().numpy()
                val_embs_ep.append(emb[0])
                val_tgts_ep.append(float(rec["spo2_gt"]))

            if val_embs_ep:
                con_model_ep.fit_linear_head(np.array(val_embs_ep), np.array(val_tgts_ep))
                torch.save(con_model_ep.state_dict(),
                           ckpt_dir / f"contrastive_{algo}_best100.pt")

            # Test
            test_preds_con_ep, test_gt_con_ep = [], []
            for rec in test_recs:
                if np.isnan(rec.get("spo2_gt", np.nan)): continue
                s = rec[f"signal_{algo}"].astype(np.float32)
                s = s[:seq_len] if len(s) >= seq_len else np.pad(s, (0, seq_len - len(s)))
                std = s.std()
                if std > 1e-6: s = (s - s.mean()) / std
                x_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = con_model_ep.predict(x_t).cpu().item()
                test_preds_con_ep.append(pred)
                test_gt_con_ep.append(float(rec["spo2_gt"]))

            if test_preds_con_ep:
                sm_con_ep = summary_metrics(np.array(test_preds_con_ep), np.array(test_gt_con_ep))
                logger.info(f"  [contrastive|{algo}|100ep] TEST → MAE: {sm_con_ep['MAE']:.3f}  "
                            f"RMSE: {sm_con_ep['RMSE']:.3f}  R²: {sm_con_ep['R2']:.4f}")
                for rec, pred_val, gt_val in zip(
                        [r for r in test_recs if not np.isnan(r.get("spo2_gt", np.nan))],
                        test_preds_con_ep, test_gt_con_ep):
                    err = abs(pred_val - gt_val)
                    all_results.append({
                        "video":          rec["video"],
                        "dl_model":       "contrastive",
                        "algorithm":      algo,
                        "spo2_estimated": round(pred_val, 3),
                        "spo2_gt":        gt_val,
                        "spo2_error":     round(err, 3),
                        "MAE_test":       round(sm_con_ep["MAE"], 3),
                        "RMSE_test":      round(sm_con_ep["RMSE"], 3),
                        "MSE_test":       round(sm_con_ep["MSE"], 3),
                        "R2_test":        round(sm_con_ep["R2"], 4),
                        "lt5pct":         float(sm_con_ep["MAE"]) < 5.0,
                        "rerun_100ep":    True,
                    })

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified rPPG + SpO2 Pipeline")
    parser.add_argument("--video-dir",   required=True)
    parser.add_argument("--gt-dir",      default="./ground-truth")
    parser.add_argument("--output-dir",  default="./pipeline_results")
    parser.add_argument("--max-videos",  type=int, default=None)
    parser.add_argument("--max-frames",  type=int, default=1800,
                        help="Max frames per video (default 1800 = 60s@30fps)")
    parser.add_argument("--workers",     type=int,
                        default=max(1, multiprocessing.cpu_count() - 2),
                        help="Parallel workers for signal extraction")
    parser.add_argument("--device",      default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--skip-dl",     action="store_true",
                        help="Skip DL model training (classical pipeline only)")
    parser.add_argument("--dl-only",     action="store_true",
                        help="Skip signal extraction entirely; re-extract BVP signals from "
                             "videos and run only DL training. Requires --video-dir pointing "
                             "to the original videos and an existing --output-dir with splits/.")
    parser.add_argument("--dl-epochs",      type=int, default=50)
    parser.add_argument("--dl-seq-len",     type=int, default=1800,
                        help="BVP sequence length for DL models (samples)")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip already-processed videos found in output CSVs")
    parser.add_argument("--cache-dir",      default="./signals_cache",
                        help="Directory for per-algorithm BVP signal cache files "
                             "(GREEN.npz / ICA.npz / POS.npz). "
                             "Set to empty string to disable caching.")
    parser.add_argument("--window-start",   type=float, default=0.0,
                        help="Start of analysis window in seconds (default 0)")
    parser.add_argument("--window-duration", type=float, default=None,
                        help="Duration of analysis window in seconds (default: full video)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    splits_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.strip() if args.cache_dir else None
    dl_cache  = load_dl_algo_cache(cache_dir)

    win_start    = args.window_start
    win_duration = args.window_duration
    if win_duration is not None:
        logger.info(f"Analysis window: {win_start}s – {win_start + win_duration}s "
                    f"({win_duration}s duration)")
    elif win_start > 0:
        logger.info(f"Analysis window: {win_start}s → end of video")

    rppg_csv     = out_dir / "rppg_results.csv"
    spo2_cls_csv = out_dir / "spo2_classical_results.csv"
    spo2_dl_csv  = out_dir / "spo2_dl_results.csv"

    # ── Device setup ──────────────────────────────────────────────────────────
    device = args.device
    if TORCH_AVAILABLE:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable → falling back to CPU")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS unavailable → falling back to CPU")
            device = "cpu"
    else:
        logger.warning("PyTorch not installed — all GPU paths disabled")
        device = "cpu"

    logger.info(f"Device: {device}  |  Workers: {args.workers}")

    # ── Load GT ───────────────────────────────────────────────────────────────
    gt_lookup = load_ground_truth(args.gt_dir)

    # ── Collect videos ────────────────────────────────────────────────────────
    video_dir = Path(args.video_dir)
    video_paths = []
    for ext in ("*.MOV", "*.mov", "*.mp4", "*.MP4", "*.avi", "*.AVI"):
        video_paths.extend(video_dir.rglob(ext))
    video_paths = sorted(set(video_paths))
    if args.max_videos:
        video_paths = video_paths[:args.max_videos]
    logger.info(f"Found {len(video_paths)} video(s)")

    # ── Auto-resume: skip already-processed videos ────────────────────────────
    done_stems = set()
    if args.resume and rppg_csv.exists():
        df_done = pd.read_csv(rppg_csv, usecols=["video"])
        done_stems = set(df_done["video"].unique())
        video_paths = [vp for vp in video_paths if vp.stem not in done_stems]
        logger.info(f"Resume: skipping {len(done_stems)} already-processed videos. "
                    f"{len(video_paths)} remaining.")

    # signal_records must always be defined before the DL section
    signal_records: List[dict] = []

    # ── DL-only fast path ─────────────────────────────────────────────────────
    if args.dl_only:
        logger.info("DL-only mode: loading BVP signals for DL training...")
        all_vp_dl = []
        for ext in ("*.MOV", "*.mov", "*.mp4", "*.MP4", "*.avi", "*.AVI"):
            all_vp_dl.extend(Path(args.video_dir).rglob(ext))
        all_vp_dl = sorted(set(all_vp_dl))
        if args.max_videos:
            all_vp_dl = all_vp_dl[:args.max_videos]

        # Serve cached videos directly; collect uncached ones for extraction
        uncached_vp = []
        for vp in all_vp_dl:
            stem = vp.stem
            if all(stem in dl_cache[a] for a in DL_ALGOS):
                gt_rec  = find_gt(stem, gt_lookup)
                signal_records.append({
                    "video":        stem,
                    "signal_GREEN": dl_cache["GREEN"][stem],
                    "signal_ICA":   dl_cache["ICA"][stem],
                    "signal_POS":   dl_cache["POS"][stem],
                    "spo2_gt":      gt_rec["spo2"] if gt_rec else np.nan,
                    "hr_gt":        gt_rec["hr"]   if gt_rec else np.nan,
                })
            else:
                uncached_vp.append(vp)

        if uncached_vp:
            logger.info(f"{len(signal_records)} from cache; extracting {len(uncached_vp)} uncached videos...")
            dl_tasks = [(str(vp), args.max_frames) for vp in uncached_vp]
            with multiprocessing.Pool(processes=args.workers) as pool:
                for stem, all_signals, fps in tqdm(
                        pool.imap_unordered(extract_signals_worker, dl_tasks),
                        total=len(dl_tasks), desc="DL signal extraction"):
                    if not all_signals:
                        continue
                    all_signals = apply_window(all_signals, fps, win_start, win_duration)
                    gt_rec  = find_gt(stem, gt_lookup)
                    skin_sigs = [all_signals[k] for k in SKIN_REGIONS
                                 if k in all_signals and all_signals[k] is not None
                                 and all_signals[k].shape[0] >= 64]
                    if not skin_sigs:
                        continue
                    min_len  = min(s.shape[0] for s in skin_sigs)
                    avg_rgb  = np.mean(np.stack([s[:min_len] for s in skin_sigs], axis=0), axis=0)
                    bvp_green = run_algo_on_signal(avg_rgb, fps, "GREEN", device)
                    bvp_ica   = run_algo_on_signal(avg_rgb, fps, "ICA", device)
                    bvp_pos   = run_algo_on_signal(avg_rgb, fps, "POS", device)
                    if bvp_pos is not None and bvp_green is not None and bvp_ica is not None:
                        dl_cache["GREEN"][stem] = bvp_green
                        dl_cache["ICA"][stem]   = bvp_ica
                        dl_cache["POS"][stem]   = bvp_pos
                        signal_records.append({
                            "video":        stem,
                            "signal_GREEN": bvp_green,
                            "signal_ICA":   bvp_ica,
                            "signal_POS":   bvp_pos,
                            "spo2_gt":      gt_rec["spo2"] if gt_rec else np.nan,
                            "hr_gt":        gt_rec["hr"]   if gt_rec else np.nan,
                        })
            if cache_dir:
                save_dl_algo_cache(cache_dir, dl_cache)

        logger.info(f"Built {len(signal_records)} signal records for DL training")
        args.skip_dl = False

    if not video_paths and not args.dl_only:
        logger.info("All videos already processed or none found.")
    elif args.dl_only:
        pass  # signal_records already built above, skip classical block
    else:
        # ── Initialise output CSVs ────────────────────────────────────────────
        rppg_header = ("video,roi,algorithm,SNR,NSQI,SE,Variance,ZCR\n")
        spo2_header = ("video,roi,algorithm,spo2_estimated,spo2_gt,spo2_error\n")
        for csv_path, header in [(rppg_csv, rppg_header), (spo2_cls_csv, spo2_header)]:
            if not csv_path.exists():
                csv_path.write_text(header)

        # ── STEP 1: Parallel signal extraction (CPU workers) ──────────────────
        tasks = [(str(vp), args.max_frames) for vp in video_paths]
        logger.info(f"Extracting signals from {len(tasks)} videos "
                    f"using {args.workers} workers...")

        # signal_records is used later for DL training
        signal_records: List[dict] = []

        with multiprocessing.Pool(processes=args.workers) as pool:
            for stem, all_signals, fps in tqdm(
                    pool.imap_unordered(extract_signals_worker, tasks),
                    total=len(tasks), desc="Extracting signals"):

                if not all_signals:
                    continue
                all_signals = apply_window(all_signals, fps, win_start, win_duration)

                gt_rec = find_gt(stem, gt_lookup)
                spo2_gt = gt_rec["spo2"] if gt_rec else np.nan
                hr_gt   = gt_rec["hr"]   if gt_rec else np.nan

                rppg_rows = []
                spo2_rows = []

                # ── STEP 2: Classical rPPG × all 31 ROIs ──────────────────────
                for roi_name, raw_rgb in all_signals.items():
                    if raw_rgb is None or raw_rgb.shape[0] < 64:
                        continue

                    for algo_name in RPPG_METHODS:
                        bvp = run_algo_on_signal(raw_rgb, fps, algo_name, device)
                        if bvp is None or len(bvp) < 30:
                            continue

                        # Blind rPPG metrics
                        m = bvp_blind_metrics(bvp, fps)
                        rppg_rows.append({
                            "video": stem, "roi": roi_name, "algorithm": algo_name,
                            "SNR":      round(m["SNR"], 4)      if not np.isnan(m["SNR"]) else "",
                            "NSQI":     round(m["NSQI"], 6)     if not np.isnan(m["NSQI"]) else "",
                            "SE":       round(m["SE"], 4)       if not np.isnan(m["SE"]) else "",
                            "Variance": round(m["Variance"], 6),
                            "ZCR":      round(m["ZCR"], 6),
                        })

                        # Classical SpO2 via spectral RoR
                        spo2_est = compute_spo2_spectral(bvp, raw_rgb, fps)
                        err = abs(spo2_est - spo2_gt) if (
                            not np.isnan(spo2_gt) and not np.isnan(spo2_est)) else np.nan
                        spo2_rows.append({
                            "video": stem, "roi": roi_name, "algorithm": algo_name,
                            "spo2_estimated": round(spo2_est, 3) if not np.isnan(spo2_est) else "",
                            "spo2_gt":        spo2_gt,
                            "spo2_error":     round(err, 3) if not np.isnan(err) else "",
                        })

                # Incrementally append to CSVs
                if rppg_rows:
                    pd.DataFrame(rppg_rows).to_csv(rppg_csv, mode="a", header=False, index=False)
                if spo2_rows:
                    pd.DataFrame(spo2_rows).to_csv(spo2_cls_csv, mode="a", header=False, index=False)

                # Build record for DL training — use algo cache if available
                skin_sigs = [all_signals[k] for k in SKIN_REGIONS if k in all_signals
                             and all_signals[k] is not None and all_signals[k].shape[0] >= 64]
                if skin_sigs:
                    if all(stem in dl_cache[a] for a in DL_ALGOS):
                        bvp_green = dl_cache["GREEN"][stem]
                        bvp_ica   = dl_cache["ICA"][stem]
                        bvp_pos   = dl_cache["POS"][stem]
                    else:
                        min_len  = min(s.shape[0] for s in skin_sigs)
                        avg_rgb  = np.mean(np.stack([s[:min_len] for s in skin_sigs], axis=0), axis=0)
                        bvp_green = run_algo_on_signal(avg_rgb, fps, "GREEN", device)
                        bvp_ica   = run_algo_on_signal(avg_rgb, fps, "ICA", device)
                        bvp_pos   = run_algo_on_signal(avg_rgb, fps, "POS", device)
                        if bvp_green is not None: dl_cache["GREEN"][stem] = bvp_green
                        if bvp_ica   is not None: dl_cache["ICA"][stem]   = bvp_ica
                        if bvp_pos   is not None: dl_cache["POS"][stem]   = bvp_pos

                    if bvp_pos is not None and bvp_green is not None and bvp_ica is not None:
                        signal_records.append({
                            "video":        stem,
                            "signal_GREEN": bvp_green,
                            "signal_ICA":   bvp_ica,
                            "signal_POS":   bvp_pos,
                            "spo2_gt":      spo2_gt,
                            "hr_gt":        hr_gt,
                        })

        if cache_dir:
            save_dl_algo_cache(cache_dir, dl_cache)
        logger.info(f"✓ Signals extracted. rPPG CSV: {rppg_csv}  |  SpO2 classical: {spo2_cls_csv}")

    # ── Summary stats for classical SpO2 ──────────────────────────────────────
    logger.info("Computing classical SpO2 summary metrics...")
    try:
        df_spo2 = pd.read_csv(spo2_cls_csv)
        df_ok   = df_spo2.dropna(subset=["spo2_error"])
        if not df_ok.empty:
            summary = df_ok.groupby("algorithm").apply(lambda g: pd.Series(
                summary_metrics(g["spo2_estimated"].values, g["spo2_gt"].values)
            )).sort_values("MAE")
            summary.to_csv(out_dir / "spo2_classical_summary.csv")
            logger.info("Classical SpO2 summary:\n" + summary.to_string())
    except Exception as e:
        logger.warning(f"Could not compute classical summary: {e}")

    # ── DL Training ───────────────────────────────────────────────────────────
    if not args.skip_dl:
        # Need signal records from all videos (including already-done ones if resuming)
        if args.resume and done_stems:
            logger.info("Resume mode: loading already-processed videos from algo cache...")
            cached_count = 0
            for stem in done_stems:
                if all(stem in dl_cache[a] for a in DL_ALGOS):
                    gt_rec = find_gt(stem, gt_lookup)
                    signal_records.append({
                        "video":        stem,
                        "signal_GREEN": dl_cache["GREEN"][stem],
                        "signal_ICA":   dl_cache["ICA"][stem],
                        "signal_POS":   dl_cache["POS"][stem],
                        "spo2_gt":      gt_rec["spo2"] if gt_rec else np.nan,
                        "hr_gt":        gt_rec["hr"]   if gt_rec else np.nan,
                    })
                    cached_count += 1
            logger.info(f"Loaded {cached_count}/{len(done_stems)} resumed videos from algo cache.")

        if len(signal_records) < 10:
            logger.warning("Too few signal records for DL training — skipping.")
        else:
            # Create data split
            all_vp_for_split = sorted(set(
                list(Path(args.video_dir).rglob("*.MOV")) +
                list(Path(args.video_dir).rglob("*.mp4"))
            ))
            if args.max_videos:
                all_vp_for_split = all_vp_for_split[:args.max_videos]
            split = make_data_split(gt_lookup, all_vp_for_split, splits_dir)

            dl_results = train_dl_models(
                signal_records, split, out_dir, ckpt_dir, device,
                epochs=args.dl_epochs, best_epochs=100, seq_len=args.dl_seq_len,
            )

            if dl_results:
                df_dl = pd.DataFrame(dl_results)
                df_dl.to_csv(spo2_dl_csv, index=False)
                logger.info(f"DL results → {spo2_dl_csv}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("Pipeline Complete")
    logger.info(f"  rPPG results:            {rppg_csv}")
    logger.info(f"  SpO2 classical results:  {spo2_cls_csv}")
    logger.info(f"  SpO2 classical summary:  {out_dir / 'spo2_classical_summary.csv'}")
    if not args.skip_dl:
        logger.info(f"  SpO2 DL results:         {spo2_dl_csv}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
