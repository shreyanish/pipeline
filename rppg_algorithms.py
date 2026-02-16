import numpy as np
from scipy.signal import detrend, butter, filtfilt
from sklearn.decomposition import FastICA, PCA
from config import FS_MIN, FS_MAX
import math

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
        ica = FastICA(n_components=3, random_state=0, max_iter=1000, tol=0.01)
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

def process_signal_ssr(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """SSR (Spatial Subspace Rotation) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # 1. Temporal Normalization
    C = raw_signal.T # shape (3, T)
    mean_C = np.mean(C, axis=1, keepdims=True)
    if np.any(mean_C == 0): return np.array([])
    Cn = C / mean_C
    
    # Parameters for windowed processing
    window_sec = 1.0
    stride_sec = 0.1 # overlap-add stride
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    if stride_len == 0: stride_len = 1
    
    T = Cn.shape[1]
    
    # Arrays for OLA
    bvp_full = np.zeros(T)
    weights = np.zeros(T)
    
    # Hanning window for smooth overlap-add
    win = np.hanning(window_len)
    
    for start in range(0, T - window_len + 1, stride_len):
        end = start + window_len
        segment = Cn[:, start:end] # (3, L)
        
        # Remove mean from segment
        segment_mean = np.mean(segment, axis=1, keepdims=True)
        segment_centered = segment - segment_mean
        
        # 2. Covariance Matrix
        R = np.dot(segment_centered, segment_centered.T) / window_len
        
        # 3. Eigen Decomposition
        eig_vals, eig_vecs = np.linalg.eigh(R)
        
        # Sort eigenvalues in descending order
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        # 4. Construct Pulse - use 2nd component
        p_segment = np.dot(eig_vecs[:, 1], segment_centered)
        
        # Align sign with Green channel
        if np.corrcoef(p_segment, segment[1, :])[0, 1] < 0:
             p_segment = -p_segment
             
        bvp_full[start:end] += p_segment * win
        weights[start:end] += win

    # Avoid division by zero
    weights[weights < 1e-6] = 1.0
    BVP = bvp_full / weights
    
    # 5. Bandpass Filtering (Zero-Phase)
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_green(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """GREEN - Simple green channel extraction"""
    if len(raw_signal) < 30: return np.array([])
    
    # Extract green channel (index 1)
    green = raw_signal[:, 1]
    
    # Detrend
    green_detrended = detrend(green)
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, green_detrended)
    
    return filtered_bvp

def process_signal_pca(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """PCA (Principal Component Analysis) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # 1. Detrend RGB signals
    signal = detrend(raw_signal, axis=0)
    
    # 2. Apply PCA
    try:
        pca = PCA(n_components=3)
        components = pca.fit_transform(signal)
    except:
        return np.array([])
    
    # 3. Select component with strongest periodicity in HR range
    nyquist = 0.5 * fs
    best_component = 0
    max_power = 0
    
    for i in range(3):
        fft_vals = np.fft.rfft(components[:, i])
        fft_freqs = np.fft.rfftfreq(len(components[:, i]), 1.0 / fs)
        
        mask = (fft_freqs >= FS_MIN) & (fft_freqs <= FS_MAX)
        power = np.sum(np.abs(fft_vals[mask])**2)
        
        if power > max_power:
            max_power = power
            best_component = i
    
    BVP = components[:, best_component]
    
    # 4. Bandpass Filtering
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_pbv(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """PBV (Pulse Blood Volume) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # Normalize each channel
    R = raw_signal[:, 0]
    G = raw_signal[:, 1]
    B = raw_signal[:, 2]
    
    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)
    
    if R_mean == 0 or G_mean == 0 or B_mean == 0:
        return np.array([])
    
    R_norm = R / R_mean
    G_norm = G / G_mean
    B_norm = B / B_mean
    
    # PBV formula: std(R)*G/std(G) - std(R)*B/std(B)
    std_R = np.std(R_norm)
    std_G = np.std(G_norm)
    std_B = np.std(B_norm)
    
    if std_G == 0 or std_B == 0:
        return np.array([])
    
    BVP = (std_R / std_G) * G_norm - (std_R / std_B) * B_norm
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_lgi(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """LGI (Local Group Invariance) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # Detrend
    signal = detrend(raw_signal, axis=0)
    
    R = signal[:, 0]
    G = signal[:, 1]
    B = signal[:, 2]
    
    # LGI transformation
    X = R - G
    Y = R + G - 2*B
    
    # Compute alpha (ratio of standard deviations)
    std_X = np.std(X)
    std_Y = np.std(Y)
    
    if std_Y == 0:
        return np.array([])
    
    alpha = std_X / std_Y
    
    # BVP signal
    BVP = X - alpha * Y
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_omit(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """OMIT (Orthogonal Matrix Image Transformation) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # Normalize
    signal = detrend(raw_signal, axis=0)
    
    # Compute covariance matrix
    C = np.cov(signal.T)
    
    # Eigendecomposition
    try:
        eig_vals, eig_vecs = np.linalg.eigh(C)
    except:
        return np.array([])
    
    # Sort by eigenvalues (descending)
    idx = eig_vals.argsort()[::-1]
    eig_vecs = eig_vecs[:, idx]
    
    # Project onto eigenvectors
    transformed = np.dot(signal, eig_vecs)
    
    # Use second component (first is usually DC/intensity)
    BVP = transformed[:, 1]
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_samc(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """SAMC (Spatial Averaging with Motion Compensation) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # Simple spatial averaging approach
    R = raw_signal[:, 0]
    G = raw_signal[:, 1]
    B = raw_signal[:, 2]
    
    # Normalize
    R_norm = (R - np.mean(R)) / (np.std(R) + 1e-8)
    G_norm = (G - np.mean(G)) / (np.std(G) + 1e-8)
    B_norm = (B - np.mean(B)) / (np.std(B) + 1e-8)
    
    # Weighted combination (green has highest weight)
    BVP = 0.5 * G_norm + 0.3 * R_norm + 0.2 * B_norm
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp

def process_signal_2sr(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """2SR (Two-Step Spatial Rotation) rPPG algorithm"""
    if len(raw_signal) < 30: return np.array([])
    
    # Temporal normalization
    C = raw_signal.T
    mean_C = np.mean(C, axis=1, keepdims=True)
    if np.any(mean_C == 0): return np.array([])
    Cn = C / mean_C
    
    # Step 1: Remove mean
    Cn_centered = Cn - np.mean(Cn, axis=1, keepdims=True)
    
    # Step 2: Build rotation matrix (simplified)
    # Project onto plane orthogonal to [1,1,1]
    P1 = np.array([[0, 1, -1], [-2, 1, 1]]) / np.sqrt(6)
    
    # Project
    S = np.dot(P1, Cn_centered)
    
    # Combine with adaptive weighting
    std_s0 = np.std(S[0, :])
    std_s1 = np.std(S[1, :])
    
    if std_s1 == 0:
        return np.array([])
    
    BVP = S[0, :] + (std_s0 / std_s1) * S[1, :]
    
    # Bandpass filter
    nyquist = 0.5 * fs
    low = FS_MIN / nyquist
    high = FS_MAX / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_bvp = filtfilt(b, a, BVP)
    
    return filtered_bvp