import numpy as np
from scipy.signal import detrend, butter, filtfilt
from sklearn.decomposition import FastICA
from config import FS_MIN, FS_MAX

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
