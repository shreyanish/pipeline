
import torch
import torch.fft
import numpy as np
from config import FS_MIN, FS_MAX

def ensure_tensor(signal, device=None):
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)
    if device and signal.device != device:
        signal = signal.to(device)
    return signal

def detrend_torch(signal, axis=0):
    """
    Simple linear detrending using least squares on GPU
    """
    n = signal.shape[axis]
    x = torch.linspace(0, 1, n, device=signal.device)
    # Combine with ones for intercept
    A = torch.stack([x, torch.ones_like(x)], dim=1)  # (N, 2)
    
    # We need to solve Ax = b for each channel
    # signal shape (N, 3) usually
    if axis != 0:
        signal = signal.transpose(0, axis)
    
    # Least squares: (A^T A)^-1 A^T y
    # A is (N, 2), y is (N, C)
    # solution is (2, C)
    solution = torch.linalg.lstsq(A, signal).solution
    
    # Fitted line
    trend = A @ solution
    detrended = signal - trend
    
    if axis != 0:
        detrended = detrended.transpose(0, axis)
        
    return detrended

def bandpass_filter_torch(signal, fs, low, high):
    """
    FFT-based ideal bandpass filter on GPU (Zero-phase)
    """
    n = signal.shape[0]
    freqs = torch.fft.rfftfreq(n, 1.0/fs, device=signal.device)
    
    # Create mask
    mask = (freqs >= low) & (freqs <= high)
    
    # FFT
    fft_vals = torch.fft.rfft(signal, dim=0)
    
    # Apply mask
    fft_vals[~mask] = 0.0
    
    # IFFT
    filtered = torch.fft.irfft(fft_vals, n=n, dim=0)
    
    return filtered

def process_signal_pos_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    # 1. Detrending
    signal = detrend_torch(signal)

    # 2. POS Algorithm
    # Standard POS projection matrix
    # P = [[0, 1, -1], [-2, 1, 1]]
    P = torch.tensor([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], device=device)
    
    # signal is (N, 3), P.T is (3, 2) -> S is (N, 2)
    S = torch.matmul(signal, P.T)
    
    # BVP = S1 + (std(S1)/std(S2)) * S2
    # S[:, 0] is S1, S[:, 1] is S2
    s1 = S[:, 0]
    s2 = S[:, 1]
    
    std1 = torch.std(s1)
    std2 = torch.std(s2)
    
    # Avoid division by zero
    if std2 == 0:
        return torch.tensor([], device=device)
        
    BVP = s1 + (std1 / std2) * s2

    # 3. Bandpass Filtering
    filtered_bvp = bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

    return filtered_bvp

def process_signal_chrom_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    # 1. Normalize (no detrend needed for CHROM usually, but raw RGB needs norm)
    # Using raw_signal directly as per CPU impl
    R = signal[:, 0]
    G = signal[:, 1]
    B = signal[:, 2]
    
    R_mean = torch.mean(R)
    G_mean = torch.mean(G)
    B_mean = torch.mean(B)
    
    if R_mean == 0 or G_mean == 0 or B_mean == 0:
        return torch.tensor([], device=device)
        
    Xn = R / R_mean
    Yn = G / G_mean
    Zn = B / B_mean
    
    # 2. Chrominance projection
    Xs = 3 * Xn - 2 * Yn
    Ys = 1.5 * Xn + Yn - 1.5 * Zn
    
    # 3. Calculate BVP
    std_Xs = torch.std(Xs)
    std_Ys = torch.std(Ys)
    
    if std_Ys == 0:
        return torch.tensor([], device=device)
        
    BVP = Xs - (std_Xs / std_Ys) * Ys
    
    # 4. Filter
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_green_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    green = signal[:, 1]
    green = detrend_torch(green.unsqueeze(1)).squeeze()
    
    return bandpass_filter_torch(green, fs, FS_MIN, FS_MAX)

def process_signal_ica_torch(raw_signal, fs, device='cuda'):
    """
    ICA implementation using SVD approximation or ported FastICA.
    True FastICA is hard to fully parallelize/port simply, 
    so we use JADE or just SVD-based pre-whitening + max-kurtosis rotation 
    OR simply PCA as a proxy if ICA is too complex.
    
    However, sklearn FastICA is iterative.
    For GPU, a common approximation is purely using SVD comparison or implementing JADE.
    
    Implementation: Symmetric FastICA in PyTorch
    """
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    # 1. Detrend
    X = detrend_torch(signal) # (N, 3)
    n_samples = X.shape[0]
    
    # 2. Centering
    X = X - torch.mean(X, dim=0)
    
    # 3. Whitening
    # Covariance
    cov = torch.cov(X.T)
    # Eigen decomposition
    D, E = torch.linalg.eigh(cov)
    # Whitening matrix: D^-1/2 * E^T
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-7))
    whitening = D_inv_sqrt @ E.T
    X_white = (whitening @ X.T).T # (N, 3)
    
    # 4. FastICA (Symmetric)
    # n_components = 3
    W = torch.randn(3, 3, device=device)
    # Orthogonalize
    U, S, Vt = torch.linalg.svd(W)
    W = U @ Vt
    
    # Iteration
    max_iter = 100
    tol = 1e-4
    alpha = 1.0
    
    for _ in range(max_iter):
        W_prev = W.clone()
        
        # g(wx) = tanh(wx)
        # g'(wx) = 1 - tanh^2(wx)
        wx = X_white @ W.T # (N, 3)
        g_wx = torch.tanh(wx)
        g_prime_wx = 1 - g_wx ** 2
        
        # Update W
        # W = E[xg(w^Tx)] - E[g'(w^Tx)]w
        # X_white is (N, 3), g_wx is (N, 3)
        term1 = (g_wx.T @ X_white) / n_samples # (3, 3)
        term2 = torch.mean(g_prime_wx, dim=0).unsqueeze(1) * W # (3, 3) via broadcasting?
        # Actually term2 is column-wise mean scaling each row of W
        term2 = torch.diag(torch.mean(g_prime_wx, dim=0)) @ W
        
        W = term1 - term2
        
        # Orthogonalize
        U, S, Vt = torch.linalg.svd(W)
        W = U @ Vt
        
        # Check convergence
        if torch.max(torch.abs(torch.abs(torch.diag(W @ W_prev.T)) - 1)) < tol:
            break
            
    # Sources
    S = X_white @ W.T
    
    # 5. Select component using FFT peak
    nyquist = 0.5 * fs
    best_component = 0
    max_power = -1.0
    
    freqs = torch.fft.rfftfreq(n_samples, 1.0/fs, device=device)
    mask = (freqs >= FS_MIN) & (freqs <= FS_MAX)
    
    fft_vals = torch.fft.rfft(S, dim=0)
    power_spectrum = torch.abs(fft_vals)**2
    
    # Sum power in physiological range
    valid_power = torch.sum(power_spectrum[mask, :], dim=0)
    best_component = torch.argmax(valid_power)
    
    BVP = S[:, best_component]
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)


def process_signal_pca_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    signal = detrend_torch(signal)
    
    # Center
    signal = signal - torch.mean(signal, dim=0)
    
    # SVD (PCA)
    # X = U S V^T
    # Components = U @ S = X @ V
    U, S, Vh = torch.linalg.svd(signal, full_matrices=False)
    
    # Components (N, 3)
    # SVD returns U(N, 3), S(3), Vh(3, 3)
    # Principal components projection: X @ V = U @ S
    components = U @ torch.diag(S)
    
    # Select component
    n_samples = signal.shape[0]
    freqs = torch.fft.rfftfreq(n_samples, 1.0/fs, device=device)
    mask = (freqs >= FS_MIN) & (freqs <= FS_MAX)
    
    fft_vals = torch.fft.rfft(components, dim=0)
    power_spectrum = torch.abs(fft_vals)**2
    valid_power = torch.sum(power_spectrum[mask, :], dim=0)
    best_component = torch.argmax(valid_power)
    
    BVP = components[:, best_component]
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)


def process_signal_ssr_torch(raw_signal, fs, device='cuda'):
    """SSR (Spatial Subspace Rotation) in PyTorch"""
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    # Temporal Normalization
    # C is (3, T)
    C = signal.T
    mean_C = torch.mean(C, dim=1, keepdim=True)
    if torch.any(mean_C == 0): return torch.tensor([], device=device)
    Cn = C / mean_C
    
    T = Cn.shape[1]
    window_sec = 1.0
    stride_sec = 0.1
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    if stride_len == 0: stride_len = 1
    
    bvp_full = torch.zeros(T, device=device)
    weights = torch.zeros(T, device=device)
    
    win = torch.hann_window(window_len, device=device)
    
    # We can perform the loop. For true GPU parallism we'd unfold, 
    # but regular loop on GPU IS faster than CPU loop for these ops.
    # To optimize more, we could use unfold to create a batch of windows.
    
    # Unfold approach:
    # Cn shape (3, T) -> unfold -> (3, N_windows, window_len)
    # But stride is non-overlapping in standard unfold? No, it has step.
    # torch.Tensor.unfold(dimension, size, step)
    
    # Let's try explicit Unfold for massive speedup
    # We need (Batch, 3, Window)
    # Cn: (3, T)
    # Unfold along T (dim 1)
    
    # Pad to ensure last window covers end?
    # For simplicity, stick to valid windows
    
    # If T < window_len, fallback
    if T < window_len:
        return torch.tensor([], device=device)

    windows = Cn.unfold(1, window_len, stride_len) # (3, N_win, Win_len)
    
    # Permute to (N_win, 3, Win_len)
    windows = windows.permute(1, 0, 2) 
    
    # Center each window
    # mean over dim 2
    means = torch.mean(windows, dim=2, keepdim=True)
    windows_centered = windows - means
    
    # Covariance: (X @ X.T) / L
    # windows_centered is (B, 3, L)
    # we want (B, 3, 3)
    covs = torch.bmm(windows_centered, windows_centered.transpose(1, 2)) / window_len
    
    # Eigen decomposition (batch support!)
    eig_vals, eig_vecs = torch.linalg.eigh(covs)
    
    # Sort eigenvalues (ascending by default in eigh) -> we want descending
    # Flip
    eig_vals = torch.flip(eig_vals, [1])
    eig_vecs = torch.flip(eig_vecs, [2]) # eigenvectors are columns
    
    # Pulse is projection onto 2nd component (index 1)
    # vec is (B, 3, 3), we want 2nd column: (B, 3)
    # But wait, `eig_vecs` returned by eigh: column v[:, i] is eigenvector.
    # We flipped dim 2, so index 1 is indeed 2nd component.
    
    vecs_p = eig_vecs[:, :, 1].unsqueeze(1) # (B, 1, 3)
    
    # Project: p = v.T @ X
    # (B, 1, 3) @ (B, 3, L) -> (B, 1, L)
    p_segments = torch.bmm(vecs_p, windows_centered).squeeze(1) # (B, L)
    
    # Sign alignment with Green channel
    # Green is channel 1 in windows (B, 3, L) -> (B, L)
    greens = windows[:, 1, :]
    
    # Correlation check
    # Dot product centered p and green
    # (B, L) . (B, L) -> (B)
    # Just sign of dot product suffices for direction
    dots = torch.sum(p_segments * greens, dim=1)
    signs = torch.sign(dots).unsqueeze(1) # (B, 1)
    # Fix signs: if sign is 0?
    signs[signs==0] = 1
    
    p_segments = p_segments * signs
    
    # Overlap Add
    # This part is sequential-ish unless we do scatter_add
    # Given the strides are strict, we can loop to add to result
    # (B, L)
    
    # We can compute indices
    # num_windows = B
    
    # Loop is fine here, it's just adding arrays
    num_windows = p_segments.shape[0]
    for i in range(num_windows):
        start = i * stride_len
        end = start + window_len
        bvp_full[start:end] += p_segments[i] * win
        weights[start:end] += win
        
    weights[weights < 1e-6] = 1.0
    BVP = bvp_full / weights
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_pbv_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    R = signal[:, 0]
    G = signal[:, 1]
    B = signal[:, 2]
    
    R_mean = torch.mean(R)
    G_mean = torch.mean(G)
    B_mean = torch.mean(B)
    
    R_norm = R / R_mean
    G_norm = G / G_mean
    B_norm = B / B_mean
    
    std_R = torch.std(R_norm)
    std_G = torch.std(G_norm)
    std_B = torch.std(B_norm)
    
    if std_G == 0 or std_B == 0:
        return torch.tensor([], device=device)
        
    BVP = (std_R / std_G) * G_norm - (std_R / std_B) * B_norm
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_lgi_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    signal = detrend_torch(signal)
    
    R = signal[:, 0]
    G = signal[:, 1]
    B = signal[:, 2]
    
    X = R - G
    Y = R + G - 2*B
    
    std_X = torch.std(X)
    std_Y = torch.std(Y)
    
    if std_Y == 0: return torch.tensor([], device=device)
    
    alpha = std_X / std_Y
    BVP = X - alpha * Y
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_samc_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    R = signal[:, 0]
    G = signal[:, 1]
    B = signal[:, 2]
    
    # Normalize (Z-scoreish)
    R_norm = (R - torch.mean(R)) / (torch.std(R) + 1e-8)
    G_norm = (G - torch.mean(G)) / (torch.std(G) + 1e-8)
    B_norm = (B - torch.mean(B)) / (torch.std(B) + 1e-8)
    
    BVP = 0.5 * G_norm + 0.3 * R_norm + 0.2 * B_norm
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_2sr_torch(raw_signal, fs, device='cuda'):
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    C = signal.T
    mean_C = torch.mean(C, dim=1, keepdim=True)
    Cn = C / mean_C
    
    Cn_centered = Cn - torch.mean(Cn, dim=1, keepdim=True)
    
    P1 = torch.tensor([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], device=device) / np.sqrt(6)
    
    S = torch.matmul(P1, Cn_centered)
    
    std_s0 = torch.std(S[0, :])
    std_s1 = torch.std(S[1, :])
    
    if std_s1 == 0: return torch.tensor([], device=device)
    
    BVP = S[0, :] + (std_s0 / std_s1) * S[1, :]
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)

def process_signal_omit_torch(raw_signal, fs, device='cuda'):
    """OMIT on GPU"""
    signal = ensure_tensor(raw_signal, device)
    if len(signal) < 30: return torch.tensor([], device=device)
    
    signal = detrend_torch(signal)
    
    cov = torch.cov(signal.T)
    
    try:
        eig_vals, eig_vecs = torch.linalg.eigh(cov)
    except:
        return torch.tensor([], device=device)
    
    # Sort descending
    eig_vals = torch.flip(eig_vals, [0])
    eig_vecs = torch.flip(eig_vecs, [1])
    
    transformed = torch.matmul(signal, eig_vecs)
    
    BVP = transformed[:, 1]
    
    return bandpass_filter_torch(BVP, fs, FS_MIN, FS_MAX)
