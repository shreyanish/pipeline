"""
Evaluation metrics for rPPG signal quality assessment
Implements multiple metrics including rBS (relative BVP similarity),
correlation-based, spectral, and temporal metrics.
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_mae(ground_truth: np.ndarray, estimated: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    if len(ground_truth) != len(estimated):
        min_len = min(len(ground_truth), len(estimated))
        ground_truth = ground_truth[:min_len]
        estimated = estimated[:min_len]
    return np.mean(np.abs(ground_truth - estimated))


def calculate_mse(ground_truth: np.ndarray, estimated: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    if len(ground_truth) != len(estimated):
        min_len = min(len(ground_truth), len(estimated))
        ground_truth = ground_truth[:min_len]
        estimated = estimated[:min_len]
    return np.mean((ground_truth - estimated) ** 2)


def calculate_rmse(ground_truth: np.ndarray, estimated: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(calculate_mse(ground_truth, estimated))


def calculate_pcc(ground_truth: np.ndarray, estimated: np.ndarray) -> float:
    """Calculate Pearson's Correlation Coefficient"""
    if len(ground_truth) != len(estimated):
        min_len = min(len(ground_truth), len(estimated))
        ground_truth = ground_truth[:min_len]
        estimated = estimated[:min_len]
    
    if np.std(ground_truth) == 0 or np.std(estimated) == 0:
        return 0.0
        
    correlation_matrix = np.corrcoef(ground_truth, estimated)
    return correlation_matrix[0, 1]


def calculate_variance(sig: np.ndarray) -> float:
    """Calculate Signal Variance (σ²)"""
    return float(np.var(sig))


def calculate_zcr(sig: np.ndarray) -> float:
    """Calculate Zero-Crossing Rate"""
    # Center signal around mean if not already
    centered = sig - np.mean(sig)
    if len(sig) <= 1:
        return 0.0
    return float(((centered[:-1] * centered[1:]) < 0).sum()) / len(sig)


def get_psd(sig: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Power Spectral Density using Welch method"""
    nperseg = min(len(sig), 256)
    if nperseg < 2:
        return np.array([]), np.array([])
    f, pxx = scipy_signal.welch(sig, fps, nperseg=nperseg)
    return f, pxx


def calculate_spectral_metrics(sig: np.ndarray, fps: float, hr_range: Tuple[float, float] = (0.75, 4.0)) -> Dict[str, float]:
    """
    Calculate spectral metrics: SNR, NSQI, PSD_peak, SE
    
    Args:
        sig: Input signal
        fps: Sampling frequency
        hr_range: Heart rate range in Hz (default 0.75-4.0 Hz / 45-240 bpm)
    """
    f, pxx = get_psd(sig, fps)
    
    if len(f) == 0 or len(pxx) == 0:
        return {'SNR': 0.0, 'NSQI': 0.0, 'PSD_peak': 0.0, 'SE': 0.0}
        
    # Define HR band mask
    mask = (f >= hr_range[0]) & (f <= hr_range[1])
    if not np.any(mask):
        return {'SNR': 0.0, 'NSQI': 0.0, 'PSD_peak': 0.0, 'SE': 0.0}
    
    f_hr = f[mask]
    pxx_hr = pxx[mask]
    
    # PSD Peak
    psd_peak_idx = np.argmax(pxx_hr)
    psd_peak_val = pxx_hr[psd_peak_idx]
    psd_peak_freq = f_hr[psd_peak_idx]
    
    # SNR calculation (simplified: power in peak vs total power in band)
    # A better SNR would define a narrow window around peak
    peak_mask = (f >= psd_peak_freq - 0.1) & (f <= psd_peak_freq + 0.1)
    signal_power = np.sum(pxx[peak_mask])
    noise_power = np.sum(pxx[mask]) - signal_power
    
    snr = 10 * np.log10(signal_power / max(noise_power, 1e-10)) if signal_power > 0 else 0.0
    
    # NSQI (Normalized Signal Quality Index)
    nsqi = signal_power / np.sum(pxx[mask]) if np.sum(pxx[mask]) > 0 else 0.0
    
    # Spectral Entropy
    norm_pxx = pxx_hr / np.sum(pxx_hr)
    se = entropy(norm_pxx)
    
    return {
        'SNR': float(snr),
        'NSQI': float(nsqi),
        'PSD_peak': float(psd_peak_val),
        'SE': float(se)
    }


def calculate_rbs(ground_truth: np.ndarray, 
                  estimated_signals: Dict[str, np.ndarray],
                  epsilon: float = 1e-10) -> Dict[str, float]:
    """Calculate rBS (relative BVP similarity) for multiple algorithms"""
    mae_scores = {}
    rmse_scores = {}
    pcc_scores = {}
    
    for algo_name, signal in estimated_signals.items():
        mae_scores[algo_name] = calculate_mae(ground_truth, signal)
        rmse_scores[algo_name] = calculate_rmse(ground_truth, signal)
        # Use abs(PCC) as per the common rBS implementation
        pcc_scores[algo_name] = np.abs(calculate_pcc(ground_truth, signal))
    
    max_mae = max(mae_scores.values()) if mae_scores else 0
    max_rmse = max(rmse_scores.values()) if rmse_scores else 0
    
    rbs_scores = {}
    for algo_name in estimated_signals.keys():
        mae = mae_scores[algo_name]
        rmse = rmse_scores[algo_name]
        pcc_abs = pcc_scores[algo_name]
        
        log_mae_term = np.log(max_mae - mae + epsilon)
        log_rmse_term = np.log(max_rmse - rmse + epsilon)
        rbs = (log_mae_term + log_rmse_term) * pcc_abs
        
        rbs_scores[algo_name] = float(rbs)
    
    return rbs_scores


def evaluate_algorithms(ground_truth: np.ndarray,
                        estimated_signals: Dict[str, np.ndarray],
                        fps: float = 30.0) -> Dict[str, Dict[str, float]]:
    """Comprehensive evaluation of multiple algorithms"""
    results = {}
    
    # Calculate individual metrics
    for algo_name, sig in estimated_signals.items():
        # Spectral metrics
        spectral = calculate_spectral_metrics(sig, fps)
        
        pcc = calculate_pcc(ground_truth, sig)
        
        results[algo_name] = {
            'MSE': calculate_mse(ground_truth, sig),
            'RMSE': calculate_rmse(ground_truth, sig),
            'MAE': calculate_mae(ground_truth, sig),
            'PCC': pcc,
            '|PCC|': np.abs(pcc),
            'SNR': spectral['SNR'],
            'NSQI': spectral['NSQI'],
            'PSD_peak': spectral['PSD_peak'],
            'SE': spectral['SE'],
            'Variance': calculate_variance(sig),
            'ZCR': calculate_zcr(sig)
        }
    
    # Calculate rBS scores
    if estimated_signals:
        rbs_scores = calculate_rbs(ground_truth, estimated_signals)
        for algo_name, rbs in rbs_scores.items():
            results[algo_name]['rBS'] = rbs
    
    return results


def evaluate_blind_metrics(sig: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    """
    Calculate metrics that don't require ground truth.
    
    Returns:
        Dictionary with SNR, NSQI, PSD_peak, SE, Variance, and ZCR.
    """
    spectral = calculate_spectral_metrics(sig, fps)
    
    return {
        'SNR': spectral['SNR'],
        'NSQI': spectral['NSQI'],
        'PSD_peak': spectral['PSD_peak'],
        'SE': spectral['SE'],
        'Variance': calculate_variance(sig),
        'ZCR': calculate_zcr(sig)
    }


def save_results_to_csv(results: Dict[str, Dict[str, float]], filepath: str):
    """Save evaluation results to a CSV file"""
    data = []
    for algo_name, metrics in results.items():
        row = {'Algorithm': algo_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    # Reorder columns to match the categories in the request
    cols = ['Algorithm', 'MSE', 'RMSE', 'MAE', 'PCC', '|PCC|', 'SNR', 'NSQI', 'PSD_peak', 'SE', 'Variance', 'ZCR', 'rBS']
    # Filter to only existing columns
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")
    return df
