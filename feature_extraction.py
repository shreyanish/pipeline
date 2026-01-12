import numpy as np
from scipy.signal import butter, filtfilt

def extract_spo2_features(raw_signal: np.ndarray, filtered_bvp: np.ndarray, fs: float) -> dict:
    if filtered_bvp.size == 0: return {}

    nyquist = 0.5 * fs
    
    # Filters for feature extraction
    b_dc, a_dc = butter(2, 0.5 / nyquist, btype='low') 
    b_ac, a_ac = butter(2, 0.7 / nyquist, btype='high') 

    # Extract components (Zero-phase)
    R_raw = raw_signal[:, 0]
    G_raw = raw_signal[:, 1]
    B_raw = raw_signal[:, 2]
    
    # DC = Baseline (Low freq)
    R_DC = np.mean(filtfilt(b_dc, a_dc, R_raw))
    G_DC = np.mean(filtfilt(b_dc, a_dc, G_raw))
    B_DC = np.mean(filtfilt(b_dc, a_dc, B_raw))
    
    # AC = Pulsatile (High freq)
    R_AC = np.std(filtfilt(b_ac, a_ac, R_raw))
    G_AC = np.std(filtfilt(b_ac, a_ac, G_raw))
    B_AC = np.std(filtfilt(b_ac, a_ac, B_raw))
    
    # Calculate ratios (with safety checks)
    R_Ratio = np.nan
    R_Green_Blue = np.nan
    
    if R_DC != 0 and B_DC != 0:
        R_Ratio = (R_AC / R_DC) / (B_AC / B_DC)
    
    if G_DC != 0 and B_DC != 0:
        R_Green_Blue = (G_AC / G_DC) / (B_AC / B_DC)
    
    return {
        'R_Ratio': R_Ratio,
        'R_Green_Blue': R_Green_Blue,
        'AC_Red': R_AC,
        'DC_Red': R_DC,
        'AC_Green': G_AC,
        'DC_Green': G_DC,
        'AC_Blue': B_AC,
        'DC_Blue': B_DC
    }
