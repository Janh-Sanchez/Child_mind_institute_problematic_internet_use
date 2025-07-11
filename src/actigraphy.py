import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def summarize_actigraphy(df, subject_id_col='Subject_ID'):
    """
    Enhanced actigraphy processing with:
    - Percentiles (10th, 25th, 75th, 90th)
    - Robust statistical measures (IQR, MAD)
    - Frequency domain features (FFT)
    """
    exclude_cols = [subject_id_col, 'timestamp']
    num_cols = [col for col in df.columns 
               if col not in exclude_cols 
               and pd.api.types.is_numeric_dtype(df[col])]
    
    # Time-domain features
    stats = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median,
        'skew': skew,
        'kurtosis': kurtosis,
        'q1': lambda x: np.percentile(x, 25),
        'q3': lambda x: np.percentile(x, 75),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'mad': lambda x: np.median(np.abs(x - np.median(x)))
    }
    
    # Frequency-domain features (simplified FFT)
    def dominant_freq(x):
        if len(x) < 2: return 0
        fft = np.abs(np.fft.fft(x))
        return np.argmax(fft[1:len(fft)//2]) + 1
    
    summary = df.groupby(subject_id_col)[num_cols].agg(stats)
    summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
    
    # Add frequency features
    freq_features = df.groupby(subject_id_col)[num_cols].agg(dominant_freq)
    freq_features.columns = [f'{col}_dominant_freq' for col in freq_features.columns]
    
    return pd.concat([summary, freq_features], axis=1).reset_index()