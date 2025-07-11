import pandas as pd
import numpy as np

def extract_time_features(df, subject_col='Subject_ID'):
    """Extrae 15 features temporales clave por sujeto"""
    features = []
    for subject_id, group in df.groupby(subject_col):
        if 'timestamp' in group.columns:
            time_diff = group['timestamp'].diff().dt.total_seconds()
            feat = {
                'Subject_ID': subject_id,
                'total_events': len(group),
                'active_hours': (time_diff < 3600).sum(),
                'night_activity': group[group['timestamp'].dt.hour.between(0, 6)]['value'].mean(),
                'max_activity': group['value'].max(),
                'std_activity': group['value'].std(),
            }
            features.append(feat)
    return pd.DataFrame(features)