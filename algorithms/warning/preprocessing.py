from typing import Dict
import pandas as pd
import numpy as np
class FloodDataPreprocessor:
    def __init__(self, timezone='UTC+8', spatial_ref='CGCS2000'):
        self.timezone = timezone
        self.spatial_ref = spatial_ref
        self.feature_scalers = {}

    def temporal_alignment(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        aligned_data = {}
        for data_name, df in data_dict.items():
            if 'timestamp' not in df.columns and not df.index.name == 'timestamp':
                df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='1H')
                df.set_index('timestamp', inplace=True)
            if isinstance(df.index, pd.DatetimeIndex):
                if 'rainfall' in data_name or '气象' in data_name:
                    aligned_df = df.resample('1H').mean()
                elif 'water_level' in data_name or '水文' in data_name:
                    aligned_df = df.resample('15T').mean().resample('1H').mean()
                elif 'static' in data_name:
                    aligned_df = df.copy()
                    aligned_df['valid_time'] = 'permanent'
                else:
                    aligned_df = df.resample('1H').mean()
                aligned_data[data_name] = aligned_df.fillna(method='ffill').fillna(method='bfill')
            else:
                aligned_data[data_name] = df
        return aligned_data

    def normalize_features(self, X: np.ndarray, feature_name: str) -> np.ndarray:
        if X.size == 0:
            return X
        if feature_name not in self.feature_scalers:
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            self.feature_scalers[feature_name] = {'min': X_min, 'max': X_max}
        else:
            X_min, X_max = self.feature_scalers[feature_name].values()
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        return X_norm

    def detect_outliers(self, X: np.ndarray, sigma=3) -> np.ndarray:
        if X.size == 0:
            return X
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
        lower_bound, upper_bound = mean - sigma * std, mean + sigma * std
        X_clean = np.where(X < lower_bound, lower_bound, X)
        X_clean = np.where(X_clean > upper_bound, upper_bound, X_clean)
        return X_clean