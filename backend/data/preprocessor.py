## **File: backend/data/preprocessor.py** (Data Preprocessing)

"""
Data Preprocessor
Advanced preprocessing for telemetry and timing data
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name)
class DataPreprocessor:
"""
Advanced data preprocessing for telemetry and timing data
"""
def __init__(self):
    self.preprocessing_stats = {}

def preprocess_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw telemetry data
    
    Operations:
    - Remove outliers
    - Smooth noisy signals
    - Interpolate missing values
    - Calculate derived features
    """
    logger.info(f"Preprocessing telemetry: {len(df)} rows")
    
    df_processed = df.copy()
    
    # Remove outliers
    df_processed = self._remove_outliers(df_processed)
    
    # Smooth signals
    df_processed = self._smooth_signals(df_processed)
    
    # Interpolate missing values
    df_processed = self._interpolate_missing(df_processed)
    
    # Calculate derived features
    df_processed = self._calculate_derived_features(df_processed)
    
    # Add lap and sector information
    df_processed = self._add_lap_sector_info(df_processed)
    
    logger.info(f"Preprocessing complete: {len(df_processed)} rows")
    
    return df_processed

def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    """Remove statistical outliers using IQR method"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Replace outliers with NaN (will be interpolated later)
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    
    return df

def _smooth_signals(self, df: pd.DataFrame) -> pd.DataFrame:
    """Smooth noisy signals using Savitzky-Golay filter"""
    signals_to_smooth = ['speed', 'nmot', 'aps', 'ath', 'steering_angle']
    
    for signal in signals_to_smooth:
        if signal in df.columns:
            try:
                # Apply Savitzky-Golay filter
                window_length = min(51, len(df) // 10)
                if window_length % 2 == 0:
                    window_length += 1
                
                if window_length >= 5:
                    df[signal] = savgol_filter(
                        df[signal].fillna(method='ffill').fillna(method='bfill'),
                        window_length=window_length,
                        polyorder=3
                    )
            except Exception as e:
                logger.warning(f"Could not smooth {signal}: {e}")
    
    return df

def _interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df[col].isna().any():
            # Linear interpolation
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # Fill any remaining NaNs with forward/backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df

def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived features from raw telemetry"""
    
    # Speed change (acceleration/deceleration)
    if 'speed' in df.columns:
        df['speed_change'] = df['speed'].diff()
        df['is_accelerating'] = df['speed_change'] > 0.1
        df['is_braking'] = df['speed_change'] < -0.1
    
    # Brake bias
    if 'pbrake_f' in df.columns and 'pbrake_r' in df.columns:
        total_brake = df['pbrake_f'] + df['pbrake_r']
        df['brake_bias'] = np.where(total_brake > 0, df['pbrake_f'] / total_brake, 0.5)
    
    # Combined g-force
    if 'accx_can' in df.columns and 'accy_can' in df.columns:
        df['total_g'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)
    
    # Steering rate (how quickly steering changes)
    if 'steering_angle' in df.columns:
        df['steering_rate'] = df['steering_angle'].diff().abs()
    
    # Throttle/brake overlap detection
    if 'ath' in df.columns and 'pbrake_f' in df.columns:
        df['throttle_brake_overlap'] = (df['ath'] > 10) & (df['pbrake_f'] > 5)
    
    return df

def _add_lap_sector_info(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add lap and sector information"""
    
    # Detect lap changes using lap trigger or distance
    if 'laptrigger_lapdist_dls' in df.columns:
        df['lap'] = 0
        lap_counter = 0
        prev_dist = 0
        
        for idx in range(len(df)):
            current_dist = df.loc[idx, 'laptrigger_lapdist_dls']
            
            # Lap completed when distance resets
            if current_dist < prev_dist - 100:
                lap_counter += 1
            
            df.loc[idx, 'lap'] = lap_counter
            prev_dist = current_dist
        
        # Calculate sector (3 sectors per lap)
        max_dist = df['laptrigger_lapdist_dls'].max()
        sector_length = max_dist / 3
        
        df['sector'] = ((df['laptrigger_lapdist_dls'] // sector_length) % 3 + 1).astype(int)
    
    return df

def preprocess_timing_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess timing/results data
    """
    logger.info(f"Preprocessing timing data: {len(df)} rows")
    
    df_processed = df.copy()
    
    # Standardize column names
    df_processed.columns = [col.lower().replace(' ', '_') for col in df_processed.columns]
    
    # Convert time strings to seconds
    time_columns = [col for col in df_processed.columns if 'time' in col.lower()]
    for col in time_columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = self._parse_time_string(df_processed[col])
    
    # Remove invalid times
    for col in time_columns:
        df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
    
    return df_processed

def _parse_time_string(self, time_series: pd.Series) -> pd.Series:
    """Parse time strings (MM:SS.mmm) to seconds"""
    def parse_time(time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        
        try:
            # Handle format MM:SS.mmm
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(time_str)
        except:
            return np.nan
    
    return time_series.apply(parse_time)

def merge_telemetry_timing(self, telemetry_df: pd.DataFrame, 
                           timing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge telemetry and timing data
    """
    # Merge on common columns (e.g., lap, driver_id)
    merge_columns = []
    
    if 'lap' in telemetry_df.columns and 'lap' in timing_df.columns:
        merge_columns.append('lap')
    
    if 'driver_id' in telemetry_df.columns and 'driver_id' in timing_df.columns:
        merge_columns.append('driver_id')
    
    if merge_columns:
        merged = pd.merge(
            telemetry_df,
            timing_df,
            on=merge_columns,
            how='left',
            suffixes=('', '_timing')
        )
        return merged
    
    return telemetry_df

def calculate_statistics(self, df: pd.DataFrame) -> Dict:
    """Calculate statistics for processed data"""
    stats = {
        "total_rows": len(df),
        "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
        "missing_values": df.isna().sum().to_dict(),
        "value_ranges": {}
    }
    
    # Calculate ranges for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats["value_ranges"][col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }
    
    return stats