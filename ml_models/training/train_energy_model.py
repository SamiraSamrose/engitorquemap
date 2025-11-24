#ml_models/training/train_energy_model.py
"""
Training script for energy prediction model
Trains XGBoost model for energy consumption prediction
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'backend'))

from config import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


def generate_synthetic_training_data(n_samples=10000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    data = {
        'speed': np.random.uniform(10, 80, n_samples),
        'gear': np.random.randint(1, 7, n_samples),
        'nmot': np.random.uniform(2000, 8000, n_samples),
        'aps': np.random.uniform(0, 100, n_samples),
        'ath': np.random.uniform(0, 100, n_samples),
        'pbrake_f': np.random.uniform(0, 100, n_samples),
        'pbrake_r': np.random.uniform(0, 80, n_samples),
        'accx_can': np.random.uniform(-1.5, 1.5, n_samples),
        'accy_can': np.random.uniform(-2.0, 2.0, n_samples),
        'braking_power': np.random.uniform(0, 150000, n_samples),
        'acceleration_power': np.random.uniform(0, 200000, n_samples),
        'yaw_energy': np.random.uniform(0, 50000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Target: energy consumption (synthetic)
    df['energy_consumption'] = (
        df['braking_power'] * 0.7 +
        df['acceleration_power'] * 0.5 +
        df['yaw_energy'] * 0.3 +
        np.random.normal(0, 5000, n_samples)
    )
    
    return df


def train_energy_model():
    """Train energy prediction model"""
    print("Generating training data...")
    df = generate_synthetic_training_data()
    
    # Features and target
    feature_cols = [
        'speed', 'gear', 'nmot', 'aps', 'ath', 
        'pbrake_f', 'pbrake_r', 'accx_can', 'accy_can',
        'braking_power', 'acceleration_power', 'yaw_energy'
    ]
    
    X = df[feature_cols].values
    y = df['energy_consumption'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Save model
    model_path = Path(settings.ENERGY_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    
    return model


if __name__ == '__main__':
    train_energy_model()