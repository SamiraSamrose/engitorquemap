#ml_models/training/train_timeshift_predictor.py
"""
Training script for time-shift predictor
Trains CNN + Gradient Boosting hybrid model
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'backend'))

from config import settings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow not available, training GB model only")


def generate_synthetic_timeshift_data(n_samples=5000):
    """Generate synthetic time-shift training data"""
    np.random.seed(42)
    
    data = {
        'speed': np.random.uniform(20, 70, n_samples),
        'brake_pressure': np.random.uniform(0, 100, n_samples),
        'throttle': np.random.uniform(0, 100, n_samples),
        'lateral_g': np.random.uniform(-2, 2, n_samples),
        'longitudinal_g': np.random.uniform(-1.5, 1.5, n_samples),
        'steering_angle': np.random.uniform(-180, 180, n_samples),
        'brake_shift': np.random.uniform(-20, 20, n_samples),
        'throttle_shift': np.random.uniform(-15, 15, n_samples),
        'apex_speed_change': np.random.uniform(-5, 5, n_samples),
        'rotation_change': np.random.uniform(-10, 10, n_samples),
        'corner_curvature': np.random.uniform(0, 0.05, n_samples),
        'elevation_change': np.random.uniform(-10, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Target: time delta (synthetic formula)
    df['time_delta'] = (
        -df['brake_shift'] * 0.015 +
        df['throttle_shift'] * 0.020 -
        df['apex_speed_change'] * 0.05 +
        np.random.normal(0, 0.05, n_samples)
    )
    
    return df


def train_gradient_boosting():
    """Train Gradient Boosting model"""
    print("Generating training data...")
    df = generate_synthetic_timeshift_data()
    
    feature_cols = [
        'speed', 'brake_pressure', 'throttle', 'lateral_g', 'longitudinal_g',
        'steering_angle', 'brake_shift', 'throttle_shift', 'apex_speed_change',
        'rotation_change', 'corner_curvature', 'elevation_change'
    ]
    
    X = df[feature_cols].values
    y = df['time_delta'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    
    # Train Gradient Boosting
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = gb_model.predict(X_train)
    test_pred = gb_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\nGradient Boosting Results:")
    print(f"Training RMSE: {train_rmse:.4f}s")
    print(f"Test RMSE: {test_rmse:.4f}s")
    print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
    
    # Save GB model
    gb_path = Path(settings.TIMESHIFT_MODEL_PATH.replace('.h5', '_gb.pkl'))
    gb_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(gb_path, 'wb') as f:
        pickle.dump(gb_model, f)
    
    print(f"\nGB model saved to: {gb_path}")
    
    return gb_model


def train_cnn_model():
    """Train CNN model for energy maps"""
    if not KERAS_AVAILABLE:
        print("Skipping CNN training (TensorFlow not available)")
        return
    
    print("\nTraining CNN model...")
    
    # Generate synthetic energy maps
    n_samples = 1000
    X_maps = np.random.randn(n_samples, 32, 32, 1)
    y = np.random.uniform(-0.5, 0.5, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_maps, y, test_size=0.2, random_state=42
    )
    
    # Build CNN
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nCNN Test MAE: {test_mae:.4f}s")
    
    # Save CNN model
    cnn_path = Path(settings.TIMESHIFT_MODEL_PATH)
    cnn_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cnn_path))
    
    print(f"CNN model saved to: {cnn_path}")


if __name__ == '__main__':
    train_gradient_boosting()
    train_cnn_model()