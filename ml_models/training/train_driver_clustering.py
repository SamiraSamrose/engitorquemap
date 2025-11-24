#ml_models/training/train_driver_clustering.py
"""
Training script for driver clustering model
Trains UMAP + HDBSCAN for driver profiling
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / 'backend'))

from config import settings
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler


def generate_synthetic_driver_data(n_drivers=100):
    """Generate synthetic driver data"""
    np.random.seed(42)
    
    driver_features = []
    driver_ids = []
    
    for i in range(n_drivers):
        driver_id = f"D{i:03d}"
        
        # Generate features for driver
        features = {
            'avg_brake_pressure': np.random.uniform(30, 80),
            'brake_aggression': np.random.uniform(5, 25),
            'avg_throttle_application_rate': np.random.uniform(2, 10),
            'throttle_aggression': np.random.uniform(5, 20),
            'steering_smoothness': np.random.uniform(0.3, 1.0),
            'avg_lateral_g': np.random.uniform(0.5, 1.5),
            'cornering_aggression': np.random.uniform(0.4, 1.2),
            'avg_speed': np.random.uniform(35, 55),
            'speed_variance': np.random.uniform(5, 15)
        }
        
        driver_features.append(list(features.values()))
        driver_ids.append(driver_id)
    
    return np.array(driver_features), driver_ids


def train_driver_clustering():
    """Train driver clustering model"""
    print("Generating driver data...")
    features, driver_ids = generate_synthetic_driver_data()
    
    print(f"Number of drivers: {len(driver_ids)}")
    print(f"Feature dimensions: {features.shape[1]}")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # UMAP dimensionality reduction
    print("Performing UMAP dimensionality reduction...")
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=3,
        metric='euclidean',
        random_state=42
    )
    
    embedding = umap_model.fit_transform(features_scaled)
    print(f"Embedding shape: {embedding.shape}")
    
    # HDBSCAN clustering
    print("Performing HDBSCAN clustering...")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean'
    )
    
    cluster_labels = hdbscan_model.fit_predict(embedding)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nClusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")
    
    # Create profiles for each driver
    profiles = {}
    for driver_id, label in zip(driver_ids, cluster_labels):
        profiles[driver_id] = {
            'cluster': int(label),
            'embedding': embedding[driver_ids.index(driver_id)].tolist()
        }
    
    # Save model
    model_path = Path(settings.DRIVER_CLUSTER_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'umap': umap_model,
        'hdbscan': hdbscan_model,
        'scaler': scaler,
        'profiles': profiles
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")


if __name__ == '__main__':
    train_driver_clustering()