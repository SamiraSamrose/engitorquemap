#backend/models/driver_signature.py (STEP 3: Driver Energy Signature)
"""
Driver Signature Model - STEP 3
Clusters drivers using telemetry + energy vectors via UMAP + HDBSCAN
Creates driver styles: late-braker, high-rotation, throttle-stabber, smooth-roller
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import pickle
from pathlib import Path
import logging

try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logging.warning("UMAP or HDBSCAN not installed. Driver clustering will be limited.")

from config import settings

logger = logging.getLogger(__name__)


class DriverSignatureModel:
    """
    Driver signature and style profiling model
    """
    
    def __init__(self):
        self.umap_model = None
        self.hdbscan_model = None
        self.scaler = StandardScaler()
        self.driver_profiles = {}
        self.cluster_labels = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained UMAP + HDBSCAN model"""
        model_path = Path(settings.DRIVER_CLUSTER_MODEL_PATH)
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.umap_model = model_data.get('umap')
                    self.hdbscan_model = model_data.get('hdbscan')
                    self.scaler = model_data.get('scaler', StandardScaler())
                    self.driver_profiles = model_data.get('profiles', {})
                logger.info("Loaded driver clustering model")
            except Exception as e:
                logger.error(f"Error loading driver model: {e}")
        else:
            logger.warning("Driver clustering model not found, will train on first use")
    
    def save_model(self):
        """Save trained model"""
        model_path = Path(settings.DRIVER_CLUSTER_MODEL_PATH)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'umap': self.umap_model,
            'hdbscan': self.hdbscan_model,
            'scaler': self.scaler,
            'profiles': self.driver_profiles
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved driver model to {model_path}")
    
    async def get_signature(self, driver_id: str) -> Dict:
        """
        Get driver energy signature and style profile
        
        Returns driver characteristics:
        - Braking style (early/late, aggressive/smooth)
        - Throttle application (progressive/stabber)
        - Cornering style (rotation/understeer)
        - Energy efficiency metrics
        """
        if driver_id in self.driver_profiles:
            return self.driver_profiles[driver_id]
        
        # If not in cache, need to analyze driver data
        logger.info(f"Analyzing driver {driver_id} signature")
        
        # Load driver telemetry data
        driver_data = await self._load_driver_data(driver_id)
        
        if driver_data is None:
            return {
                "driver_id": driver_id,
                "status": "insufficient_data",
                "message": "Not enough data to generate signature"
            }
        
        # Compute signature
        signature = await self.analyze_style(driver_data)
        
        # Cache the profile
        self.driver_profiles[driver_id] = signature
        
        return signature
    
    async def _load_driver_data(self, driver_id: str) -> Optional[pd.DataFrame]:
        """Load all telemetry data for a driver"""
        data_dir = Path(settings.PROCESSED_DATA_DIR)
        
        # Look for files matching driver_id
        driver_files = list(data_dir.glob(f"*{driver_id}*.csv")) + list(data_dir.glob(f"*{driver_id}*.parquet"))
        
        if not driver_files:
            logger.warning(f"No data files found for driver {driver_id}")
            return None
        
        # Load and combine all files
        dfs = []
        for file in driver_files:
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
        
        if not dfs:
            return None
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    async def analyze_style(self, driver_data: Dict) -> Dict:
        """
        Analyze driver style from telemetry and energy data
        
        Extracts features:
        - Braking point distribution (early/late)
        - Braking pressure characteristics
        - Throttle application timing and aggression
        - Steering smoothness
        - Energy efficiency
        """
        # If driver_data is a dict (single snapshot), convert to DataFrame
        if isinstance(driver_data, dict):
            df = pd.DataFrame([driver_data])
        else:
            df = driver_data
        
        # Extract key features
        features = self._extract_driver_features(df)
        
        # Classify style
        style_classification = self._classify_driver_style(features)
        
        # Generate signature
        signature = {
            "driver_id": driver_data.get('driver_id', 'unknown') if isinstance(driver_data, dict) else 'unknown',
            "style": style_classification,
            "features": features,
            "recommendations": self._generate_recommendations(features, style_classification)
        }
        
        return signature
    
    def _extract_driver_features(self, df: pd.DataFrame) -> Dict:
        """Extract characteristic features from driver telemetry"""
        
        features = {}
        
        # Braking features
        if 'pbrake_f' in df.columns:
            brake_data = df['pbrake_f'].values
            brake_events = brake_data > 5.0  # Threshold for braking
            
            features['avg_brake_pressure'] = float(np.mean(brake_data[brake_events])) if np.any(brake_events) else 0.0
            features['max_brake_pressure'] = float(np.max(brake_data))
            features['brake_aggression'] = float(np.std(brake_data[brake_events])) if np.any(brake_events) else 0.0
        
        # Throttle features
        if 'ath' in df.columns or 'aps' in df.columns:
            throttle_col = 'ath' if 'ath' in df.columns else 'aps'
            throttle_data = df[throttle_col].values
            
            # Throttle application rate
            throttle_diff = np.diff(throttle_data)
            features['avg_throttle_application_rate'] = float(np.mean(throttle_diff[throttle_diff > 0])) if len(throttle_diff[throttle_diff > 0]) > 0 else 0.0
            features['throttle_aggression'] = float(np.std(throttle_diff))
            features['avg_throttle_position'] = float(np.mean(throttle_data))
        
        # Steering features
        if 'Steering_Angle' in df.columns:
            steering_data = df['Steering_Angle'].values
            steering_diff = np.diff(steering_data)
            
            features['avg_steering_angle'] = float(np.mean(np.abs(steering_data)))
            features['steering_smoothness'] = float(1.0 / (np.std(steering_diff) + 0.001))  # Inverse of std = smoothness
            features['max_steering_angle'] = float(np.max(np.abs(steering_data)))
        
        # Lateral g features (cornering)
        if 'accy_can' in df.columns:
            lateral_g = df['accy_can'].values
            features['avg_lateral_g'] = float(np.mean(np.abs(lateral_g)))
            features['max_lateral_g'] = float(np.max(np.abs(lateral_g)))
            features['cornering_aggression'] = float(np.percentile(np.abs(lateral_g), 90))
        
        # Longitudinal g features
        if 'accx_can' in df.columns:
            long_g = df['accx_can'].values
            features['avg_longitudinal_g'] = float(np.mean(long_g))
            features['max_acceleration'] = float(np.max(long_g))
            features['max_deceleration'] = float(np.min(long_g))
        
        # Speed management
        if 'speed' in df.columns:
            speed_data = df['speed'].values
            features['avg_speed'] = float(np.mean(speed_data))
            features['min_speed'] = float(np.min(speed_data))
            features['speed_variance'] = float(np.std(speed_data))
        
        return features
    
    def _classify_driver_style(self, features: Dict) -> str:
        """
        Classify driver into style categories:
        - late_braker: Late braking points, high brake pressure
        - early_braker: Early braking, smoother deceleration
        - high_rotation: Aggressive steering, high lateral g
        - throttle_stabber: Aggressive throttle application
        - smooth_roller: Smooth inputs, consistent speed
        """
        # Decision tree based on features
        
        # Braking style
        brake_aggression = features.get('brake_aggression', 0)
        brake_pressure = features.get('avg_brake_pressure', 0)
        
        # Throttle style
        throttle_aggression = features.get('throttle_aggression', 0)
        throttle_rate = features.get('avg_throttle_application_rate', 0)
        
        # Steering style
        steering_smoothness = features.get('steering_smoothness', 1.0)
        cornering_aggression = features.get('cornering_aggression', 0)
        
        # Classification logic
        if brake_pressure > 50 and brake_aggression > 10:
            primary_style = "late_braker"
        elif brake_pressure < 30 and brake_aggression < 5:
            primary_style = "early_braker"
        elif throttle_aggression > 15 and throttle_rate > 5:
            primary_style = "throttle_stabber"
        elif cornering_aggression > 0.8 and steering_smoothness < 0.5:
            primary_style = "high_rotation"
        elif steering_smoothness > 0.7 and throttle_aggression < 10:
            primary_style = "smooth_roller"
        else:
            primary_style = "balanced"
        
        return primary_style
    
    def _generate_recommendations(self, features: Dict, style: str) -> List[str]:
        """Generate personalized recommendations based on style"""
        recommendations = []
        
        if style == "late_braker":
            recommendations.append("Consider earlier brake application to reduce tire stress")
            recommendations.append("Smoother brake release can improve rotation")
        
        elif style == "early_braker":
            recommendations.append("Try braking slightly later to carry more speed")
            recommendations.append("You have good brake control - can afford more aggression")
        
        elif style == "throttle_stabber":
            recommendations.append("Progressive throttle application will improve traction")
            recommendations.append("Smoother inputs reduce tire wear and improve exit speed")
        
        elif style == "high_rotation":
            recommendations.append("Excellent rotation - ensure tire temps stay optimal")
            recommendations.append("Consider slightly earlier throttle for better drive")
        
        elif style == "smooth_roller":
            recommendations.append("Great consistency - try pushing limits in key corners")
            recommendations.append("Your smoothness is an asset for tire management")
        
        # Feature-specific recommendations
        if features.get('steering_smoothness', 1.0) < 0.3:
            recommendations.append("Steering inputs could be smoother for better balance")
        
        if features.get('brake_aggression', 0) > 20:
            recommendations.append("High brake aggression detected - check for lock-ups")
        
        return recommendations
    
    async def get_clusters(self) -> Dict:
        """Get all driver clusters and their characteristics"""
        
        if not self.driver_profiles:
            return {"clusters": [], "message": "No driver data available yet"}
        
        # Group drivers by style
        clusters = {}
        for driver_id, profile in self.driver_profiles.items():
            style = profile.get('style', 'unknown')
            
            if style not in clusters:
                clusters[style] = {
                    "style_name": style,
                    "drivers": [],
                    "count": 0,
                    "avg_features": {}
                }
            
            clusters[style]['drivers'].append(driver_id)
            clusters[style]['count'] += 1
        
        return {"clusters": list(clusters.values())}
    
    def train_clustering(self, all_driver_data: pd.DataFrame):
        """
        Train UMAP + HDBSCAN clustering on driver telemetry
        
        Args:
            all_driver_data: DataFrame with telemetry from all drivers
                             Must include 'driver_id' column
        """
        logger.info("Training driver clustering model...")
        
        # Extract features for all drivers
        driver_ids = all_driver_data['driver_id'].unique()
        
        feature_matrix = []
        driver_labels = []
        
        for driver_id in driver_ids:
            driver_df = all_driver_data[all_driver_data['driver_id'] == driver_id]
            
            if len(driver_df) < 100:  # Minimum data requirement
                continue
            
            features = self._extract_driver_features(driver_df)
            
            # Convert features to vector
            feature_vector = [
                features.get('avg_brake_pressure', 0),
                features.get('brake_aggression', 0),
                features.get('avg_throttle_application_rate', 0),
                features.get('throttle_aggression', 0),
                features.get('steering_smoothness', 0),
                features.get('avg_lateral_g', 0),
                features.get('cornering_aggression', 0),
                features.get('avg_speed', 0),
                features.get('speed_variance', 0)
            ]
            
            feature_matrix.append(feature_vector)
            driver_labels.append(driver_id)
        
        feature_matrix = np.array(feature_matrix)
        
        # Normalize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # UMAP dimensionality reduction
        self.umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=3,
            metric='euclidean',
            random_state=42
        )
        
        embedding = self.umap_model.fit_transform(feature_matrix_scaled)
        
        # HDBSCAN clustering
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean'
        )
        
        cluster_labels = self.hdbscan_model.fit_predict(embedding)
        
        # Store cluster assignments
        for i, driver_id in enumerate(driver_labels):
            self.cluster_labels[driver_id] = int(cluster_labels[i])
        
        logger.info(f"Clustering complete. Found {len(set(cluster_labels))} clusters")
        
        # Save model
        self.save_model()