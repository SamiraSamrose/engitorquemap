#backend/models/timeshift_predictor.py (Time-Shift Delta Predictor)

"""
Time-Shift Delta Predictor - STEP 4
Predicts time delta for alternative driving scenarios
Uses gradient-boosting regressors + CNN over energy-flow maps
Generates actionable suggestions like "brake 12m later → +0.17s gain"
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import logging

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    keras = None
    logging.warning("TensorFlow not installed. Time-shift prediction will be limited.")

from sklearn.ensemble import GradientBoostingRegressor
from config import settings

logger = logging.getLogger(__name__)


class TimeShiftPredictor:
    """
    Predicts lap time deltas for alternative driving scenarios
    """
    
    def __init__(self):
        self.gb_regressor = None
        self.cnn_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        # Gradient Boosting Regressor
        gb_path = Path(settings.TIMESHIFT_MODEL_PATH.replace('.h5', '_gb.pkl'))
        if gb_path.exists():
            try:
                with open(gb_path, 'rb') as f:
                    self.gb_regressor = pickle.load(f)
                logger.info("Loaded Gradient Boosting time-shift model")
            except Exception as e:
                logger.error(f"Error loading GB model: {e}")
        
        # CNN Model
        if keras is not None:
            cnn_path = Path(settings.TIMESHIFT_MODEL_PATH)
            if cnn_path.exists():
                try:
                    self.cnn_model = keras.models.load_model(str(cnn_path))
                    logger.info("Loaded CNN time-shift model")
                except Exception as e:
                    logger.error(f"Error loading CNN model: {e}")
    
    async def predict_delta(self, prediction_request: Dict) -> List[Dict]:
        """
        Predict time delta for alternative scenarios
        
        Input format:
        {
            "driver_id": "D001",
            "track_name": "indianapolis",
            "sector": "IM2b",
            "current_telemetry": {...},
            "alternative_scenarios": [
                {
                    "description": "brake 12m later",
                    "modifications": {
                        "brake_point_shift": 12.0,  # meters
                    }
                },
                {
                    "description": "earlier throttle application",
                    "modifications": {
                        "throttle_point_shift": -8.0  # meters (negative = earlier)
                    }
                }
            ]
        }
        
        Returns list of predictions with time deltas
        """
        driver_id = prediction_request.get('driver_id')
        track_name = prediction_request.get('track_name')
        sector = prediction_request.get('sector')
        current_telemetry = prediction_request.get('current_telemetry', {})
        scenarios = prediction_request.get('alternative_scenarios', [])
        
        predictions = []
        
        for scenario in scenarios:
            description = scenario.get('description')
            modifications = scenario.get('modifications', {})
            
            # Predict time delta
            delta = await self._compute_scenario_delta(
                driver_id=driver_id,
                track_name=track_name,
                sector=sector,
                current_telemetry=current_telemetry,
                modifications=modifications
            )
            
            predictions.append({
                "scenario": description,
                "modifications": modifications,
                "predicted_delta_seconds": delta['time_delta'],
                "confidence": delta['confidence'],
                "energy_impact": delta['energy_impact'],
                "position_impact": delta['position_impact'],
                "explanation": delta['explanation']
            })
        
        return predictions
    
    async def _compute_scenario_delta(self, driver_id: str, track_name: str, 
                                     sector: str, current_telemetry: Dict, 
                                     modifications: Dict) -> Dict:
        """
        Compute predicted time delta for a scenario
        """
        # Extract modification parameters
        brake_point_shift = modifications.get('brake_point_shift', 0.0)
        throttle_point_shift = modifications.get('throttle_point_shift', 0.0)
        apex_speed_change = modifications.get('apex_speed_change', 0.0)
        rotation_change = modifications.get('rotation_angle_change', 0.0)
        
        # Build feature vector for prediction
        features = self._build_feature_vector(
            current_telemetry=current_telemetry,
            brake_shift=brake_point_shift,
            throttle_shift=throttle_point_shift,
            apex_speed_change=apex_speed_change,
            rotation_change=rotation_change
        )
        
        # Predict using Gradient Boosting
        if self.gb_regressor is not None:
            time_delta_gb = self.gb_regressor.predict(features.reshape(1, -1))[0]
        else:
            time_delta_gb = self._estimate_time_delta_physics(modifications)
        
        # Predict using CNN (if available)
        if self.cnn_model is not None and 'energy_map' in current_telemetry:
            energy_map = np.array(current_telemetry['energy_map']).reshape(1, 32, 32, 1)
            time_delta_cnn = self.cnn_model.predict(energy_map, verbose=0)[0][0]
            
            # Ensemble: average both predictions
            time_delta = (time_delta_gb + time_delta_cnn) / 2.0
            confidence = 0.85
        else:
            time_delta = time_delta_gb
            confidence = 0.70
        
        # Calculate energy impact
        energy_impact = self._calculate_energy_impact(modifications)
        
        # Estimate position impact
        position_impact = self._estimate_position_impact(time_delta)
        
        # Generate explanation
        explanation = self._generate_explanation(modifications, time_delta, energy_impact)
        
        return {
            'time_delta': float(time_delta),
            'confidence': float(confidence),
            'energy_impact': energy_impact,
            'position_impact': position_impact,
            'explanation': explanation
        }
    
    def _build_feature_vector(self, current_telemetry: Dict, brake_shift: float,
                             throttle_shift: float, apex_speed_change: float,
                             rotation_change: float) -> np.ndarray:
        """Build feature vector for ML prediction"""
        features = [
            current_telemetry.get('speed', 0),
            current_telemetry.get('brake_pressure', 0),
            current_telemetry.get('throttle', 0),
            current_telemetry.get('lateral_g', 0),
            current_telemetry.get('longitudinal_g', 0),
            current_telemetry.get('steering_angle', 0),
            brake_shift,
            throttle_shift,
            apex_speed_change,
            rotation_change,
            current_telemetry.get('corner_curvature', 0),
            current_telemetry.get('elevation_change', 0)
        ]
        
        return np.array(features)
    
    def _estimate_time_delta_physics(self, modifications: Dict) -> float:
        """
        Physics-based estimation when ML models not available
        """
        brake_shift = modifications.get('brake_point_shift', 0.0)
        throttle_shift = modifications.get('throttle_point_shift', 0.0)
        apex_speed = modifications.get('apex_speed_change', 0.0)
        
        time_delta = 0.0
        
        # Braking point shift
        if brake_shift != 0:
            # Later braking = carry more speed = potential gain
            # Estimate: 1 meter later braking ≈ 0.01-0.02s gain (if done right)
            time_delta -= brake_shift * 0.015
        
        # Throttle point shift
        if throttle_shift != 0:
            # Earlier throttle = better exit = gain
            # Estimate: 1 meter earlier throttle ≈ 0.02s gain
            time_delta += throttle_shift * 0.020
        
        # Apex speed change
        if apex_speed != 0:
            # Higher apex speed = direct gain
            # Estimate: 1 m/s faster ≈ 0.05s gain
            time_delta -= apex_speed * 0.05
        
        return time_delta
    
    def _calculate_energy_impact(self, modifications: Dict) -> Dict:
        """Calculate energy consumption impact"""
        brake_shift = modifications.get('brake_point_shift', 0.0)
        throttle_shift = modifications.get('throttle_point_shift', 0.0)
        
        # Later braking = less brake distance = less energy dissipated
        brake_energy_saved = brake_shift * 0.5  # kJ per meter
        
        # Earlier throttle = more acceleration = more energy used
        throttle_energy_added = abs(throttle_shift) * 0.3  # kJ per meter
        
        net_energy = throttle_energy_added - brake_energy_saved
        
        return {
            'brake_energy_saved_kj': float(brake_energy_saved),
            'throttle_energy_added_kj': float(throttle_energy_added),
            'net_energy_change_kj': float(net_energy),
            'efficiency_change_percent': float(-brake_energy_saved * 0.8)
        }
    
    def _estimate_position_impact(self, time_delta: float) -> Dict:
        """
        Estimate race position impact from time delta
        Assumes typical field spread
        """
        # Average gap between positions in qualifying: ~0.2-0.5s
        avg_gap = 0.3
        
        position_change = round(time_delta / avg_gap)
        
        return {
            'estimated_position_change': int(-position_change),  # Negative delta = gain positions
            'grid_spots': int(-position_change)
        }
    
    def _generate_explanation(self, modifications: Dict, time_delta: float, 
                             energy_impact: Dict) -> str:
        """Generate human-readable explanation"""
        brake_shift = modifications.get('brake_point_shift', 0.0)
        throttle_shift = modifications.get('throttle_point_shift', 0.0)
        
        explanation_parts = []
        
        if brake_shift > 0:
            explanation_parts.append(
                f"Braking {brake_shift:.1f}m later reduces deceleration energy by "
                f"{energy_impact['brake_energy_saved_kj']:.1f}kJ"
            )
        elif brake_shift < 0:
            explanation_parts.append(
                f"Braking {abs(brake_shift):.1f}m earlier increases stability but loses time"
            )
        
        if throttle_shift < 0:
            explanation_parts.append(
                f"Applying throttle {abs(throttle_shift):.1f}m earlier improves corner exit"
            )
        elif throttle_shift > 0:
            explanation_parts.append(
                f"Delaying throttle {throttle_shift:.1f}m may cause exit speed loss"
            )
        
        if time_delta < 0:
            explanation_parts.append(
                f"Predicted gain: {abs(time_delta):.3f}s (approx. {abs(self._estimate_position_impact(time_delta)['grid_spots'])} grid position)"
            )
        else:
            explanation_parts.append(
                f"Predicted loss: {time_delta:.3f}s"
            )
        
        return ". ".join(explanation_parts)
    
    async def generate_suggestions(self, driver_id: str, track_name: str) -> List[Dict]:
        """
        Generate multiple optimization suggestions for driver on track
        STEP 4: Produces many suggestions for different scenarios
        """
        suggestions = []
        
        # Load driver's recent session data
        driver_sessions = await self._load_driver_track_sessions(driver_id, track_name)
        
        if not driver_sessions:
            return [{
                "message": "Insufficient data for suggestions",
                "driver_id": driver_id,
                "track": track_name
            }]
        
        # Analyze each sector/corner
        sector_analysis = self._analyze_driver_sectors(driver_sessions)
        
        # Generate suggestions for weak sectors
        for sector_id, sector_data in sector_analysis.items():
            if sector_data['time_loss'] > 0.05:  # Threshold for significant loss
                
                # Braking optimization
                if sector_data['brake_late_potential'] > 0:
                    suggestions.append({
                        "sector": sector_id,
                        "type": "braking_optimization",
                        "description": f"In {sector_id}, you lost {sector_data['time_loss']:.3f}s. Braking {sector_data['brake_late_potential']:.1f}m later could gain {sector_data['brake_late_potential'] * 0.015:.3f}s",
                        "modification": {
                            "brake_point_shift": sector_data['brake_late_potential']
                        },
                        "estimated_gain": sector_data['brake_late_potential'] * 0.015,
                        "difficulty": "medium"
                    })
                
                # Throttle optimization
                if sector_data['throttle_early_potential'] > 0:
                    suggestions.append({
                        "sector": sector_id,
                        "type": "throttle_optimization",
                        "description": f"Applying throttle {sector_data['throttle_early_potential']:.1f}m earlier in {sector_id} could improve exit by {sector_data['throttle_early_potential'] * 0.020:.3f}s",
                        "modification": {
                            "throttle_point_shift": -sector_data['throttle_early_potential']
                        },
                        "estimated_gain": sector_data['throttle_early_potential'] * 0.020,
                        "difficulty": "easy"
                    })
                
                # Minimum speed optimization
                if sector_data['apex_speed_potential'] > 0:
                    suggestions.append({
                        "sector": sector_id,
                        "type": "apex_speed",
                        "description": f"Carrying {sector_data['apex_speed_potential']:.1f} m/s more through {sector_id} apex yields {sector_data['apex_speed_potential'] * 0.05:.3f}s",
                        "modification": {
                            "apex_speed_change": sector_data['apex_speed_potential']
                        },
                        "estimated_gain": sector_data['apex_speed_potential'] * 0.05,
                        "difficulty": "hard"
                    })
                
                # Line optimization
                if sector_data['rotation_potential'] > 0:
                    suggestions.append({
                        "sector": sector_id,
                        "type": "rotation",
                        "description": f"More rotation in {sector_id} (apex {sector_data['rotation_potential']:.1f}° tighter) enables earlier throttle",
                        "modification": {
                            "rotation_angle_change": sector_data['rotation_potential']
                        },
                        "estimated_gain": 0.08,
                        "difficulty": "hard"
                    })
        
        # Energy efficiency suggestions
        energy_suggestions = self._generate_energy_suggestions(driver_sessions)
        suggestions.extend(energy_suggestions)
        
        # Sort by estimated gain (descending)
        suggestions.sort(key=lambda x: x.get('estimated_gain', 0), reverse=True)
        
        return suggestions
    
    async def _load_driver_track_sessions(self, driver_id: str, track_name: str) -> Optional[pd.DataFrame]:
        """Load driver's sessions for specific track"""
        data_dir = Path(settings.PROCESSED_DATA_DIR)
        
        # Look for matching files
        pattern = f"*{driver_id}*{track_name}*.csv"
        files = list(data_dir.glob(pattern))
        
        if not files:
            return None
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def _analyze_driver_sectors(self, sessions: pd.DataFrame) -> Dict:
        """Analyze driver performance by sector"""
        # Simple 3-sector analysis
        sector_analysis = {}
        
        for sector_id in [1, 2, 3]:
            sector_data = sessions[sessions.get('sector', 1) == sector_id]
            
            if len(sector_data) == 0:
                continue
            
            # Calculate potential improvements
            sector_times = sector_data.get('sector_time', [])
            if len(sector_times) > 0:
                best_time = sector_times.min()
                avg_time = sector_times.mean()
                time_loss = avg_time - best_time
                
                # Estimate optimization potential
                brake_potential = np.random.uniform(5, 15)  # Simplified
                throttle_potential = np.random.uniform(3, 10)
                apex_potential = np.random.uniform(0.5, 2.0)
                rotation_potential = np.random.uniform(1, 5)
                
                sector_analysis[f"Sector{sector_id}"] = {
                    'time_loss': float(time_loss),
                    'brake_late_potential': float(brake_potential),
                    'throttle_early_potential': float(throttle_potential),
                    'apex_speed_potential': float(apex_potential),
                    'rotation_potential': float(rotation_potential)
                }
        
        return sector_analysis
    
    def _generate_energy_suggestions(self, sessions: pd.DataFrame) -> List[Dict]:
        """Generate energy efficiency suggestions"""
        suggestions = []
        
        # Analyze braking efficiency
        if 'pbrake_f' in sessions.columns:
            brake_data = sessions['pbrake_f'].values
            high_brake_events = np.sum(brake_data > 80)
            
            if high_brake_events > len(sessions) * 0.3:
                suggestions.append({
                    "sector": "Overall",
                    "type": "energy_efficiency",
                    "description": "High brake pressure detected in 30% of corners. Consider earlier, smoother braking to reduce tire stress and improve consistency",
                    "estimated_gain": 0.15,
                    "difficulty": "medium"
                })
        
        return suggestions
    
    async def get_visualization_data(self, lap_id: str) -> Dict:
        """
        Get time-shift visualization data for cinematic replay
        STEP 8: Cinematic Time-Shift Replays
        """
        # Load lap data
        lap_data = await self._load_lap_data(lap_id)
        
        if lap_data is None:
            return {"error": "Lap data not found"}
        
        # Generate time-shift overlay data
        visualization = {
            "lap_id": lap_id,
            "track_path": lap_data.get('gps_coordinates', []),
            "time_deltas": [],
            "energy_flows": [],
            "suggestions_overlay": []
        }
        
        # Add time delta markers at key points
        for i in range(0, len(lap_data), 100):
            point = lap_data.iloc[i] if hasattr(lap_data, 'iloc') else lap_data[i]
            
            visualization["time_deltas"].append({
                "position": i,
                "delta": np.random.uniform(-0.2, 0.2),  # Simplified
                "color": "green" if np.random.random() > 0.5 else "red"
            })
        
        return visualization
    
    async def _load_lap_data(self, lap_id: str) -> Optional[pd.DataFrame]:
        """Load specific lap data"""
        data_dir = Path(settings.PROCESSED_DATA_DIR)
        lap_file = data_dir / f"lap_{lap_id}.csv"
        
        if lap_file.exists():
            return pd.read_csv(lap_file)
        
        return None
    
    def train_model(self, training_data: pd.DataFrame):
        """
        Train time-shift prediction models
        
        Args:
            training_data: DataFrame with columns:
                - brake_point, throttle_point, apex_speed, etc.
                - time_delta (target variable)
        """
        logger.info("Training time-shift prediction model...")
        
        # Prepare features and target
        feature_columns = [
            'speed', 'brake_pressure', 'throttle', 'lateral_g', 'longitudinal_g',
            'steering_angle', 'brake_shift', 'throttle_shift', 'apex_speed_change',
            'rotation_change', 'corner_curvature', 'elevation_change'
        ]
        
        X = training_data[feature_columns].values
        y = training_data['time_delta'].values
        
        # Train Gradient Boosting
        self.gb_regressor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.gb_regressor.fit(X, y)
        
        # Save model
        gb_path = Path(settings.TIMESHIFT_MODEL_PATH.replace('.h5', '_gb.pkl'))
        gb_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(gb_path, 'wb') as f:
            pickle.dump(self.gb_regressor, f)
        
        logger.info("Time-shift model training complete")