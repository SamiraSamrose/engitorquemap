#backend/services/energy_vectors.py (Energy Flow Vectors)
"""
Energy Vector Service - STEP 2
Derives energy flow vectors from telemetry using physics approximations
Computes braking power, acceleration power, yaw-energy, tire load
Uses XGBoost + LSTM hybrids to predict energy consumption patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import pickle
from pathlib import Path

from config import settings
from utils.physics import (
    calculate_braking_power,
    calculate_acceleration_power,
    calculate_yaw_energy,
    calculate_tire_load,
    calculate_aerodynamic_drag,
    calculate_rolling_resistance
)

logger = logging.getLogger(__name__)


class EnergyVectorService:
    """
    Computes energy vectors from telemetry data
    """
    
    def __init__(self):
        self.xgboost_model = None
        self.lstm_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained XGBoost and LSTM models"""
        try:
            xgb_path = Path(settings.ENERGY_MODEL_PATH)
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    self.xgboost_model = pickle.load(f)
                logger.info("Loaded XGBoost energy model")
            else:
                logger.warning(f"XGBoost model not found at {xgb_path}")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
        
        try:
            from tensorflow import keras
            lstm_path = Path(settings.TIMESHIFT_MODEL_PATH)
            if lstm_path.exists():
                self.lstm_model = keras.models.load_model(str(lstm_path))
                logger.info("Loaded LSTM energy model")
            else:
                logger.warning(f"LSTM model not found at {lstm_path}")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
    
    async def compute_vectors(self, telemetry_data: Dict[str, Any]) -> Dict:
        """
        Compute energy vectors from telemetry snapshot
        
        Input telemetry parameters:
        - speed: vehicle speed (m/s or km/h)
        - gear: current gear
        - nmot: engine RPM
        - aps: accelerator pedal position (%)
        - ath: throttle position (%)
        - pbrake_f: front brake pressure (bar)
        - pbrake_r: rear brake pressure (bar)
        - accx_can: longitudinal acceleration (g)
        - accy_can: lateral acceleration (g)
        - Steering_Angle: steering angle (degrees)
        - VBOX_Long_Minutes: GPS longitude
        - VBOX_Lat_Min: GPS latitude
        - Laptrigger_lapdist_dls: lap distance
        """
        
        # Extract telemetry values
        speed = telemetry_data.get('speed', 0.0)  # m/s
        gear = telemetry_data.get('gear', 0)
        rpm = telemetry_data.get('nmot', 0.0)
        aps = telemetry_data.get('aps', 0.0)
        throttle = telemetry_data.get('ath', 0.0)
        brake_f = telemetry_data.get('pbrake_f', 0.0)
        brake_r = telemetry_data.get('pbrake_r', 0.0)
        acc_x = telemetry_data.get('accx_can', 0.0)
        acc_y = telemetry_data.get('accy_can', 0.0)
        steering = telemetry_data.get('Steering_Angle', 0.0)
        
        # Convert speed to m/s if in km/h
        if speed > 100:  # Likely km/h
            speed = speed / 3.6
        
        # Calculate energy components
        
        # 1. Braking Power (negative, energy dissipated)
        braking_power = calculate_braking_power(
            speed=speed,
            brake_pressure_f=brake_f,
            brake_pressure_r=brake_r,
            deceleration=abs(min(acc_x, 0))
        )
        
        # 2. Acceleration Power (positive, energy input)
        acceleration_power = calculate_acceleration_power(
            speed=speed,
            throttle=throttle,
            rpm=rpm,
            gear=gear,
            acceleration=max(acc_x, 0)
        )
        
        # 3. Yaw Energy (rotational energy in corners)
        yaw_energy = calculate_yaw_energy(
            speed=speed,
            lateral_acc=acc_y,
            steering_angle=steering
        )
        
        # 4. Tire Load (energy from tire deformation)
        tire_load = calculate_tire_load(
            speed=speed,
            lateral_acc=acc_y,
            longitudinal_acc=acc_x
        )
        
        # 5. Aerodynamic Drag
        aero_drag = calculate_aerodynamic_drag(speed)
        
        # 6. Rolling Resistance
        rolling_resistance = calculate_rolling_resistance(speed)
        
        # Total energy rate (power)
        total_power = acceleration_power - braking_power - aero_drag - rolling_resistance
        
        energy_vectors = {
            "timestamp": telemetry_data.get('timestamp', 0),
            "speed": float(speed),
            "braking_power": float(braking_power),
            "acceleration_power": float(acceleration_power),
            "yaw_energy": float(yaw_energy),
            "tire_load_energy": float(tire_load),
            "aerodynamic_drag": float(aero_drag),
            "rolling_resistance": float(rolling_resistance),
            "net_power": float(total_power),
            "kinetic_energy": float(0.5 * settings.VEHICLE_MASS * speed**2),
            "energy_efficiency": self._calculate_efficiency(acceleration_power, braking_power)
        }
        
        # Predict energy consumption using ML models
        if self.xgboost_model is not None:
            ml_prediction = self._predict_energy_ml(telemetry_data, energy_vectors)
            energy_vectors["ml_predicted_consumption"] = ml_prediction
        
        return energy_vectors
    
    def _calculate_efficiency(self, accel_power: float, brake_power: float) -> float:
        """Calculate instantaneous energy efficiency"""
        total_input = accel_power + 0.001  # Avoid division by zero
        energy_lost = brake_power
        efficiency = max(0.0, min(1.0, 1.0 - (energy_lost / total_input)))
        return float(efficiency)
    
    def _predict_energy_ml(self, telemetry: Dict, energy_vectors: Dict) -> float:
        """
        Predict energy consumption using XGBoost + LSTM hybrid
        """
        try:
            # Prepare features for XGBoost
            features = np.array([
                telemetry.get('speed', 0),
                telemetry.get('gear', 0),
                telemetry.get('nmot', 0),
                telemetry.get('aps', 0),
                telemetry.get('ath', 0),
                telemetry.get('pbrake_f', 0),
                telemetry.get('pbrake_r', 0),
                telemetry.get('accx_can', 0),
                telemetry.get('accy_can', 0),
                energy_vectors['braking_power'],
                energy_vectors['acceleration_power'],
                energy_vectors['yaw_energy']
            ]).reshape(1, -1)
            
            prediction = self.xgboost_model.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return 0.0
    
    async def analyze_session(self, session_id: str) -> Dict:
        """
        Analyze complete session energy usage
        Returns comprehensive energy analysis
        """
        # Load session telemetry
        session_data = await self._load_session_data(session_id)
        
        if session_data is None:
            raise ValueError(f"Session {session_id} not found")
        
        # Compute energy vectors for each telemetry point
        energy_timeline = []
        total_energy_consumed = 0.0
        total_energy_recovered = 0.0
        
        for idx, row in session_data.iterrows():
            telemetry = row.to_dict()
            energy = await self.compute_vectors(telemetry)
            
            energy_timeline.append(energy)
            
            if energy['net_power'] < 0:
                total_energy_consumed += abs(energy['net_power'])
            else:
                total_energy_recovered += energy['net_power']
        
        # Analyze energy hotspots
        hotspots = self._identify_energy_hotspots(energy_timeline)
        
        # Calculate sector-based energy usage
        sector_analysis = self._analyze_sector_energy(energy_timeline)
        
        return {
            "session_id": session_id,
            "total_energy_consumed": float(total_energy_consumed),
            "total_energy_recovered": float(total_energy_recovered),
            "net_energy": float(total_energy_consumed - total_energy_recovered),
            "average_efficiency": float(np.mean([e['energy_efficiency'] for e in energy_timeline])),
            "energy_timeline": energy_timeline,
            "hotspots": hotspots,
            "sector_analysis": sector_analysis
        }
    
    async def _load_session_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """Load session telemetry data"""
        # Try to load from processed data directory
        session_file = Path(settings.PROCESSED_DATA_DIR) / f"{session_id}.parquet"
        
        if session_file.exists():
            return pd.read_parquet(session_file)
        
        # Try CSV
        session_file_csv = Path(settings.PROCESSED_DATA_DIR) / f"{session_id}.csv"
        if session_file_csv.exists():
            return pd.read_csv(session_file_csv)
        
        logger.warning(f"Session data not found: {session_id}")
        return None
    
    def _identify_energy_hotspots(self, energy_timeline: List[Dict]) -> List[Dict]:
        """
        Identify zones with high energy consumption or inefficiency
        """
        hotspots = []
        
        # Convert to numpy array for analysis
        braking_power = np.array([e['braking_power'] for e in energy_timeline])
        efficiency = np.array([e['energy_efficiency'] for e in energy_timeline])
        
        # Find high braking zones (threshold: 75th percentile)
        braking_threshold = np.percentile(braking_power, 75)
        high_braking_indices = np.where(braking_power > braking_threshold)[0]
        
        # Find low efficiency zones
        efficiency_threshold = np.percentile(efficiency, 25)
        low_efficiency_indices = np.where(efficiency < efficiency_threshold)[0]
        
        # Combine into hotspot regions
        for idx in high_braking_indices:
            if idx < len(energy_timeline):
                hotspots.append({
                    "index": int(idx),
                    "type": "high_braking",
                    "severity": float(braking_power[idx]),
                    "timestamp": energy_timeline[idx]['timestamp']
                })
        
        for idx in low_efficiency_indices:
            if idx < len(energy_timeline):
                hotspots.append({
                    "index": int(idx),
                    "type": "low_efficiency",
                    "severity": float(1.0 - efficiency[idx]),
                    "timestamp": energy_timeline[idx]['timestamp']
                })
        
        return hotspots
    
    def _analyze_sector_energy(self, energy_timeline: List[Dict]) -> List[Dict]:
        """Analyze energy usage by sector"""
        # Simple 3-sector split for now
        n = len(energy_timeline)
        sector_size = n // 3
        
        sectors = []
        for i in range(3):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size if i < 2 else n
            
            sector_data = energy_timeline[start_idx:end_idx]
            
            sectors.append({
                "sector_id": i + 1,
                "average_power": float(np.mean([e['net_power'] for e in sector_data])),
                "total_energy": float(np.sum([e['net_power'] for e in sector_data])),
                "average_efficiency": float(np.mean([e['energy_efficiency'] for e in sector_data])),
                "max_braking": float(max([e['braking_power'] for e in sector_data]))
            })
        
        return sectors
    
    async def process_telemetry(self, message: Dict) -> Dict:
        """
        Process streaming telemetry message from Kafka
        STEP 6: Real-time FlowGrid processing
        """
        energy_vectors = await self.compute_vectors(message)
        
        # Add metadata
        energy_vectors['session_id'] = message.get('session_id')
        energy_vectors['driver_id'] = message.get('driver_id')
        energy_vectors['lap'] = message.get('lap')
        
        return energy_vectors
    
    async def forensic_replay(self, session_id: str) -> Dict:
        """
        Post-event forensic energy replay
        STEP 7: Forensic Analysis Layer
        """
        session_analysis = await self.analyze_session(session_id)
        
        # Add forensic-specific analysis
        energy_timeline = session_analysis['energy_timeline']
        
        # Detect energy drift patterns
        drift_patterns = self._detect_energy_drift(energy_timeline)
        
        # Detect grip failures (sudden efficiency drops)
        grip_failures = self._detect_grip_failures(energy_timeline)
        
        # Tire stress accumulation
        tire_stress = self._calculate_tire_stress_accumulation(energy_timeline)
        
        forensic_data = {
            **session_analysis,
            "drift_patterns": drift_patterns,
            "grip_failures": grip_failures,
            "tire_stress_accumulation": tire_stress
        }
        
        return forensic_data
    
    def _detect_energy_drift(self, energy_timeline: List[Dict]) -> List[Dict]:
        """Detect gradual energy efficiency degradation"""
        efficiencies = np.array([e['energy_efficiency'] for e in energy_timeline])
        
        # Calculate moving average
        window_size = min(100, len(efficiencies) // 10)
        moving_avg = np.convolve(efficiencies, np.ones(window_size)/window_size, mode='valid')
        
        # Detect downward trend
        drift_rate = np.gradient(moving_avg)
        
        drifts = []
        for i, rate in enumerate(drift_rate):
            if rate < -0.001:  # Threshold for significant drift
                drifts.append({
                    "index": int(i),
                    "drift_rate": float(rate),
                    "efficiency": float(moving_avg[i])
                })
        
        return drifts
    
    def _detect_grip_failures(self, energy_timeline: List[Dict]) -> List[Dict]:
        """Detect sudden grip loss events"""
        efficiencies = np.array([e['energy_efficiency'] for e in energy_timeline])
        lateral_energy = np.array([e['yaw_energy'] for e in energy_timeline])
        
        # Detect sudden drops in efficiency with high lateral load
        failures = []
        
        for i in range(1, len(efficiencies)):
            efficiency_drop = efficiencies[i-1] - efficiencies[i]
            
            if efficiency_drop > 0.15 and lateral_energy[i] > np.mean(lateral_energy):
                failures.append({
                    "index": int(i),
                    "efficiency_drop": float(efficiency_drop),
                    "lateral_load": float(lateral_energy[i]),
                    "timestamp": energy_timeline[i]['timestamp']
                })
        
        return failures
    
    def _calculate_tire_stress_accumulation(self, energy_timeline: List[Dict]) -> Dict:
        """Calculate cumulative tire stress over session"""
        tire_loads = np.array([e['tire_load_energy'] for e in energy_timeline])
        
        cumulative_stress = np.cumsum(tire_loads)
        
        return {
            "total_stress": float(cumulative_stress[-1]),
            "average_stress": float(np.mean(tire_loads)),
            "peak_stress": float(np.max(tire_loads)),
            "stress_timeline": cumulative_stress.tolist()
        }
    
    async def analyze_drift_patterns(self, session_id: str) -> Dict:
        """
        Analyze energy drift patterns for forensic analysis
        STEP 7: Energy drift patterns
        """
        forensic_data = await self.forensic_replay(session_id)
        return {
            "session_id": session_id,
            "drift_patterns": forensic_data['drift_patterns'],
            "grip_failures": forensic_data['grip_failures']
        }