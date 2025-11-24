#engitorquemap/backend/models/energy_models.py
"""
Energy Calculation Models (STEP 2)
Computes braking, acceleration, yaw, tire load energy and predictions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path
from backend.config import settings


@dataclass
class TelemetryPoint:
    """Single telemetry sample at 100Hz"""
    timestamp: float
    speed: float  # m/s
    gear: int
    rpm: float
    throttle: float  # 0-1
    brake_pressure: float  # bar
    long_accel: float  # g
    lat_accel: float  # g
    steering_angle: float  # degrees
    gps_lat: float
    gps_lon: float


@dataclass
class EnergyVectors:
    """Computed energy components"""
    braking_power: float  # kW
    acceleration_power: float  # kW
    yaw_energy: float  # kJ
    tire_load_energy: float  # kJ
    total_consumption: float  # kJ
    efficiency: float  # 0-1


class EnergyCalculator:
    """
    STEP 2: Energy Flow Vectors
    Computes energy components from telemetry at 100Hz
    """
    
    def __init__(self):
        self.mass = settings.VEHICLE_MASS
        self.brake_eff = settings.BRAKE_EFFICIENCY
        self.gear_eff = settings.GEAR_EFFICIENCY
        self.g = 9.81  # m/s^2
        
        # Load XGBoost model for energy consumption prediction
        model_path = Path(settings.MODEL_PATH) / settings.XGBOOST_ENERGY_MODEL
        if model_path.exists():
            self.xgb_model = joblib.load(model_path)
        else:
            self.xgb_model = None
    
    def compute_braking_power(self, telemetry: TelemetryPoint) -> float:
        """
        Calculate braking power using mass, deceleration, and brake pressure
        Power = Force × Velocity = (m × a) × v
        """
        # Convert g-force to m/s^2
        deceleration = abs(min(telemetry.long_accel, 0)) * self.g
        
        # Braking force considering brake pressure and efficiency
        braking_force = self.mass * deceleration * self.brake_eff
        
        # Power in kW
        power = (braking_force * telemetry.speed) / 1000.0
        
        return power
    
    def compute_acceleration_power(self, telemetry: TelemetryPoint) -> float:
        """
        Calculate acceleration power from throttle, RPM, gear, and longitudinal accel
        Considers gear efficiency and engine characteristics
        """
        if telemetry.throttle < 0.01 or telemetry.long_accel <= 0:
            return 0.0
        
        # Acceleration force
        accel = max(telemetry.long_accel, 0) * self.g
        accel_force = self.mass * accel
        
        # Power with gear efficiency
        power = (accel_force * telemetry.speed * telemetry.throttle * self.gear_eff) / 1000.0
        
        return power
    
    def compute_yaw_energy(self, telemetry: TelemetryPoint) -> float:
        """
        Determine yaw energy from lateral acceleration and steering angle
        Uses rotational kinetics: E = 0.5 × I × ω^2
        """
        # Estimate moment of inertia (simplified for race car)
        wheelbase = 3.6  # meters (typical F1)
        moment_inertia = self.mass * (wheelbase ** 2) / 12.0
        
        # Angular velocity from steering and lateral accel
        lat_accel_ms2 = telemetry.lat_accel * self.g
        
        if telemetry.speed > 1.0:
            angular_velocity = lat_accel_ms2 / telemetry.speed
        else:
            angular_velocity = 0.0
        
        # Rotational kinetic energy in kJ
        yaw_energy = 0.5 * moment_inertia * (angular_velocity ** 2) / 1000.0
        
        return yaw_energy
    
    def compute_tire_load_energy(self, telemetry: TelemetryPoint) -> float:
        """
        Calculate tire load energy from combined g-forces
        Higher combined g-force = more tire stress and energy dissipation
        """
        # Combined g-force
        combined_g = np.sqrt(telemetry.long_accel**2 + telemetry.lat_accel**2)
        
        # Normal force variation due to combined acceleration
        normal_force = self.mass * self.g * (1 + combined_g)
        
        # Tire deformation energy (simplified model)
        tire_stiffness = 200000  # N/m (typical race tire)
        deformation = normal_force / tire_stiffness
        
        # Energy in kJ
        tire_energy = 0.5 * tire_stiffness * (deformation ** 2) / 1000.0
        
        return tire_energy
    
    def extract_features(self, telemetry: TelemetryPoint) -> np.ndarray:
        """
        Extract 12 telemetry features for XGBoost prediction
        """
        features = [
            telemetry.speed,
            telemetry.gear,
            telemetry.rpm,
            telemetry.throttle,
            telemetry.brake_pressure,
            telemetry.long_accel,
            telemetry.lat_accel,
            telemetry.steering_angle,
            abs(telemetry.long_accel),
            abs(telemetry.lat_accel),
            np.sqrt(telemetry.long_accel**2 + telemetry.lat_accel**2),
            telemetry.throttle * telemetry.rpm / 1000.0
        ]
        return np.array(features).reshape(1, -1)
    
    def predict_consumption(self, telemetry: TelemetryPoint) -> float:
        """
        Predict energy consumption using XGBoost model trained on 12 features
        """
        if self.xgb_model is None:
            # Fallback: sum of computed energies
            braking = self.compute_braking_power(telemetry)
            accel = self.compute_acceleration_power(telemetry)
            yaw = self.compute_yaw_energy(telemetry)
            tire = self.compute_tire_load_energy(telemetry)
            return braking + accel + yaw + tire
        
        features = self.extract_features(telemetry)
        consumption = self.xgb_model.predict(features)[0]
        return max(consumption, 0.0)
    
    def compute_all_vectors(self, telemetry: TelemetryPoint) -> EnergyVectors:
        """
        Compute all energy vectors for a single telemetry point
        """
        braking_power = self.compute_braking_power(telemetry)
        accel_power = self.compute_acceleration_power(telemetry)
        yaw_energy = self.compute_yaw_energy(telemetry)
        tire_energy = self.compute_tire_load_energy(telemetry)
        total_consumption = self.predict_consumption(telemetry)
        
        # Efficiency metric (useful power / total consumption)
        useful_power = accel_power
        efficiency = useful_power / total_consumption if total_consumption > 0 else 0.0
        efficiency = min(efficiency, 1.0)
        
        return EnergyVectors(
            braking_power=braking_power,
            acceleration_power=accel_power,
            yaw_energy=yaw_energy,
            tire_load_energy=tire_energy,
            total_consumption=total_consumption,
            efficiency=efficiency
        )
    
    def process_telemetry_batch(self, telemetry_batch: List[TelemetryPoint]) -> List[EnergyVectors]:
        """
        Process multiple telemetry points at once
        """
        return [self.compute_all_vectors(t) for t in telemetry_batch]


class EnergyGrid:
    """
    STEP 1 continuation: 1-meter resolution energy grid
    Maps potential energy from elevation and kinetic energy from corner speeds
    """
    
    def __init__(self, track_length: float, resolution: float = 1.0):
        self.resolution = resolution
        self.grid_size = int(track_length / resolution)
        self.potential_energy = np.zeros(self.grid_size)
        self.kinetic_energy = np.zeros(self.grid_size)
        self.total_energy = np.zeros(self.grid_size)
    
    def set_elevation_energy(self, positions: np.ndarray, elevations: np.ndarray):
        """
        Calculate potential energy: PE = m × g × h
        """
        for pos, elev in zip(positions, elevations):
            idx = int(pos / self.resolution)
            if 0 <= idx < self.grid_size:
                self.potential_energy[idx] = settings.VEHICLE_MASS * 9.81 * elev / 1000.0  # kJ
    
    def set_corner_energy(self, positions: np.ndarray, corner_speeds: np.ndarray):
        """
        Calculate kinetic energy: KE = 0.5 × m × v^2
        """
        for pos, speed in zip(positions, corner_speeds):
            idx = int(pos / self.resolution)
            if 0 <= idx < self.grid_size:
                self.kinetic_energy[idx] = 0.5 * settings.VEHICLE_MASS * (speed ** 2) / 1000.0  # kJ
    
    def compute_total_energy(self):
        """
        Combine potential and kinetic energy
        """
        self.total_energy = self.potential_energy + self.kinetic_energy
    
    def get_energy_at_position(self, position: float) -> Tuple[float, float, float]:
        """
        Retrieve energy values at a specific track position
        Returns: (potential, kinetic, total)
        """
        idx = int(position / self.resolution)
        if 0 <= idx < self.grid_size:
            return (
                self.potential_energy[idx],
                self.kinetic_energy[idx],
                self.total_energy[idx]
            )
        return (0.0, 0.0, 0.0)