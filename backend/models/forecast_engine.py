#backend/models/forecast_engine.py (Pre-Event Prediction)
"""
Forecast Engine - STEP 5
Pre-event prediction using weather, historical timing, and track-energy grids
Uses Random Forest Ensembles + Bayesian Regression
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import pickle
from pathlib import Path
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from config import settings

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Predicts race conditions and optimal strategies before events
    """
    
    def __init__(self):
        self.rf_model = None
        self.bayesian_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained forecast models"""
        model_path = Path(settings.FORECAST_MODEL_PATH)
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
                    self.rf_model = models.get('random_forest')
                    self.bayesian_model = models.get('bayesian')
                logger.info("Loaded forecast models")
            except Exception as e:
                logger.error(f"Error loading forecast models: {e}")
        else:
            logger.warning("Forecast models not found")
    
    async def predict_event(self, event_data: Dict) -> Dict:
        """
        Predict event conditions and outcomes
        
        Input:
        {
            "track_name": "indianapolis",
            "date": "2024-06-15",
            "time": "14:00",
            "weather": {
                "temperature": 28,
                "humidity": 65,
                "wind_speed": 15,
                "precipitation_chance": 20
            },
            "historical_data": {...}
        }
        
        Returns predictions for:
        - Expected grip evolution
        - Optimal braking zones
        - Energy-saving lines
        - Expected fastest sector combinations
        """
        track_name = event_data.get('track_name')
        weather = event_data.get('weather', {})
        
        # Load historical data for track
        historical = await self._load_historical_data(track_name)
        
        # Predict grip evolution
        grip_evolution = self._predict_grip_evolution(weather, historical)
        
        # Predict optimal braking zones
        braking_zones = self._predict_optimal_braking_zones(track_name, weather, grip_evolution)
        
        # Predict energy-saving lines
        energy_lines = self._predict_energy_lines(track_name, weather)
        
        # Predict fastest sector combinations
        fastest_sectors = self._predict_fastest_sectors(track_name, historical, weather)
        
        # Predict lap times
        lap_time_forecast = self._forecast_lap_times(track_name, weather, historical)
        
        forecast = {
            "track": track_name,
            "event_date": event_data.get('date'),
            "weather_conditions": weather,
            "grip_evolution": grip_evolution,
            "optimal_braking_zones": braking_zones,
            "energy_saving_lines": energy_lines,
            "fastest_sector_combinations": fastest_sectors,
            "lap_time_forecast": lap_time_forecast,
            "recommendations": self._generate_event_recommendations(
                grip_evolution, weather, fastest_sectors
            )
        }
        
        return forecast
    
    async def _load_historical_data(self, track_name: str) -> pd.DataFrame:
        """Load historical timing and weather data"""
        data_dir = Path(settings.PROCESSED_DATA_DIR)
        hist_file = data_dir / f"{track_name}_historical.csv"
        
        if hist_file.exists():
            return pd.read_csv(hist_file)
        
        # Return empty DataFrame if no historical data
        return pd.DataFrame()
    
    def _predict_grip_evolution(self, weather: Dict, historical: pd.DataFrame) -> Dict:
        """
        Predict how track grip will evolve during session
        """
        temp = weather.get('temperature', 25)
        humidity = weather.get('humidity', 50)
        wind_speed = weather.get('wind_speed', 10)
        
        # Initial grip (normalized 0-1)
        if temp < 15:
            initial_grip = 0.75
        elif temp < 25:
            initial_grip = 0.90
        else:
            initial_grip = 0.95
        
        # Adjust for humidity
        initial_grip *= (1.0 - humidity / 200.0)
        
        # Predict evolution over session (minutes)
        session_duration = 60  # minutes
        time_points = np.linspace(0, session_duration, 20)
        
        grip_timeline = []
        for t in time_points:
            # Grip increases as rubber goes down
            rubber_factor = min(0.10, t / session_duration * 0.10)
            
            # Temperature effect
            temp_factor = 0.02 * (temp - 25) / 10.0  # Deviation from optimal
            
            grip = initial_grip + rubber_factor - abs(temp_factor)
            grip = max(0.6, min(1.0, grip))
            
            grip_timeline.append({
                "time_minutes": float(t),
                "grip_level": float(grip),
                "confidence": 0.75
            })
        
        return {
            "initial_grip": float(initial_grip),
            "peak_grip": float(max([g['grip_level'] for g in grip_timeline])),
            "timeline": grip_timeline,
            "factors": {
                "temperature_impact": float(-abs(temp_factor)),
                "rubber_buildup": float(rubber_factor),
                "weather_impact": float(-humidity / 200.0)
            }
        }
    
    def _predict_optimal_braking_zones(self, track_name: str, weather: Dict, 
                                      grip_evolution: Dict) -> List[Dict]:
        """
        Predict optimal braking zones based on conditions
        """
        # Load track geometry
        from services.track_geometry import TrackGeometryService
        track_service = TrackGeometryService()
        
        # Simplified: generate braking zones for key corners
        braking_zones = []
        
        for corner_id in range(1, 11):  # Assume 10 major corners
            current_grip = grip_evolution['initial_grip']
            
            # Calculate optimal braking point
            # Higher grip = later braking
            base_distance = 100.0  # meters before corner
            grip_adjustment = (current_grip - 0.85) * 20.0  # +/- 20m
            
            optimal_point = base_distance - grip_adjustment
            
            braking_zones.append({
                "corner_id": corner_id,
                "corner_name": f"Turn {corner_id}",
                "optimal_brake_point_meters": float(optimal_point),
                "brake_pressure_recommendation": float(70 + current_grip * 20),
                "confidence": 0.80
            })
        
        return braking_zones
    
    def _predict_energy_lines(self, track_name: str, weather: Dict) -> Dict:
        """
        Predict energy-saving racing lines
        """
        # Simulate energy-optimal lines based on conditions
        temp = weather.get('temperature', 25)
        
        # Higher temp = prefer shade/inside lines
        # Lower temp = prefer sun/outside lines for tire temp
        
        if temp > 30:
            line_strategy = "conservative_inside"
            energy_savings = 5.0  # percent
        elif temp < 15:
            line_strategy = "aggressive_outside"
            energy_savings = 3.0
        else:
            line_strategy = "balanced"
            energy_savings = 7.0
        
        return {
            "recommended_line_strategy": line_strategy,
            "estimated_energy_savings_percent": float(energy_savings),
            "tire_management_benefit": "high" if temp > 30 else "medium",
            "key_corners": [
                {
                    "corner": "Turn 1",
                    "line": "outside" if temp < 20 else "inside",
                    "reason": "Optimal for tire temperature management"
                }
            ]
        }
    
    def _predict_fastest_sectors(self, track_name: str, historical: pd.DataFrame, 
                                weather: Dict) -> List[Dict]:
        """
        Predict fastest possible sector times and combinations
        """
        # Use historical data if available
        if not historical.empty and 'sector_time' in historical.columns:
            best_s1 = historical[historical['sector'] == 1]['sector_time'].min()
            best_s2 = historical[historical['sector'] == 2]['sector_time'].min()
            best_s3 = historical[historical['sector'] == 3]['sector_time'].min()
        else:
            # Use estimates
            best_s1 = 28.5
            best_s2 = 32.1
            best_s3 = 29.8
        
        # Adjust for weather
        temp = weather.get('temperature', 25)
        temp_factor = 1.0 + (abs(temp - 25) / 100.0)
        
        return [
            {
                "sector": 1,
                "predicted_best_time": float(best_s1temp_factor),
"confidence": 0.82,
"key_factors": ["grip_level", "temperature"]
},
{
"sector": 2,
"predicted_best_time": float(best_s2 * temp_factor),
"confidence": 0.85,
"key_factors": ["technical_section", "tire_wear"]
},
{
"sector": 3,
"predicted_best_time": float(best_s3 * temp_factor),
"confidence": 0.80,
"key_factors": ["top_speed", "exit_speed"]
}
]
def _forecast_lap_times(self, track_name: str, weather: Dict,
historical: pd.DataFrame) -> Dict:
"""
Forecast lap time distribution for event
"""
# Base lap time from historical data
if not historical.empty and 'lap_time' in historical.columns:
base_lap_time = historical['lap_time'].min()
else:
base_lap_time = 90.0  # Default estimate

# Weather adjustment
 temp = weather.get('temperature', 25)
 humidity = weather.get('humidity', 50)
 
 weather_adjustment = (abs(temp - 25) * 0.05) + (humidity / 1000.0)
 
 predicted_pole = base_lap_time + weather_adjustment
 predicted_mean = predicted_pole + 1.5
 predicted_slowest = predicted_pole + 5.0
 
 return {
     "predicted_pole_time": float(predicted_pole),
     "predicted_mean_time": float(predicted_mean),
     "predicted_field_spread": float(predicted_slowest - predicted_pole),
     "confidence_interval": {
         "lower": float(predicted_pole - 0.3),
         "upper": float(predicted_pole + 0.3)
     },
     "factors": {
         "temperature_impact": float((abs(temp - 25) * 0.05)),
         "humidity_impact": float(humidity / 1000.0),
         "track_evolution": 0.2
     }
 }
 def _generate_event_recommendations(self, grip_evolution: Dict,
weather: Dict,
fastest_sectors: List[Dict]) -> List[str]:
"""Generate actionable recommendations for event"""
recommendations = []
# Grip-based recommendations
 if grip_evolution['initial_grip'] < 0.80:
     recommendations.append(
         "Low initial grip expected. Consider conservative first lap strategy."
     )
 
 if grip_evolution['peak_grip'] > 0.95:
     recommendations.append(
         "High grip expected in final sessions. Save best runs for late in session."
     )
 
 # Weather-based recommendations
 temp = weather.get('temperature', 25)
 if temp > 30:
     recommendations.append(
         "High temperature forecast. Focus on tire management and cooling."
     )
 elif temp < 15:
     recommendations.append(
         "Cool conditions expected. Aggressive tire warming strategy recommended."
     )
 
 # Wind considerations
 wind = weather.get('wind_speed', 0)
 if wind > 20:
     recommendations.append(
         "Strong winds forecast. Adjust braking points and stability setup."
     )
 
 return recommendations

 async def weather_forecast(self, track_name: str, date: str) -> Dict:
"""
Get weather-based performance forecast
"""
# In production, integrate with weather API
# For now, simulate weather forecast

weather_forecast = {
     "track": track_name,
     "date": date,
     "hourly_forecast": []
 }
 
 # Generate hourly forecast for race day (6am - 6pm)
 base_temp = 20.0
 for hour in range(6, 19):
     # Temperature curve (peaks at 2pm)
     temp = base_temp + 8 * np.sin((hour - 6) * np.pi / 12)
     humidity = 60 - (hour - 6) * 2
     
     weather_forecast["hourly_forecast"].append({
         "hour": hour,
         "temperature_celsius": float(temp),
         "humidity_percent": float(max(30, humidity)),
         "wind_speed_kph": float(10 + np.random.uniform(-3, 3)),
         "precipitation_chance": 10.0,
         "track_temp_celsius": float(temp + 8),
         "grip_prediction": float(0.85 + (temp - 20) / 100)
     })
 
 # Performance impact analysis
 weather_forecast["performance_impact"] = {
     "best_session_time": "10:00-12:00",
     "optimal_temperature_window": "22-26°C",
     "tire_strategy_recommendation": "medium_compound",
     "estimated_lap_time_variance": "±0.5s"
 }
 
 return weather_forecast

 def train_models(self, training_data: pd.DataFrame):
"""
Train forecast models
Args:
     training_data: Historical event data with weather and results
 """
 logger.info("Training forecast models...")
 
 # Prepare features
 feature_columns = [
     'temperature', 'humidity', 'wind_speed', 
     'track_temperature', 'precipitation',
     'session_time', 'rubber_level'
 ]
 
 X = training_data[feature_columns].values
 y_lap_time = training_data['lap_time'].values
 y_grip = training_data['grip_level'].values
 
 # Train Random Forest for lap time prediction
 self.rf_model = RandomForestRegressor(
     n_estimators=100,
     max_depth=10,
     random_state=42
 )
 self.rf_model.fit(X, y_lap_time)
 
 # Train Bayesian Ridge for grip prediction
 self.bayesian_model = BayesianRidge()
 self.bayesian_model.fit(X, y_grip)
 
 # Save models
 model_path = Path(settings.FORECAST_MODEL_PATH)
 model_path.parent.mkdir(parents=True, exist_ok=True)
 
 with open(model_path, 'wb') as f:
     pickle.dump({
         'random_forest': self.rf_model,
         'bayesian': self.bayesian_model
     }, f)
 
 logger.info("Forecast models training complete")