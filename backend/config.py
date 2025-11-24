#engitorquemap/backend/config.py
"""
Configuration Management for EngiTorqueMap
Handles all environment variables and system settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "EngiTorqueMap"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "engitorque"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis (STEP 6: Real-time caching with 5-minute TTL)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 300  # 5 minutes
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Kafka (STEP 6: Streaming at 100Hz)
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TELEMETRY_TOPIC: str = "telemetry-stream"
    KAFKA_ENERGY_TOPIC: str = "energy-updates"
    KAFKA_CONSUMER_GROUP: str = "energy-processor"
    TELEMETRY_FREQUENCY: int = 100  # Hz
    
    # WebSocket (STEP 6: Real-time broadcasting)
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8001
    
    # ML Models
    MODEL_PATH: str = "./ml_models/trained"
    XGBOOST_ENERGY_MODEL: str = "energy_consumption.pkl"
    GRADIENT_BOOST_TIMESHIFT: str = "timeshift_predictor.pkl"
    CNN_VALIDATOR_MODEL: str = "energy_map_cnn.onnx"
    RANDOM_FOREST_FORECAST: str = "forecast_engine.pkl"
    BAYESIAN_GRIP_MODEL: str = "grip_predictor.pkl"
    
    # STEP 1: Track Geometry
    TRACK_DATA_PATH: str = "./data/raw"
    PROCESSED_DATA_PATH: str = "./data/processed"
    BSPLINE_SMOOTHING: float = 0.5
    ENERGY_GRID_RESOLUTION: float = 1.0  # meters
    
    # STEP 2: Energy Vectors
    VEHICLE_MASS: float = 798.0  # kg (F1 car minimum weight)
    BRAKE_EFFICIENCY: float = 0.85
    GEAR_EFFICIENCY: float = 0.95
    
    # STEP 3: Driver Clustering
    UMAP_N_COMPONENTS: int = 3
    UMAP_METRIC: str = "euclidean"
    HDBSCAN_MIN_CLUSTER_SIZE: int = 5
    
    # STEP 4: Time-Shift Predictor
    GRADIENT_BOOST_ESTIMATORS: int = 200
    BRAKE_SHIFT_ENERGY_FACTOR: float = 0.5  # kJ/m
    THROTTLE_SHIFT_ENERGY_FACTOR: float = 0.3  # kJ/m
    POSITION_GAP_TIME: float = 0.3  # seconds
    
    # STEP 5: Forecast Engine
    RANDOM_FOREST_ESTIMATORS: int = 100
    OPTIMAL_TEMPERATURE: float = 25.0  # Celsius
    
    # STEP 6: FlowGrid Rendering
    GRID_CELL_SIZE: int = 20  # pixels
    ENERGY_WINDOW_SIZE: int = 100  # samples for moving average
    
    # STEP 7: Forensic Analysis
    DRIFT_THRESHOLD: float = -0.001
    GRIP_FAILURE_THRESHOLD: float = 0.15  # 15% efficiency drop
    
    # STEP 9: Strategy Thresholds
    TIRE_AGE_PIT_SOON: int = 20  # laps
    TIRE_AGE_PIT_WINDOW: int = 15  # laps
    ATTACK_GAP_THRESHOLD: float = 1.0  # seconds
    DEFEND_GAP_THRESHOLD: float = 1.5  # seconds
    TIRE_TEMP_HIGH: float = 110.0  # Celsius
    TIRE_TEMP_LOW: float = 80.0  # Celsius
    FUEL_CRITICAL: float = 15.0  # percent
    TIRE_AGE_CRITICAL: int = 22  # laps
    
    # STEP 10: Multi-Agent System
    BRAKING_POWER_ANOMALY: float = 150.0  # kW
    EFFICIENCY_ANOMALY: float = 0.5
    LLM_MODEL: str = "gpt-4"  # or "claude-3-opus"
    LLM_API_KEY: Optional[str] = None
    
    # Vector Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    VECTOR_DB_PATH: str = "./data/embeddings"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/engitorque.log"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()