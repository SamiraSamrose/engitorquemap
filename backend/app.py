#backend/app.py

"""
Main FastAPI Application for EngiTorqueMap
Orchestrates all API endpoints and services
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from typing import List, Dict, Any, Optional
import json
import logging

from config import settings, setup_directories
from services.track_geometry import TrackGeometryService
from services.energy_vectors import EnergyVectorService
from services.kafka_service import KafkaService
from services.redis_service import RedisService
from services.embedding_service import EmbeddingService
from services.llm_rag_service import LLMRAGService
from models.driver_signature import DriverSignatureModel
from models.timeshift_predictor import TimeShiftPredictor
from models.forecast_engine import ForecastEngine
from models.ml_agents import MultiAgentSystem
from data.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")


manager = ConnectionManager()

# Global service instances
track_service: Optional[TrackGeometryService] = None
energy_service: Optional[EnergyVectorService] = None
kafka_service: Optional[KafkaService] = None
redis_service: Optional[RedisService] = None
embedding_service: Optional[EmbeddingService] = None
llm_service: Optional[LLMRAGService] = None
driver_model: Optional[DriverSignatureModel] = None
timeshift_model: Optional[TimeShiftPredictor] = None
forecast_engine: Optional[ForecastEngine] = None
agent_system: Optional[MultiAgentSystem] = None
data_loader: Optional[DataLoader] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - startup and shutdown"""
    global track_service, energy_service, kafka_service, redis_service
    global embedding_service, llm_service, driver_model, timeshift_model
    global forecast_engine, agent_system, data_loader
    
    logger.info("Starting EngiTorqueMap application...")
    
    # Setup directories
    setup_directories()
    
    # Initialize services
    logger.info("Initializing services...")
    track_service = TrackGeometryService()
    energy_service = EnergyVectorService()
    kafka_service = KafkaService()
    redis_service = RedisService()
    embedding_service = EmbeddingService()
    llm_service = LLMRAGService(embedding_service)
    
    # Initialize ML models
    logger.info("Loading ML models...")
    driver_model = DriverSignatureModel()
    timeshift_model = TimeShiftPredictor()
    forecast_engine = ForecastEngine()
    
    # Initialize multi-agent system
    logger.info("Initializing Multi-Agent System...")
    agent_system = MultiAgentSystem(
        energy_service=energy_service,
        timeshift_model=timeshift_model,
        llm_service=llm_service
    )
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Start background tasks
    logger.info("Starting background tasks...")
    asyncio.create_task(kafka_consumer_task())
    
    logger.info("Application startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if kafka_service:
        await kafka_service.close()
    if redis_service:
        await redis_service.close()
    logger.info("Application shutdown complete!")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")


async def kafka_consumer_task():
    """Background task to consume Kafka messages"""
    async for message in kafka_service.consume_telemetry():
        try:
            # Process telemetry and compute energy vectors
            energy_data = await energy_service.process_telemetry(message)
            
            # Cache in Redis
            await redis_service.cache_energy_data(energy_data)
            
            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "energy_update",
                "data": energy_data
            })
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    with open("../frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


# ============================================================================
# STEP 1: Track Geometry API
# ============================================================================

@app.get("/api/v1/tracks")
async def get_tracks():
    """Get list of all available tracks"""
    return {"tracks": settings.TRACKS}


@app.get("/api/v1/tracks/{track_name}/geometry")
async def get_track_geometry(track_name: str):
    """
    Get track geometry including polygons, elevation, and energy grid
    STEP 1: Track-Energy Geometry
    """
    try:
        geometry = await track_service.get_track_geometry(track_name)
        return JSONResponse(content=geometry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tracks/{track_name}/sectors")
async def get_track_sectors(track_name: str):
    """Get track sector definitions"""
    try:
        sectors = await track_service.get_sectors(track_name)
        return {"track": track_name, "sectors": sectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tracks/{track_name}/energy-grid")
async def get_energy_grid(track_name: str):
    """
    Get 1-meter resolution energy grid for track
    STEP 1: Energy Grid Layer
    """
    try:
        grid = await track_service.get_energy_grid(track_name)
        return JSONResponse(content=grid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 2: Energy Vectors API
# ============================================================================

@app.post("/api/v1/energy/compute")
async def compute_energy_vectors(telemetry_data: Dict[str, Any]):
    """
    Compute energy vectors from telemetry data
    STEP 2: Energy Flow Vectors
    """
    try:
        energy_vectors = await energy_service.compute_vectors(telemetry_data)
        return {"energy_vectors": energy_vectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/energy/session/{session_id}")
async def get_session_energy(session_id: str):
    """Get energy analysis for a complete session"""
    try:
        analysis = await energy_service.analyze_session(session_id)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 3: Driver Signature API
# ============================================================================

@app.get("/api/v1/drivers/{driver_id}/signature")
async def get_driver_signature(driver_id: str):
    """
    Get driver energy signature and style profile
    STEP 3: Driver Energy Signature
    """
    try:
        signature = await driver_model.get_signature(driver_id)
        return JSONResponse(content=signature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/drivers/clusters")
async def get_driver_clusters():
    """Get all driver clusters (late-braker, smooth-roller, etc.)"""
    try:
        clusters = await driver_model.get_clusters()
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/drivers/analyze")
async def analyze_driver_style(driver_data: Dict[str, Any]):
    """Analyze driver style from telemetry"""
    try:
        analysis = await driver_model.analyze_style(driver_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 4: Time-Shift Prediction API
# ============================================================================

@app.post("/api/v1/timeshift/predict")
async def predict_time_delta(prediction_request: Dict[str, Any]):
    """
    Predict time delta for alternative driving scenarios
    STEP 4: Time-Shift Delta Predictor
    
    Example: "If you braked 12m later in IM2b, predicted delta = -0.17s"
    """
    try:
        predictions = await timeshift_model.predict_delta(prediction_request)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/timeshift/suggestions/{driver_id}/{track_name}")
async def get_improvement_suggestions(driver_id: str, track_name: str):
    """
    Get AI-generated improvement suggestions for driver on specific track
    STEP 4: Generate multiple optimization suggestions
    """
    try:
        suggestions = await timeshift_model.generate_suggestions(driver_id, track_name)
        return {"driver_id": driver_id, "track": track_name, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 5: Pre-Event Prediction API
# ============================================================================

@app.post("/api/v1/forecast/event")
async def forecast_event(event_data: Dict[str, Any]):
    """
    Pre-event prediction including grip evolution, optimal zones
    STEP 5: Forecast Layer
    """
    try:
        forecast = await forecast_engine.predict_event(event_data)
        return JSONResponse(content=forecast)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast/weather/{track_name}/{date}")
async def get_weather_forecast(track_name: str, date: str):
    """Get weather-based performance forecast"""
    try:
        forecast = await forecast_engine.weather_forecast(track_name, date)
        return JSONResponse(content=forecast)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 6: Real-Time FlowGrid API (WebSocket)
# ============================================================================

@app.websocket("/ws/flowgrid")
async def websocket_flowgrid(websocket: WebSocket):
    """
    WebSocket endpoint for real-time FlowGrid energy visualization
    STEP 6: Real-Time FlowGrid Engine
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command.get("type") == "subscribe":
                track = command.get("track")
                session = command.get("session")
                logger.info(f"Client subscribed to {track}/{session}")
            
            # Keep connection alive
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/v1/flowgrid/stream/start")
async def start_flowgrid_stream(stream_config: Dict[str, Any]):
    """Start real-time FlowGrid streaming"""
    try:
        await kafka_service.start_streaming(stream_config)
        return {"status": "streaming_started", "config": stream_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/flowgrid/stream/stop")
async def stop_flowgrid_stream():
    """Stop real-time FlowGrid streaming"""
    try:
        await kafka_service.stop_streaming()
        return {"status": "streaming_stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 7: Post-Event Forensic Analysis API
# ============================================================================

@app.get("/api/v1/forensic/replay/{session_id}")
async def get_forensic_replay(session_id: str):
    """
    Post-event forensic energy replay
    STEP 7: Forensic Energy Replay
    """
    try:
        replay_data = await energy_service.forensic_replay(session_id)
        return JSONResponse(content=replay_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forensic/drift-patterns/{session_id}")
async def get_drift_patterns(session_id: str):
    """Analyze energy drift patterns and grip failures"""
    try:
        patterns = await energy_service.analyze_drift_patterns(session_id)
        return JSONResponse(content=patterns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 8: 3D Visualization Data API
# ============================================================================

@app.get("/api/v1/visualization/hologram/{track_name}")
async def get_hologram_data(track_name: str):
    """
    Get 3D holographic energy stream data
    STEP 8: Energy-Flow Holograms
    """
    try:
        hologram = await track_service.generate_hologram_data(track_name)
        return JSONResponse(content=hologram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/visualization/timeshift/{lap_id}")
async def get_timeshift_visualization(lap_id: str):
    """Get cinematic time-shift replay visualization data"""
    try:
        viz_data = await timeshift_model.get_visualization_data(lap_id)
        return JSONResponse(content=viz_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 9: IoT Strategy Command Center API
# ============================================================================

@app.get("/api/v1/strategy/alerts")
async def get_strategy_alerts():
    """
    Get real-time strategy alerts for pit wall
    STEP 9: IoT Strategy Command Center
    """
    try:
        alerts = await agent_system.get_strategy_alerts()
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/strategy/opponent-analysis/{opponent_id}")
async def analyze_opponent(opponent_id: str):
    """Predict opponent energy-use tendencies"""
    try:
        analysis = await agent_system.analyze_opponent(opponent_id)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/strategy/recommendation")
async def get_strategy_recommendation(race_state: Dict[str, Any]):
    """Get AI-powered pit/pace recommendations"""
    try:
        recommendation = await agent_system.get_recommendation(race_state)
        return JSONResponse(content=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 10: Multi-Agent System API
# ============================================================================

@app.post("/api/v1/agents/query")
async def query_agent_system(query: Dict[str, Any]):
    """
    Query the multi-agent ML+LLM system
    STEP 10: Multi-Agent Hybrid Architecture
    """
    try:
        response = await agent_system.process_query(query)
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/status")
async def get_agent_status():
    """Get status of all agents in the system"""
    try:
        status = await agent_system.get_status()
        return {"agents": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Management API
# ============================================================================

@app.post("/api/v1/data/download")
async def download_track_data(track_name: str):
    """Download and process track data from TRD"""
    try:
        result = await data_loader.download_track_data(track_name)
        return {"status": "success", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/process")
async def process_raw_data(processing_config: Dict[str, Any]):
    """Process raw data files"""
    try:
        result = await data_loader.process_data(processing_config)
        return {"status": "success", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/status")
async def get_data_status():
    """Get status of available data"""
    try:
        status = await data_loader.get_status()
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Community Detection & Network Analysis API
# ============================================================================

@app.get("/api/v1/network/community/{track_name}")
async def get_community_network(track_name: str):
    """
    Get driver community/connection detection network
    Advanced Feature: Community Detection
    """
    try:
        from data.network_detector import CommunityDetector
        detector = CommunityDetector()
        network = await detector.detect_communities(track_name)
        return JSONResponse(content=network)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )