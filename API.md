# EngiTorqueMap API Documentation

Complete API reference for EngiTorqueMap backend services.

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication

Currently no authentication required for development. Production deployment should implement API keys or OAuth2.

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "app": "EngiTorqueMap",
  "version": "1.0.0"
}
```

---

### Track Geometry (STEP 1)

**GET** `/tracks`

Get list of available tracks.

**Response:**
```json
{
  "tracks": ["indianapolis", "barber-motorsports-park", ...]
}
```

**GET** `/tracks/{track_name}/geometry`

Get complete track geometry including polygons, elevation, and energy grid.

**Parameters:**
- `track_name` (path): Track identifier

**Response:**
```json
{
  "track_name": "indianapolis",
  "center_line": [[lon, lat], ...],
  "elevation_profile": [elevation, ...],
  "curvature": [curvature, ...],
  "corners": [...],
  "boundaries": {
    "inner": [[lon, lat], ...],
    "outer": [[lon, lat], ...]
  },
  "polygon": {...},
  "sectors": [...],
  "length_meters": 4000.0
}
```

**GET** `/tracks/{track_name}/energy-grid`

Get 1-meter resolution energy grid.

**Response:**
```json
{
  "track_name": "indianapolis",
  "resolution_meters": 1.0,
  "grid_points": 4000,
  "grid": [
    {
      "index": 0,
      "position": [lon, lat],
      "elevation": 220.5,
      "curvature": 0.002,
      "potential_energy": 2650000,
      "kinetic_energy": 1500000,
      "total_energy": 4150000
    },
    ...
  ]
}
```

---

### Energy Vectors (STEP 2)

**POST** `/energy/compute`

Compute energy vectors from telemetry snapshot.

**Request Body:**
```json
{
  "speed": 45.0,
  "gear": 4,
  "nmot": 5000,
  "aps": 80,
  "ath": 75,
  "pbrake_f": 60,
  "pbrake_r": 40,
  "accx_can": 0.5,
  "accy_can": 0.8,
  "Steering_Angle": 15
}
```

**Response:**
```json
{
  "energy_vectors": {
    "timestamp": 1234567890,
    "speed": 45.0,
    "braking_power": 85000.0,
    "acceleration_power": 120000.0,
    "yaw_energy": 25000.0,
    "tire_load_energy": 15000.0,
    "aerodynamic_drag": 8000.0,
    "rolling_resistance": 600.0,
    "net_power": 25400.0,
    "kinetic_energy": 1215000.0,
    "energy_efficiency": 0.85
  }
}
```

**GET** `/energy/session/{session_id}`

Get complete energy analysis for a session.

---

### Driver Signature (STEP 3)

**GET** `/drivers/{driver_id}/signature`

Get driver energy signature and style profile.

**Response:**
```json
{
  "driver_id": "D001",
  "style": "late_braker",
  "features": {
    "avg_brake_pressure": 65.5,
    "brake_aggression": 18.2,
    "avg_throttle_application_rate": 6.5,
    "throttle_aggression": 12.1,
    "steering_smoothness": 0.72,
    "avg_lateral_g": 1.15,
    "cornering_aggression": 0.95
  },
  "recommendations": [
    "Consider earlier brake application to reduce tire stress",
    "Smoother brake release can improve rotation"
  ]
}
```

**GET** `/drivers/clusters`

Get all driver clusters.

---

### Time-Shift Prediction (STEP 4)

**POST** `/timeshift/predict`

Predict time delta for alternative scenarios.

**Request Body:**
```json
{
  "driver_id": "D001",
  "track_name": "indianapolis",
  "sector": "IM2b",
  "current_telemetry": {...},
  "alternative_scenarios": [
    {
      "description": "brake 12m later",
      "modifications": {
        "brake_point_shift": 12.0
      }
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "scenario": "brake 12m later",
      "predicted_delta_seconds": -0.170,
      "confidence": 0.85,
      "energy_impact": {
        "brake_energy_saved_kj": 6.0,
        "net_energy_change_kj": 3.6
      },
      "position_impact": {
        "estimated_position_change": 1,
        "grid_spots": 1
      },
      "explanation": "Braking 12.0m later reduces deceleration energy by 6.0kJ. Predicted gain: 0.170s"
    }
  ]
}
```

**GET** `/timeshift/suggestions/{driver_id}/{track_name}`

Get AI-generated improvement suggestions.

---

### Pre-Event Forecast (STEP 5)

**POST** `/forecast/event`

Get pre-event prediction.

**Request Body:**
```json
{
  "track_name": "indianapolis",
  "date": "2024-06-15",
  "time": "14:00",
  "weather": {
    "temperature": 28,
    "humidity": 65,
    "wind_speed": 15
  }
}
```

---

### Real-Time FlowGrid (STEP 6)

**POST** `/flowgrid/stream/start`

Start real-time FlowGrid streaming.

**POST** `/flowgrid/stream/stop`

Stop streaming.

**WebSocket** `/ws/flowgrid`

WebSocket endpoint for real-time energy updates.

---

### Forensic Analysis (STEP 7)

**GET** `/forensic/replay/{session_id}`

Get post-event forensic replay data.

**GET** `/forensic/drift-patterns/{session_id}`

Get energy drift patterns.

---

### 3D Visualization (STEP 8)

**GET** `/visualization/hologram/{track_name}`

Get 3D holographic energy stream data.

---

### Strategy Center (STEP 9)

**GET** `/strategy/alerts`

Get real-time strategy alerts.

**GET** `/strategy/opponent-analysis/{opponent_id}`

Analyze opponent.

**POST** `/strategy/recommendation`

Get AI-powered strategy recommendation.

---

### Multi-Agent System (STEP 10)

**POST** `/agents/query`

Query the multi-agent system.

**Request Body:**
```json
{
  "type": "performance_analysis",
  "data": {...}
}
```

**GET** `/agents/status`

Get status of all agents.

---

## Error Responses

All endpoints return standard error responses:
```json
{
  "detail": "Error message description"
}
```

Status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error