# EngiTorqueMap: Real-Time Energy Field Maps + Predictive Driver Time-Shift Optimizer.

Racing telemetry analysis system with ML-powered performance optimization.

## Overview

EngiTorqueMap combines track geometry analysis, energy flow computation, driver profiling and AI-powered recommendations to provide real-time racing insights and predictive time-shift suggestions using Toyota Gazoo Racing Datasets.

### Key Features

- **Track-Energy Geometry**: 1-meter resolution energy grids with spline interpolation
- **Energy Flow Vectors**: Physics-based energy computation from telemetry
- **Driver Signatures**: ML clustering (UMAP + HDBSCAN) for driver profiling
- **Time-Shift Prediction**: "Brake 12m later → +0.17s" actionable insights
- **Pre-Event Forecasting**: Weather-based performance prediction
- **Real-Time FlowGrid**: Kafka streaming with live energy visualization
- **Forensic Replay**: Post-event energy analysis
- **3D Visualization**: Three.js holographic energy streams
- **IoT Command Center**: Real-time strategy alerts
- **Multi-Agent AI**: 4 coordinated ML+LLM agents


## Links

- **Live Site Demo**: https://samirasamrose.github.io/engitorquemap/
- **Source Code**: https://github.com/SamiraSamrose/engitorquemap
- **Video Demo**: https://youtu.be/XmS7dvTu4JQ


### Technology Stack


**Languages**: Python 3.10, JavaScript ES6, HTML5, CSS3, SQL

**Backend Frameworks**: FastAPI, Uvicorn, Pydantic

**Frontend Libraries**: D3.js, Three.js, Plotly.js, Leaflet.js

**Machine Learning**: XGBoost, TensorFlow, Keras, scikit-learn, UMAP, HDBSCAN, NumPy, Pandas, SciPy

**Deep Learning**: LSTM, CNN, Gradient Boosting Regressor, Random Forest, Bayesian Ridge

**NLP & AI**: LangChain, Sentence-Transformers, OpenAI GPT-4, Anthropic Claude, ChromaDB

**Geospatial**: GDAL, Shapely, Geopandas, Rasterio, Pyproj, Folium

**Databases**: PostgreSQL 14, TimescaleDB, Redis 7

**Message Streaming**: Apache Kafka 3.0, Zookeeper

**Containerization**: Docker, Docker Compose

**Web Server**: Nginx

**APIs**: OpenAI API, Anthropic API

**Development Tools**: Git, pytest, black, flake8, mypy

**Visualization**: Plotly, Matplotlib, Seaborn

**Deployment**: Kubernetes, ONNX Runtime

**Data Integrations**: TRD Hackathon 2025 API (track maps), USAC Timing System API (race results)

**Datasets**:
Telemetry CSV files (speed, gear, RPM, throttle, brake pressure, accelerations, steering, GPS), Timing results CSV (lap times, sector times, positions), Weather data CSV (temperature, humidity, wind), Track maps (boundary coordinates, elevation profiles), Analysis with Sections data (sector-based timing)

## Comprehensive Project Description

EngiTorqueMap is a real-time motorsport telemetry analysis system that transforms raw racing data into actionable driver performance insights and strategic race decisions. The system processes track geometry from Toyota Gazoo Racing data (ZIP archives) containing GPS boundary coordinates and elevation data, applying B-spline interpolation to generate smooth 1-meter resolution energy grids that map the potential energy landscape of each circuit. Telemetry streams at 100Hz capture vehicle dynamics including speed, gear position, engine RPM, throttle and brake pressures, three-axis accelerations, steering angle, and GPS coordinates, which feed into physics-based energy vector calculations determining braking power dissipation, acceleration power delivery, rotational yaw energy, and tire deformation energy at each instant.

Machine learning models include XGBoost regressors predicting energy consumption from 12 telemetry features, LSTM networks capturing temporal patterns in power delivery, and CNN architectures processing 32x32 energy map images for spatial pattern recognition. Driver profiling employs UMAP dimensionality reduction transforming 9-dimensional feature spaces (brake aggression, throttle rates, steering smoothness, cornering behavior, speed consistency) into 3D embeddings, followed by HDBSCAN density-based clustering identifying driver style archetypes. The time-shift prediction engine uses gradient boosting with 200 decision trees to model the relationship between technique modifications (brake point shifts measured in meters, throttle application timing changes, apex speed variations, rotation angle adjustments) and resulting lap time deltas, generating specific recommendations like "braking 12m later in Turn 3 saves 6kJ brake energy and gains 0.17s lap time."

Pre-event forecasting combines Random Forest ensembles processing weather parameters (temperature, humidity, wind speed, track surface temperature) with Bayesian regression models incorporating historical race results to predict grip coefficient evolution, optimal tire compound selection, and expected lap time distributions. The real-time FlowGrid visualization system uses Kafka message streaming to publish telemetry to topic partitions, consumer threads compute energy vectors on each message, Redis caching stores computed results with 5-minute TTL, and WebSocket connections broadcast updates to browser clients rendering canvas-based particle systems showing energy flows as colored vectors overlaid on track maps.

Post-event forensic analysis processes complete session telemetry computing cumulative tire stress integrals, detecting energy drift patterns through moving average efficiency calculations, identifying grip loss events from sudden efficiency drops correlated with high lateral loading, and validating predicted time-shift deltas against actual sector time improvements. The multi-agent architecture coordinates four specialized components: Energy Field Agent monitoring power flows and detecting anomalies exceeding threshold values, Driver Time-Shift Agent generating technique optimization suggestions ranked by predicted time gain, Strategy Agent analyzing race state for pit window timing and pace recommendations, and Opponent Dynamics Agent profiling competitor energy signatures to predict strategic decisions and identify performance weaknesses.

LLM integration uses LangChain to orchestrate GPT-4 or Claude API calls with structured prompts synthesizing multi-agent outputs into natural language explanations, while Sentence-Transformers generate 384-dimensional embeddings for telemetry descriptions enabling semantic similarity search across historical sessions stored in ChromaDB vector database. The frontend provides interactive visualizations using D3.js for time-series energy charts, Plotly.js for multi-dimensional performance analysis, Leaflet.js for track map rendering with GeoJSON polygon overlays, and Three.js for 3D holographic energy flow animations where arrow helpers indicate power vector directions colored by magnitude thresholds.

The technology stack includes Python 3.10 backend services with FastAPI framework handling REST API routing and WebSocket management, PostgreSQL 14 with TimescaleDB extension optimizing time-series telemetry storage in hypertables partitioned by timestamp, Redis 7 providing sub-millisecond in-memory caching for session state and real-time metrics, and Apache Kafka 3.0 managing distributed message streaming with topic partitioning for horizontal scalability. Geospatial processing uses GDAL for raster track map manipulation, Shapely for polygon geometry operations, and Geopandas for coordinate system transformations from GPS minutes format to decimal degrees. Deployment leverages Docker containerization with multi-stage builds, Docker Compose orchestrating service dependencies, Kubernetes for production container orchestration with horizontal pod autoscaling, and ONNX Runtime enabling optimized ML model inference on edge devices with CPU execution providers.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ (with TimescaleDB extension)
- Redis 7+
- Kafka 3.0+ (optional, for real-time streaming)
- Node.js 18+ (for frontend build tools)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/engitorquemap.git
cd engitorquemap
```

2. **Install Python dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Setup database**
```bash
python scripts/setup_db.py
```

5. **Download track data**
```bash
python scripts/download_data.py
```

6. **Run backend**
```bash
python backend/app.py
```

7. **Access dashboard**
````
Open browser: http://localhost:8000
````

### Project Structure
See SETUP.md for detailed project structure and component descriptions.

### Data Sources

Track Maps: https://trddev.com/hackathon-2025/
Timing Results: http://usac.alkamelna.com/

### API Documentation

API documentation available at: http://localhost:8000/docs

### Architecture

See docs/ARCHITECTURE.md for detailed system architecture.

### License

### MIT License - see LICENSE file for details.
