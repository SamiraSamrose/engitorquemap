# EngiTorqueMap System Architecture

## Overview

EngiTorqueMap is a distributed, microservices-based system for real-time racing telemetry analysis with ML-powered performance optimization.

## Architecture Diagram

┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  HTML5 + JavaScript + D3.js + Three.js + Plotly.js + Leaflet  │
└──────────────────────────┬──────────────────────────────────────┘
│ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                       API Gateway (FastAPI)                      │
│                     http://localhost:8000                        │
└──────┬───────────┬──────────┬──────────┬────────────┬───────────┘
│           │          │          │            │
▼           ▼          ▼          ▼            ▼
┌──────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│  Track   │ │ Energy  │ │Driver  │ │Time-   │ │Multi-    │
│ Geometry │ │ Vectors │ │Signa-│ │Shift   │ │Agent     │
│ Service  │ │ Service │ │ture    │ │Predict │ │System    │
└────┬─────┘ └────┬────┘ └───┬────┘ └───┬────┘ └────┬─────┘
│            │           │          │           │
└────────────┴───────────┴──────────┴───────────┘
│
┌────────────────┼────────────────┐
│                │                │
▼                ▼                ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│PostgreSQL│    │  Redis   │    │  Kafka   │
│TimescaleDB    │  Cache   │    │Streaming │
└──────────┘    └──────────┘    └──────────┘
│
▼
┌──────────┐
│ ML Models│
│XGBoost   │
│LSTM/CNN  │
│UMAP+HDBSCAN
│LLM+RAG   │
└──────────┘

## System Components

### 1. Frontend Layer

**Technology**: HTML5, CSS3, JavaScript (ES6+)

**Libraries**:
- D3.js for data visualizations
- Three.js for 3D holographic displays
- Plotly.js for interactive charts
- Leaflet.js for map rendering

**Features**:
- Real-time FlowGrid visualization
- Time-Shift predictor interface
- Strategy command center
- Forensic replay viewer
- 3D energy hologram

**Communication**:
- REST API calls via Fetch API
- WebSocket connections for real-time updates

### 2. API Gateway

**Technology**: FastAPI (Python 3.10+)

**Responsibilities**:
- Route HTTP requests to appropriate services
- Handle WebSocket connections
- Request validation
- Response formatting
- CORS management

**Endpoints**: 50+ REST endpoints organized by feature

### 3. Service Layer

#### Track Geometry Service (STEP 1)
- Loads and processes track map ZIP files
- Reconstructs circuit polygons with spline interpolation
- Extracts curvature and classifies corners
- Generates 1-meter resolution energy grids
- Tools: GDAL, Shapely, NumPy, SciPy

#### Energy Vector Service (STEP 2)
- Computes energy vectors from telemetry
- Calculates braking power, acceleration power, yaw energy
- Physics-based approximations
- Real-time energy flow computation
- XGBoost + LSTM hybrid predictions

#### Driver Signature Service (STEP 3)
- Clusters drivers using UMAP + HDBSCAN
- Profiles driver styles (late-braker, smooth-roller, etc.)
- Feature extraction from telemetry
- Recommendation generation

#### Time-Shift Predictor (STEP 4)
- Predicts lap time deltas for alternative scenarios
- Gradient Boosting + CNN hybrid
- Generates actionable suggestions
- Energy impact analysis

#### Forecast Engine (STEP 5)
- Pre-event predictions
- Weather-based grip evolution
- Random Forest + Bayesian Regression
- Optimal braking zone prediction

#### Kafka Service (STEP 6)
- Real-time telemetry streaming
- Producer/consumer management
- Message serialization
- Stream processing

#### LLM + RAG Service (STEP 10)
- AI reasoning layer
- LangChain integration
- Vector embeddings (Sentence-Transformers)
- ChromaDB for knowledge base
- Prompt engineering

### 4. Data Storage Layer

#### PostgreSQL + TimescaleDB
- Primary database for telemetry
- Time-series optimization
- Session data
- Driver profiles
- Historical results

**Schema**:
```sql
telemetry (
    id, session_id, driver_id, timestamp,
    speed, gear, brake_pressure, throttle,
    accelerations, GPS, lap, sector
)

energy_vectors (
    id, session_id, timestamp,
    braking_power, acceleration_power,
    net_power, efficiency
)

sessions (
    session_id, track_name, event_date
)

driver_profiles (
    driver_id, style, features, cluster_id
)
```

#### Redis
- Real-time data caching
- Session state management
- Energy vector cache
- Metrics storage
- TTL-based expiration

**Key Patterns**:
- `energy:{session_id}:{driver_id}:{timestamp}`
- `session:state:{session_id}`
- `metric:{metric_name}`

#### Kafka
- Telemetry streaming topic: `telemetry-stream`
- Energy vectors topic: `energy-vectors`
- Alert topics: `strategy-alerts`

### 5. ML Models

#### XGBoost Energy Model
- Input: 12 telemetry features
- Output: Energy consumption prediction
- Training: Gradient boosting on historical data
- Format: Pickle (.pkl)

#### UMAP + HDBSCAN Driver Clustering
- Dimensionality reduction: 9D → 3D
- Clustering: Community detection
- Output: Driver style classifications
- Format: Pickle (.pkl)

#### Gradient Boosting Time-Shift
- Input: Current state + modifications
- Output: Time delta prediction
- Ensemble with CNN for energy maps
- Format: Pickle (.pkl) + Keras (.h5)

#### Random Forest Forecast
- Input: Weather + historical data
- Output: Grip evolution, lap time forecast
- Bayesian regression for uncertainty
- Format: Pickle (.pkl)

#### LLM (OpenAI GPT-4 / Anthropic Claude)
- Synthesis of multi-agent results
- Natural language explanations
- Strategy recommendations
- API-based integration

### 6. Multi-Agent Architecture

**Energy Field Agent**:
- Monitors FlowGrid
- Detects energy anomalies
- Updates energy fields

**Driver Time-Shift Agent**:
- Generates optimization suggestions
- Calculates time deltas
- Prioritizes improvements

**Strategy Agent**:
- Pit/pace recommendations
- Tire management
- Risk assessment

**Opponent Dynamics Agent**:
- Profiles opponents
- Predicts rival strategies
- Counter-strategy generation

**Coordination**:
- Asynchronous task execution
- Shared context via Redis
- LLM-powered synthesis

## Data Flow

### Real-Time Streaming Flow

Telemetry Source → Kafka Producer → telemetry-stream topic
↓
Kafka Consumer
↓
Energy Vector Service
↓
Redis Cache
↓
WebSocket Broadcast
↓
Frontend FlowGrid

### Analysis Flow

User Request → API Gateway → Service Layer → ML Models
↓
Database Query
↓
Data Processing
↓
Response Format
↓
JSON Response

### Training Pipeline

Raw Data → Data Loader → Preprocessor → Feature Engineering
↓
Train/Test Split
↓
Model Training
↓
Validation
↓
Model Export
↓
Model Repository

## Deployment Architecture

### Docker Compose (Development)
```yaml
services:
  - postgres (TimescaleDB)
  - redis
  - zookeeper
  - kafka
  - backend (FastAPI)
```

### Production (Kubernetes)

Load Balancer
↓
Ingress Controller
↓
┌─────────────────────────────────┐
│     FastAPI Pods (3 replicas)   │
└─────────────────────────────────┘
↓                    ↓
PostgreSQL         Redis Cluster
(Managed)          (3 nodes)
↓
Kafka Cluster
(3 brokers)

### Scalability Considerations

**Horizontal Scaling**:
- FastAPI: Multiple replicas behind load balancer
- Kafka: Multiple brokers and partitions
- Redis: Cluster mode with sharding

**Vertical Scaling**:
- ML inference: GPU acceleration
- Database: Read replicas
- Cache: Increased memory

**Performance Optimization**:
- Connection pooling (PostgreSQL)
- Request batching (ML inference)
- CDN for static assets
- Gzip compression
- Database indexing

## Security

**API Security**:
- HTTPS/TLS encryption
- API key authentication (production)
- Rate limiting
- CORS policies
- Input validation

**Data Security**:
- Database encryption at rest
- Connection encryption (SSL)
- Sensitive data hashing
- Access control lists
- Audit logging

**Network Security**:
- Private VPC
- Security groups
- Firewall rules
- DDoS protection

## Monitoring & Observability

**Metrics**:
- Prometheus for metrics collection
- Grafana for visualization
- Custom metrics per service

**Logging**:
- Structured JSON logging
- Centralized log aggregation (ELK stack)
- Log levels: DEBUG, INFO, WARNING, ERROR

**Tracing**:
- OpenTelemetry for distributed tracing
- Request correlation IDs
- Performance profiling

**Alerting**:
- Alert rules for critical metrics
- PagerDuty integration
- Slack notifications

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, D3.js, Three.js |
| API | FastAPI, Python 3.10+ |
| Database | PostgreSQL 14, TimescaleDB |
| Cache | Redis 7 |
| Streaming | Kafka 3.0+ |
| ML | XGBoost, TensorFlow, scikit-learn |
| Clustering | UMAP, HDBSCAN |
| NLP | LangChain, Sentence-Transformers |
| Vector DB | ChromaDB |
| Geospatial | GDAL, Shapely, Geopandas |
| Container | Docker, Docker Compose |
| Orchestration | Kubernetes (production) |

## Future Enhancements

1. **Mobile App**: React Native for iOS/Android
2. **Real-Time Collaboration**: Multi-user session sharing
3. **Advanced ML**: Reinforcement learning for strategy
4. **Edge Computing**: On-car processing with edge devices
5. **Cloud Integration**: AWS/GCP/Azure deployment
6. **API Marketplace**: Public API for third-party integrations