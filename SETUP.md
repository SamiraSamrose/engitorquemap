## **File: SETUP.md**

# EngiTorqueMap Setup Guide

Complete setup instructions for development and production environments.

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- OS: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ with WSL2

### Recommended Requirements
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 500 GB NVMe SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for ML training)

## Step-by-Step Setup

### 1. Python Environment
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 2. Database Setup (PostgreSQL + TimescaleDB)
```bash
# Install PostgreSQL
sudo apt-get install postgresql-14 postgresql-contrib

# Install TimescaleDB
sudo add-apt-repository ppa:timescale/timescaledb-ppa
sudo apt-get update
sudo apt-get install timescaledb-postgresql-14

# Initialize database
sudo -u postgres psql
CREATE DATABASE engitorquemap;
CREATE USER engitorque_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE engitorquemap TO engitorque_user;
\c engitorquemap
CREATE EXTENSION IF NOT EXISTS timescaledb;
\q

# Run migrations
python scripts/setup_db.py
```

### 3. Redis Setup
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
# Should return: PONG
```

### 4. Kafka Setup (Optional, for real-time streaming)
```bash
# Using Docker Compose
cd kafka
docker-compose up -d

# Verify Kafka is running
docker-compose ps
```

### 5. Download Track Data
```bash
# Download all track data
python scripts/download_data.py --all

# Or download specific track
python scripts/download_data.py --track indianapolis

# Process raw data
python backend/app.py process-data --track indianapolis
```

### 6. Train ML Models (Optional)
```bash
# Train all models
cd ml_models/training
python train_energy_model.py
python train_driver_clustering.py
python train_timeshift_predictor.py

# Models will be saved to ml_models/trained/
```

### 7. Configure API Keys

Edit `.env` file:
```bash
# For LLM features
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=your-anthropic-key-here

# For embeddings (optional, uses local model by default)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 8. Run Application
```bash
# Development mode
cd backend
python app.py

# Production mode with Uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# With auto-reload (development)
uvicorn app:app --reload
```

### 9. Access Application

- **Dashboard**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc

## Docker Deployment
```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## Testing
```bash
# Run unit tests
pytest backend/tests/

# Run with coverage
pytest --cov=backend backend/tests/

# Run specific test
pytest backend/tests/test_energy.py
```

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt --force-reinstall
```

**2. Database connection failed**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection string in .env
DATABASE_URL=postgresql://user:password@localhost:5432/engitorquemap
```

**3. Kafka connection issues**
```bash
# Check Kafka is running
docker-compose -f kafka/docker-compose.yml ps

# Restart Kafka
docker-compose -f kafka/docker-compose.yml restart
```

**4. Model loading errors**
```bash
# Download pre-trained models or train new ones
python ml_models/training/train_energy_model.py
```

## Performance Optimization

### Database
```sql
-- Create indexes for faster queries
CREATE INDEX idx_telemetry_timestamp ON telemetry(timestamp);
CREATE INDEX idx_telemetry_driver ON telemetry(driver_id);
CREATE INDEX idx_energy_session ON energy_vectors(session_id);
```

### Redis Caching
```bash
# Increase Redis memory limit
sudo nano /etc/redis/redis.conf
# Set: maxmemory 4gb
sudo systemctl restart redis
```

### Kafka Performance
```bash
# Increase partition count for better parallelism
kafka-topics --bootstrap-server localhost:9092 --alter --topic telemetry-stream --partitions 8
```

## Monitoring

### Application Metrics
```bash
# View metrics endpoint
curl http://localhost:8000/api/v1/health

# Prometheus metrics (if enabled)
curl http://localhost:8000/metrics
```

### Logs
```bash
# View application logs
tail -f logs/engitorquemap.log

# Docker logs
docker-compose logs -f backend
```

## Security

1. **Change default passwords** in `.env`
2. **Enable HTTPS** in production
3. **Configure firewall** rules
4. **Regular backups** of database
5. **API rate limiting** (configured in FastAPI)

## Next Steps

- Review API documentation: `/docs`
- Train custom ML models with your data
- Configure real-time streaming
- Setup production deployment
- Enable monitoring and alerting

For more information, see:
- ARCHITECTURE.md - System design
- API.md - API reference
- DEPLOYMENT.md - Production deployment guide
````

---

Let me continue with the frontend and remaining files in the next response.