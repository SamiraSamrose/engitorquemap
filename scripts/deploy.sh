## engitorquemap/scripts/deploy.sh

#!/bin/bash

# EngiTorqueMap Deployment Script
# Deploys the complete racing telemetry analysis system

set -e  # Exit on error

echo "================================================"
echo "EngiTorqueMap Deployment Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}Environment: ${ENVIRONMENT}${NC}"
echo -e "${GREEN}Project Root: ${PROJECT_ROOT}${NC}"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}Loading environment variables...${NC}"
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}Warning: .env file not found. Using defaults.${NC}"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${GREEN}Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites satisfied${NC}"

# Create necessary directories
echo -e "\n${GREEN}Creating directories...${NC}"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/embeddings"
mkdir -p "$PROJECT_ROOT/ml_models/trained"
mkdir -p "$PROJECT_ROOT/logs"

# Step 1: Setup Python virtual environment
echo -e "\n${GREEN}Setting up Python environment...${NC}"
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install Python dependencies
echo -e "\n${GREEN}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Step 2: Start infrastructure services (Kafka, Redis, PostgreSQL)
echo -e "\n${GREEN}Starting infrastructure services...${NC}"
cd "$PROJECT_ROOT/kafka"
docker-compose down
docker-compose up -d

echo -e "${YELLOW}Waiting for Kafka to be ready (30 seconds)...${NC}"
sleep 30

# Check if Kafka is running
if docker ps | grep -q engitorque-kafka; then
    echo -e "${GREEN}✓ Kafka is running${NC}"
else
    echo -e "${RED}Error: Kafka failed to start${NC}"
    exit 1
fi

# Check if Redis is running
if docker ps | grep -q engitorque-redis; then
    echo -e "${GREEN}✓ Redis is running${NC}"
else
    echo -e "${RED}Error: Redis failed to start${NC}"
    exit 1
fi

# Step 3: Initialize database
echo -e "\n${GREEN}Initializing database...${NC}"
cd "$PROJECT_ROOT"
python scripts/setup_db.py
echo -e "${GREEN}✓ Database initialized${NC}"

# Step 4: Create Kafka topics
echo -e "\n${GREEN}Creating Kafka topics...${NC}"
docker exec engitorque-kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic telemetry-stream \
    --partitions 3 \
    --replication-factor 1 \
    --if-not-exists

docker exec engitorque-kafka kafka-topics --create \
    --bootstrap-server localhost:9092 \
    --topic energy-updates \
    --partitions 3 \
    --replication-factor 1 \
    --if-not-exists

echo -e "${GREEN}✓ Kafka topics created${NC}"

# Step 5: Build Docker images
echo -e "\n${GREEN}Building Docker images...${NC}"
cd "$PROJECT_ROOT/docker"

docker build -f Dockerfile.backend -t engitorque-backend:latest ..
docker build -f Dockerfile.frontend -t engitorque-frontend:latest ..

echo -e "${GREEN}✓ Docker images built${NC}"

# Step 6: Start application containers
echo -e "\n${GREEN}Starting application containers...${NC}"
docker-compose up -d

echo -e "${YELLOW}Waiting for services to be ready (15 seconds)...${NC}"
sleep 15

# Step 7: Health checks
echo -e "\n${GREEN}Performing health checks...${NC}"

# Check backend API
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ Backend API is healthy${NC}"
else
    echo -e "${RED}Warning: Backend API health check failed${NC}"
fi

# Check frontend
if curl -s http://localhost:8080 > /dev/null; then
    echo -e "${GREEN}✓ Frontend is accessible${NC}"
else
    echo -e "${RED}Warning: Frontend health check failed${NC}"
fi

# Step 8: Start background services
echo -e "\n${GREEN}Starting background services...${NC}"

# Start energy consumer
cd "$PROJECT_ROOT"
nohup python kafka/consumers/energy_consumer.py --ws > logs/energy_consumer.log 2>&1 &
echo $! > logs/energy_consumer.pid
echo -e "${GREEN}✓ Energy consumer started (PID: $(cat logs/energy_consumer.pid))${NC}"

# Optional: Start telemetry simulator for testing
if [ "$ENVIRONMENT" = "development" ]; then
    echo -e "\n${YELLOW}Starting telemetry simulator for testing...${NC}"
    nohup python kafka/producers/telemetry_producer.py --mode simulate --duration 300 > logs/telemetry_producer.log 2>&1 &
    echo $! > logs/telemetry_producer.pid
    echo -e "${GREEN}✓ Telemetry simulator started (PID: $(cat logs/telemetry_producer.pid))${NC}"
fi

# Display status summary
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\nService URLs:"
echo -e "  Backend API:     ${GREEN}http://localhost:8000${NC}"
echo -e "  Frontend:        ${GREEN}http://localhost:8080${NC}"
echo -e "  Kafka UI:        ${GREEN}http://localhost:8090${NC}"
echo -e "  WebSocket:       ${GREEN}ws://localhost:8001${NC}"
echo -e "\nKafka Topics:"
echo -e "  - telemetry-stream"
echo -e "  - energy-updates"
echo -e "\nLogs:"
echo -e "  Backend:         ${YELLOW}logs/backend.log${NC}"
echo -e "  Energy Consumer: ${YELLOW}logs/energy_consumer.log${NC}"
echo -e "  Telemetry Prod:  ${YELLOW}logs/telemetry_producer.log${NC}"

echo -e "\n${GREEN}To stop services:${NC}"
echo -e "  ./scripts/stop.sh"

echo -e "\n${GREEN}To view logs:${NC}"
echo -e "  tail -f logs/energy_consumer.log"
echo -e "  docker-compose logs -f"

echo -e "\n${GREEN}================================================${NC}"
