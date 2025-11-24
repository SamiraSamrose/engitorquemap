#engitorquemap/kafka/consumers/energy_consumer.py

"""
Energy Consumer (STEP 6)
Processes telemetry messages, computes energy vectors, caches in Redis
"""

import json
import logging
from kafka import KafkaConsumer
import redis
import sys
from pathlib import Path
from typing import Dict
import asyncio
import websockets

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.models.energy_models import EnergyCalculator, TelemetryPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyConsumer:
    """
    Consumes telemetry from Kafka, computes energy vectors, caches in Redis
    Broadcasts updates via WebSocket
    """
    
    def __init__(self):
        self.consumer = KafkaConsumer(
            settings.KAFKA_TELEMETRY_TOPIC,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=100
        )
        
        # Redis connection for caching (5-minute TTL)
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
        
        # Energy calculator
        self.calculator = EnergyCalculator()
        
        # WebSocket clients for broadcasting
        self.ws_clients = set()
        
        logger.info(f"EnergyConsumer initialized: topic='{settings.KAFKA_TELEMETRY_TOPIC}', group='{settings.KAFKA_CONSUMER_GROUP}'")
    
    def process_message(self, message: Dict) -> Dict:
        """
        Process single telemetry message and compute energy vectors
        """
        session_id = message['session_id']
        driver_id = message['driver_id']
        timestamp = message['timestamp']
        data = message['data']
        
        # Convert to TelemetryPoint
        telemetry = TelemetryPoint(
            timestamp=timestamp,
            speed=data['speed'],
            gear=data['gear'],
            rpm=data['rpm'],
            throttle=data['throttle'],
            brake_pressure=data['brake_pressure'],
            long_accel=data['long_accel'],
            lat_accel=data['lat_accel'],
            steering_angle=data['steering_angle'],
            gps_lat=data['gps_lat'],
            gps_lon=data['gps_lon']
            )

        # Compute energy vectors
    energy = self.calculator.compute_all_vectors(telemetry)
    
    # Prepare result
    result = {
        'session_id': session_id,
        'driver_id': driver_id,
        'timestamp': timestamp,
        'position': {
            'lat': telemetry.gps_lat,
            'lon': telemetry.gps_lon
        },
        'energy': {
            'braking_power': energy.braking_power,
            'acceleration_power': energy.acceleration_power,
            'yaw_energy': energy.yaw_energy,
            'tire_load_energy': energy.tire_load_energy,
            'total_consumption': energy.total_consumption,
            'efficiency': energy.efficiency
        }
    }
    
    # Cache in Redis with pattern: energy:{session}:{driver}:{timestamp}
    redis_key = f"energy:{session_id}:{driver_id}:{int(timestamp*1000)}"
    self.redis_client.setex(
        redis_key,
        settings.REDIS_TTL,  # 5-minute TTL
        json.dumps(result)
    )
    
    return result

async def broadcast_update(self, energy_data: Dict):
    """
    Broadcast energy update to all connected WebSocket clients
    """
    if self.ws_clients:
        message = json.dumps(energy_data)
        # Send to all connected clients
        await asyncio.gather(
            *[client.send(message) for client in self.ws_clients],
            return_exceptions=True
        )

def consume_and_process(self):
    """
    Main consumption loop
    """
    logger.info("Starting consumption loop...")
    processed_count = 0
    
    try:
        for message in self.consumer:
            try:
                energy_data = self.process_message(message.value)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} messages")
                    
                    # Log sample energy data
                    logger.debug(f"Sample: braking={energy_data['energy']['braking_power']:.2f}kW, "
                               f"efficiency={energy_data['energy']['efficiency']:.3f}")
                
                # Broadcast to WebSocket clients (would need async integration)
                # For now, just process and cache
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue
                
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    finally:
        self.consumer.close()
        self.redis_client.close()
        logger.info(f"Consumer closed. Total processed: {processed_count}")

def get_cached_energy(self, session_id: str, driver_id: str, timestamp: float) -> Dict:
    """
    Retrieve cached energy data from Redis
    """
    redis_key = f"energy:{session_id}:{driver_id}:{int(timestamp*1000)}"
    data = self.redis_client.get(redis_key)
    
    if data:
        return json.loads(data)
    return None

def get_session_stats(self, session_id: str, driver_id: str) -> Dict:
    """
    Get aggregated statistics for a session
    """
    pattern = f"energy:{session_id}:{driver_id}:*"
    keys = self.redis_client.keys(pattern)
    
    if not keys:
        return {'count': 0}
    
    total_braking = 0
    total_accel = 0
    total_consumption = 0
    efficiencies = []
    
    for key in keys:
        data = json.loads(self.redis_client.get(key))
        energy = data['energy']
        total_braking += energy['braking_power']
        total_accel += energy['acceleration_power']
        total_consumption += energy['total_consumption']
        efficiencies.append(energy['efficiency'])
    
    count = len(keys)
    avg_efficiency = sum(efficiencies) / count if efficiencies else 0
    
    return {
        'count': count,
        'avg_braking_power': total_braking / count,
        'avg_accel_power': total_accel / count,
        'total_consumption': total_consumption,
        'avg_efficiency': avg_efficiency
    }

async def websocket_server(consumer: EnergyConsumer):
"""
WebSocket server for broadcasting energy updates
"""
async def handler(websocket, path):
consumer.ws_clients.add(websocket)
logger.info(f"WebSocket client connected. Total clients: {len(consumer.ws_clients)}")
try:
        await websocket.wait_closed()
    finally:
        consumer.ws_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(consumer.ws_clients)}")

async with websockets.serve(handler, settings.WS_HOST, settings.WS_PORT):
    logger.info(f"WebSocket server started on {settings.WS_HOST}:{settings.WS_PORT}")
    await asyncio.Future()  # Run forever

def main():
"""CLI entry point"""
import argparse
parser = argparse.ArgumentParser(description='Energy Kafka Consumer')
parser.add_argument('--ws', action='store_true', help='Enable WebSocket server')

args = parser.parse_args()

consumer = EnergyConsumer()

if args.ws:
    # Run both consumer and WebSocket server
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_server(consumer))
    consumer.consume_and_process()
else:
    # Just run consumer
    consumer.consume_and_process()

if name == 'main':
main()