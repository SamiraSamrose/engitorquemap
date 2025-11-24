## **File: backend/services/kafka_service.py** (STEP 6: Real-Time Streaming)

"""
Kafka Service - STEP 6
Real-time telemetry streaming using Kafka
Handles producer and consumer for FlowGrid engine
"""
import asyncio
from typing import Dict, AsyncGenerator, Optional
import json
import logging

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("kafka-python not installed. Streaming features limited.")

from config import settings

logger = logging.getLogger(__name__)


class KafkaService:
    """
    Handles Kafka streaming for real-time telemetry
    """
    
    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.is_streaming = False
        
        if KAFKA_AVAILABLE:
            self._initialize_kafka()
        else:
            logger.warning("Kafka not available, using mock streaming")
    
    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            # Producer for sending telemetry
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            
            logger.info(f"Kafka producer initialized: {settings.KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
    
    async def start_streaming(self, stream_config: Dict):
        """
        Start streaming telemetry data
        STEP 6: Real-Time FlowGrid streaming
        """
        self.is_streaming = True
        session_id = stream_config.get('session_id')
        
        logger.info(f"Starting streaming for session {session_id}")
        
        # In production, this would read from actual telemetry source
        # For now, simulate streaming
        asyncio.create_task(self._simulate_telemetry_stream(session_id))
    
    async def stop_streaming(self):
        """Stop telemetry streaming"""
        self.is_streaming = False
        logger.info("Streaming stopped")
    
    async def _simulate_telemetry_stream(self, session_id: str):
        """Simulate telemetry streaming (for testing)"""
        sample_rate = 1.0 / settings.TELEMETRY_SAMPLE_RATE  # seconds per sample
        
        while self.is_streaming:
            # Generate synthetic telemetry
            telemetry = {
                "session_id": session_id,
                "timestamp": asyncio.get_event_loop().time(),
                "speed": 45.0 + np.random.uniform(-5, 5),
                "gear": np.random.randint(2, 6),
                "nmot": 5000 + np.random.uniform(-500, 500),
                "aps": np.random.uniform(0, 100),
                "ath": np.random.uniform(0, 100),
                "pbrake_f": np.random.uniform(0, 80),
                "pbrake_r": np.random.uniform(0, 60),
                "accx_can": np.random.uniform(-1.5, 1.5),
                "accy_can": np.random.uniform(-2.0, 2.0),
                "Steering_Angle": np.random.uniform(-180, 180)
                }
            # Publish to Kafka
        await self.publish_telemetry(telemetry)
        
        await asyncio.sleep(sample_rate)

async def publish_telemetry(self, telemetry: Dict):
    """Publish telemetry message to Kafka"""
    if self.producer and KAFKA_AVAILABLE:
        try:
            future = self.producer.send(
                settings.KAFKA_TOPIC_TELEMETRY,
                value=telemetry
            )
            # Wait for send to complete
            future.get(timeout=10)
        except KafkaError as e:
            logger.error(f"Failed to publish telemetry: {e}")
    else:
        # Mock publishing
        logger.debug(f"Mock publish: {telemetry['timestamp']}")

async def consume_telemetry(self) -> AsyncGenerator[Dict, None]:
    """
    Consume telemetry messages from Kafka
    Yields telemetry dictionaries
    """
    if not KAFKA_AVAILABLE:
        # Mock consumer
        while True:
            await asyncio.sleep(0.1)
            yield {
                "timestamp": asyncio.get_event_loop().time(),
                "speed": 45.0,
                "mock": True
            }
    
    # Initialize consumer if not exists
    if self.consumer is None:
        self.consumer = KafkaConsumer(
            settings.KAFKA_TOPIC_TELEMETRY,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(','),
            group_id=settings.KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
    
    # Consume messages
    for message in self.consumer:
        yield message.value

async def publish_energy_vectors(self, energy_data: Dict):
    """Publish computed energy vectors to Kafka"""
    if self.producer and KAFKA_AVAILABLE:
        try:
            future = self.producer.send(
                settings.KAFKA_TOPIC_ENERGY,
                value=energy_data
            )
            future.get(timeout=10)
        except KafkaError as e:
            logger.error(f"Failed to publish energy data: {e}")

async def close(self):
    """Close Kafka connections"""
    if self.producer:
        self.producer.close()
    if self.consumer:
        self.consumer.close()
    logger.info("Kafka connections closed")