#engitorquemap/kafka/producers/telemetry_producer.py
"""
Telemetry Producer (STEP 6)
Publishes telemetry at 100Hz to Kafka telemetry-stream topic
"""

import json
import time
import numpy as np
from kafka import KafkaProducer
from typing import Dict, Optional
import logging
from dataclasses import asdict
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.config import settings
from backend.models.energy_models import TelemetryPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryProducer:
    """
    Streams racing telemetry to Kafka at 100Hz frequency
    """
    
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            batch_size=16384,
            linger_ms=10,  # Small latency for real-time feel
            acks=1  # Balance between performance and reliability
        )
        self.topic = settings.KAFKA_TELEMETRY_TOPIC
        self.frequency = settings.TELEMETRY_FREQUENCY
        self.interval = 1.0 / self.frequency  # 0.01 seconds for 100Hz
        
        logger.info(f"TelemetryProducer initialized: {self.frequency}Hz on topic '{self.topic}'")
    
    def send_telemetry(self, telemetry: TelemetryPoint, session_id: str, driver_id: str):
        """
        Send single telemetry point to Kafka
        """
        message = {
            'session_id': session_id,
            'driver_id': driver_id,
            'timestamp': telemetry.timestamp,
            'data': {
                'speed': telemetry.speed,
                'gear': telemetry.gear,
                'rpm': telemetry.rpm,
                'throttle': telemetry.throttle,
                'brake_pressure': telemetry.brake_pressure,
                'long_accel': telemetry.long_accel,
                'lat_accel': telemetry.lat_accel,
                'steering_angle': telemetry.steering_angle,
                'gps_lat': telemetry.gps_lat,
                'gps_lon': telemetry.gps_lon
            }
        }
        
        future = self.producer.send(self.topic, value=message)
        return future
    
    def stream_from_file(self, filepath: str, session_id: str, driver_id: str, realtime: bool = True):
        """
        Stream telemetry from CSV/parquet file
        If realtime=True, respects 100Hz timing; otherwise sends as fast as possible
        """
        import pandas as pd
        
        logger.info(f"Starting stream from {filepath}")
        
        # Load telemetry data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        start_time = time.time()
        messages_sent = 0
        
        for idx, row in df.iterrows():
            telemetry = TelemetryPoint(
                timestamp=row['timestamp'],
                speed=row['speed'],
                gear=int(row['gear']),
                rpm=row['rpm'],
                throttle=row['throttle'],
                brake_pressure=row['brake_pressure'],
                long_accel=row['long_accel'],
                lat_accel=row['lat_accel'],
                steering_angle=row['steering_angle'],
                gps_lat=row['gps_lat'],
                gps_lon=row['gps_lon']
            )
            
            self.send_telemetry(telemetry, session_id, driver_id)
            messages_sent += 1
            
            if realtime:
                # Maintain 100Hz timing
                expected_time = start_time + (messages_sent * self.interval)
                current_time = time.time()
                sleep_time = expected_time - current_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            if messages_sent % 1000 == 0:
                logger.info(f"Sent {messages_sent} messages")
        
        self.producer.flush()
        elapsed = time.time() - start_time
        logger.info(f"Streaming complete: {messages_sent} messages in {elapsed:.2f}s ({messages_sent/elapsed:.1f} msg/s)")
    
    def simulate_live_telemetry(self, session_id: str, driver_id: str, duration: int = 60):
        """
        Generate synthetic telemetry for testing (sinusoidal patterns)
        """
        logger.info(f"Starting synthetic telemetry simulation for {duration}s")
        
        start_time = time.time()
        messages_sent = 0
        
        while (time.time() - start_time) < duration:
            t = time.time() - start_time
            
            # Synthetic telemetry with realistic patterns
            telemetry = TelemetryPoint(
                timestamp=t,
                speed=50 + 30 * np.sin(0.5 * t),  # 50-80 m/s oscillation
                gear=int(3 + 2 * np.sin(0.3 * t)),  # Gears 1-5
                rpm=8000 + 4000 * np.sin(0.7 * t),  # 4000-12000 RPM
                throttle=max(0, 0.5 + 0.5 * np.sin(0.4 * t)),  # 0-1
                brake_pressure=max(0, 50 * (1 - np.sin(0.4 * t))),  # 0-100 bar
                long_accel=2 * np.sin(0.6 * t),  # -2 to 2 g
                lat_accel=1.5 * np.cos(0.8 * t),  # -1.5 to 1.5 g
                steering_angle=30 * np.sin(1.2 * t),  # -30 to 30 degrees
                gps_lat=40.5 + 0.001 * np.sin(0.1 * t),
                gps_lon=-3.6 + 0.001 * np.cos(0.1 * t)
            )
            
            self.send_telemetry(telemetry, session_id, driver_id)
            messages_sent += 1
            
            # Maintain 100Hz
            expected_time = start_time + (messages_sent * self.interval)
            current_time = time.time()
            sleep_time = expected_time - current_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.producer.flush()
        logger.info(f"Simulation complete: {messages_sent} messages sent")
    
    def close(self):
        """Close producer connection"""
        self.producer.close()
        logger.info("TelemetryProducer closed")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Telemetry Kafka Producer')
    parser.add_argument('--mode', choices=['file', 'simulate'], default='simulate', help='Data source mode')
    parser.add_argument('--file', type=str, help='Telemetry file path (CSV or Parquet)')
    parser.add_argument('--session', type=str, default='test_session', help='Session ID')
    parser.add_argument('--driver', type=str, default='driver_001', help='Driver ID')
    parser.add_argument('--duration', type=int, default=60, help='Simulation duration (seconds)')
    parser.add_argument('--realtime', action='store_true', help='Maintain 100Hz timing')
    
    args = parser.parse_args()
    
    producer = TelemetryProducer()
    
    try:
        if args.mode == 'file':
            if not args.file:
                logger.error("File path required for file mode")
                return
            producer.stream_from_file(args.file, args.session, args.driver, args.realtime)
        else:
            producer.simulate_live_telemetry(args.session, args.driver, args.duration)
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    finally:
        producer.close()


if __name__ == '__main__':
    main()