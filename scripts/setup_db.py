## **File: scripts/setup_db.py** (Database Setup Script)

"""
Database setup script
Creates tables and initializes TimescaleDB
"""
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from config import settings
import asyncpg


async def setup_database():
    print("Setting up database...")
    
    try:
        # Connect to database
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        # Create tables
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                driver_id VARCHAR(100),
                timestamp TIMESTAMPTZ NOT NULL,
                speed FLOAT,
                gear INTEGER,
                nmot FLOAT,
                aps FLOAT,
                ath FLOAT,
                pbrake_f FLOAT,
                pbrake_r FLOAT,
                accx_can FLOAT,
                accy_can FLOAT,
                steering_angle FLOAT,
                gps_longitude FLOAT,
                gps_latitude FLOAT,
                lap INTEGER,
                sector INTEGER
            );
        ''')
        
        # Convert to hypertable (TimescaleDB)
        try:
            await conn.execute(
                "SELECT create_hypertable('telemetry', 'timestamp', if_not_exists => TRUE);"
            )
            print("✓ Created hypertable for telemetry")
        except Exception as e:
            print(f"Note: {e}")
        
        # Create energy_vectors table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS energy_vectors (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100),
                driver_id VARCHAR(100),
                timestamp TIMESTAMPTZ NOT NULL,
                speed FLOAT,
                braking_power FLOAT,
                acceleration_power FLOAT,
                yaw_energy FLOAT,
                tire_load_energy FLOAT,
                aerodynamic_drag FLOAT,
                rolling_resistance FLOAT,
                net_power FLOAT,
                energy_efficiency FLOAT
            );
        ''')
        
        # Create sessions table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(100) UNIQUE,
                track_name VARCHAR(100),
                event_date DATE,
                session_type VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        ''')
        
        # Create driver_profiles table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS driver_profiles (
                id SERIAL PRIMARY KEY,
                driver_id VARCHAR(100) UNIQUE,
                style VARCHAR(50),
                features JSONB,
                cluster_id INTEGER,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        ''')
        
        # Create indexes
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry(session_id);')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_driver ON telemetry(driver_id);')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_energy_session ON energy_vectors(session_id);')
        
        print("✓ Database setup complete")
        
        await conn.close()
        
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(setup_database())
