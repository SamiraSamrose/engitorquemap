## **File: backend/services/redis_service.py** (Real-Time Caching)

"""
Redis Service
Real-time data caching using Redis
Stores energy data, session state, and real-time metrics
"""
import json
from typing import Dict, Optional
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not installed. Caching features limited.")

from config import settings

logger = logging.getLogger(__name__)


class RedisService:
    """
    Handles Redis caching for real-time data
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        
        if REDIS_AVAILABLE:
            self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            logger.info(f"Redis client initialized: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
    
    async def cache_energy_data(self, energy_data: Dict):
        """Cache energy vector data"""
        if not self.redis_client:
            return
        
        try:
            session_id = energy_data.get('session_id')
            driver_id = energy_data.get('driver_id')
            timestamp = energy_data.get('timestamp')
            
            # Cache with key pattern: energy:{session_id}:{driver_id}:{timestamp}
            key = f"energy:{session_id}:{driver_id}:{timestamp}"
            
            await self.redis_client.setex(
                key,
                300,  # 5 minute TTL
                json.dumps(energy_data)
            )
            
            # Also maintain a list of recent energy updates
            list_key = f"energy:recent:{session_id}"
            await self.redis_client.lpush(list_key, key)
            await self.redis_client.ltrim(list_key, 0, 999)  # Keep last 1000
            await self.redis_client.expire(list_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Failed to cache energy data: {e}")
    
    async def get_energy_data(self, session_id: str, driver_id: str, 
                             timestamp: Optional[float] = None) -> Optional[Dict]:
        """Retrieve cached energy data"""
        if not self.redis_client:
            return None
        
        try:
            if timestamp:
                key = f"energy:{session_id}:{driver_id}:{timestamp}"
                data = await self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                # Get most recent
                list_key = f"energy:recent:{session_id}"
                keys = await self.redis_client.lrange(list_key, 0, 0)
                if keys:
                    data = await self.redis_client.get(keys[0])
                    if data:
                        return json.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve energy data: {e}")
            return None
    
    async def cache_session_state(self, session_id: str, state: Dict):
        """Cache current session state"""
        if not self.redis_client:
            return
        
        try:
            key = f"session:state:{session_id}"
            await self.redis_client.setex(
                key,
                3600,  # 1 hour TTL
                json.dumps(state)
            )
        except Exception as e:
            logger.error(f"Failed to cache session state: {e}")
    
    async def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve session state"""
        if not self.redis_client:
            return None
        
        try:
            key = f"session:state:{session_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve session state: {e}")
            return None
    
    async def cache_metric(self, metric_name: str, value: float, tags: Dict = None):
        """Cache real-time metric"""
        if not self.redis_client:
            return
        
        try:
            key = f"metric:{metric_name}"
            metric_data = {
                "value": value,
                "timestamp": asyncio.get_event_loop().time(),
                "tags": tags or {}
            }
            
            await self.redis_client.setex(
                key,
                60,  # 1 minute TTL
                json.dumps(metric_data)
            )
        except Exception as e:
            logger.error(f"Failed to cache metric: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
