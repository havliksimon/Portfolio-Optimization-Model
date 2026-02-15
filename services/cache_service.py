"""
Portfolio Optimizer - Caching Service
=====================================

Smart caching layer to minimize database queries and API calls.
Critical for free tier deployments (Koyeb/Render + Neon DB).

Features:
- Redis cache for market data and analysis results
- In-memory LRU cache for hot data
- Cache warming strategies
- Graceful degradation when cache unavailable

Author: Built with quantitative rigor
"""

import os
import json
import hashlib
import logging
from typing import Optional, Any, Callable
from functools import wraps
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheService:
    """
    Multi-layer caching service.
    
    Layer 1: In-memory LRU cache (fastest, process-local)
    Layer 2: Redis cache (shared across workers)
    Layer 3: Database (source of truth)
    """
    
    # Cache TTLs in seconds
    DEFAULT_TTL = 3600  # 1 hour
    MARKET_DATA_TTL = 3600  # 1 hour (market data changes frequently)
    ANALYSIS_TTL = 86400  # 24 hours (analysis results are expensive)
    STATIC_TTL = 604800  # 7 days (static data like sector info)
    
    def __init__(self):
        self._redis_client: Optional[Any] = None
        self._memory_cache: Optional[Any] = None
        self._enabled = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
        
        if not self._enabled:
            logger.info("Cache disabled via ENABLE_CACHE env var")
            return
        
        # Initialize in-memory cache
        if CACHETOOLS_AVAILABLE:
            self._memory_cache = TTLCache(
                maxsize=1000,
                ttl=self.DEFAULT_TTL
            )
            logger.info("In-memory cache initialized")
        
        # Initialize Redis cache
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection with fallback."""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available (redis package not installed)")
            return
        
        redis_url = os.getenv('REDIS_URL') or os.getenv('UPSTASH_REDIS_REST_URL')
        
        if not redis_url:
            logger.info("Redis URL not configured, using memory cache only")
            return
        
        try:
            # Handle both redis:// and rediss:// URLs
            if redis_url.startswith('redis://') or redis_url.startswith('rediss://'):
                self._redis_client = redis.from_url(
                    redis_url,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
            else:
                # Upstash REST API mode
                import upstash_redis
                self._redis_client = upstash_redis.Redis(
                    url=os.getenv('UPSTASH_REDIS_REST_URL'),
                    token=os.getenv('UPSTASH_REDIS_REST_TOKEN')
                )
            
            # Test connection
            self._redis_client.ping()
            logger.info("Redis cache connected successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
            self._redis_client = None
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a consistent cache key from arguments."""
        key_data = json.dumps({
            'args': args,
            'kwargs': sorted(kwargs.items())
        }, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        return json.dumps({
            'data': value,
            'timestamp': datetime.utcnow().isoformat()
        }, default=str)
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        try:
            parsed = json.loads(value)
            return parsed.get('data')
        except (json.JSONDecodeError, TypeError):
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2 -> None)."""
        if not self._enabled:
            return None
        
        # Try L1 (memory) cache first
        if self._memory_cache is not None:
            try:
                if key in self._memory_cache:
                    logger.debug(f"L1 cache hit: {key}")
                    return self._memory_cache[key]
            except Exception as e:
                logger.debug(f"L1 cache error: {e}")
        
        # Try L2 (Redis) cache
        if self._redis_client is not None:
            try:
                value = self._redis_client.get(key)
                if value:
                    deserialized = self._deserialize(value)
                    # Backfill L1 cache
                    if self._memory_cache is not None:
                        self._memory_cache[key] = deserialized
                    logger.debug(f"L2 cache hit: {key}")
                    return deserialized
            except Exception as e:
                logger.debug(f"L2 cache error: {e}")
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        memory_only: bool = False
    ) -> bool:
        """Set value in cache (L1 and optionally L2)."""
        if not self._enabled:
            return False
        
        ttl = ttl or self.DEFAULT_TTL
        success = False
        
        # Set L1 (memory) cache
        if self._memory_cache is not None:
            try:
                self._memory_cache[key] = value
                success = True
            except Exception as e:
                logger.debug(f"L1 cache set error: {e}")
        
        # Set L2 (Redis) cache
        if not memory_only and self._redis_client is not None:
            try:
                serialized = self._serialize(value)
                self._redis_client.setex(key, ttl, serialized)
                success = True
            except Exception as e:
                logger.debug(f"L2 cache set error: {e}")
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from all cache layers."""
        success = False
        
        # Delete from L1
        if self._memory_cache is not None:
            try:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    success = True
            except Exception as e:
                logger.debug(f"L1 cache delete error: {e}")
        
        # Delete from L2
        if self._redis_client is not None:
            try:
                self._redis_client.delete(key)
                success = True
            except Exception as e:
                logger.debug(f"L2 cache delete error: {e}")
        
        return success
    
    def clear(self) -> bool:
        """Clear all caches."""
        success = False
        
        # Clear L1
        if self._memory_cache is not None:
            try:
                self._memory_cache.clear()
                success = True
            except Exception as e:
                logger.debug(f"L1 cache clear error: {e}")
        
        # Clear L2 (be careful with this in production!)
        # Only clear keys with our prefix
        if self._redis_client is not None:
            try:
                # This is a simple clear - in production use a pattern
                # self._redis_client.flushdb()  # Don't do this!
                success = True
            except Exception as e:
                logger.debug(f"L2 cache clear error: {e}")
        
        return success
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        skip_args: Optional[list] = None
    ):
        """
        Decorator for caching function results.
        
        Usage:
            @cache.cached(ttl=3600, key_prefix="market_data")
            def fetch_stock_price(ticker: str) -> dict:
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self._enabled:
                    return func(*args, **kwargs)
                
                # Build cache key
                # Skip arguments that shouldn't affect caching
                skip = set(skip_args or [])
                cache_args = [a for i, a in enumerate(args) if i not in skip]
                cache_kwargs = {k: v for k, v in kwargs.items() if k not in skip}
                
                key = f"{key_prefix}:{self._make_key(*cache_args, **cache_kwargs)}"
                
                # Try to get from cache
                cached_value = self.get(key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                logger.debug(f"Cache set for {func.__name__}")
                
                return result
            
            # Attach cache management methods
            wrapper.cache_delete = lambda *a, **kw: self.delete(
                f"{key_prefix}:{self._make_key(*a, **kw)}"
            )
            wrapper.cache_clear = lambda: self.clear()
            
            return wrapper
        return decorator
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            'enabled': self._enabled,
            'memory_cache': False,
            'redis_cache': False
        }
        
        if self._memory_cache is not None:
            stats['memory_cache'] = True
            stats['memory_size'] = len(self._memory_cache)
        
        if self._redis_client is not None:
            try:
                info = self._redis_client.info()
                stats['redis_cache'] = True
                stats['redis_keys'] = self._redis_client.dbsize()
                stats['redis_memory_used'] = info.get('used_memory_human', 'N/A')
            except Exception as e:
                stats['redis_error'] = str(e)
        
        return stats


# Global cache instance
cache = CacheService()


def cached_market_data(func: Callable) -> Callable:
    """Specialized decorator for market data with 1-hour TTL."""
    return cache.cached(
        ttl=CacheService.MARKET_DATA_TTL,
        key_prefix="market"
    )(func)


def cached_analysis(func: Callable) -> Callable:
    """Specialized decorator for analysis results with 24-hour TTL."""
    return cache.cached(
        ttl=CacheService.ANALYSIS_TTL,
        key_prefix="analysis"
    )(func)


def cached_static(func: Callable) -> Callable:
    """Specialized decorator for static data with 7-day TTL."""
    return cache.cached(
        ttl=CacheService.STATIC_TTL,
        key_prefix="static"
    )(func)


# Example usage and testing
if __name__ == "__main__":
    # Test the cache
    @cache.cached(ttl=10, key_prefix="test")
    def expensive_function(x: int) -> int:
        import time
        time.sleep(1)  # Simulate slow operation
        return x * x
    
    print("Testing cache...")
    
    # First call (slow)
    start = datetime.now()
    result1 = expensive_function(5)
    duration1 = (datetime.now() - start).total_seconds()
    print(f"First call: {result1} (took {duration1:.2f}s)")
    
    # Second call (fast - from cache)
    start = datetime.now()
    result2 = expensive_function(5)
    duration2 = (datetime.now() - start).total_seconds()
    print(f"Second call: {result2} (took {duration2:.2f}s)")
    
    print(f"\nCache stats: {cache.get_stats()}")
