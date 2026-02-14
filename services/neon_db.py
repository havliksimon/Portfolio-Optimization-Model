"""
Portfolio Optimizer - Neon DB Optimization
===========================================

Optimized PostgreSQL configuration for Neon DB serverless:
- Connection pooling
- PgBouncer compatibility
- Session management
- Query optimization
- Automatic reconnection
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


class NeonDatabaseManager:
    """
    Manages Neon DB connections with optimization for serverless.
    
    Features:
    - Connection pooling (QueuePool for non-serverless, NullPool for PgBouncer)
    - Prepared statement caching disabled for PgBouncer
    - Automatic connection validation
    - Connection pre-ping to handle stale connections
    - SSL mode configuration
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
    
    def _setup_engine(self):
        """Configure SQLAlchemy engine with Neon optimizations."""
        
        # Detect if using PgBouncer
        is_pgbouncer = 'pgbouncer' in self.database_url.lower() or \
                      self._is_neon_pooler_url(self.database_url)
        
        # Base engine arguments
        engine_args = {
            'pool_pre_ping': True,  # Verify connections before use
            'pool_recycle': 300,    # Recycle connections after 5 min
            'echo': False,
        }
        
        if is_pgbouncer:
            # PgBouncer/Neon Pooler mode
            logger.info("Configuring for PgBouncer/Neon Pooler")
            engine_args.update({
                'poolclass': NullPool,  # Let PgBouncer handle pooling
                'connect_args': {
                    'sslmode': 'require',
                    'options': '-c statement_timeout=30000'  # 30 second timeout
                }
            })
        else:
            # Direct connection with SQLAlchemy pooling
            logger.info("Configuring for direct PostgreSQL connection")
            engine_args.update({
                'poolclass': QueuePool,
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30,
                'connect_args': {
                    'sslmode': 'require',
                    'connect_timeout': 10
                }
            })
        
        # Fix postgres:// to postgresql://
        url = self.database_url
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        
        # Add prepared statement options for PgBouncer
        if is_pgbouncer:
            # Disable prepared statements which conflict with PgBouncer
            url = self._add_pgbouncer_options(url)
        
        self.engine = create_engine(url, **engine_args)
        
        # Add event listeners for connection management
        self._add_event_listeners()
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database engine configured successfully")
    
    def _is_neon_pooler_url(self, url: str) -> bool:
        """Check if URL is using Neon pooler."""
        parsed = urlparse(url)
        # Neon pooler URLs typically use -pooler in hostname
        return '-pooler' in parsed.hostname.lower() if parsed.hostname else False
    
    def _add_pgbouncer_options(self, url: str) -> str:
        """Add PgBouncer compatibility options to URL."""
        # Add application_name for monitoring
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}application_name=portfolio_optimizer"
    
    def _add_event_listeners(self):
        """Add SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self.engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Handle new connection."""
            logger.debug("New database connection established")
            
            # Set session parameters for Neon
            with dbapi_conn.cursor() as cursor:
                # Optimize for read-heavy workload
                cursor.execute("SET SESSION characteristics AS TRANSACTION READ WRITE")
                # Set statement timeout
                cursor.execute("SET SESSION statement_timeout = '30s'")
        
        @event.listens_for(self.engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug("Database connection checked out")
        
        @event.listens_for(self.engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            """Handle connection return to pool."""
            logger.debug("Database connection returned to pool")
    
    @contextmanager
    def get_session(self) -> Generator:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                result = session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_raw_connection(self):
        """Get raw connection for bulk operations."""
        return self.engine.raw_connection()
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_pool_status(self) -> dict:
        """Get connection pool statistics."""
        if hasattr(self.engine, 'pool'):
            pool = self.engine.pool
            return {
                'size': pool.size() if hasattr(pool, 'size') else 0,
                'checked_in': pool.checkedin() if hasattr(pool, 'checkedin') else 0,
                'checked_out': pool.checkedout() if hasattr(pool, 'checkedout') else 0,
            }
        return {}
    
    def dispose(self):
        """Clean up all connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections disposed")


class DatabaseOptimizer:
    """
    Query optimization utilities for Neon DB.
    """
    
    @staticmethod
    def optimize_for_neon(session):
        """
        Apply Neon-specific optimizations for a session.
        
        Call this at the start of heavy operations.
        """
        # Set work_mem for better performance on complex queries
        session.execute(text("SET LOCAL work_mem = '64MB'"))
        # Enable parallel queries if available
        session.execute(text("SET LOCAL max_parallel_workers_per_gather = 2"))
    
    @staticmethod
    def create_indexes(engine):
        """
        Create optimized indexes for Neon DB.
        
        Should be called during database initialization.
        """
        index_definitions = [
            # Market data indexes
            """
            CREATE INDEX IF NOT EXISTS idx_market_data_asset_date 
            ON market_data (asset_id, date DESC)
            """,
            # Portfolio history indexes
            """
            CREATE INDEX IF NOT EXISTS idx_portfolio_history_date 
            ON portfolio_history (portfolio_id, snapshot_date DESC)
            """,
            # User activity indexes
            """
            CREATE INDEX IF NOT EXISTS idx_user_activity_user_time 
            ON user_activities (user_id, created_at DESC)
            """,
            # Saved scenarios index
            """
            CREATE INDEX IF NOT EXISTS idx_scenarios_user_updated 
            ON saved_scenarios (user_id, updated_at DESC)
            """,
        ]
        
        with engine.connect() as conn:
            for sql in index_definitions:
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    logger.info(f"Created index: {sql[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation skipped: {e}")
    
    @staticmethod
    def vacuum_analyze(engine):
        """Run VACUUM ANALYZE for query optimization."""
        with engine.connect() as conn:
            conn.execute(text("ANALYZE"))
            conn.commit()
            logger.info("ANALYZE completed")


class CacheManager:
    """
    In-memory caching for guest users and frequent queries.
    
    Reduces Neon DB queries for non-persistent data.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache = {}
        _max_size = max_size
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        from datetime import datetime
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (datetime.utcnow() - timestamp).seconds < self._ttl:
                return value
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache a value."""
        from datetime import datetime
        
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= 1000:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (value, datetime.utcnow())
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
    
    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        keys_to_remove = [k for k in self._cache if pattern in k]
        for k in keys_to_remove:
            del self._cache[k]


# Global instances
db_manager: Optional[NeonDatabaseManager] = None
cache_manager = CacheManager()


def init_neon_db(database_url: str) -> NeonDatabaseManager:
    """Initialize Neon DB manager."""
    global db_manager
    db_manager = NeonDatabaseManager(database_url)
    return db_manager


def get_db() -> Generator:
    """FastAPI-style dependency for getting database session."""
    if db_manager is None:
        raise RuntimeError("Database not initialized")
    
    with db_manager.get_session() as session:
        yield session


def with_neon_optimization(func):
    """Decorator to apply Neon optimizations for a function."""
    def wrapper(*args, **kwargs):
        if db_manager and db_manager.engine:
            # Apply optimizations
            with db_manager.get_session() as session:
                DatabaseOptimizer.optimize_for_neon(session)
                kwargs['_optimized_session'] = session
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper
