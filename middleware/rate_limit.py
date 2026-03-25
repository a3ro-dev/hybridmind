"""
Rate limiting middleware for HybridMind.
Implements token bucket algorithm for request throttling.
"""

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with per-client tracking.
    
    Features:
    - Configurable requests per minute
    - Per-client rate limiting (by IP or API key)
    - Token refill over time
    - Statistics tracking
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Base rate limit
            burst_size: Max tokens (burst capacity). Defaults to 2x rpm
        """
        self.rpm = requests_per_minute
        self.burst_size = burst_size or (requests_per_minute * 2)
        
        # Token buckets per client
        self.tokens: dict = defaultdict(lambda: float(self.burst_size))
        self.last_update: dict = defaultdict(float)
        self.lock = Lock()
        
        # Statistics
        self.total_requests = 0
        self.throttled_requests = 0
        
        logger.info(
            f"RateLimiter initialized: {requests_per_minute} req/min, "
            f"burst={self.burst_size}"
        )
    
    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if request is allowed for the given client.
        
        Args:
            client_id: Unique client identifier (IP or API key)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        with self.lock:
            now = time.time()
            self.total_requests += 1
            
            # Calculate token refill since last request
            elapsed = now - self.last_update[client_id]
            self.last_update[client_id] = now
            
            # Refill tokens (rpm/60 tokens per second)
            refill_rate = self.rpm / 60.0
            refill = elapsed * refill_rate
            self.tokens[client_id] = min(
                self.burst_size,
                self.tokens[client_id] + refill
            )
            
            # Check and consume token
            if self.tokens[client_id] >= 1.0:
                self.tokens[client_id] -= 1.0
                
                return True, {
                    "remaining": int(self.tokens[client_id]),
                    "limit": self.rpm,
                    "reset_seconds": int((self.burst_size - self.tokens[client_id]) / refill_rate)
                }
            
            # Request throttled
            self.throttled_requests += 1
            retry_after = int((1.0 - self.tokens[client_id]) / refill_rate) + 1
            
            return False, {
                "remaining": 0,
                "limit": self.rpm,
                "retry_after": retry_after
            }
    
    def get_client_tokens(self, client_id: str) -> float:
        """Get remaining tokens for a client."""
        with self.lock:
            # Update tokens first
            now = time.time()
            elapsed = now - self.last_update[client_id]
            refill_rate = self.rpm / 60.0
            refill = elapsed * refill_rate
            return min(self.burst_size, self.tokens[client_id] + refill)
    
    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        with self.lock:
            throttle_rate = (
                self.throttled_requests / self.total_requests
                if self.total_requests > 0 else 0.0
            )
            
            return {
                "total_requests": self.total_requests,
                "throttled_requests": self.throttled_requests,
                "throttle_rate": round(throttle_rate, 4),
                "active_clients": len(self.tokens),
                "requests_per_minute": self.rpm,
                "burst_size": self.burst_size
            }
    
    def reset_client(self, client_id: str):
        """Reset rate limit for a specific client."""
        with self.lock:
            self.tokens[client_id] = float(self.burst_size)
            self.last_update[client_id] = time.time()
    
    def reset_all(self):
        """Reset all rate limits."""
        with self.lock:
            self.tokens.clear()
            self.last_update.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request rate limiting.
    
    Applies token bucket rate limiting based on client IP
    or X-API-Key header.
    """
    
    # Paths exempt from rate limiting
    EXEMPT_PATHS = {"/health", "/live", "/ready", "/docs", "/redoc", "/openapi.json", "/admin/compact", "/snapshot"}
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: Optional[int] = None,
        enabled: bool = True
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Rate limit
            burst_size: Burst capacity
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute, burst_size)
        self.enabled = enabled
    
    def _get_client_id(self, request: Request) -> str:
        """
        Extract client identifier from request.
        Prefers X-API-Key header, falls back to client IP.
        """
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in the chain is the original client
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        # Direct connection
        if request.client:
            return f"ip:{request.client.host}"
        
        return "ip:unknown"
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip if disabled or exempt path
        if not self.enabled or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        allowed, info = self.limiter.is_allowed(client_id)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": info["retry_after"]
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "Retry-After": str(info["retry_after"])
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        
        return response


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(rpm: int = 100) -> RateLimiter:
    """Get or create rate limiter singleton."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_minute=rpm)
    
    return _rate_limiter

