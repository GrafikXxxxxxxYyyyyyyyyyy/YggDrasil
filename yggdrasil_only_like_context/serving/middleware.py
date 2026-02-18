"""Middleware for the FastAPI serving layer."""
from __future__ import annotations

import time
import logging
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger("yggdrasil.serving")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware.
    
    Set YGGDRASIL_API_KEY environment variable to enable.
    Pass key via X-API-Key header or ?api_key= query parameter.
    """
    
    def __init__(self, app, api_key: Optional[str] = None):
        super().__init__(app)
        self.api_key = api_key
    
    async def dispatch(self, request: Request, call_next):
        if self.api_key is None:
            return await call_next(request)
        
        # Skip auth for health check
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)
        
        # Check header
        key = request.headers.get("X-API-Key")
        if not key:
            key = request.query_params.get("api_key")
        
        if key != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting.
    
    Limits requests per IP address.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._request_log: dict = {}  # ip -> list of timestamps
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old entries
        if client_ip in self._request_log:
            self._request_log[client_ip] = [
                t for t in self._request_log[client_ip]
                if now - t < 60
            ]
        else:
            self._request_log[client_ip] = []
        
        # Check limit
        if len(self._request_log[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        
        self._request_log[client_ip].append(now)
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing."""
    
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        
        response = await call_next(request)
        
        elapsed = time.time() - start
        logger.info(
            f"{request.method} {request.url.path} "
            f"[{response.status_code}] {elapsed:.3f}s"
        )
        
        return response
