"""
Production Safety Manager
Ensures RAG pipeline safety with health checks, circuit breakers, and monitoring.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from enum import Enum
import traceback

from config import *


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Simple circuit breaker for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 name: str = "breaker"):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            name: Identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result of function call
        
        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} opened after "
                    f"{self.failure_count} failures"
                )
    
    def get_state(self) -> Dict:
        """Get circuit breaker state."""
        with self.lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time.isoformat() 
                                     if self.last_failure_time else None
            }


class HealthCheck:
    """Health check for a specific component."""
    
    def __init__(self, 
                 name: str,
                 check_fn: Callable,
                 timeout: int = 5,
                 critical: bool = False):
        """
        Initialize health check.
        
        Args:
            name: Component name
            check_fn: Function that returns True if healthy
            timeout: Check timeout in seconds
            critical: If True, component failure fails entire system
        """
        self.name = name
        self.check_fn = check_fn
        self.timeout = timeout
        self.critical = critical
        
        self.last_check_time = None
        self.last_status = HealthStatus.UNKNOWN
        self.last_error = None
        self.lock = threading.RLock()
    
    def check(self) -> HealthStatus:
        """
        Perform health check.
        
        Returns:
            HealthStatus
        """
        try:
            result = self.check_fn()
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
        except Exception as e:
            logger.error(f"Health check {self.name} failed: {e}")
            status = HealthStatus.UNHEALTHY
            self.last_error = str(e)
        
        with self.lock:
            self.last_check_time = datetime.now()
            self.last_status = status
        
        return status
    
    def get_status(self) -> Dict:
        """Get current status."""
        with self.lock:
            return {
                'name': self.name,
                'status': self.last_status.value,
                'last_check': self.last_check_time.isoformat() 
                             if self.last_check_time else None,
                'error': self.last_error,
                'critical': self.critical
            }


class ProductionSafetyManager:
    """
    Comprehensive safety monitoring for RAG pipeline.
    
    Features:
    - Health checks for all components
    - Circuit breakers for external services
    - Request rate limiting
    - Error tracking and recovery
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize safety manager."""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.total_requests = 0
        self.failed_requests = 0
        self.total_tokens_used = 0
        
        self.lock = threading.RLock()
        self.last_health_status = HealthStatus.UNKNOWN
    
    def register_health_check(self,
                             name: str,
                             check_fn: Callable,
                             timeout: int = 5,
                             critical: bool = False):
        """
        Register a health check.
        
        Args:
            name: Component name
            check_fn: Function that returns True if healthy
            timeout: Check timeout
            critical: If True, failure fails entire system
        """
        self.health_checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            timeout=timeout,
            critical=critical
        )
        logger.info(f"Registered health check: {name}")
    
    def register_circuit_breaker(self,
                                name: str,
                                failure_threshold: int = 5,
                                recovery_timeout: int = 60):
        """
        Register a circuit breaker.
        
        Args:
            name: Service name
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before retry
        """
        self.circuit_breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name
        )
        logger.info(f"Registered circuit breaker: {name}")
    
    def perform_health_checks(self) -> HealthStatus:
        """
        Perform all health checks.
        
        Returns:
            Overall health status
        """
        statuses = []
        
        for check in self.health_checks.values():
            status = check.check()
            statuses.append((status, check.critical))
        
        # Determine overall status
        if any(status == HealthStatus.UNHEALTHY and critical 
               for status, critical in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.UNHEALTHY 
                for status, _ in statuses):
            overall_status = HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY 
                for status, _ in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        with self.lock:
            self.last_health_status = overall_status
        
        return overall_status
    
    def get_health_report(self) -> Dict:
        """Get detailed health report."""
        checks = [check.get_status() for check in self.health_checks.values()]
        breakers = [breaker.get_state() for breaker in self.circuit_breakers.values()]
        
        with self.lock:
            return {
                'overall_status': self.last_health_status.value,
                'timestamp': datetime.now().isoformat(),
                'health_checks': checks,
                'circuit_breakers': breakers,
                'metrics': {
                    'total_requests': self.total_requests,
                    'failed_requests': self.failed_requests,
                    'failure_rate': self.failed_requests / max(self.total_requests, 1),
                    'total_tokens_used': self.total_tokens_used
                }
            }
    
    def record_request(self, success: bool, tokens: int = 0):
        """Record request metrics."""
        with self.lock:
            self.total_requests += 1
            if not success:
                self.failed_requests += 1
            self.total_tokens_used += tokens
    
    def call_with_circuit_breaker(self,
                                  service_name: str,
                                  func: Callable,
                                  *args,
                                  **kwargs):
        """
        Call a function with circuit breaker protection.
        
        Args:
            service_name: Service identifier
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        if service_name not in self.circuit_breakers:
            # Create breaker if not exists
            self.register_circuit_breaker(service_name)
        
        breaker = self.circuit_breakers[service_name]
        return breaker.call(func, *args, **kwargs)
    
    def is_healthy(self) -> bool:
        """Check if system is in healthy state."""
        with self.lock:
            return self.last_health_status in (
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED
            )
    
    def is_critical(self) -> bool:
        """Check if system is in critical state."""
        with self.lock:
            return self.last_health_status == HealthStatus.UNHEALTHY


# Global safety manager instance
_safety_manager: Optional[ProductionSafetyManager] = None
_safety_lock = threading.Lock()


def get_safety_manager() -> ProductionSafetyManager:
    """Get or create global safety manager."""
    global _safety_manager
    
    if _safety_manager is None:
        with _safety_lock:
            if _safety_manager is None:
                _safety_manager = ProductionSafetyManager()
    
    return _safety_manager
