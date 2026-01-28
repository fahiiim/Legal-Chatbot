"""
Production Request Validation and Rate Limiting
Protects the RAG pipeline from abuse and invalid inputs.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with per-user and global limits.
    """
    
    def __init__(self,
                 global_rate: int = 100,
                 window_seconds: int = 60,
                 per_user_rate: int = 20):
        """
        Initialize rate limiter.
        
        Args:
            global_rate: Max requests globally per window
            window_seconds: Window duration in seconds
            per_user_rate: Max requests per user per window
        """
        self.global_rate = global_rate
        self.window_seconds = window_seconds
        self.per_user_rate = per_user_rate
        
        self.global_tokens = global_rate
        self.last_global_refill = time.time()
        
        self.user_tokens = defaultdict(lambda: per_user_rate)
        self.user_last_refill = defaultdict(time.time)
        
        self.lock = threading.RLock()
    
    def _refill_tokens(self):
        """Refill global token bucket."""
        now = time.time()
        elapsed = now - self.last_global_refill
        
        if elapsed >= self.window_seconds:
            self.global_tokens = self.global_rate
            self.last_global_refill = now
    
    def _refill_user_tokens(self, user_id: str):
        """Refill user token bucket."""
        now = time.time()
        elapsed = now - self.user_last_refill[user_id]
        
        if elapsed >= self.window_seconds:
            self.user_tokens[user_id] = self.per_user_rate
            self.user_last_refill[user_id] = now
    
    def is_allowed(self, user_id: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Check if request is allowed.
        
        Args:
            user_id: Optional user identifier
        
        Returns:
            Tuple of (allowed: bool, info: Dict with limit details)
        """
        with self.lock:
            self._refill_tokens()
            
            # Check global limit
            if self.global_tokens <= 0:
                return False, {
                    'reason': 'Global rate limit exceeded',
                    'global_tokens': self.global_tokens,
                    'per_user_tokens': 0
                }
            
            # Check per-user limit
            if user_id:
                self._refill_user_tokens(user_id)
                if self.user_tokens[user_id] <= 0:
                    return False, {
                        'reason': f'User rate limit exceeded for {user_id}',
                        'global_tokens': self.global_tokens,
                        'per_user_tokens': self.user_tokens[user_id]
                    }
                
                self.user_tokens[user_id] -= 1
            
            self.global_tokens -= 1
            
            return True, {
                'allowed': True,
                'global_tokens': self.global_tokens,
                'per_user_tokens': self.user_tokens.get(user_id, self.per_user_rate) if user_id else None
            }
    
    def get_status(self) -> Dict:
        """Get current limiter status."""
        with self.lock:
            return {
                'global_tokens': self.global_tokens,
                'global_rate': self.global_rate,
                'window_seconds': self.window_seconds,
                'active_users': len(self.user_tokens),
                'per_user_rate': self.per_user_rate
            }


class InputValidator:
    """
    Validates inputs for security and quality.
    Prevents common attack patterns and invalid requests.
    """
    
    MAX_QUERY_LENGTH = 5000
    MIN_QUERY_LENGTH = 3
    
    # Malicious patterns to block
    BLOCKED_PATTERNS = [
        'DROP TABLE',
        'DELETE FROM',
        'INSERT INTO',
        'UPDATE ',
        '<script',
        'javascript:',
        'onclick=',
        'onerror=',
    ]
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a legal query.
        
        Args:
            query: Query string to validate
        
        Returns:
            Tuple of (valid: bool, error_message: str or None)
        """
        # Length checks
        if not query or len(query) < InputValidator.MIN_QUERY_LENGTH:
            return False, "Query is too short (minimum 3 characters)"
        
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            return False, f"Query is too long (maximum {InputValidator.MAX_QUERY_LENGTH} characters)"
        
        # Check for malicious patterns
        query_upper = query.upper()
        for pattern in InputValidator.BLOCKED_PATTERNS:
            if pattern in query_upper:
                return False, "Query contains invalid pattern"
        
        # Check for excessive special characters
        special_count = sum(1 for c in query if not c.isalnum() and c not in ' ?!')
        if special_count > len(query) * 0.3:  # More than 30% special chars
            return False, "Query contains too many special characters"
        
        return True, None
    
    @staticmethod
    def validate_doc_type(doc_type: str, supported: list) -> Tuple[bool, Optional[str]]:
        """
        Validate document type filter.
        
        Args:
            doc_type: Document type to validate
            supported: List of supported document types
        
        Returns:
            Tuple of (valid: bool, error_message: str or None)
        """
        if doc_type not in supported:
            return False, f"Unsupported document type: {doc_type}"
        return True, None
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """
        Sanitize query string.
        Removes leading/trailing whitespace and normalizes spacing.
        
        Args:
            query: Query to sanitize
        
        Returns:
            Sanitized query
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        # Remove leading/trailing spaces
        return query.strip()


class RequestLimiter:
    """
    Monitors and limits requests per IP/user to prevent abuse.
    """
    
    def __init__(self,
                 max_concurrent: int = 10,
                 max_daily: int = 1000):
        """
        Initialize request limiter.
        
        Args:
            max_concurrent: Max concurrent requests
            max_daily: Max requests per day per user
        """
        self.max_concurrent = max_concurrent
        self.max_daily = max_daily
        
        self.concurrent_requests = defaultdict(int)
        self.daily_requests = defaultdict(lambda: {'count': 0, 'date': None})
        
        self.lock = threading.RLock()
    
    def increment_concurrent(self, user_id: str) -> bool:
        """
        Increment concurrent request counter.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if allowed, False if limit exceeded
        """
        with self.lock:
            if self.concurrent_requests[user_id] >= self.max_concurrent:
                return False
            
            self.concurrent_requests[user_id] += 1
            return True
    
    def decrement_concurrent(self, user_id: str):
        """Decrement concurrent request counter."""
        with self.lock:
            self.concurrent_requests[user_id] = max(0, self.concurrent_requests[user_id] - 1)
    
    def increment_daily(self, user_id: str) -> bool:
        """
        Increment daily request counter.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if allowed, False if limit exceeded
        """
        with self.lock:
            today = datetime.now().date()
            
            daily_record = self.daily_requests[user_id]
            
            # Reset if new day
            if daily_record['date'] != today:
                daily_record['count'] = 0
                daily_record['date'] = today
            
            if daily_record['count'] >= self.max_daily:
                return False
            
            daily_record['count'] += 1
            return True
    
    def get_status(self, user_id: str) -> Dict:
        """Get request status for user."""
        with self.lock:
            today = datetime.now().date()
            daily_record = self.daily_requests[user_id]
            
            # Reset if new day
            if daily_record['date'] != today:
                daily_record['count'] = 0
                daily_record['date'] = today
            
            return {
                'user_id': user_id,
                'concurrent_requests': self.concurrent_requests[user_id],
                'max_concurrent': self.max_concurrent,
                'daily_requests': daily_record['count'],
                'max_daily': self.max_daily
            }


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_request_limiter: Optional[RequestLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        with _limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    
    return _rate_limiter


def get_request_limiter() -> RequestLimiter:
    """Get or create global request limiter."""
    global _request_limiter
    
    if _request_limiter is None:
        with _limiter_lock:
            if _request_limiter is None:
                _request_limiter = RequestLimiter()
    
    return _request_limiter
