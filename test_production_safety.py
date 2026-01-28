"""
Test Suite for Production Safety Features
Tests background indexing, rate limiting, health checks, and circuit breakers.
"""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import safety modules
from background_indexer import (
    BackgroundIndexer, 
    IndexingJob, 
    IndexingState,
    compute_file_hash
)
from production_safety import (
    ProductionSafetyManager,
    CircuitBreaker,
    CircuitBreakerState,
    HealthCheck,
    HealthStatus
)
from request_limiter import (
    RateLimiter,
    InputValidator,
    RequestLimiter
)


class TestBackgroundIndexer:
    """Test background indexing functionality."""
    
    def test_indexing_state_persistence(self, tmp_path):
        """Test that indexing state persists to disk."""
        state_file = tmp_path / "test_state.json"
        state = IndexingState(str(state_file))
        
        # Add a job
        job = IndexingJob(
            job_id="test_1",
            status="completed",
            document_name="test.pdf",
            documents_processed=10,
            chunks_created=50
        )
        state.add_job(job)
        
        # Create new state instance (simulating restart)
        state2 = IndexingState(str(state_file))
        retrieved_job = state2.get_job("test_1")
        
        assert retrieved_job is not None
        assert retrieved_job.status == "completed"
        assert retrieved_job.documents_processed == 10
    
    def test_document_hash_tracking(self, tmp_path):
        """Test document change detection."""
        state_file = tmp_path / "test_state.json"
        state = IndexingState(str(state_file))
        
        # Set initial hash
        state.update_document_hash("doc.pdf", "abc123")
        
        # Check hash is stored
        hash_val = state.get_document_hash("doc.pdf")
        assert hash_val == "abc123"
        
        # Update hash
        state.update_document_hash("doc.pdf", "def456")
        hash_val = state.get_document_hash("doc.pdf")
        assert hash_val == "def456"
    
    def test_indexing_job_creation(self):
        """Test IndexingJob creation and serialization."""
        job = IndexingJob(
            job_id="job_1",
            status="pending",
            document_name="test.pdf"
        )
        
        # Test serialization
        job_dict = job.to_dict()
        assert job_dict['job_id'] == "job_1"
        assert job_dict['status'] == "pending"
        
        # Test deserialization
        job2 = IndexingJob.from_dict(job_dict)
        assert job2.job_id == job.job_id
        assert job2.status == job.status
    
    def test_background_indexer_initialization(self):
        """Test indexer initialization."""
        indexer = BackgroundIndexer()
        
        assert not indexer.running
        assert indexer.max_workers == 1
        assert len(indexer.workers) == 0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1
        )
        
        def failing_func():
            raise Exception("Service error")
        
        # Fail until circuit opens
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Circuit should be open
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Should reject immediately without calling function
        with pytest.raises(Exception) as exc_info:
            breaker.call(failing_func)
        assert "open" in str(exc_info.value).lower()
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1
        )
        
        def failing_func():
            raise Exception("Service error")
        
        def working_func():
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should attempt recovery (HALF_OPEN)
        result = breaker.call(working_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_success_resets_count(self):
        """Test that successful calls reset failure count."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise Exception("Error")
        
        def working_func():
            return "ok"
        
        # Fail twice
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Success resets counter
        result = breaker.call(working_func)
        assert result == "ok"
        assert breaker.failure_count == 0


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_execution(self):
        """Test health check execution."""
        check_fn_called = False
        
        def check_func():
            nonlocal check_fn_called
            check_fn_called = True
            return True
        
        check = HealthCheck("test_component", check_func)
        status = check.check()
        
        assert check_fn_called
        assert status == HealthStatus.HEALTHY
    
    def test_health_check_failure(self):
        """Test health check failure handling."""
        def failing_check():
            raise Exception("Check failed")
        
        check = HealthCheck("test_component", failing_check)
        status = check.check()
        
        assert status == HealthStatus.UNHEALTHY
        assert check.last_error is not None
    
    def test_health_check_critical_flag(self):
        """Test critical flag on health check."""
        def dummy_check():
            return True
        
        critical_check = HealthCheck("critical", dummy_check, critical=True)
        non_critical = HealthCheck("optional", dummy_check, critical=False)
        
        assert critical_check.critical is True
        assert non_critical.critical is False


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_global_limit(self):
        """Test global rate limit enforcement."""
        limiter = RateLimiter(global_rate=3, window_seconds=1)
        
        # Should allow 3 requests
        for _ in range(3):
            allowed, info = limiter.is_allowed()
            assert allowed is True
        
        # 4th request should fail
        allowed, info = limiter.is_allowed()
        assert allowed is False
        assert "Global rate limit" in info['reason']
    
    def test_rate_limiter_per_user(self):
        """Test per-user rate limiting."""
        limiter = RateLimiter(per_user_rate=2, window_seconds=1)
        
        # User 1 can make 2 requests
        for _ in range(2):
            allowed, info = limiter.is_allowed("user1")
            assert allowed is True
        
        # User 1's 3rd request should fail
        allowed, info = limiter.is_allowed("user1")
        assert allowed is False
        
        # User 2 should have separate limit
        allowed, info = limiter.is_allowed("user2")
        assert allowed is True
    
    def test_rate_limiter_reset(self):
        """Test rate limiter window reset."""
        limiter = RateLimiter(global_rate=1, window_seconds=1)
        
        # Use first token
        allowed, _ = limiter.is_allowed()
        assert allowed is True
        
        # Should be rate limited
        allowed, _ = limiter.is_allowed()
        assert allowed is False
        
        # Wait for window reset (add buffer)
        time.sleep(1.1)
        
        # Should allow again
        allowed, _ = limiter.is_allowed()
        assert allowed is True


class TestInputValidator:
    """Test input validation."""
    
    def test_validate_query_length(self):
        """Test query length validation."""
        # Too short
        valid, error = InputValidator.validate_query("ab")
        assert valid is False
        assert "too short" in error.lower()
        
        # Valid length
        valid, error = InputValidator.validate_query("What is a contract?")
        assert valid is True
        
        # Too long
        long_query = "a" * 6000
        valid, error = InputValidator.validate_query(long_query)
        assert valid is False
        assert "too long" in error.lower()
    
    def test_validate_query_sql_injection(self):
        """Test SQL injection detection."""
        malicious_queries = [
            "DROP TABLE documents;",
            "DELETE FROM users WHERE 1=1;",
            "INSERT INTO data VALUES (1, 2);",
        ]
        
        for query in malicious_queries:
            valid, error = InputValidator.validate_query(query)
            assert valid is False
            assert "invalid pattern" in error.lower()
    
    def test_sanitize_query(self):
        """Test query sanitization."""
        # Test whitespace normalization
        query = "  What    is   a   contract?  "
        sanitized = InputValidator.sanitize_query(query)
        assert sanitized == "What is a contract?"
        
        # Test multiple spaces
        query = "Question  with   extra    spaces"
        sanitized = InputValidator.sanitize_query(query)
        assert sanitized == "Question with extra spaces"


class TestRequestLimiter:
    """Test request limiting functionality."""
    
    def test_concurrent_limit(self):
        """Test concurrent request limiting."""
        limiter = RequestLimiter(max_concurrent=2)
        
        # Allow 2 concurrent
        assert limiter.increment_concurrent("user1") is True
        assert limiter.increment_concurrent("user1") is True
        
        # Reject 3rd
        assert limiter.increment_concurrent("user1") is False
        
        # Decrement and allow again
        limiter.decrement_concurrent("user1")
        assert limiter.increment_concurrent("user1") is True
    
    def test_daily_limit(self):
        """Test daily request limiting."""
        limiter = RequestLimiter(max_daily=2)
        
        # Allow 2 requests
        assert limiter.increment_daily("user1") is True
        assert limiter.increment_daily("user1") is True
        
        # Reject 3rd
        assert limiter.increment_daily("user1") is False
    
    def test_concurrent_per_user(self):
        """Test concurrent limits are per-user."""
        limiter = RequestLimiter(max_concurrent=1)
        
        # Each user has separate limit
        assert limiter.increment_concurrent("user1") is True
        assert limiter.increment_concurrent("user2") is True
        
        # Both can't exceed limit
        assert limiter.increment_concurrent("user1") is False
        assert limiter.increment_concurrent("user2") is False


class TestProductionSafetyManager:
    """Test overall safety manager."""
    
    def test_safety_manager_health_report(self):
        """Test safety manager generates health reports."""
        manager = ProductionSafetyManager()
        
        # Register a check
        manager.register_health_check(
            "test",
            lambda: True,
            critical=True
        )
        
        # Perform checks
        status = manager.perform_health_checks()
        assert status == HealthStatus.HEALTHY
        
        # Get report
        report = manager.get_health_report()
        assert report['overall_status'] == 'healthy'
        assert len(report['health_checks']) > 0
    
    def test_safety_manager_circuit_breaker(self):
        """Test circuit breaker through safety manager."""
        manager = ProductionSafetyManager()
        
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise Exception("Error")
            return "success"
        
        # Register breaker
        manager.register_circuit_breaker("test", failure_threshold=3)
        
        # Should fail then open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                manager.call_with_circuit_breaker("test", test_func)
        
        # Should now reject (circuit open)
        with pytest.raises(Exception):
            manager.call_with_circuit_breaker("test", test_func)
    
    def test_safety_manager_metrics(self):
        """Test safety manager tracks metrics."""
        manager = ProductionSafetyManager()
        
        # Record some requests
        manager.record_request(success=True, tokens=100)
        manager.record_request(success=True, tokens=150)
        manager.record_request(success=False, tokens=50)
        
        report = manager.get_health_report()
        metrics = report['metrics']
        
        assert metrics['total_requests'] == 3
        assert metrics['failed_requests'] == 1
        assert metrics['total_tokens_used'] == 300
        assert metrics['failure_rate'] > 0


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_request_validation_chain(self):
        """Test complete request validation chain."""
        # Create limiter
        limiter = RateLimiter(global_rate=10)
        
        # Validate input
        valid, error = InputValidator.validate_query("What is a contract?")
        assert valid
        
        # Check rate limit
        allowed, info = limiter.is_allowed("user1")
        assert allowed
    
    def test_health_check_with_circuit_breaker(self):
        """Test health check monitors circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        def check_breaker_closed():
            return breaker.state == CircuitBreakerState.CLOSED
        
        health = HealthCheck("breaker", check_breaker_closed)
        
        # Initially healthy
        status = health.check()
        assert status == HealthStatus.HEALTHY
        
        # Trigger failure
        with pytest.raises(Exception):
            breaker.call(lambda: 1/0)
        
        # Should now be unhealthy
        status = health.check()
        assert status == HealthStatus.UNHEALTHY


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
