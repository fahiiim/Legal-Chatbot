# Production-Safe Legal RAG Pipeline

## Overview

This document describes the production-safety features implemented in the Michigan Legal RAG Chatbot, ensuring reliable, scalable service delivery with built-in error recovery and monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Request Layer  │  │ Safety Layer │  │ Indexing Layer │  │
│  ├────────────────┤  ├──────────────┤  ├────────────────┤  │
│  │ - Validation   │  │ - Health     │  │ - Background   │  │
│  │ - Rate Limit   │  │   Checks     │  │   Indexing     │  │
│  │ - Input Check  │  │ - Breakers   │  │ - Change       │  │
│  │ - Sanitize     │  │ - Monitoring │  │   Detection    │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
│           │                  │                   │           │
│           └──────────────────┼───────────────────┘           │
│                              │                               │
│                    ┌─────────▼──────────┐                   │
│                    │   RAG Engine       │                   │
│                    ├────────────────────┤                   │
│                    │ - Query Processing │                   │
│                    │ - Retrieval        │                   │
│                    │ - Generation       │                   │
│                    └────────┬───────────┘                   │
│                             │                                │
│           ┌─────────────────┼──────────────────┐             │
│           │                 │                  │             │
│    ┌──────▼─────┐  ┌────────▼────────┐  ┌────▼──────┐     │
│    │ Vector      │  │ LLM API         │  │ Citation  │     │
│    │ Store       │  │ (OpenAI)        │  │ Extractor │     │
│    │ (ChromaDB)  │  │                 │  │           │     │
│    └─────────────┘  └─────────────────┘  └───────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Background Indexing

**Purpose**: Asynchronously index documents without blocking API requests

**Features**:
- Non-blocking document processing
- Automatic change detection (SHA256 hashing)
- Incremental batch processing
- Error recovery with exponential backoff
- State persistence
- Job queue management

**Usage**:
```bash
# Start indexing all documents
POST /indexing/start

# Index specific document
POST /indexing/document/{document_name}

# Check job status
GET /indexing/status/{job_id}

# Get comprehensive stats
GET /indexing/stats

# List all jobs
GET /indexing/jobs
```

### 2. Health Monitoring

**Purpose**: Real-time system health tracking and alerting

**Components**:
- **RAG Engine Health**: Verifies initialization and functionality
- **Vector Store Health**: Checks database connectivity
- **Circuit Breakers**: Automatic fault isolation
- **Request Metrics**: Tracks success/failure rates

**Endpoints**:
```bash
# Quick health check
GET /health

# Detailed health report
GET /health/detailed
```

**Health Levels**:
- `healthy`: All systems operational
- `degraded`: Non-critical components failing
- `unhealthy`: Critical failures
- `unknown`: Status unknown

### 3. Circuit Breakers

**Purpose**: Prevent cascading failures in external service calls

**Configuration**:
- Failure Threshold: 3-5 failures
- Recovery Timeout: 30-60 seconds
- Automatic State Management

**States**:
1. **CLOSED**: Normal operation (service available)
2. **OPEN**: Service failing, reject requests (fail fast)
3. **HALF_OPEN**: Testing recovery (limited traffic)

**Protected Services**:
- OpenAI API calls
- Vector database operations

### 4. Rate Limiting

**Purpose**: Prevent abuse and resource exhaustion

**Implementation**:
- **Token Bucket Algorithm**: Smooth traffic distribution
- **Global Limits**: 100 requests/minute
- **Per-User Limits**: 20 requests/minute
- **Daily Limits**: 1000 requests/user/day
- **Concurrent Limits**: 10 concurrent requests/user

**Headers**:
```
X-User-Id: user123  # Optional user identifier
```

### 5. Input Validation & Sanitization

**Purpose**: Secure the application from malicious inputs

**Checks**:
- Query length validation (3-5000 characters)
- SQL injection pattern detection
- XSS attack prevention
- Special character filtering
- Query sanitization (whitespace normalization)

**Error Responses**:
```json
{
  "detail": "Query is too long (maximum 5000 characters)"
}
```

## Production Configuration

### Memory Management
```python
# Vector store uses disk-based storage (ChromaDB)
# Documents chunked to manage token limits
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 100  # tokens
MAX_CONTEXT_TOKENS = 12000
```

### Concurrency
```python
# Single worker thread for sequential processing
# Thread-safe operations with locks
MAX_WORKERS = 1
```

### Error Recovery
```python
# Automatic retry with exponential backoff
MAX_RETRIES = 3
CIRCUIT_BREAKER_TIMEOUT = 60  # seconds
```

## Monitoring & Metrics

### Key Metrics
```json
{
  "total_requests": 1000,
  "failed_requests": 5,
  "failure_rate": 0.005,
  "total_tokens_used": 50000,
  "average_response_time": 2.5,
  "queue_size": 0,
  "active_users": 45
}
```

### Health Report Structure
```json
{
  "overall_status": "healthy",
  "timestamp": "2024-01-28T10:30:00",
  "health_checks": [
    {
      "name": "rag_engine",
      "status": "healthy",
      "critical": true
    }
  ],
  "circuit_breakers": [
    {
      "state": "closed",
      "failure_count": 0
    }
  ],
  "metrics": {
    "total_requests": 1000,
    "failure_rate": 0.005
  }
}
```

## Handling Failures

### Degraded Service (Non-Critical Failure)
1. System returns `degraded` status
2. API requests continue with reduced functionality
3. User receives warning about limited results
4. Automatic recovery attempts in background

### Critical Failure
1. System returns `unhealthy` status
2. API requests rejected with 503 Service Unavailable
3. Manual intervention required
4. Check `/health/detailed` for root cause

### Rate Limiting
```
HTTP 429 Too Many Requests
{
  "detail": "Rate limit exceeded"
}
```

## State Persistence

**State File**: `./indexing_state.json`

**Stored Information**:
- Completed indexing jobs
- Document hashes (change detection)
- Last successful index time
- Indexed document list

**Recovery**: Persisted state survives application restarts

## Best Practices

### For API Users
1. **Provide User ID**: Include `X-User-Id` header for better rate limiting
2. **Handle Rate Limits**: Implement exponential backoff on 429 responses
3. **Monitor Health**: Check `/health` before critical operations
4. **Use Specific Endpoints**: Choose appropriate response format (json/html/markdown)

### For Operators
1. **Monitor Health**: Set up alerts on `/health/detailed` unhealthy status
2. **Track Indexing**: Monitor `/indexing/stats` for document freshness
3. **Review Metrics**: Check failure rates and token usage
4. **Plan Scaling**: Monitor concurrent request limits

### For Maintenance
1. **Document Updates**: Use `/indexing/start?force=true` for re-indexing
2. **Circuit Breaking**: Monitor circuit breaker states for patterns
3. **Logs**: Check application logs for errors and warnings
4. **Resource Cleanup**: Background jobs auto-cleanup old job records

## Example Usage

### Complete Query with Safety
```python
import requests

headers = {
    "X-User-Id": "attorney_123"
}

# Check health first
health = requests.get("http://api/health").json()
if health['status'] != 'healthy':
    print("System degraded, may have limited results")

# Submit query
response = requests.post(
    "http://api/query",
    json={
        "query": "What is Michigan's rule on criminal procedure?",
        "include_sources": True,
        "include_citations": True
    },
    headers=headers
)

if response.status_code == 429:
    print("Rate limited, retry after delay")
elif response.status_code == 503:
    print("System unavailable")
else:
    result = response.json()
    print(result['answer'])
```

### Managing Background Indexing
```python
# Trigger full re-indexing
response = requests.post("http://api/indexing/start?force=true")
job_ids = response.json()['job_ids']

# Monitor specific job
job_status = requests.get(f"http://api/indexing/status/{job_ids[0]}").json()

# Get comprehensive stats
stats = requests.get("http://api/indexing/stats").json()
print(f"Indexed documents: {stats['stats']['indexed_documents']}")
print(f"Completed jobs: {stats['stats']['completed_jobs']}")
print(f"Failed jobs: {stats['stats']['failed_jobs']}")
```

## Performance Tuning

### Optimize Vector Search
```python
TOP_K_RETRIEVAL = 4  # Number of chunks to retrieve
# Lower = faster but less context
# Higher = slower but more comprehensive
```

### Optimize Token Usage
```python
CHUNK_SIZE = 800  # Tokens per chunk
MAX_CONTEXT_TOKENS = 12000  # Total context limit
# Smaller chunks = more precise but more overhead
# Larger chunks = less precise but faster
```

### Batch Processing
```python
# Vector store uses batch_size=50 for updates
# Larger batches = faster but more memory
# Smaller batches = slower but less memory
```

## Troubleshooting

### Circuit Breaker Open
**Cause**: Too many failures to external service
**Solution**: Check service health, review logs, wait for recovery timeout

### High Failure Rate
**Cause**: Invalid inputs, service issues, or resource exhaustion
**Solution**: Review error logs, check rate limits, increase resources

### Slow Responses
**Cause**: Large context, slow vector search, or API latency
**Solution**: Reduce chunk overlap, limit context size, optimize retrieval

### Memory Issues
**Cause**: Large batches or unreleased documents
**Solution**: Reduce batch size, use disk-based storage (already configured)

## Future Enhancements

1. **Distributed Indexing**: Multi-worker processing with work distribution
2. **Caching Layer**: Response caching for common queries
3. **Load Balancing**: Multiple API instances with shared state
4. **Metrics Export**: Prometheus/Grafana integration
5. **Advanced Monitoring**: Distributed tracing and profiling
6. **Auto-scaling**: Dynamic resource allocation based on load
