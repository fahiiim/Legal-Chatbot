# Production Configuration Guide

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration (REQUIRED)
OPENAI_API_KEY=sk-...

# Optional: Set OpenAI model
OPENAI_MODEL=gpt-4o

# Optional: Set embedding model
EMBEDDING_MODEL=text-embedding-3-small

# Vector Store Configuration
VECTOR_STORE_PERSIST_DIR=./chroma_db
COLLECTION_NAME=legal_documents

# PDF Knowledge Base
PDF_DIRECTORY=knowledge_base

# Optional: Application Settings
LOG_LEVEL=INFO
MAX_WORKERS=1
```

## Application Configuration

Edit `config.py` to adjust production parameters:

### Rate Limiting

```python
# In request_limiter.py (when initializing)

rate_limiter = RateLimiter(
    global_rate=100,        # Max requests per minute (global)
    window_seconds=60,      # Time window
    per_user_rate=20        # Max per user per window
)
```

### Request Limits

```python
# Concurrent and daily request limits

request_limiter = RequestLimiter(
    max_concurrent=10,      # Max simultaneous requests per user
    max_daily=1000          # Max daily requests per user
)
```

### Circuit Breaker Configuration

```python
# In app.py lifespan

safety_manager.register_circuit_breaker(
    "openai_api",
    failure_threshold=3,    # Failures before opening
    recovery_timeout=30     # Seconds before retry
)
```

### Health Check Configuration

```python
# Register custom health checks

def check_database():
    """Check if database is accessible."""
    return os.path.exists(VECTOR_STORE_PERSIST_DIR)

safety_manager.register_health_check(
    "database",
    check_database,
    timeout=5,
    critical=True  # Failure fails entire system
)
```

### Background Indexing

```python
# Adjust indexing behavior

indexer = BackgroundIndexer(
    vector_store_dir=VECTOR_STORE_PERSIST_DIR,
    pdf_directory=PDF_DIRECTORY,
    max_workers=1,          # Number of worker threads
    state_file="./indexing_state.json"
)
```

## Production Deployment

### Docker Configuration

Example `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Ensure knowledge base exists
RUN mkdir -p knowledge_base chroma_db

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

Example `docker-compose.yml`:

```yaml
version: '3.8'

services:
  legal-rag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./knowledge_base:/app/knowledge_base
      - ./chroma_db:/app/chroma_db
      - ./indexing_state.json:/app/indexing_state.json
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
```

### Kubernetes Deployment

Example `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legal-rag-chatbot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: legal-rag
  template:
    metadata:
      labels:
        app: legal-rag
    spec:
      containers:
      - name: legal-rag
        image: legal-rag-chatbot:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: legal-rag-secrets
              key: openai-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5
        volumeMounts:
        - name: vector-store
          mountPath: /app/chroma_db
        - name: knowledge-base
          mountPath: /app/knowledge_base
      volumes:
      - name: vector-store
        persistentVolumeClaim:
          claimName: vector-store-pvc
      - name: knowledge-base
        persistentVolumeClaim:
          claimName: knowledge-base-pvc
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: legal-rag-service
spec:
  selector:
    app: legal-rag
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring Setup

### Prometheus Metrics

Add to `app.py`:

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
query_counter = Counter(
    'legal_rag_queries_total',
    'Total number of queries',
    ['status']
)

query_duration = Histogram(
    'legal_rag_query_duration_seconds',
    'Query duration in seconds'
)

active_requests = Gauge(
    'legal_rag_active_requests',
    'Number of active requests'
)

# Use in endpoints
@app.post("/query")
async def query_legal_question(request: QueryRequest):
    active_requests.inc()
    start = time.time()
    try:
        # ... query logic
        query_counter.labels(status='success').inc()
    except Exception:
        query_counter.labels(status='error').inc()
        raise
    finally:
        query_duration.observe(time.time() - start)
        active_requests.dec()
```

### Alert Rules

Example Prometheus alerts in `alerts.yml`:

```yaml
groups:
- name: legal_rag_alerts
  rules:
  - alert: HighFailureRate
    expr: rate(legal_rag_queries_total{status="error"}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High query failure rate"
      description: "Error rate above 5%"
  
  - alert: SystemUnhealthy
    expr: legal_rag_system_health == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "System health check failed"
  
  - alert: RateLimitExceeded
    expr: increase(legal_rag_rate_limit_exceeded[5m]) > 10
    labels:
      severity: warning
    annotations:
      summary: "Rate limits being exceeded"
```

### Logging Configuration

```python
# In app.py or separate logging config

import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {name} {message}',
            'style': '{'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(timestamp)s %(level)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/legal_rag.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## Performance Tuning

### Batch Size Optimization

```python
# In vector_store.py add_documents method

def add_documents(self, documents, batch_size=50):
    """
    Add documents with optimal batch size.
    
    Performance guidelines:
    - batch_size=10: Lower memory, slower
    - batch_size=50: Balanced (default)
    - batch_size=100: Higher throughput, more memory
    """
    # ... implementation
```

### Chunk Size Optimization

```python
# In config.py

# For faster retrieval (less accurate):
CHUNK_SIZE = 400
TOP_K_RETRIEVAL = 2

# For better quality (slower):
CHUNK_SIZE = 1200
TOP_K_RETRIEVAL = 6

# Balanced (recommended):
CHUNK_SIZE = 800
TOP_K_RETRIEVAL = 4
```

### Context Limit Optimization

```python
# Adjust maximum context sent to LLM

MAX_CONTEXT_TOKENS = 8000   # Stricter limit (faster)
MAX_CONTEXT_TOKENS = 12000  # Balanced (default)
MAX_CONTEXT_TOKENS = 16000  # More context (slower)
```

## Security Configuration

### CORS Configuration

```python
# In app.py

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    # Production: specify exact origins
    allow_origins=[
        "https://your-domain.com",
        "https://app.your-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-User-Id"]
)
```

### HTTPS Configuration

```python
# Use with reverse proxy (nginx, traefik)

# Or enable SSL directly in uvicorn:
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=8000,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem",
    ssl_version=2  # TLS 1.2
)
```

### Rate Limiting Configuration

```python
# Adjust for your use case

# Strict (low usage)
global_rate=50
per_user_rate=10
max_daily=500

# Moderate (medium usage) - default
global_rate=100
per_user_rate=20
max_daily=1000

# Permissive (high-volume)
global_rate=500
per_user_rate=100
max_daily=5000
```

## Backup and Recovery

### State File Backup

```bash
# Backup indexing state
cp indexing_state.json indexing_state.json.backup

# Backup vector store
tar -czf chroma_db_backup.tar.gz chroma_db/

# Restore
tar -xzf chroma_db_backup.tar.gz
```

### Document Recovery

If documents are lost:

```python
# Force re-indexing
curl -X POST "http://localhost:8000/indexing/start?force=true"

# Or programmatically
import requests
response = requests.post(
    "http://localhost:8000/indexing/start",
    params={"force": True}
)
```

## Troubleshooting

### Check Application Health

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed health report
curl http://localhost:8000/health/detailed

# Indexing statistics
curl http://localhost:8000/indexing/stats
```

### View Logs

```bash
# Docker
docker logs legal-rag-service

# Kubernetes
kubectl logs deployment/legal-rag-chatbot

# Local file
tail -f logs/legal_rag.log
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
uvicorn app:app --log-level debug
```
