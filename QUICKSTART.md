# Quick Start: Production-Safe RAG Pipeline

## 5-Minute Setup

### 1. Prerequisites
```bash
# Ensure Python 3.10+ is installed
python --version

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Create .env file with required settings
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 4. Run Production Validation
```bash
python production_init.py
```

Expected output:
```
✓ Environment Variables
✓ Directory Structure
✓ Python Dependencies
✓ Configuration
✓ State Initialization
✓ PDF Files
✓ Vector Store
✓ OpenAI API

Total: 8/8 checks passed!
```

### 5. Start the Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Using the Production-Safe Features

### Health Monitoring
```bash
# Quick health check
curl http://localhost:8000/health

# Detailed health report
curl http://localhost:8000/health/detailed
```

### Background Indexing
```bash
# Start background indexing of all documents
curl -X POST http://localhost:8000/indexing/start

# Check indexing status
curl http://localhost:8000/indexing/stats

# Index specific document
curl -X POST "http://localhost:8000/indexing/document/federal-rules-of-evidence-dec-1-2024_0.pdf"
```

### Query with Safety Features
```bash
# Include user ID for rate limiting
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-User-Id: attorney_123" \
  -d '{
    "query": "What are Michigans rules on criminal procedure?",
    "include_sources": true,
    "include_citations": true
  }'
```

## Key Features Explained

### 1. Background Indexing
- Automatically indexes documents in the background
- Non-blocking: Doesn't interfere with query requests
- Change detection: Only re-indexes modified documents
- Recoverable: Persists state to disk

**Why it matters**: Large document sets can take time to index. Background indexing ensures the API remains responsive while documents are being processed.

### 2. Health Checks
- Monitors system health in real-time
- Identifies component failures
- Supports graceful degradation

**Response statuses**:
- `healthy`: All systems operational
- `degraded`: Non-critical components failing
- `unhealthy`: Critical failures

### 3. Circuit Breakers
- Prevents cascading failures
- Auto-recovers when services stabilize
- Fails fast on known failures

**Example**: If OpenAI API fails 3 times, circuit opens and rejects further requests to avoid timeout delays.

### 4. Rate Limiting
- Prevents API abuse
- Per-user and global limits
- Token bucket algorithm (smooth distribution)

**Defaults**:
- Global: 100 requests/minute
- Per-user: 20 requests/minute
- Daily: 1000 requests/user

### 5. Input Validation
- Prevents injection attacks
- Validates query length and format
- Detects malicious patterns

## Production Checklist

- [ ] `.env` file configured with `OPENAI_API_KEY`
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Validation script passes: `python production_init.py`
- [ ] PDFs added to `knowledge_base/` directory
- [ ] Application starts: `uvicorn app:app --host 0.0.0.0 --port 8000`
- [ ] Health check responds: `curl http://localhost:8000/health`
- [ ] Indexing completes: Check `/indexing/stats`

## Monitoring in Production

### Health Dashboard
Monitor these endpoints every 30 seconds:

```bash
#!/bin/bash
while true; do
  echo "System Health:"
  curl -s http://localhost:8000/health | jq .
  
  echo "Indexing Stats:"
  curl -s http://localhost:8000/indexing/stats | jq .
  
  sleep 30
done
```

### Key Metrics to Watch
1. **Health Status**: Should remain `healthy` or `degraded`
2. **Query Success Rate**: Monitor failure_rate in health report
3. **Indexing Progress**: Check completed_jobs vs total_jobs
4. **Rate Limit Hits**: Track 429 responses
5. **Token Usage**: Monitor total_tokens_used

## Common Issues

### Issue: "OPENAI_API_KEY not set"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY=sk-your-key-here
# Or create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Issue: "RAG engine not initialized"
```bash
# Solution: Wait for startup or check health
curl http://localhost:8000/health
# Should show engine_initialized: true
```

### Issue: "Rate limit exceeded" (HTTP 429)
```bash
# Solution: Provide user ID header to distribute limit
curl -H "X-User-Id: your-unique-id" ...
# Wait 60 seconds for rate limit window to reset
```

### Issue: "System in critical state"
```bash
# Solution: Check detailed health
curl http://localhost:8000/health/detailed | jq .

# Restart indexing
curl -X POST "http://localhost:8000/indexing/start?force=true"
```

## Performance Tuning

### For High Throughput
```python
# In config.py
TOP_K_RETRIEVAL = 2      # Fewer documents
CHUNK_SIZE = 400         # Smaller chunks
MAX_CONTEXT_TOKENS = 8000  # Less context
```

### For Better Quality
```python
# In config.py
TOP_K_RETRIEVAL = 6      # More documents
CHUNK_SIZE = 1200        # Larger chunks
MAX_CONTEXT_TOKENS = 16000  # More context
```

### For Balanced Performance (default)
```python
# In config.py (current values)
TOP_K_RETRIEVAL = 4
CHUNK_SIZE = 800
MAX_CONTEXT_TOKENS = 12000
```

## Docker Deployment

### Quick Start with Docker
```bash
# Build image
docker build -t legal-rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  -v $(pwd)/knowledge_base:/app/knowledge_base \
  legal-rag-chatbot
```

### Using Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Scaling Considerations

### Single Instance (Current)
- Suitable for: < 100 requests/day
- Max concurrency: 10 users
- Memory: 2-4 GB
- CPU: 1-2 cores

### Multiple Instances (Future)
- Would require: Load balancer, shared vector store
- Need: Redis for distributed state
- Benefits: High availability, load distribution

## Support & Documentation

- **Detailed Documentation**: See [PRODUCTION_SAFETY.md](PRODUCTION_SAFETY.md)
- **Configuration Guide**: See [PRODUCTION_CONFIG.md](PRODUCTION_CONFIG.md)
- **API Documentation**: Visit `http://localhost:8000/docs`
- **Alternative Docs**: Visit `http://localhost:8000/redoc`

## Next Steps

1. **Test the API**: Try the interactive docs at `/docs`
2. **Monitor Health**: Set up continuous health checks
3. **Optimize Performance**: Adjust chunk sizes based on your queries
4. **Configure Monitoring**: Set up Prometheus metrics (optional)
5. **Plan Scaling**: Consider load balancing if needed

---

**Version**: 2.1.0
**Last Updated**: January 2026
**Status**: Production-Ready
