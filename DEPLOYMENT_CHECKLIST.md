# Deployment Checklist: Production-Safe Legal RAG Pipeline

## Pre-Deployment (Development Environment)

### Code Quality
- [ ] All tests pass: `pytest test_production_safety.py -v`
- [ ] No import errors: `python -c "from app import app"`
- [ ] Type hints verified (optional): `mypy *.py`
- [ ] Code style checked: `flake8 *.py`

### Functionality Testing
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] Query endpoint works: Test `/query` with sample request
- [ ] Indexing endpoints work: Test `/indexing/start`
- [ ] Rate limiting works: Exceed limits and verify 429 response
- [ ] Input validation works: Try malicious inputs
- [ ] Circuit breaker works: Trigger failures and observe behavior

### Configuration
- [ ] `.env` file created with `OPENAI_API_KEY`
- [ ] All required environment variables set
- [ ] `config.py` reviewed for your use case
- [ ] Rate limits adjusted if needed
- [ ] Chunk sizes optimized for query types

### Documentation
- [ ] QUICKSTART.md reviewed
- [ ] PRODUCTION_SAFETY.md reviewed
- [ ] PRODUCTION_CONFIG.md reviewed
- [ ] Team familiar with monitoring endpoints

## Environment Setup (Staging/Production)

### System Requirements
- [ ] Python 3.10+ installed
- [ ] Sufficient disk space for vector store (min 2GB)
- [ ] Sufficient RAM (min 2GB, recommended 4GB+)
- [ ] Network access to OpenAI API
- [ ] Outbound HTTPS (port 443) available

### Dependencies
- [ ] All packages installed: `pip install -r requirements.txt`
- [ ] OpenAI client verified: `python -c "from openai import OpenAI"`
- [ ] LangChain verified: `python -c "from langchain import *"`
- [ ] ChromaDB verified: `python -c "from chromadb import *"`

### Database Setup
- [ ] Vector store directory exists and writable
- [ ] `chroma_db/` directory created
- [ ] Permissions correct (read/write for application user)
- [ ] Disk space adequate for full vector store

### API Credentials
- [ ] OpenAI API key obtained
- [ ] API key added to `.env` or environment
- [ ] API key validated: `python production_init.py`
- [ ] API billing quota checked
- [ ] Rate limits understood (default: 100 req/min global)

## Initial Deployment

### Pre-Flight Checks
```bash
# Run comprehensive validation
python production_init.py
```
Expected output: All 8 checks passed ✓

### Application Start
```bash
# Start the application
uvicorn app:app --host 0.0.0.0 --port 8000
```
- [ ] Application starts without errors
- [ ] Lifespan events complete
- [ ] Health check responsive

### Verify Core Functionality
- [ ] Health endpoint returns `healthy`: `curl http://localhost:8000/health`
- [ ] API documentation loads: Visit `http://localhost:8000/docs`
- [ ] Detailed health report loads: `curl http://localhost:8000/health/detailed`

### Initial Indexing
```bash
# Trigger background indexing
curl -X POST http://localhost:8000/indexing/start
```
- [ ] Indexing job queued successfully
- [ ] Job ID returned
- [ ] Status tracking works: `GET /indexing/status/{job_id}`
- [ ] Indexing completes without errors

### Sample Queries
- [ ] Submit test query: `POST /query`
- [ ] Receive response with answer
- [ ] Check tier classification works
- [ ] Verify citations extracted
- [ ] Confirm sources returned

## Monitoring Setup

### Health Monitoring
- [ ] Set up alerts for unhealthy status
- [ ] Monitor `/health` endpoint every 30 seconds
- [ ] Alert on status change from healthy → degraded/unhealthy
- [ ] Alert on multiple circuit breaker opens

### Metrics Monitoring
- [ ] Monitor total request count
- [ ] Monitor failure rate (alert if > 5%)
- [ ] Monitor token usage (track costs)
- [ ] Monitor average response time
- [ ] Monitor queue size (should be near 0)

### Log Monitoring
- [ ] Application logs directed to file or service
- [ ] Error logs monitored for exceptions
- [ ] Warning logs monitored for issues
- [ ] Log rotation configured (if file-based)
- [ ] Log analysis tool connected (optional)

### Circuit Breaker Monitoring
- [ ] Monitor circuit breaker states
- [ ] Alert on breaker open events
- [ ] Track recovery attempts
- [ ] Monitor failure counts

## Load Testing

### Baseline Performance
- [ ] Single query response time measured
- [ ] Memory usage at idle measured
- [ ] CPU usage at idle measured
- [ ] Indexing time per document measured

### Stress Testing
- [ ] 10 concurrent requests: No failures
- [ ] 20 concurrent requests: Observe behavior
- [ ] Rate limit enforcement verified
- [ ] System recovers cleanly after stress

### Capacity Planning
- [ ] Daily request estimate: _____ requests
- [ ] Peak concurrent users: _____ users
- [ ] Estimated monthly tokens: _____ tokens
- [ ] Estimated monthly costs: _____ USD

## Security

### Access Control
- [ ] API behind authentication (if applicable)
- [ ] Rate limiting enforced
- [ ] Input validation working
- [ ] CORS configured correctly
- [ ] HTTPS enabled (production)

### Secret Management
- [ ] API keys not in version control
- [ ] `.env` file in `.gitignore`
- [ ] Environment variables used for secrets
- [ ] Secrets rotated according to policy

### Input Security
- [ ] SQL injection attempts blocked
- [ ] XSS attempts blocked
- [ ] Query length limits enforced
- [ ] Special character filtering active

## Backup & Recovery

### State Backup
- [ ] Backup strategy defined
- [ ] `indexing_state.json` backed up
- [ ] Backup frequency scheduled
- [ ] Restore procedure tested

### Vector Store Backup
- [ ] `chroma_db/` included in backups
- [ ] Backup size estimated
- [ ] Backup retention policy defined
- [ ] Restore procedure documented

### Disaster Recovery
- [ ] Recovery time objective (RTO): _____ minutes
- [ ] Recovery point objective (RPO): _____ hours
- [ ] Full restore procedure tested
- [ ] Fallback plan in place

## Documentation

### Runbooks
- [ ] How to restart application documented
- [ ] How to check health documented
- [ ] How to trigger re-indexing documented
- [ ] Troubleshooting guide accessible

### Operational Handoff
- [ ] Operations team trained on system
- [ ] Monitoring alerts configured
- [ ] On-call rotation established
- [ ] Escalation procedures defined

### Change Log
- [ ] Deployment version documented: _____.
- [ ] Deployment date documented: _____
- [ ] Changes from previous version documented
- [ ] Configuration changes documented

## Post-Deployment (First 24 Hours)

### Monitoring Verification
- [ ] Health checks running continuously
- [ ] No alerts triggered for non-issues
- [ ] Metrics being collected properly
- [ ] Logs being written correctly

### Performance Verification
- [ ] Response times meet expectations
- [ ] Resource usage within limits
- [ ] Error rate < 1%
- [ ] No memory leaks observed

### User Validation
- [ ] Sample queries produce correct answers
- [ ] Citations properly extracted
- [ ] Rate limiting works as expected
- [ ] User feedback positive

### System Stability
- [ ] No unexpected restarts
- [ ] No error spikes
- [ ] No resource exhaustion
- [ ] Circuit breakers not triggering frequently

## Ongoing Operations

### Daily
- [ ] [ ] Check health endpoint
- [ ] [ ] Review error logs
- [ ] [ ] Monitor failure rate
- [ ] [ ] Verify indexing progress (if ongoing)

### Weekly
- [ ] [ ] Review metrics summary
- [ ] [ ] Check token usage trend
- [ ] [ ] Test backup/restore procedure
- [ ] [ ] Review performance alerts

### Monthly
- [ ] [ ] Plan for updates
- [ ] [ ] Review capacity vs. actual usage
- [ ] [ ] Analyze user patterns
- [ ] [ ] Plan optimizations

### Quarterly
- [ ] [ ] Security audit
- [ ] [ ] Performance optimization
- [ ] [ ] Capacity planning update
- [ ] [ ] Documentation review

## Rollback Plan

If critical issues occur:

1. **Stop Application**
   ```bash
   # Kill the running process
   pkill -f "uvicorn app:app"
   ```

2. **Revert Code** (if necessary)
   ```bash
   # Restore previous version from version control
   git checkout <previous-version>
   ```

3. **Restore State**
   ```bash
   # If state corrupted, restore from backup
   cp indexing_state.json.backup indexing_state.json
   ```

4. **Restart Application**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Verify Functionality**
   ```bash
   # Check health
   curl http://localhost:8000/health
   ```

6. **Notify Stakeholders**
   - Document what went wrong
   - Explain remediation
   - Plan prevention

## Sign-Off

### Development Team
- Name: ________________________
- Date: _______________________
- Signature: ___________________

### QA/Testing
- Name: ________________________
- Date: _______________________
- Signature: ___________________

### Operations/DevOps
- Name: ________________________
- Date: _______________________
- Signature: ___________________

### Project Manager
- Name: ________________________
- Date: _______________________
- Signature: ___________________

---

## Quick Reference

### Essential Endpoints
```
GET  /health           - Quick health check
GET  /health/detailed  - Full health report
POST /query            - Submit legal query
GET  /indexing/stats   - Indexing statistics
GET  /docs             - API documentation
```

### Troubleshooting Hotline
1. Check: `GET /health/detailed`
2. Review: Application logs
3. Validate: `python production_init.py`
4. Restart: Stop and start application

### Emergency Contacts
- On-Call Engineer: _________________ Phone: _________________
- Platform Team: ____________________ Phone: _________________
- Vendor Support: ___________________ Phone: _________________

### Important Thresholds
- Health Status Alert: unhealthy
- Failure Rate Alert: > 5%
- Response Time Alert: > 10 seconds
- Circuit Breaker Alert: Open state

---

**Deployment Version**: 2.1.0
**Last Updated**: January 28, 2026
**Ready for Production**: YES ✓
