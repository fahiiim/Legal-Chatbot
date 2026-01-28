"""
FastAPI Application for Michigan Legal RAG Chatbot
Provides REST API endpoints for legal question answering.

Production-Safe Features:
- Background document indexing
- Health monitoring and circuit breakers
- Rate limiting and request validation
- Error recovery and graceful degradation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import uvicorn
import re
import logging
from datetime import datetime

from rag_engine import get_rag_engine
from tier_router import TierRouter
from report_generator import LegalReportGenerator
from config import *

# Import production safety features
from background_indexer import get_background_indexer
from production_safety import get_safety_manager, HealthStatus
from request_limiter import (
    get_rate_limiter, 
    get_request_limiter,
    InputValidator
)

logger = logging.getLogger(__name__)


def format_answer_for_display(answer: str) -> str:
    """
    Format the answer to ensure proper line breaks and clean display.
    Converts markdown-style formatting to clean readable text.
    """
    if not answer:
        return answer
    
    # Normalize line endings
    formatted = answer.replace('\r\n', '\n').replace('\r', '\n')
    
    # Clean up excessive newlines (more than 2 consecutive)
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    # Ensure proper spacing after headers (## or ###)
    formatted = re.sub(r'(#{1,3}\s+[^\n]+)\n(?!\n)', r'\1\n\n', formatted)
    
    # Ensure proper spacing before bullet points
    formatted = re.sub(r'([^\n])\n([-•*]\s)', r'\1\n\n\2', formatted)
    
    return formatted.strip()


def convert_markdown_to_html(text: str) -> str:
    """
    Convert markdown text to HTML with proper formatting.
    Handles headers, bold, italic, lists, and line breaks.
    """
    if not text:
        return text
    
    import html as html_lib
    
    # Escape HTML special characters first
    text = html_lib.escape(text)
    
    # Convert headers (### -> h3, ## -> h2, # -> h1)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Convert bold (**text** or __text__)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    
    # Convert italic (*text* or _text_)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', text)
    
    # Convert bullet points (-, *, •)
    text = re.sub(r'^[-•*]\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive <li> items in <ul>
    text = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', text)
    
    # Convert numbered lists
    text = re.sub(r'^(\d+)\.\s+(.+)$', r'<li>\2</li>', text, flags=re.MULTILINE)
    
    # Convert line breaks (double newline -> paragraph, single -> <br>)
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if p:
            # Don't wrap if already wrapped in HTML tags
            if not (p.startswith('<h') or p.startswith('<ul') or p.startswith('<ol')):
                p = p.replace('\n', '<br>\n')
                if not p.startswith('<'):
                    p = f'<p>{p}</p>'
            formatted_paragraphs.append(p)
    
    return '\n\n'.join(formatted_paragraphs)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for legal queries."""
    query: str = Field(..., description="Legal question to ask", min_length=1)
    include_sources: bool = Field(default=True, description="Include source documents in response")
    include_citations: bool = Field(default=True, description="Include legal citations in response")
    doc_type_filter: Optional[str] = Field(default=None, description="Filter by document type")


class QueryResponse(BaseModel):
    """Response model for legal queries."""
    query: str
    answer: str
    tier: int
    tier_description: str
    tier_reasoning: str
    tier_recommendation: str
    citations: List[Dict] = []
    sources: List[Dict] = []
    num_sources: int = 0
    is_legal: bool = True
    usage: Optional[Dict] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine_initialized: bool
    timestamp: str


class StatsResponse(BaseModel):
    """Statistics response."""
    stats: Dict
    supported_documents: List[str]

# Global RAG engine instance
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global rag_engine
    
    # Startup
    print("Initializing Michigan Legal RAG Engine...")
    rag_engine = get_rag_engine(force_reload=False)
    print("RAG Engine ready!")
    
    # Initialize safety manager
    print("Initializing production safety manager...")
    safety_manager = get_safety_manager()
    
    # Register health checks
    def check_rag_engine():
        return rag_engine and rag_engine.is_initialized
    
    def check_vector_store():
        return rag_engine and hasattr(rag_engine, 'vectorstore') and rag_engine.vectorstore.vectorstore is not None
    
    safety_manager.register_health_check(
        "rag_engine",
        check_rag_engine,
        critical=True
    )
    safety_manager.register_health_check(
        "vector_store",
        check_vector_store,
        critical=True
    )
    
    # Register circuit breakers
    safety_manager.register_circuit_breaker("openai_api", failure_threshold=3, recovery_timeout=30)
    safety_manager.register_circuit_breaker("vector_search", failure_threshold=5, recovery_timeout=60)
    
    # Start background indexer
    print("Starting background indexer...")
    indexer = get_background_indexer()
    indexer.start()
    
    # Initial indexing if needed
    indexed = indexer.state.get_indexed_documents()
    if not indexed:
        print("Performing initial document indexing...")
        indexer.index_all_documents(force=True)
    
    print("Application startup complete!")
    
    yield
    
    # Shutdown
    print("Shutting down background indexer...")
    indexer.stop(timeout=10)
    print("Application shutdown complete!")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Michigan Legal RAG Chatbot",
    description="AI-powered legal document search and question answering",
    version="2.1.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Michigan Legal RAG Chatbot API",
        "version": "2.0.0",
        "endpoints": {
            "POST /query": "Submit a legal query (JSON response)",
            "POST /query/formatted": "Submit a legal query (HTML formatted response)",
            "POST /query/markdown": "Submit a legal query (Markdown response)",
            "POST /query/report": "Submit a legal query (full formatted report)",
            "POST /query/summary-report": "Submit a legal query (summary report only)",
            "GET /health": "Health check",
            "GET /stats": "Get system statistics",
            "GET /documents": "List supported documents",
            "POST /search": "Search documents",
            "POST /reload": "Reload documents (admin)"
        },
        "supported_documents": SUPPORTED_PDFS,
        "report_endpoints": {
            "description": "For attorney review and case documentation",
            "/query/report": "Complete professional legal report with all citations and sources",
            "/query/summary-report": "Quick summary for initial review",
            "/query/formatted": "HTML formatted response with proper line breaks",
            "/query/markdown": "Clean markdown response"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with comprehensive status.
    Includes engine status, safety checks, and circuit breaker state.
    """
    safety_manager = get_safety_manager()
    overall_status = safety_manager.perform_health_checks()
    
    return HealthResponse(
        status=overall_status.value,
        engine_initialized=rag_engine.is_initialized if rag_engine else False,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health/detailed")
async def health_check_detailed():
    """
    Detailed health report for monitoring and debugging.
    Includes all component statuses and metrics.
    """
    safety_manager = get_safety_manager()
    health_report = safety_manager.get_health_report()
    
    indexer = get_background_indexer()
    indexing_stats = indexer.get_indexing_stats()
    
    return {
        'health': health_report,
        'indexing': indexing_stats,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    stats = rag_engine.get_stats()
    return StatsResponse(
        stats=stats,
        supported_documents=SUPPORTED_PDFS
    )


@app.post("/query", response_model=QueryResponse)
async def query_legal_question(request: QueryRequest, x_user_id: Optional[str] = Header(None)):
    """
    Process a legal query and return an answer with citations (JSON body).
    
    Production-safe with:
    - Input validation
    - Rate limiting
    - Request limiting
    - Health checks
    - Circuit breaker protection
    
    Args:
        request: QueryRequest with the legal question
        x_user_id: Optional user ID header for rate limiting
    
    Returns:
        QueryResponse with answer, tier classification, citations, and sources
    """
    user_id = x_user_id or "anonymous"
    
    # Initialize safety and limiting systems
    safety_manager = get_safety_manager()
    rate_limiter = get_rate_limiter()
    request_limiter = get_request_limiter()
    
    # Check health status
    if safety_manager.is_critical():
        raise HTTPException(
            status_code=503,
            detail="System in critical state, please retry later"
        )
    
    # Validate input
    valid, error = InputValidator.validate_query(request.query)
    if not valid:
        safety_manager.record_request(success=False)
        raise HTTPException(status_code=400, detail=error)
    
    # Check rate limits
    rate_allowed, rate_info = rate_limiter.is_allowed(user_id)
    if not rate_allowed:
        logger.warning(f"Rate limit exceeded for user {user_id}: {rate_info}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Check concurrent request limits
    if not request_limiter.increment_concurrent(user_id):
        logger.warning(f"Concurrent limit exceeded for user {user_id}")
        raise HTTPException(status_code=429, detail="Too many concurrent requests")
    
    # Check daily limits
    if not request_limiter.increment_daily(user_id):
        request_limiter.decrement_concurrent(user_id)
        logger.warning(f"Daily limit exceeded for user {user_id}")
        raise HTTPException(status_code=429, detail="Daily request limit exceeded")
    
    try:
        if not rag_engine or not rag_engine.is_initialized:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Sanitize query
        sanitized_query = InputValidator.sanitize_query(request.query)
        
        # Query the RAG engine with circuit breaker protection
        def execute_query():
            if request.doc_type_filter:
                return rag_engine.query_with_filter(
                    question=sanitized_query,
                    doc_type=request.doc_type_filter
                )
            else:
                return rag_engine.query(
                    question=sanitized_query,
                    return_sources=request.include_sources,
                    return_citations=request.include_citations
                )
        
        try:
            result = safety_manager.call_with_circuit_breaker(
                "openai_api",
                execute_query
            )
        except Exception as e:
            logger.error(f"Circuit breaker triggered: {e}")
            safety_manager.record_request(success=False)
            raise HTTPException(status_code=503, detail="Temporary service unavailable")
        
        # Handle errors
        if "error" in result:
            safety_manager.record_request(success=False)
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Only classify tier for legitimate legal queries
        is_legal = result.get("is_legal", True)
        is_harmful = result.get("is_harmful", False)
        
        if is_legal and not is_harmful:
            tier, tier_desc, tier_reasoning = TierRouter.classify_tier(sanitized_query)
            tier_recommendation = TierRouter.get_tier_recommendation(tier)
        else:
            # Don't provide tier info for non-legal or harmful queries
            tier = 0
            tier_desc = "Not applicable - query not related to legal matters" if not is_harmful else "Not applicable - inappropriate query"
            tier_reasoning = "Query was not processed as a legal question"
            tier_recommendation = "Please submit a legitimate legal question about Michigan or federal law."
        
        # Build response - format the answer for proper display
        formatted_answer = format_answer_for_display(result.get("answer", ""))
        
        # Record successful request
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        safety_manager.record_request(success=True, tokens=tokens_used)
        
        response = QueryResponse(
            query=request.query,
            answer=formatted_answer,
            tier=tier,
            tier_description=tier_desc,
            tier_reasoning=tier_reasoning,
            tier_recommendation=tier_recommendation,
            citations=result.get("citations", []),
            sources=result.get("sources", []),
            num_sources=result.get("num_sources", 0),
            is_legal=is_legal,
            usage=result.get("usage"),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    finally:
        # Always decrement concurrent counter
        request_limiter.decrement_concurrent(user_id)


@app.post("/query/formatted", response_class=HTMLResponse)
async def query_legal_question_formatted(request: QueryRequest):
    """
    Process a legal query and return an HTML formatted response with proper line breaks.
    This endpoint renders the answer with proper formatting for display.
    
    Args:
        request: QueryRequest with the legal question
    
    Returns:
        HTML formatted response with proper line breaks and styling
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Query the RAG engine
    if request.doc_type_filter:
        result = rag_engine.query_with_filter(
            question=request.query,
            doc_type=request.doc_type_filter
        )
    else:
        result = rag_engine.query(
            question=request.query,
            return_sources=request.include_sources,
            return_citations=request.include_citations
        )
    
    # Handle errors
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Get and format the answer
    answer = format_answer_for_display(result.get("answer", ""))
    
    # Convert markdown to HTML
    html_answer = convert_markdown_to_html(answer)
    
    # Classify tier
    is_legal = result.get("is_legal", True)
    is_harmful = result.get("is_harmful", False)
    
    if is_legal and not is_harmful:
        tier, tier_desc, tier_reasoning = TierRouter.classify_tier(request.query)
        tier_recommendation = TierRouter.get_tier_recommendation(tier)
    else:
        tier = 0
        tier_desc = "Not applicable"
        tier_reasoning = "Query was not processed as a legal question"
        tier_recommendation = "Please submit a legitimate legal question."
    
    # Build HTML response
    html_response = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Legal Query Response</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #2c5282; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 25px; }}
        h3 {{ color: #4a5568; }}
        .query {{ background: #ebf8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #3182ce; margin: 20px 0; }}
        .answer {{ background: #f7fafc; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .tier-info {{ background: #faf5ff; padding: 15px; border-radius: 5px; border-left: 4px solid #805ad5; margin: 20px 0; }}
        .metadata {{ color: #718096; font-size: 0.9em; margin-top: 30px; padding-top: 15px; border-top: 1px solid #e2e8f0; }}
        ul, ol {{ margin-left: 20px; }}
        li {{ margin-bottom: 8px; }}
        strong {{ color: #2d3748; }}
        code {{ background: #edf2f7; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Michigan Legal RAG Response</h1>
        
        <div class="query">
            <strong>Your Question:</strong><br>
            {request.query}
        </div>
        
        <h2>Legal Analysis</h2>
        <div class="answer">
            {html_answer}
        </div>
        
        <div class="tier-info">
            <h3>Case Classification</h3>
            <p><strong>Tier {tier}:</strong> {tier_desc}</p>
            <p><strong>Reasoning:</strong> {tier_reasoning}</p>
            <p><strong>Recommendation:</strong> {tier_recommendation}</p>
        </div>
        
        <div class="metadata">
            <p><strong>Sources Used:</strong> {result.get("num_sources", 0)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
"""
    return html_response


@app.post("/query/markdown", response_class=PlainTextResponse)
async def query_legal_question_markdown(request: QueryRequest):
    """
    Process a legal query and return a clean markdown response.
    Line breaks are properly formatted for markdown rendering.
    
    Args:
        request: QueryRequest with the legal question
    
    Returns:
        Clean markdown formatted response
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Query the RAG engine
    if request.doc_type_filter:
        result = rag_engine.query_with_filter(
            question=request.query,
            doc_type=request.doc_type_filter
        )
    else:
        result = rag_engine.query(
            question=request.query,
            return_sources=request.include_sources,
            return_citations=request.include_citations
        )
    
    # Handle errors
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Get and format the answer
    answer = format_answer_for_display(result.get("answer", ""))
    
    # Classify tier
    is_legal = result.get("is_legal", True)
    is_harmful = result.get("is_harmful", False)
    
    if is_legal and not is_harmful:
        tier, tier_desc, tier_reasoning = TierRouter.classify_tier(request.query)
        tier_recommendation = TierRouter.get_tier_recommendation(tier)
    else:
        tier = 0
        tier_desc = "Not applicable"
        tier_reasoning = "Query was not processed as a legal question"
        tier_recommendation = "Please submit a legitimate legal question."
    
    # Build markdown response
    markdown_response = f"""# Michigan Legal RAG Response

## Your Question

{request.query}

---

## Legal Analysis

{answer}

---

## Case Classification

**Tier {tier}:** {tier_desc}

**Reasoning:** {tier_reasoning}

**Recommendation:** {tier_recommendation}

---

*Sources Used: {result.get("num_sources", 0)}*

*Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*
"""
    return markdown_response


@app.post("/query/report", response_class=PlainTextResponse)
async def query_legal_question_with_report(request: QueryRequest):
    """
    Process a legal query and return a complete formatted legal report.
    Ideal for attorneys who want a professional document for case review.
    
    Args:
        request: QueryRequest with the legal question
    
    Returns:
        Plain text report formatted for attorney review
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Query the RAG engine
    if request.doc_type_filter:
        result = rag_engine.query_with_filter(
            question=request.query,
            doc_type=request.doc_type_filter
        )
    else:
        result = rag_engine.query(
            question=request.query,
            return_sources=request.include_sources,
            return_citations=request.include_citations
        )
    
    # Handle errors
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Classify tier
    is_legal = result.get("is_legal", True)
    is_harmful = result.get("is_harmful", False)
    
    if is_legal and not is_harmful:
        tier, tier_desc, tier_reasoning = TierRouter.classify_tier(request.query)
        tier_recommendation = TierRouter.get_tier_recommendation(tier)
    else:
        tier = 0
        tier_desc = "Not applicable - query not related to legal matters" if not is_harmful else "Not applicable - inappropriate query"
        tier_reasoning = "Query was not processed as a legal question"
        tier_recommendation = "Please submit a legitimate legal question about Michigan or federal law."
    
    # Generate and return full report
    report = LegalReportGenerator.generate_report(
        query=request.query,
        answer=result.get("answer", ""),
        tier=tier,
        tier_description=tier_desc,
        tier_reasoning=tier_reasoning,
        tier_recommendation=tier_recommendation,
        citations=result.get("citations", []),
        sources=result.get("sources", []),
        usage=result.get("usage")
    )
    
    return report


@app.post("/query/summary-report", response_class=PlainTextResponse)
async def query_legal_question_summary(request: QueryRequest):
    """
    Process a legal query and return a concise summary report.
    Useful for quick attorney review.
    
    Args:
        request: QueryRequest with the legal question
    
    Returns:
        Plain text summary report
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Query the RAG engine
    result = rag_engine.query(
        question=request.query,
        return_sources=False,
        return_citations=False
    )
    
    # Handle errors
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Classify tier
    is_legal = result.get("is_legal", True)
    is_harmful = result.get("is_harmful", False)
    
    if is_legal and not is_harmful:
        tier, tier_desc, tier_reasoning = TierRouter.classify_tier(request.query)
    else:
        tier = 0
        tier_desc = "Not applicable"
        tier_reasoning = "Query was not processed as a legal question"
    
    # Generate and return summary report
    report = LegalReportGenerator.generate_summary_report(
        query=request.query,
        answer=result.get("answer", ""),
        tier=tier,
        tier_description=tier_desc,
        tier_reasoning=tier_reasoning
    )
    
    return report


# ============================================
# Background Indexing Endpoints
# ============================================

@app.post("/indexing/start")
async def start_indexing(force: bool = False):
    """
    Trigger background indexing of all documents.
    
    Args:
        force: Force re-indexing even if documents unchanged
    
    Returns:
        Job IDs and status
    """
    indexer = get_background_indexer()
    job_ids = indexer.index_all_documents(force=force)
    
    return {
        'message': 'Indexing jobs queued',
        'job_ids': job_ids,
        'count': len(job_ids),
        'timestamp': datetime.now().isoformat()
    }


@app.post("/indexing/document/{document_name}")
async def index_document(document_name: str, force: bool = False):
    """
    Trigger indexing of a specific document.
    
    Args:
        document_name: Name of the PDF file
        force: Force re-indexing
    
    Returns:
        Job ID and status
    """
    indexer = get_background_indexer()
    
    try:
        job_id = indexer.index_document(document_name, force=force)
        return {
            'message': f'Indexing job queued for {document_name}',
            'job_id': job_id,
            'document': document_name,
            'timestamp': datetime.now().isoformat()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/indexing/status/{job_id}")
async def get_indexing_status(job_id: str):
    """
    Get status of an indexing job.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Job status and details
    """
    indexer = get_background_indexer()
    status = indexer.get_job_status(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {
        'job': status,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/indexing/stats")
async def get_indexing_stats():
    """
    Get comprehensive indexing statistics.
    
    Returns:
        Indexing job statistics and metrics
    """
    indexer = get_background_indexer()
    stats = indexer.get_indexing_stats()
    
    return {
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/indexing/jobs")
async def get_all_indexing_jobs():
    """
    Get all indexing jobs.
    
    Returns:
        List of all jobs with their status
    """
    indexer = get_background_indexer()
    jobs = indexer.get_all_jobs()
    
    return {
        'jobs': jobs,
        'total': len(jobs),
        'timestamp': datetime.now().isoformat()
    }


async def reload_documents(background_tasks: BackgroundTasks):
    """
    Reload all documents and rebuild vector store.
    This is an admin endpoint for updating the knowledge base.
    """
    global rag_engine
    
    def reload_task():
        global rag_engine
        print("Reloading documents...")
        rag_engine = get_rag_engine(force_reload=True)
        print("Reload complete!")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Document reload initiated in background",
        "status": "processing"
    }


@app.get("/documents")
async def list_documents():
    """List all supported documents."""
    return {
        "documents": SUPPORTED_PDFS,
        "count": len(SUPPORTED_PDFS),
        "directory": PDF_DIRECTORY
    }


@app.post("/search")
async def search_documents(query: str, k: int = 5):
    """
    Search for relevant document chunks without generating an answer.
    Useful for debugging and understanding retrieval.
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    docs = rag_engine.get_relevant_documents(query, k=k)
    
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            "metadata": doc.metadata
        })
    
    return {
        "query": query,
        "num_results": len(results),
        "results": results
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
