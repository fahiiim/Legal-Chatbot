"""
FastAPI Application for Michigan Legal RAG Chatbot
Provides REST API endpoints for legal question answering.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
from datetime import datetime

from rag_engine import get_rag_engine
from tier_router import TierRouter
from config import *


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


# Initialize FastAPI app
app = FastAPI(
    title="Michigan Legal RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for Michigan legal questions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global RAG engine instance
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup."""
    global rag_engine
    print("Initializing Michigan Legal RAG Engine...")
    rag_engine = get_rag_engine(force_reload=False)
    print("RAG Engine ready!")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Michigan Legal RAG Chatbot API",
        "version": "2.0.0",
        "endpoints": {
            "POST /query": "Submit a legal query",
            "GET /health": "Health check",
            "GET /stats": "Get system statistics",
            "POST /reload": "Reload documents (admin)"
        },
        "supported_documents": SUPPORTED_PDFS
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if rag_engine and rag_engine.is_initialized else "initializing",
        engine_initialized=rag_engine.is_initialized if rag_engine else False,
        timestamp=datetime.now().isoformat()
    )


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
async def query_legal_question(request: QueryRequest):
    """
    Process a legal query and return an answer with citations.
    
    Args:
        request: QueryRequest with the legal question
    
    Returns:
        QueryResponse with answer, tier classification, citations, and sources
    """
    if not rag_engine or not rag_engine.is_initialized:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    # Classify tier
    tier, tier_desc, tier_reasoning = TierRouter.classify_tier(request.query)
    tier_recommendation = TierRouter.get_tier_recommendation(tier)
    
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
    
    # Build response
    response = QueryResponse(
        query=request.query,
        answer=result.get("answer", ""),
        tier=tier,
        tier_description=tier_desc,
        tier_reasoning=tier_reasoning,
        tier_recommendation=tier_recommendation,
        citations=result.get("citations", []),
        sources=result.get("sources", []),
        num_sources=result.get("num_sources", 0),
        is_legal=result.get("is_legal", True),
        usage=result.get("usage"),
        timestamp=datetime.now().isoformat()
    )
    
    return response


@app.post("/reload")
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
