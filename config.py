import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# PDF Knowledge Base
PDF_DIRECTORY = "knowledge_base"
SUPPORTED_PDFS = [
    "federal-rules-of-criminal-procedure-dec-1-2024_0.pdf",
    "federal-rules-of-evidence-dec-1-2024_0.pdf",
    "federal-rules-of-civil-procedure-dec-1-2024_0.pdf",
    "criminal-jury-instructions.pdf",
    "model-civil-jury-instructions.pdf",
    "michigan-court-rules.pdf"
]

# RAG Parameters
EMBEDDING_DIM = 1536
CHUNK_SIZE = 800  # Tokens per chunk (reduced to avoid token limits)
CHUNK_OVERLAP = 100  # Tokens overlap
TOP_K_RETRIEVAL = 4  # Number of top chunks to retrieve (reduced to limit context)

# Maximum context tokens to send to LLM (leave room for response)
MAX_CONTEXT_TOKENS = 12000

# Vector Store Configuration
VECTOR_STORE_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "legal_documents"

# Response Configuration
NON_LEGAL_RESPONSE = "I only assist with legal matters under Michigan or federal law. Please describe your legal issue."
NO_PROVISION_RESPONSE = "No relevant provision found in the Michigan Court Rules, Federal Rules of Criminal Procedure, Federal Rules of Evidence, or Michigan Model Jury Instructions."

# Tier Classification Thresholds
TIER_KEYWORDS = {
    1: ["traffic", "infraction", "name change", "small claims", "uncontested", "ticket", "ordinance"],
    2: ["felony", "jail", "motion", "contested", "probation", "custody", "divorce", "landlord", "tenant", "harass", "misdemeanor"],
    3: ["homicide", "federal", "violent", "csc", "armed", "constitutional", "death", "kill", "murder", "assault", "weapon", "gun", "knife", "sex", "rape", "strangle", "suffocate", "arson", "burn", "fire"],
    4: ["appellate", "supreme court", "capital", "rico", "precedent", "class action", "complex litigation"]
}

# ========================================
# Phase 1 Enhancement Configuration
# ========================================

# Reranker Configuration
USE_RERANKER = True  # Enable cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RERANKER_TOP_K = 5  # Number of documents to return after reranking
LEGAL_CITATION_BOOST = 0.1  # Score boost for documents with legal citations

# Response Validation Configuration
USE_RESPONSE_VALIDATION = True  # Validate answers against sources
VALIDATION_MODEL = "gpt-4o-mini"  # Cheaper model for validation
STRICT_VALIDATION = True  # Flag any unsupported claims
AUTO_CORRECT_RESPONSES = False  # Automatically correct validation failures

# Evaluation Configuration
USE_EVALUATION = True  # Track retrieval and generation metrics
EVALUATION_MODEL = "gpt-4o-mini"  # Model for LLM-as-judge evaluation
SAVE_EVALUATION_HISTORY = True  # Save evaluations to file
EVALUATION_HISTORY_PATH = "evaluation_history.json"

# Semantic Cache Configuration
USE_SEMANTIC_CACHE = True  # Enable semantic caching
CACHE_SIMILARITY_THRESHOLD = 0.95  # Similarity threshold for cache hit (0-1)
CACHE_MAX_SIZE = 1000  # Maximum number of cached queries
CACHE_TTL_HOURS = 24  # Cache entry time-to-live in hours
CACHE_PERSIST_PATH = "./cache_data"  # Path to persist cache

# Advanced Retrieval Configuration
INITIAL_RETRIEVAL_K = 10  # Retrieve more docs before reranking
MMR_LAMBDA = 0.5  # MMR diversity parameter (0=max diversity, 1=max relevance)
