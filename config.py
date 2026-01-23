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
