# Michigan Legal RAG Chatbot - Technical Documentation

## Architecture Overview

This is a sophisticated Retrieval-Augmented Generation (RAG) system designed specifically for Michigan legal queries. It uses LangChain orchestration with advanced legal document processing.

## System Components

### 1. Document Processing Pipeline

#### **legal_preprocessor.py**
- Detects document types (Federal Rules, Michigan Court Rules, Jury Instructions)
- Extracts legal structure (articles, sections, rules)
- Cleans and normalizes legal text
- Extracts metadata (effective dates, rule numbers)

**Key Features:**
- Regex patterns for different legal document formats
- Section detection (Federal Rule 12, MCR 6.101, etc.)
- Citation extraction
- OCR error correction

#### **document_loader.py**
- Loads PDFs using pdfplumber (primary) and PyPDF2 (fallback)
- Extracts text with page markers
- Creates structured `LegalDocument` objects
- Converts to LangChain Document format

**Key Features:**
- Dual PDF extraction for reliability
- Metadata preservation
- Section-based document creation
- Progress tracking with tqdm

### 2. Intelligent Chunking System

#### **legal_chunker.py**
- Article-by-article chunking (NOT fixed-size)
- Token-aware splitting using tiktoken
- Hierarchical chunking strategy:
  1. Section-level (primary)
  2. Subsection-level ((a), (1), (i))
  3. Paragraph-level
  4. Sentence-level (fallback)

**Chunking Logic:**
```python
# Chunks respect legal structure
- Keep sections together when possible
- Split by subsections if too large
- Maintain context with overlaps
- Preserve section headers in all chunks
```

**Key Features:**
- Token counting (not character counting)
- Semantic boundary detection
- Context preservation
- Configurable chunk size (default: 1000 tokens)

### 3. Vector Store & Retrieval

#### **vector_store.py**
- ChromaDB for persistent vector storage
- OpenAI embeddings (text-embedding-3-small)
- Multiple retrieval strategies:
  - **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity
  - **Similarity Search**: Pure semantic similarity
  - **Score Threshold**: Minimum similarity filtering
  - **Hybrid Retrieval**: Combines vector + keyword search

**Key Features:**
- Metadata filtering by doc_type, section_number
- Contextual compression retrieval
- Persistent storage (survives restarts)
- Collection statistics

### 4. RAG Engine

#### **rag_engine.py**
- LangChain orchestration with custom prompts
- Retrieval chain with legal-specific context
- Citation extraction and validation
- Source tracking and attribution
- Token usage monitoring

**Pipeline:**
```
Query → Legal Check → Retrieve (MMR) → LLM (GPT-4o) → Extract Citations → Format Response
```

**Key Features:**
- Legal query validation
- OpenAI callback tracking
- Metadata filtering
- Error handling
- Global singleton instance

### 5. Citation & Source Management

#### **citation_extractor.py**
- Extracts citations from responses
- Validates citations against source documents
- Formats citations (MCR, FRCP, FRCrP, FRE)
- Creates citation lists

**Supported Citation Formats:**
- MCR 6.101(A)
- Fed. R. Civ. P. 12(b)(6)
- Fed. R. Crim. P. 11
- Fed. R. Evid. 404(b)
- M Civ JI 3.01
- M Crim JI 10.01

### 6. Tier Classification

#### **tier_router.py**
- 4-tier complexity classification
- Keyword-based scoring
- Recommendations based on tier

**Tiers:**
1. Routine (traffic tickets, small claims)
2. Moderate (felonies, contested hearings)
3. High-Stakes (violent crimes, federal)
4. Complex (appellate, constitutional)

### 7. FastAPI Application

#### **app.py**
- RESTful API endpoints
- Request/response validation with Pydantic
- CORS support
- Background task processing
- Swagger documentation (auto-generated)

**Endpoints:**
- `POST /query` - Main query endpoint
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /reload` - Reload documents
- `POST /search` - Debug retrieval

## Data Flow

```
PDFs in knowledge_base/
    ↓
document_loader.py (Extract text + metadata)
    ↓
legal_preprocessor.py (Detect structure, clean)
    ↓
legal_chunker.py (Chunk by article/section)
    ↓
vector_store.py (Embed + Store in ChromaDB)
    ↓
User Query → rag_engine.py
    ↓
Retrieve relevant chunks (MMR)
    ↓
LLM generates answer with citations
    ↓
Response with tier classification
```

## Advanced Features

### 1. Article-Based Chunking
Unlike traditional RAG systems that use fixed-size chunks, this system:
- Respects legal document structure
- Keeps entire rules/sections together when possible
- Splits intelligently by subsections
- Maintains hierarchical context

### 2. Metadata-Rich Retrieval
Every chunk includes:
- Document type
- Section number
- Section title
- Chunk type (section/subsection/paragraph)
- Source filename
- Page numbers

### 3. Hybrid Search
Combines:
- Semantic similarity (embeddings)
- Keyword matching
- Configurable weighting (alpha parameter)

### 4. Contextual Compression
Uses LLM to extract only relevant portions from retrieved chunks, reducing noise.

### 5. Citation Validation
Extracts citations from responses and validates them against source documents.

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
```

### config.py Parameters
```python
CHUNK_SIZE = 1000           # Tokens per chunk
CHUNK_OVERLAP = 200         # Overlap tokens
TOP_K_RETRIEVAL = 5         # Chunks to retrieve
OPENAI_MODEL = "gpt-4o"     # Generation model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## Performance Considerations

### Token Management
- Uses tiktoken for accurate counting
- Prevents context window overflow
- Monitors usage with callbacks

### Caching
- Vector store persists to disk
- Avoids re-embedding on restart
- Can force reload with `/reload` endpoint

### Optimization
- MMR reduces redundant retrievals
- Chunk overlap ensures context continuity
- Metadata filtering speeds up queries

## Setup & Deployment

### Installation
```bash
pip install -r requirements.txt
```

### Initialization
```bash
python setup.py  # Check system readiness
```

### Run Server
```bash
python -m uvicorn app:app --reload
```

### API Documentation
Visit: `http://localhost:8000/docs`

## Example Usage

### Python Client
```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the requirements for a motion to dismiss in federal court?",
    "include_sources": True,
    "include_citations": True
})

data = response.json()
print(data["answer"])
print(data["citations"])
```

### cURL
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MCR 6.101?"}'
```

## Maintenance

### Adding New Documents
1. Place PDF in `knowledge_base/`
2. Add filename to `config.SUPPORTED_PDFS`
3. Call `POST /reload` endpoint

### Updating Embeddings
```bash
# Force reload
curl -X POST "http://localhost:8000/reload"
```

### Monitoring
- Check `/health` for system status
- Check `/stats` for collection info
- Monitor token usage in responses

## Future Enhancements

1. **Semantic Chunking**: Use embeddings to group related sections
2. **BM25 Integration**: Better keyword search
3. **Multi-query Retrieval**: Generate multiple query variants
4. **Reranking**: Use cross-encoder for better results
5. **Streaming Responses**: Stream LLM output
6. **Conversation Memory**: Multi-turn dialogue support
7. **Fine-tuned Embeddings**: Domain-specific embeddings
8. **Query Expansion**: Add legal synonyms and related terms

## Troubleshooting

### Common Issues

**1. "Vector store not initialized"**
- Check if PDFs exist in knowledge_base/
- Run initialization: `python setup.py`
- Check logs for loading errors

**2. "No relevant documents found"**
- Query may be too vague
- Try more specific legal terms
- Check document coverage

**3. "OpenAI API error"**
- Verify API key in .env
- Check API quota/billing
- Reduce chunk size if rate limited

**4. "Poor answer quality"**
- Increase TOP_K_RETRIEVAL for more context
- Check if relevant PDFs are loaded
- Review chunk sizes (may be too small/large)

## Dependencies

Core:
- langchain (orchestration)
- chromadb (vector store)
- fastapi (API server)
- pdfplumber (PDF extraction)
- tiktoken (token counting)

See requirements.txt for full list.
