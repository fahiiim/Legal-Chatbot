# Michigan Legal RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that answers legal queries based on Michigan state law and federal criminal/civil procedure rules.

## 🌟 Key Features

- **Advanced Legal Document Processing**: Article-by-article chunking that respects legal document structure
- **LangChain Orchestration**: Full RAG pipeline with contextual compression and hybrid retrieval
- **Intelligent Chunking**: Token-aware chunking by legal sections, not fixed-size blocks
- **ChromaDB Vector Store**: Persistent embeddings with metadata filtering
- **Citation Extraction**: Automatic extraction and validation of legal citations
- **Tier Classification**: 4-tier complexity classification for legal queries
- **RESTful API**: FastAPI-based with Swagger documentation

## 📁 Project Structure

```
michigan-legal-rag/
├── app.py                      # FastAPI application with endpoints
├── rag_engine.py               # Main RAG orchestration engine
├── document_loader.py          # Advanced PDF loader with metadata extraction
├── legal_chunker.py            # Article-by-article intelligent chunking
├── legal_preprocessor.py       # Legal document structure detection
├── vector_store.py             # ChromaDB integration and retrieval
├── citation_extractor.py       # Citation extraction and validation
├── tier_router.py              # Query complexity classification
├── config.py                   # Configuration and constants
├── setup.py                    # System initialization script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create from template)
├── .env.template               # Environment template
├── knowledge_base/             # PDF documents directory
├── chroma_db/                  # Vector database (auto-created)
├── USER_GUIDE.md               # User documentation
├── TECHNICAL_DOCS.md           # Technical documentation
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Verify Setup

```powershell
python setup.py
```

This checks for:
- ✓ Required PDF files in `knowledge_base/`
- ✓ Environment configuration
- ✓ Installed dependencies

### 4. Run the Application

```powershell
python -m uvicorn app:app --reload
```

Server available at: `http://localhost:8000`  
API docs at: `http://localhost:8000/docs`

## 📚 Required Documents

Place these PDFs in the `knowledge_base/` directory:

- `federal-rules-of-criminal-procedure-dec-1-2024_0.pdf`
- `federal-rules-of-evidence-dec-1-2024_0.pdf`
- `federal-rules-of-civil-procedure-dec-1-2024_0.pdf`
- `criminal-jury-instructions.pdf` (Michigan Model Criminal Jury Instructions)
- `model-civil-jury-instructions.pdf` (Michigan Model Civil Jury Instructions)
- `michigan-court-rules.pdf`

## 💡 Usage Example

### REST API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the elements of armed robbery in Michigan?"}'
```

### Python Client

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What is the burden of proof in a criminal trial?",
    "include_sources": True,
    "include_citations": True
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Tier: {result['tier']} - {result['tier_description']}")
print(f"Citations: {result['citations']}")
```

## 🎯 API Endpoints

### Core Endpoints

- **POST /query** - Submit a legal query
  - Returns: answer, tier classification, citations, sources
  
- **GET /health** - Health check
  - Returns: system status

- **GET /stats** - System statistics
  - Returns: document count, model info

- **GET /documents** - List loaded documents

- **POST /search** - Debug retrieval (returns raw chunks)

- **POST /reload** - Reload documents (admin)

## 🏗️ Architecture Highlights

### 1. **Article-Based Chunking**
Unlike traditional fixed-size chunking, our system:
- Detects legal structure (rules, sections, articles)
- Keeps complete legal provisions together
- Splits intelligently by subsections when needed
- Maintains hierarchical context

### 2. **LangChain Orchestration**
- Retrieval chain with custom legal prompts
- Contextual compression for relevance
- MMR (Maximal Marginal Relevance) for diversity
- Source tracking and citation validation

### 3. **Advanced Retrieval**
- **Vector Search**: Semantic similarity with embeddings
- **Metadata Filtering**: Filter by doc_type, section_number
- **Hybrid Search**: Combines vector + keyword search
- **MMR Reranking**: Balances relevance with diversity

### 4. **Legal-Specific Processing**
- Rule number extraction (MCR 6.101, Fed. R. Crim. P. 12)
- Citation formatting and validation
- Section title and metadata preservation
- Page number tracking

## 📊 Tier Classification

Queries are automatically classified into 4 tiers:

1. **Tier 1 - Routine**: Traffic tickets, small claims, civil infractions
2. **Tier 2 - Moderate**: Felonies, contested hearings, motion practice  
3. **Tier 3 - High-Stakes**: Violent crimes, federal cases, constitutional issues
4. **Tier 4 - Complex**: Appellate, Supreme Court, capital cases

## 🔧 Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 1000           # Tokens per chunk
CHUNK_OVERLAP = 200         # Overlap tokens  
TOP_K_RETRIEVAL = 5         # Number of chunks to retrieve
OPENAI_MODEL = "gpt-4o"     # Generation model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## 📖 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Detailed usage guide with examples
- **[TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)** - Architecture and technical details
- **[API Docs](http://localhost:8000/docs)** - Interactive Swagger documentation (when running)

### Tier 3 — High-Stakes / Complex
- Long-term incarceration, federal charges, violent felony, expert witnesses, jury trial likely
- Examples: homicide, CSC, armed robbery, high-asset divorce, federal crimes

### Tier 4 — Extreme / Specialized
- Appellate-only, Supreme Court, precedent-setting, capital cases, RICO
- Examples: appeals, constitutional challenges, capital cases

**Routing Rule:** If uncertain about tier, the system **always refers UP** (never down).

## Key Features

✅ **Legal-Only Responses**: Rejects non-legal queries with:
```
"I only assist with legal matters under Michigan or federal law. Please describe your legal issue."
```

✅ **Citation-Based Answers**: All responses include source citations:
- Format: `(Mich. Ct. R. 6.110, p. 5)` or `(Fed. R. Crim. P. 5.1, p. 3)`

✅ **No Hallucination**: If no relevant provision exists:
```
"No relevant provision found in the Michigan Court Rules, Federal Rules of Criminal Procedure, Federal Rules of Evidence, or Michigan Model Jury Instructions."
```

✅ **Semantic Search**: Uses SentenceTransformer embeddings to retrieve relevant PDF excerpts

✅ **Neutral, Procedural Responses**: Never suggests strategy, predicts outcomes, or acts as an attorney

## Response Behavior Rules

1. **Never hallucinate**: Only cite provisions from the provided PDFs
2. **Neutral and procedural**: Explain applicable rules without legal advice
3. **Full citations**: Always include document name, rule/instruction number, and page
4. **No strategy**: Never suggest litigation strategies or predict outcomes
5. **Conservative tier routing**: If uncertain, route to higher tier (safety-first approach)

## Configuration

Edit `config.py` to customize:
- `OPENAI_MODEL`: OpenAI model to use (default: `gpt-4o`)
- `EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `TOP_K_RETRIEVAL`: Number of PDF chunks to retrieve (default: 5)
- `CHUNK_SIZE`: Characters per PDF chunk (default: 500)

## Example Queries

### Tier 1 Example
```json
{
  "query": "What happens for a first traffic violation in Michigan?"
}
```

### Tier 2 Example
```json
{
  "query": "I was arrested for drug possession. What are my rights?"
}
```

### Tier 3 Example
```json
{
  "query": "I'm facing federal charges for wire fraud. What are the evidence rules?"
}
```

## Troubleshooting

### PDFs not loading
- Ensure `knowledge_base/` directory exists
- Check PDF filenames match exactly in `config.py`
- Verify PDFs are readable (not corrupted)

### API key errors
- Verify `OPENAI_API_KEY` is set in `.env`
- Ensure key has appropriate permissions for chat and embedding models

### Slow responses
- First query takes time to load and embed PDFs
- Subsequent queries are faster as embeddings are cached in memory
- For production, consider using external vector database (Pinecone, Weaviate, etc.)

## Limitations

- **In-memory embeddings**: Embeddings are cached in RAM; not persisted
- **Single-session**: Restarts clear cached embeddings
- **No multi-user session management**: Stateless API
- **PDF extraction limits**: Complex PDF layouts may extract imperfectly

## Future Enhancements

- [ ] Persist embeddings to vector database
- [ ] Add filtering by tier-specific documents
- [ ] Multi-language support
- [ ] User feedback loop for response quality
- [ ] Integration with legal case management systems
- [ ] WebSocket for real-time queries
- [ ] Advanced citation tracking and validation

## License

Confidential for legal research purposes.

## Contact

For questions or issues, contact the development team.
