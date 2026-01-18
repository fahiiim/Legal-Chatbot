# Michigan Legal RAG Chatbot - Testing Guide

## System Status

✅ **RAG Engine Ready**
- 6 Legal Documents Loaded (16,972 chunks)
- Embeddings cached and persisted to disk
- Semantic search operational
- Tier classification system active

## Running the Test Script

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the interactive test
python test_rag.py
```

## Test Mode Options

### 1. **Chunks-Only Mode** (Default)
Shows retrieved chunks without OpenAI API calls. Perfect for validating retrieval quality.

```
> llm off
[CONFIG] ✓ LLM response generation disabled (chunks only)
> Enter legal query: What is second-degree murder in Michigan?
```

### 2. **LLM Response Mode**
Generates full chatbot responses with citations using OpenAI API.

```
> llm on
[CONFIG] ✓ LLM response generation enabled
> Enter legal query: What is second-degree murder in Michigan?
```

### 3. **Utility Commands**
```
cache          - Show loaded embeddings and statistics
save           - Persist current embeddings to disk
quit/exit      - Exit the test script
llm on/off     - Toggle LLM response generation
```

## Example Test Queries

### Tier 1 (Routine)
```
What are the rules for a traffic violation hearing?
How do I file for a name change in Michigan?
What is a small claims court?
```

### Tier 2 (Moderate Litigation)
```
I was arrested for drug possession. What are my rights?
What happens during a probation violation hearing?
Explain custody disputes in family court.
```

### Tier 3 (High-Stakes/Complex)
```
What constitutes criminal sexual conduct in Michigan?
Explain first-degree murder charges and penalties.
What are federal charges and how are they handled?
```

### Tier 4 (Appellate/Specialized)
```
What is the process for filing an appeal?
How do capital cases work?
```

## Validation Checklist

When testing, verify:

1. **Legal Query Detection**
   - Non-legal queries → "I only assist with legal matters..."
   - Legal queries → Tier classification and retrieval

2. **Tier Classification**
   - Is the tier appropriate for the query?
   - Is reasoning provided?

3. **Chunk Retrieval**
   - Relevance scores > 0.5 (good matches)
   - Multiple documents represented
   - Source documents and page numbers correct

4. **Response Quality** (when using LLM mode)
   - Response grounded in retrieved chunks
   - Citations included with document name and page
   - No hallucinated legal provisions
   - Neutral, procedural language (no legal advice)

## File Structure

```
michigan-legal-rag/
├── test_rag.py                    # Standalone test script
├── rag_engine.py                  # RAG core logic
├── tier_router.py                 # Case classification
├── config.py                      # Configuration
├── embeddings_cache.pkl           # Persisted embeddings (created after first run)
├── embeddings_metadata.json       # Embeddings metadata
├── knowledge_base/                # PDF documents
│   ├── federal-rules-of-criminal-procedure-dec-1-2024_0.pdf
│   ├── federal-rules-of-evidence-dec-1-2024_0.pdf
│   ├── federal-rules-of-civil-procedure-dec-1-2024_0.pdf
│   ├── criminal-jury-instructions.pdf
│   ├── model-civil-jury-instructions.pdf
│   └── michigan-court-rules.pdf
└── venv/                          # Virtual environment
```

## Performance Notes

- **First Load:** 2-3 minutes (processes all PDFs, creates embeddings)
- **Subsequent Loads:** < 1 second (loads from cache)
- **Total Chunks:** 16,972 (100% of all documents)
- **Embedding Dimension:** 384 (MiniLM model)
- **Retrieval Speed:** ~100-200ms per query

## Troubleshooting

### No chunks retrieved for query
- Try simpler legal keywords
- Query may not be legal-related
- Check embeddings_cache.pkl exists

### LLM response error
- OpenAI API key may be invalid
- Use `llm off` to see just chunks
- Check .env file for API key

### Cache not loading
- Delete embeddings_cache.pkl and embeddings_metadata.json
- Run script again to rebuild cache
- Takes 2-3 minutes on first run

## Next Steps

1. ✅ Test chunk retrieval quality (chunks-only mode)
2. ✅ Validate tier classification logic
3. ✅ Test LLM response generation (when OpenAI key configured)
4. ⏳ Deploy via FastAPI (app.py)
5. ⏳ Add persistent vector database (Pinecone/Weaviate)
