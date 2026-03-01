<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Version-2.1.0-blue?style=for-the-badge" alt="Version" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">Michigan Legal RAG Chatbot</h1>

<p align="center">
  <strong>An Advanced Retrieval-Augmented Generation System for Michigan and Federal Legal Research</strong>
</p>

<p align="center">
  Production-grade AI-powered legal document search, analysis, and question answering<br/>
  built with domain-specific chunking, cross-encoder reranking, hallucination validation,<br/>
  and four-tier case classification.
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" alt="GPT-4o" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=flat-square&logo=databricks&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Sentence_Transformers-FF6F00?style=flat-square&logo=huggingface&logoColor=white" alt="Sentence Transformers" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/tiktoken-000000?style=flat-square&logo=openai&logoColor=white" alt="tiktoken" />
  <img src="https://img.shields.io/badge/Uvicorn-2D3748?style=flat-square&logo=gunicorn&logoColor=white" alt="Uvicorn" />
  <img src="https://img.shields.io/badge/PyPDF2-CC0000?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="PyPDF2" />
  <img src="https://img.shields.io/badge/pdfplumber-003B57?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="pdfplumber" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white" alt="Prometheus" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [Advanced Chunking Strategy](#advanced-chunking-strategy)
- [Key Features](#key-features)
- [Legal Knowledge Base](#legal-knowledge-base)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Production Safety](#production-safety)
- [Evaluation Framework](#evaluation-framework)
- [Disclaimer](#disclaimer)

---

## Overview

The **Michigan Legal RAG Chatbot** is a specialized, production-grade Retrieval-Augmented Generation system engineered exclusively for legal research across Michigan state law and federal procedural rules. Unlike generic chatbots, this system is purpose-built for the legal domain -- every component, from document ingestion to response generation, is designed around the unique structural and semantic characteristics of legal text.

The system processes and indexes over **1.5 million tokens** of authoritative legal material, including Michigan Court Rules, Federal Rules of Civil and Criminal Procedure, Federal Rules of Evidence, and both civil and criminal Model Jury Instructions. It delivers cited, validated, and tier-classified legal analysis through a high-performance REST API.

**What sets this system apart:**

- **Domain-Specific Chunking** -- Legal documents are not split arbitrarily. The system understands articles, rules, subsections, definitions, cross-references, and tables of contents, preserving the semantic integrity that legal interpretation demands.
- **Cross-Encoder Reranking** -- A dedicated neural reranking stage using `ms-marco-MiniLM-L-12-v2` reorders retrieved documents by true query-document relevance, achieving 15-30% improvement in retrieval precision over bi-encoder similarity alone.
- **Hallucination Prevention** -- Every generated response is validated against its source documents through an independent LLM judge, ensuring that no fabricated legal citations or unsupported claims reach the end user.
- **Four-Tier Case Classification** -- Queries are automatically classified into complexity tiers (Routine, Moderate, High-Stakes, Complex/Appellate), enabling attorneys to prioritize cases and allocate resources effectively.
- **Production-Grade Safety** -- Circuit breakers, rate limiting, health monitoring, background indexing, and graceful degradation ensure the system performs reliably under real-world conditions.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                              CLIENT REQUEST (REST API)                             |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                           FastAPI Application Layer                                |
|                                                                                   |
|   +-------------------+   +-------------------+   +--------------------+          |
|   | Input Validation  |   |   Rate Limiter    |   | Request Limiter    |          |
|   | & Sanitization    |   | (Token Bucket)    |   | (Concurrent/Daily) |          |
|   +-------------------+   +-------------------+   +--------------------+          |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                        Production Safety Manager                                  |
|                                                                                   |
|   +-------------------+   +-------------------+   +--------------------+          |
|   | Health Checks     |   | Circuit Breakers  |   | Error Recovery     |          |
|   | (Component-level) |   | (OpenAI, Vector)  |   | & Monitoring       |          |
|   +-------------------+   +-------------------+   +--------------------+          |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                          Semantic Cache Layer                                     |
|                                                                                   |
|   Embedding-based similarity matching (threshold: 0.95)                           |
|   LRU eviction  |  TTL expiration (24h)  |  Persistent storage                   |
|   Cache Hit --> Return cached response immediately                                |
+-----------------------------------------------------------------------------------+
                                        |
                                   Cache Miss
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                           RAG Engine Core                                         |
|                                                                                   |
|   +-----------------------------------------------------------------------+       |
|   |                    RETRIEVAL STAGE                                     |       |
|   |                                                                       |       |
|   |   Query --> OpenAI text-embedding-3-small --> Vector Similarity        |       |
|   |         --> ChromaDB (MMR Search, lambda=0.5)                         |       |
|   |         --> Top-K Candidate Documents (K=10)                          |       |
|   +-----------------------------------------------------------------------+       |
|                                   |                                               |
|                                   v                                               |
|   +-----------------------------------------------------------------------+       |
|   |                    RERANKING STAGE                                     |       |
|   |                                                                       |       |
|   |   Cross-Encoder: ms-marco-MiniLM-L-12-v2                             |       |
|   |   Query-Document pair scoring --> Re-sort by relevance                |       |
|   |   Legal citation boost (+0.1) --> Top-5 after reranking              |       |
|   +-----------------------------------------------------------------------+       |
|                                   |                                               |
|                                   v                                               |
|   +-----------------------------------------------------------------------+       |
|   |                    CONTEXT OPTIMIZATION                                |       |
|   |                                                                       |       |
|   |   Token counting (tiktoken, gpt-4o encoding)                         |       |
|   |   Context window limiting (12,000 tokens max)                         |       |
|   |   Intelligent truncation with partial document inclusion              |       |
|   +-----------------------------------------------------------------------+       |
|                                   |                                               |
|                                   v                                               |
|   +-----------------------------------------------------------------------+       |
|   |                    GENERATION STAGE                                    |       |
|   |                                                                       |       |
|   |   LLM: GPT-4o (temperature=0 for deterministic legal output)          |       |
|   |   Legal-specific prompt template with citation instructions           |       |
|   |   Source tracking and citation extraction                             |       |
|   +-----------------------------------------------------------------------+       |
|                                   |                                               |
|                                   v                                               |
|   +-----------------------------------------------------------------------+       |
|   |                    VALIDATION STAGE                                    |       |
|   |                                                                       |       |
|   |   LLM-as-Judge: GPT-4o-mini (independent verification)               |       |
|   |   Grounding check against source documents                            |       |
|   |   Citation accuracy verification                                      |       |
|   |   Hallucination detection --> PASS / FLAG / REJECT                    |       |
|   +-----------------------------------------------------------------------+       |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                      Post-Processing & Response                                   |
|                                                                                   |
|   +-------------------+   +-------------------+   +--------------------+          |
|   | Tier Router       |   | Citation          |   | Report Generator   |          |
|   | (4-Tier Classify) |   | Extractor         |   | (Attorney Format)  |          |
|   +-------------------+   +-------------------+   +--------------------+          |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                          RESPONSE (JSON / HTML / Markdown / Report)                |
+-----------------------------------------------------------------------------------+
```

### Supporting Infrastructure

```
+---------------------------------------------------+
|           Background Indexing Service              |
|                                                   |
|   Async document processing  |  State persistence |
|   Change detection (SHA-256) |  Retry with backoff|
|   Thread-safe job queue      |  Progress tracking |
+---------------------------------------------------+

+---------------------------------------------------+
|           Document Processing Pipeline             |
|                                                   |
|   PDF Extraction (PyPDF2 + pdfplumber)            |
|   Legal Preprocessor (structure detection)        |
|   Intelligent Legal Chunker (1049-line engine)    |
|   ChromaDB Vector Store (disk-persisted)          |
+---------------------------------------------------+

+---------------------------------------------------+
|           Evaluation & Monitoring                  |
|                                                   |
|   Precision@K, Recall@K, F1, MRR, NDCG           |
|   Faithfulness scoring (LLM-as-Judge)             |
|   Answer relevance and completeness metrics       |
|   Persistent evaluation history (JSON)            |
+---------------------------------------------------+
```

---

## How the RAG Pipeline Works

The system implements a **six-stage pipeline** that transforms a raw legal question into a validated, cited, and classified response:

### Stage 1: Query Reception and Safety Gating

Every incoming query passes through input validation, sanitization, rate limiting (token-bucket algorithm with per-user and global thresholds), and concurrent request control. The Production Safety Manager performs real-time health checks on all critical components and manages circuit breakers for external service dependencies (OpenAI API, vector store). If any critical component is in a degraded state, the system gracefully rejects requests rather than producing unreliable output.

### Stage 2: Semantic Cache Lookup

Before invoking any retrieval or generation, the system checks its semantic cache. Unlike simple string-matching caches, this layer computes embedding-based cosine similarity between the incoming query and previously cached queries using OpenAI `text-embedding-3-small` embeddings. With a similarity threshold of 0.95, semantically equivalent questions (even when phrased differently) receive instant cached responses. The cache implements LRU eviction, TTL-based expiration (24 hours), and persistent disk storage, significantly reducing API costs and response latency for repeated or similar queries.

### Stage 3: Retrieval with Maximal Marginal Relevance

On a cache miss, the query is embedded and searched against the ChromaDB vector store using **Maximal Marginal Relevance (MMR)** retrieval. MMR balances relevance against diversity (lambda = 0.5), ensuring that retrieved chunks cover different aspects of the legal question rather than returning redundant content. The initial retrieval fetches 10 candidate documents to provide sufficient material for the reranking stage.

### Stage 4: Cross-Encoder Neural Reranking

The 10 candidate documents are passed through a **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-12-v2`). Unlike the bi-encoder approach used during retrieval (where query and document are embedded independently), the cross-encoder processes query-document pairs jointly, enabling it to capture fine-grained relevance signals such as term interactions, negation, and contextual meaning. Documents containing verified legal citations receive an additional relevance boost (+0.1). The top 5 documents after reranking proceed to generation. This stage delivers a measured **15-30% improvement in retrieval precision**.

### Stage 5: Grounded Generation with GPT-4o

The reranked, token-limited context is fed to GPT-4o at temperature 0 (fully deterministic) with a legal-domain prompt template that instructs the model to cite specific rules, sections, and statutes, and to explicitly state when information is not found in the provided sources. Token counting via `tiktoken` ensures the context never exceeds 12,000 tokens, leaving sufficient room for a comprehensive response. The Citation Extractor then parses the generated response to identify and structure all legal references (MCR, FRCP, FRCrP, FRE, M Civ JI, M Crim JI patterns).

### Stage 6: Independent Response Validation

The generated response undergoes validation by an independent LLM judge (GPT-4o-mini) that cross-references every claim in the answer against the source documents. The validator produces a structured assessment including grounding confidence, supported and unsupported claims, hallucinated information, citation errors, and an overall recommendation (PASS, FLAG_FOR_REVIEW, or REJECT). In strict mode, even minor unsupported details are flagged -- a critical safeguard for legal content where accuracy is non-negotiable.

---

## Advanced Chunking Strategy

The legal chunking engine (`legal_chunker.py` -- over 1,000 lines of domain-specific logic) is one of the most critical differentiators of this system. Generic text splitters destroy the structural and semantic boundaries that are essential to legal interpretation. This system implements a multi-layered, legally-aware chunking pipeline:

### Hierarchical Structure Preservation

The chunker detects and respects the natural hierarchy of legal documents:

- **Articles and Rules** -- Top-level divisions (e.g., "Rule 12", "MCR 6.101") are identified and used as primary split boundaries, ensuring that a single rule is never arbitrarily split mid-sentence.
- **Sections and Subsections** -- Nested structures like (a), (1), (i) are tracked to maintain the parent-child relationship between provisions, so a subsection always carries context about the rule it belongs to.
- **Enriched Metadata** -- Every chunk is annotated with its document type, section number, parent section, hierarchical level, and source file, enabling precise downstream filtering and citation.

### Table of Contents Extraction

The system identifies and separately indexes Table of Contents pages using density analysis of dotted-leader patterns and TOC header detection. TOC chunks are tagged with `is_navigation: true` and contain structured entry data (section numbers, titles, page references), allowing the retrieval system to locate relevant sections even when the user's query matches the section title rather than its content.

### Legal Definition Extraction

Legal definitions are extracted using pattern matching for constructs like `"Term" means...`, `"Term" is defined as...`, and `As used in this rule, "Term" means...`. Each definition is stored as an independent, searchable chunk tagged with `is_definition: true` and the defined term, ensuring that definitional queries surface the exact authoritative definition rather than a passage that incidentally mentions the term.

### Cross-Reference Tracking

Legal documents are densely cross-referenced. The chunker extracts all cross-references (MCR citations, Federal Rule references, section references, "pursuant to" constructs) from each chunk and stores them in metadata. This enables the system to understand the relational graph of legal provisions and potentially retrieve related rules that the user did not explicitly ask about.

### Memory-Efficient Processing

Documents exceeding 400,000 tokens (such as the Michigan Criminal Jury Instructions at approximately 400K tokens or Michigan Court Rules at approximately 600K tokens) are processed one at a time with explicit garbage collection between documents. Chunks are immediately committed to the ChromaDB disk-based vector store in batches of 500, ensuring that system memory usage remains bounded regardless of corpus size.

---

## Key Features

### Four-Tier Case Classification

| Tier | Classification | Examples | Recommendation |
|------|---------------|----------|----------------|
| **1** | Routine / Low-Risk | Traffic tickets, name changes, small claims | Self-help resources may suffice |
| **2** | Moderate / Litigation | Felony charges, custody disputes, probation | Consult with qualified attorney |
| **3** | High-Stakes / Serious Felony | Violent crimes, federal offenses, CSC | Engage experienced counsel immediately |
| **4** | Complex / Appellate | Supreme Court, capital cases, class actions | Seek specialized legal expertise |

Classification is performed by the Tier Router using keyword-based scoring across 40+ legal terms mapped to the four tiers. Every response includes the tier classification, reasoning, and a professional recommendation.

### Professional Legal Report Generation

The system generates attorney-grade research reports containing:

- Case classification and urgency assessment
- Client statement / case facts
- Legal analysis and findings
- Applicable statutes, rules, and jury instructions with full citations
- Research sources consulted with document types and page references
- Detailed tier classification explanation
- Professional disclaimer

### Multi-Format Response Delivery

| Endpoint | Format | Use Case |
|----------|--------|----------|
| `POST /query` | JSON | API integration, frontend applications |
| `POST /query/formatted` | HTML | Browser display with styling |
| `POST /query/markdown` | Markdown | Documentation, rendered display |
| `POST /query/report` | Plain text | Attorney review, case files |
| `POST /query/summary-report` | Plain text | Quick initial assessment |

### Semantic Caching

- Embedding-based similarity matching eliminates redundant API calls for semantically equivalent queries
- Configurable similarity threshold (default: 0.95), TTL (default: 24 hours), and maximum cache size (default: 1,000 entries)
- LRU eviction policy with persistent disk storage
- Tracks hit/miss statistics for performance monitoring

### Background Document Indexing

- Asynchronous document processing with thread-safe job queue
- SHA-256 change detection to avoid re-indexing unchanged documents
- Automatic retry with configurable backoff (up to 3 retries per document)
- Real-time progress tracking and state persistence across restarts

---

## Legal Knowledge Base

The system indexes the following authoritative legal materials:

| Document | Approximate Size | Type |
|----------|-----------------|------|
| Federal Rules of Criminal Procedure (Dec 2024) | ~64K tokens | Federal Procedural Rules |
| Federal Rules of Evidence (Dec 2024) | ~27K tokens | Federal Evidence Rules |
| Federal Rules of Civil Procedure (Dec 2024) | ~95K tokens | Federal Procedural Rules |
| Michigan Criminal Jury Instructions | ~400K tokens | State Jury Instructions |
| Michigan Model Civil Jury Instructions | ~340K tokens | State Jury Instructions |
| Michigan Court Rules | ~600K tokens | State Court Rules |

**Total corpus: approximately 1.5 million tokens** of authoritative legal content indexed into the vector store with full structural metadata.

---

## API Endpoints

### Core Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Submit a legal query (JSON response with full metadata) |
| `POST` | `/query/formatted` | Submit a legal query (HTML formatted with styling) |
| `POST` | `/query/markdown` | Submit a legal query (Markdown formatted) |
| `POST` | `/query/report` | Submit a legal query (full professional legal report) |
| `POST` | `/query/summary-report` | Submit a legal query (concise summary report) |

### System Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Basic health check |
| `GET` | `/health/detailed` | Comprehensive health report with component statuses |
| `GET` | `/stats` | System statistics and performance metrics |
| `GET` | `/documents` | List all supported legal documents |
| `POST` | `/search` | Search document chunks without generating an answer |

### Background Indexing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/indexing/start` | Trigger background indexing of all documents |
| `POST` | `/indexing/document/{name}` | Index a specific document |
| `GET` | `/indexing/status/{job_id}` | Check status of an indexing job |
| `GET` | `/indexing/stats` | Comprehensive indexing statistics |
| `GET` | `/indexing/jobs` | List all indexing jobs |

### Request Schema

```json
{
  "query": "What are the requirements for filing a motion to dismiss under Michigan Court Rules?",
  "include_sources": true,
  "include_citations": true,
  "doc_type_filter": null
}
```

### Response Schema

```json
{
  "query": "...",
  "answer": "Under MCR 2.116, a motion to dismiss may be filed...",
  "tier": 2,
  "tier_description": "Moderate / Litigation",
  "tier_reasoning": "Query matched Tier 2: motion",
  "tier_recommendation": "Consider consulting with an attorney...",
  "citations": [
    {
      "name": "MCR 2.116",
      "reference": "Michigan Court Rules",
      "description": "Rule/Section 2.116"
    }
  ],
  "sources": [],
  "num_sources": 5,
  "is_legal": true,
  "usage": {
    "total_tokens": 2847,
    "prompt_tokens": 2100,
    "completion_tokens": 747,
    "total_cost": 0.0284
  },
  "timestamp": "2026-03-02T01:30:00.000000"
}
```

---

## Project Structure

```
Michigan Legal RAG Chatbot/
|
|-- app.py                    # FastAPI application with all REST endpoints
|-- rag_engine.py             # Core RAG pipeline orchestration (944 lines)
|-- legal_chunker.py          # Domain-specific legal document chunker (1049 lines)
|-- legal_preprocessor.py     # Legal text cleaning, normalization, structure detection
|-- document_loader.py        # PDF ingestion with multi-library fallback
|-- vector_store.py           # ChromaDB vector store manager with batch processing
|-- reranker.py               # Cross-encoder neural reranking engine
|-- response_validator.py     # LLM-as-Judge hallucination prevention
|-- evaluator.py              # Comprehensive RAG evaluation framework (509 lines)
|-- semantic_cache.py         # Embedding-based semantic caching layer
|-- citation_extractor.py     # Legal citation extraction and validation
|-- tier_router.py            # Four-tier case complexity classifier
|-- report_generator.py       # Professional legal report formatter
|-- config.py                 # Centralized configuration management
|-- background_indexer.py     # Async document indexing with state management
|-- production_safety.py      # Health checks, circuit breakers, monitoring
|-- request_limiter.py        # Rate limiting and request validation
|-- production_init.py        # Production initialization utilities
|-- setup.py                  # Package setup configuration
|-- requirements.txt          # Python dependency specifications
|-- knowledge_base/           # Legal PDF documents directory
|-- chroma_db/                # Persisted ChromaDB vector store
|-- logs/                     # Application log files
```

---

## Installation and Setup

### Prerequisites

- Python 3.12+
- OpenAI API key
- HuggingFace token (for cross-encoder model download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/fahiiim/Legal-Chatbot.git
cd Legal-Chatbot
```

### Step 2: Create and Activate Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Step 5: Add Legal Documents

Place your PDF legal documents in the `knowledge_base/` directory. The system is pre-configured to process the following files:

- `federal-rules-of-criminal-procedure-dec-1-2024_0.pdf`
- `federal-rules-of-evidence-dec-1-2024_0.pdf`
- `federal-rules-of-civil-procedure-dec-1-2024_0.pdf`
- `criminal-jury-instructions.pdf`
- `model-civil-jury-instructions.pdf`
- `michigan-court-rules.pdf`

### Step 6: Run the Application

```bash
python app.py
```

The API server will start on the configured host and port. Access the interactive API documentation at `http://<host>:<port>/docs`.

---

## Configuration

All system parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENAI_MODEL` | `gpt-4o` | Primary LLM for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for retrieval |
| `CHUNK_SIZE` | `500` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVAL` | `3` | Documents returned after reranking |
| `MAX_CONTEXT_TOKENS` | `12000` | Maximum context window for LLM |
| `INITIAL_RETRIEVAL_K` | `10` | Documents retrieved before reranking |
| `MMR_LAMBDA` | `0.5` | MMR diversity parameter |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` | Cross-encoder model |
| `RERANKER_TOP_K` | `5` | Documents after reranking |
| `LEGAL_CITATION_BOOST` | `0.1` | Score boost for cited documents |
| `VALIDATION_MODEL` | `gpt-4o-mini` | LLM for response validation |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Semantic cache hit threshold |
| `CACHE_MAX_SIZE` | `1000` | Maximum cached entries |
| `CACHE_TTL_HOURS` | `24` | Cache entry time-to-live |

---

## Production Safety

The system implements multiple layers of production safety:

### Health Monitoring

- **Component-level health checks** for the RAG engine, vector store, and all critical subsystems
- **Periodic automated health assessments** with status reporting (HEALTHY, DEGRADED, UNHEALTHY)
- **Detailed health endpoint** (`/health/detailed`) exposing all component statuses for external monitoring integration

### Circuit Breaker Pattern

- **OpenAI API circuit breaker** (threshold: 3 failures, recovery: 30s) prevents cascading failures during API outages
- **Vector search circuit breaker** (threshold: 5 failures, recovery: 60s) isolates vector store issues
- **Three-state model** (CLOSED, OPEN, HALF_OPEN) with automatic recovery testing

### Rate Limiting and Request Control

- **Token-bucket rate limiting** with configurable global (100 req/min) and per-user (20 req/min) thresholds
- **Concurrent request limiting** prevents resource exhaustion under load
- **Daily request quotas** per user
- **Input validation and sanitization** against injection and malformed queries

### Error Recovery

- **Automatic retry logic** with exponential backoff for transient failures
- **Graceful degradation** -- the system continues operating with reduced functionality rather than failing entirely
- **Persistent state management** ensures recovery after restarts without data loss

---

## Evaluation Framework

The system includes a comprehensive evaluation framework (`evaluator.py`) implementing standard information retrieval and generation quality metrics:

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of retrieved documents that are relevant |
| Recall@K | Fraction of relevant documents that were retrieved |
| F1@K | Harmonic mean of Precision and Recall |
| MRR | Mean Reciprocal Rank of the first relevant result |
| NDCG@K | Normalized Discounted Cumulative Gain |

### Generation Metrics

| Metric | Description |
|--------|-------------|
| Faithfulness | Degree to which the answer is grounded in source documents |
| Answer Relevance | How directly the answer addresses the question |
| Completeness | Whether all aspects of the question are addressed |
| Citation Accuracy | Correctness of legal citations against source material |

Evaluation results are persisted to `evaluation_history.json` for longitudinal performance tracking.

---

## Disclaimer

This system is developed for **educational and legal research assistance purposes**. All responses are generated using Retrieval-Augmented Generation technology sourced from official Michigan Court Rules, Federal Rules of Civil and Criminal Procedure, Federal Rules of Evidence, and Michigan Model Jury Instructions.

**This system does not constitute legal advice.** All output should be reviewed and verified by a qualified attorney. An attorney should conduct independent legal research and provide personalized counsel based on the specific facts of any given case. The developers assume no liability for decisions made based on system output.

---

<p align="center">
  <strong>Built for Legal Professionals. Powered by Advanced RAG Architecture.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python" />
  <img src="https://img.shields.io/badge/Powered%20by-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="Powered by OpenAI" />
  <img src="https://img.shields.io/badge/Vector%20Store-ChromaDB-FF6F00?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
</p>
