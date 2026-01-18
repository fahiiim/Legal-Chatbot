# Michigan Legal RAG Chatbot - User Guide

## Quick Start

### 1. Installation

```powershell
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Verify Setup

```powershell
python setup.py
```

This will check:
- ✓ All required PDFs are present
- ✓ Environment is configured
- ✓ Dependencies are installed

### 4. Start the Server

```powershell
python -m uvicorn app:app --reload
```

The server will be available at `http://localhost:8000`

### 5. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

## Using the API

### Basic Query

**Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the elements of armed robbery in Michigan?"}'
```

**Response:**
```json
{
  "query": "What are the elements of armed robbery in Michigan?",
  "answer": "According to M Crim JI 17.10, the elements of armed robbery are...",
  "tier": 3,
  "tier_description": "High-Stakes / Serious Felony",
  "tier_reasoning": "Query matched Tier 3: armed, robbery",
  "tier_recommendation": "This is a serious matter with significant consequences...",
  "citations": [
    {
      "citation_text": "M Crim JI 17.10",
      "source": "m_crim_ji",
      "section_number": "17.10"
    }
  ],
  "sources": [...],
  "num_sources": 5,
  "is_legal": true
}
```

### Query with Options

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What is the discovery process in civil litigation?",
    "include_sources": True,      # Include source documents
    "include_citations": True,    # Extract citations
    "doc_type_filter": "federal_civil_rules"  # Filter by document type
})

print(response.json()["answer"])
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Get Statistics

```bash
curl http://localhost:8000/stats
```

## Query Tips

### Best Practices

1. **Be Specific**
   - ❌ "Tell me about trials"
   - ✅ "What are the jury selection procedures in Michigan criminal trials?"

2. **Use Legal Terms**
   - ❌ "Can the police search my car?"
   - ✅ "What are the requirements for a warrantless vehicle search under Michigan law?"

3. **Reference Specific Rules**
   - ✅ "What does MCR 6.101 say about criminal procedures?"
   - ✅ "Explain Fed. R. Evid. 404(b)"

4. **Ask About Procedures**
   - ✅ "What is the process for filing a motion to suppress evidence?"
   - ✅ "What are the requirements for expert witness disclosure under FRCP?"

### Document Coverage

The system includes:
- **Federal Rules of Criminal Procedure** (FRCrP)
- **Federal Rules of Civil Procedure** (FRCP)
- **Federal Rules of Evidence** (FRE)
- **Michigan Court Rules** (MCR)
- **Michigan Model Criminal Jury Instructions**
- **Michigan Model Civil Jury Instructions**

## Understanding Tier Classifications

### Tier 1: Routine / Low-Risk
- Traffic tickets
- Civil infractions
- Name changes
- Small claims
- Uncontested matters

**Example:** "How do I contest a parking ticket in Michigan?"

### Tier 2: Moderate / Litigation
- Felony charges
- Contested hearings
- Motion practice
- Custody disputes
- Probation violations

**Example:** "What is the process for a preliminary examination in a felony case?"

### Tier 3: High-Stakes / Serious Felony
- Violent crimes
- Federal charges
- Constitutional issues
- CSC (Criminal Sexual Conduct)
- Armed offenses

**Example:** "What are the elements of first-degree murder in Michigan?"

### Tier 4: Complex / Appellate
- Appellate procedures
- Supreme Court matters
- Capital cases
- RICO charges
- Precedent-setting issues

**Example:** "What is the standard of review for appeals in federal court?"

## API Endpoints

### POST /query
Submit a legal query

**Request Body:**
```json
{
  "query": "string",
  "include_sources": true,
  "include_citations": true,
  "doc_type_filter": "string (optional)"
}
```

### GET /health
Check system status

### GET /stats
Get system statistics

### GET /documents
List all loaded documents

### POST /search
Search for relevant chunks (debugging)

**Parameters:**
- `query`: Search query
- `k`: Number of results (default: 5)

### POST /reload
Reload all documents (admin)

## Python Client Example

```python
import requests

class LegalChatbot:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def ask(self, question):
        """Ask a legal question."""
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": question}
        )
        return response.json()
    
    def is_healthy(self):
        """Check if service is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()["status"] == "healthy"

# Usage
bot = LegalChatbot()

if bot.is_healthy():
    result = bot.ask("What is the burden of proof in a criminal trial?")
    print(f"Answer: {result['answer']}")
    print(f"Tier: {result['tier']} - {result['tier_description']}")
    print(f"Citations: {result['citations']}")
```

## Advanced Features

### Filtering by Document Type

```python
# Search only in Federal Criminal Rules
response = requests.post("http://localhost:8000/query", json={
    "query": "What is the discovery deadline?",
    "doc_type_filter": "federal_criminal_rules"
})
```

Available filters:
- `federal_criminal_rules`
- `federal_civil_rules`
- `federal_evidence_rules`
- `michigan_court_rules`
- `michigan_criminal_jury_instructions`
- `michigan_civil_jury_instructions`

### Debugging Retrieval

Use the `/search` endpoint to see what documents are retrieved:

```python
response = requests.post("http://localhost:8000/search", params={
    "query": "motion to dismiss",
    "k": 3
})

for result in response.json()["results"]:
    print(result["metadata"])
```

## Troubleshooting

### "I only assist with legal matters..."

This means your query doesn't appear to be legal-related. Try:
- Adding legal keywords (law, court, rule, etc.)
- Rephrasing as a legal question
- Referencing specific laws or procedures

### "No relevant provision found..."

The answer isn't in the available documents. Try:
- Checking the document list with GET /documents
- Broadening your query
- Asking about a different jurisdiction (if applicable)

### Slow First Query

The first query after startup takes longer because:
1. Documents are being loaded
2. Embeddings are being created
3. Vector store is being initialized

Subsequent queries are much faster.

### API Errors

**503 Service Unavailable**
- Server is still initializing
- Wait a few seconds and retry

**500 Internal Server Error**
- Check server logs
- Verify API key is valid
- Ensure PDFs are accessible

## Best Practices for Production

1. **Set specific CORS origins** in app.py
2. **Add authentication** for sensitive deployments
3. **Rate limiting** to prevent abuse
4. **Monitoring** with logging and metrics
5. **Regular backups** of chroma_db/
6. **Keep PDFs updated** with latest rules

## Getting Help

- Check `/docs` for API documentation
- Review `TECHNICAL_DOCS.md` for architecture details
- Check logs for error messages
- Verify setup with `python setup.py`

## Example Queries

```
1. "What is the Miranda warning requirement in Michigan?"
2. "Explain the hearsay rule under Federal Rules of Evidence"
3. "What are the jury instruction requirements for self-defense?"
4. "How do I file a motion to suppress evidence in federal court?"
5. "What is MCR 2.116(C)(10)?"
6. "Explain the elements of breaking and entering under Michigan law"
7. "What are the discovery obligations in civil litigation?"
8. "What is the standard for granting summary judgment?"
```
