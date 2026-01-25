"""
RAG Engine with LangChain Orchestration
Main engine for retrieval-augmented generation with legal documents.
"""

from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain.callbacks import get_openai_callback
except ImportError:
    try:
        from langchain_openai import get_openai_callback
    except ImportError:
        # Fallback - define a dummy callback context manager
        from contextlib import contextmanager
        @contextmanager
        def get_openai_callback():
            class CallbackData:
                def __init__(self):
                    self.total_tokens = 0
                    self.prompt_tokens = 0
                    self.completion_tokens = 0
                    self.total_cost = 0
            yield CallbackData()

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    try:
        from langchain_core.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.chains import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        # Fallback for older versions
        create_stuff_documents_chain = None
        create_retrieval_chain = None
        ChatPromptTemplate = None

from document_loader import load_legal_knowledge_base
from legal_chunker import create_legal_chunks
from vector_store import VectorStoreManager, HybridRetriever
from citation_extractor import CitationExtractor, SourceTracker
from config import *

# Import Phase 1 enhancements
from reranker import LegalReranker
from response_validator import LegalResponseValidator
from evaluator import RAGEvaluator
from semantic_cache import SemanticCache

# Import tiktoken for token counting
try:
    import tiktoken
    TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
except:
    TOKENIZER = None

# Maximum context tokens (default if not in config)
try:
    MAX_CONTEXT = MAX_CONTEXT_TOKENS
except:
    MAX_CONTEXT = 12000


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    return len(text) // 4  # Approximate


def limit_context(documents: List[Document], max_tokens: int = MAX_CONTEXT) -> List[Document]:
    """Limit the context to a maximum number of tokens."""
    if not documents:
        return documents
    
    limited_docs = []
    total_tokens = 0
    
    for doc in documents:
        doc_tokens = count_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_tokens:
            # Truncate this document if partial fit
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 200:  # Worth including partial
                # Truncate content
                chars_to_keep = remaining_tokens * 4  # Approximate
                truncated_content = doc.page_content[:chars_to_keep] + "..."
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                )
                limited_docs.append(truncated_doc)
            break
        limited_docs.append(doc)
        total_tokens += doc_tokens
    
    return limited_docs


class LegalRAGEngine:
    """
    Advanced RAG Engine for legal document Q&A.
    Orchestrates the full pipeline from document loading to response generation.
    """
    
    def __init__(self, 
                 pdf_directory: str = PDF_DIRECTORY,
                 pdf_files: List[str] = SUPPORTED_PDFS,
                 model_name: str = OPENAI_MODEL,
                 embedding_model: str = EMBEDDING_MODEL,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG engine.
        
        Args:
            pdf_directory: Directory containing PDF files
            pdf_files: List of PDF filenames to load
            model_name: OpenAI model for generation
            embedding_model: OpenAI embedding model
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.pdf_directory = pdf_directory
        self.pdf_files = pdf_files
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.vector_store_manager = VectorStoreManager(embedding_model=embedding_model)
        self.citation_extractor = CitationExtractor()
        self.source_tracker = SourceTracker()
        
        # Phase 1 Enhancement Components
        self.reranker = None
        self.validator = None
        self.evaluator = None
        self.cache = None
        
        # Initialize Phase 1 components based on config
        if USE_RERANKER:
            try:
                self.reranker = LegalReranker(
                    model_name=RERANKER_MODEL,
                    use_reranker=True,
                    citation_boost=LEGAL_CITATION_BOOST
                )
                print("✓ Reranker initialized")
            except Exception as e:
                print(f"Warning: Could not initialize reranker: {e}")
        
        if USE_RESPONSE_VALIDATION:
            try:
                self.validator = LegalResponseValidator(
                    llm_model=VALIDATION_MODEL,
                    strict_mode=STRICT_VALIDATION
                )
                print("✓ Response validator initialized")
            except Exception as e:
                print(f"Warning: Could not initialize validator: {e}")
        
        if USE_EVALUATION:
            try:
                self.evaluator = RAGEvaluator(llm_model=EVALUATION_MODEL)
                print("✓ Evaluator initialized")
            except Exception as e:
                print(f"Warning: Could not initialize evaluator: {e}")
        
        if USE_SEMANTIC_CACHE:
            try:
                self.cache = SemanticCache(
                    embedding_model=embedding_model,
                    similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
                    max_cache_size=CACHE_MAX_SIZE,
                    ttl_hours=CACHE_TTL_HOURS,
                    persist_path=CACHE_PERSIST_PATH
                )
                print("✓ Semantic cache initialized")
            except Exception as e:
                print(f"Warning: Could not initialize cache: {e}")
        
        # Will be populated during initialization
        self.retriever = None
        self.qa_chain = None
        
        # Initialize flag
        self.is_initialized = False
    
    def initialize(self, force_reload: bool = False) -> bool:
        """
        Initialize the RAG engine by loading documents and creating vector store.
        
        Args:
            force_reload: Force reload of documents even if vector store exists
        
        Returns:
            True if successful
        """
        try:
            # Try to load existing vector store
            if not force_reload:
                vectorstore = self.vector_store_manager.load_vectorstore()
                if vectorstore:
                    print("Loaded existing vector store")
                    self.retriever = self.vector_store_manager.get_retriever(
                        search_type="mmr",
                        k=TOP_K_RETRIEVAL,
                        lambda_mult=0.5
                    )
                    self._create_qa_chain()
                    self.is_initialized = True
                    return True
            
            # Load documents
            print("Loading legal documents...")
            documents = load_legal_knowledge_base(self.pdf_directory, self.pdf_files)
            
            if not documents:
                print("Error: No documents loaded")
                return False
            
            # Chunk documents
            print("Chunking documents...")
            chunks = create_legal_chunks(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            print(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Create vector store
            print("Creating embeddings and vector store...")
            self.vector_store_manager.create_vectorstore(chunks)
            
            # Create retriever
            self.retriever = self.vector_store_manager.get_retriever(
                search_type="mmr",
                k=TOP_K_RETRIEVAL,
                lambda_mult=0.5
            )
            
            # Create QA chain
            self._create_qa_chain()
            
            self.is_initialized = True
            print("RAG Engine initialized successfully!")
            return True
        
        except Exception as e:
            print(f"Error initializing RAG engine: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt."""
        
        if create_retrieval_chain is not None and ChatPromptTemplate is not None:
            # Modern approach for newer versions
            system_prompt = """You are a seasoned Michigan criminal defense and civil litigation attorney providing legal guidance.

Your role is to provide accurate, professionally-formatted legal information based ONLY on the provided context from official legal documents.

RESPONSE FORMAT REQUIREMENTS:
You MUST structure EVERY response using this exact format:

## **Legal Assessment: [Brief Title Based on Query]**

**IMPORTANT:** [One sentence highlighting urgency or key consideration]

---

### **[Relevant Legal Category - e.g., "Applicable Criminal Charges" or "Relevant Legal Standards"]**

[For each applicable law/rule, provide:]
**1. [Statute/Rule Name] ([Citation - e.g., MCL 750.xxx or MCR x.xxx])**
- [Explanation of what the law covers]
- Per [Jury Instruction Citation, e.g., M Crim JI xx.x], [relevant legal standard or definition]
- [How it applies to the user's situation]

[Repeat for additional applicable laws]

---

### **Protective Measures / Procedural Options**

[If applicable, explain available legal remedies:]
- **[Option Name]** under **[Citation]**: [Brief explanation]
- Filing process and requirements
- What relief it provides

---

### **Recommended Next Steps**

1. **[Action Item]**
   - [Specific details and instructions]

2. **[Action Item]**
   - [Specific details and instructions]

[Continue as needed]

---

### **Summary**

[2-3 sentence summary of key points and urgency]

---

*This information is based on Michigan Model Criminal Jury Instructions, Michigan Compiled Laws, and Michigan Court Rules. It is intended as general legal guidance and does not constitute formal legal advice. You should consult with a licensed Michigan attorney regarding your specific circumstances.*

CRITICAL RULES:
1. ONLY answer questions related to Michigan law, Federal Rules of Criminal Procedure, Federal Rules of Civil Procedure, Federal Rules of Evidence, and Michigan Court Rules
2. Base your answers STRICTLY on the provided context
3. If the context doesn't contain relevant information, say "I don't have sufficient information in the available legal documents to fully address this question."
4. For non-legal questions, respond with: "{non_legal_response}"
5. ALWAYS cite specific statutes (MCL), court rules (MCR), jury instructions (M Crim JI / M Civ JI), or federal rules with precise numbers
6. Use DEFINITIVE language: "Michigan law permits..." or "You may be eligible..." NOT "You might have grounds..."
7. Maintain a confident, precise, client-focused attorney tone
8. Include practical, actionable next steps in every response
9. When multiple legal provisions apply, address each one separately with its own citation

Context from legal documents:
{{context}}

Question: {{input}}

Provide your response following the EXACT format specified above:"""

            prompt = ChatPromptTemplate.from_template(system_prompt.format(
                non_legal_response=NON_LEGAL_RESPONSE,
                context="{context}",
                input="{input}"
            ))
            
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            self.qa_chain = create_retrieval_chain(self.retriever, document_chain)
        else:
            # Fallback: use simple runnable chain
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            
            template = """You are a seasoned Michigan criminal defense and civil litigation attorney providing legal guidance.

RESPONSE FORMAT - Structure your response with:
## **Legal Assessment: [Title]**
### **Applicable Legal Standards** (with MCL, MCR, M Crim JI citations)
### **Protective Measures / Options** (if applicable)
### **Recommended Next Steps** (numbered, actionable)
### **Summary**

CONTEXT:
{context}

QUESTION: {question}

Provide your response in the professional attorney format specified above, with precise Michigan legal citations:"""
            
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # Create a simple RAG chain
            self.qa_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
    
    def query(self, 
             question: str,
             return_sources: bool = True,
             return_citations: bool = True) -> Dict:
        """
        Query the RAG engine with Phase 1 enhancements.
        
        Args:
            question: User question
            return_sources: Whether to return source documents
            return_citations: Whether to extract and return citations
        
        Returns:
            Dictionary with answer, sources, citations, and metadata
        """
        if not self.is_initialized:
            return {
                "error": "RAG Engine not initialized. Call initialize() first.",
                "answer": "",
                "sources": [],
                "citations": []
            }
        
        # PHASE 1: Check semantic cache first
        if self.cache:
            cached_response = self.cache.get(question)
            if cached_response:
                print("✓ Cache hit - returning cached response")
                return cached_response
        
        # Check for harmful queries first
        if self._is_harmful_query(question):
            return {
                "answer": "I cannot assist with questions about committing crimes or evading law enforcement. If you are in crisis, please contact emergency services or a mental health hotline.",
                "sources": [],
                "citations": [],
                "is_legal": False,
                "is_harmful": True
            }
        
        # Check if question is legal-related
        if not self._is_legal_query(question):
            return {
                "answer": NON_LEGAL_RESPONSE,
                "sources": [],
                "citations": [],
                "is_legal": False
            }
        
        try:
            # Track OpenAI API usage
            with get_openai_callback() as cb:
                # PHASE 1 ENHANCED RETRIEVAL PIPELINE
                
                # Step 1: Retrieve more documents for reranking
                initial_k = INITIAL_RETRIEVAL_K if self.reranker else TOP_K_RETRIEVAL
                retrieval_kwargs = {"k": initial_k, "lambda_mult": MMR_LAMBDA}
                
                # Update retriever if needed
                temp_retriever = self.vector_store_manager.get_retriever(
                    search_type="mmr",
                    k=initial_k,
                    lambda_mult=MMR_LAMBDA
                )
                
                source_documents = temp_retriever.invoke(question)
                print(f"Retrieved {len(source_documents)} initial documents")
                
                # Step 2: RERANKING - Improve relevance by 15-30%
                reranker_metadata = {}
                if self.reranker:
                    source_documents, reranker_metadata = self.reranker.rerank_with_metadata(
                        question, 
                        source_documents, 
                        top_k=RERANKER_TOP_K
                    )
                    print(f"✓ Reranked to {len(source_documents)} most relevant documents")
                    if reranker_metadata.get("order_changed"):
                        print(f"  Position changes: {reranker_metadata.get('position_changes', 0)}")
                
                # Step 3: Limit context to avoid token overflow
                source_documents = limit_context(source_documents, MAX_CONTEXT)
                
                # Step 4: Format context
                context_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])
                
                # Step 5: Create prompt and get response
                system_prompt = f"""You are a seasoned Michigan criminal defense and civil litigation attorney providing legal guidance.

Your role is to provide accurate, professionally-formatted legal information based ONLY on the provided context from official legal documents.

RESPONSE FORMAT REQUIREMENTS:
You MUST structure EVERY response using this exact format:

## **Legal Assessment: [Brief Title Based on Query]**

**IMPORTANT:** [One sentence highlighting urgency or key consideration based on the severity of the situation]

---

### **[Relevant Legal Category - e.g., "Applicable Criminal Charges" or "Relevant Legal Standards"]**

[For each applicable law/rule, provide:]
**1. [Statute/Rule Name] ([Citation - e.g., MCL 750.xxx or MCR x.xxx])**
- [Explanation of what the law covers]
- Per [Jury Instruction Citation, e.g., M Crim JI xx.x], [relevant legal standard or definition]
- [How it applies to the user's situation]

[Repeat for additional applicable laws]

---

### **Protective Measures / Procedural Options**

[If applicable, explain available legal remedies such as PPOs, motions, filings:]
- **[Option Name]** under **[Citation]**: [Brief explanation]
- Filing process and requirements under Michigan Court Rules
- What relief it provides

---

### **Recommended Next Steps**

1. **[Action Item]**
   - [Specific details and instructions]

2. **[Action Item]**
   - [Specific details and instructions]

3. **[Action Item]**
   - [Specific details and instructions]

[Continue as needed - be practical and actionable]

---

### **Summary**

[2-3 sentence summary emphasizing key legal provisions and recommended actions. If situation is urgent/serious, convey appropriate urgency.]

---

*This information is based on Michigan Model Criminal Jury Instructions, Michigan Compiled Laws, and Michigan Court Rules. It is intended as general legal guidance and does not constitute formal legal advice. You should consult with a licensed Michigan attorney regarding your specific circumstances.*

CRITICAL RULES:
1. ONLY answer questions related to Michigan law, Federal Rules, and Michigan Court Rules
2. Base your answers STRICTLY on the provided context - do not invent laws or citations
3. If the context doesn't contain relevant information, acknowledge the limitation clearly
4. ALWAYS cite specific statutes (MCL), court rules (MCR), jury instructions (M Crim JI / M Civ JI), or federal rules with PRECISE numbers
5. Use DEFINITIVE language: "Michigan law permits..." or "You may be eligible..." NOT "You might have grounds..." or "could potentially"
6. Maintain a confident, precise, client-focused attorney tone throughout
7. Include practical, actionable next steps in EVERY response
8. When the situation involves violence, threats, or urgent safety concerns, convey appropriate urgency

Context from legal documents:
{context_text}

Question: {question}

Provide your response following the EXACT format specified above:"""
                
                # Call LLM directly
                response = self.llm.invoke(system_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # PHASE 1: RESPONSE VALIDATION - Prevent hallucinations
                validation_result = {}
                if self.validator:
                    print("✓ Validating response...")
                    validation_result = self.validator.validate_with_fallback(
                        answer, 
                        source_documents, 
                        question
                    )
                    
                    # Handle validation failures
                    if not validation_result.get("validation_passed", True):
                        print(f"⚠ Validation warning: {validation_result.get('recommendation', 'FLAG')}")
                        
                        # Auto-correct if enabled
                        if AUTO_CORRECT_RESPONSES and validation_result.get("recommendation") != "PASS":
                            correction = self.validator.handle_validation_failure(
                                answer, validation_result, source_documents
                            )
                            if correction.get("action") == "CORRECTED":
                                answer = correction["corrected_answer"]
                                print("✓ Response auto-corrected")
                
                # Extract citations
                citations = []
                if return_citations:
                    answer, citations = self.citation_extractor.enrich_response_with_citations(
                        answer, 
                        source_documents
                    )
                
                # Format sources
                formatted_sources = []
                if return_sources and source_documents:
                    formatted_sources = self.source_tracker.get_source_metadata(source_documents)
                
                # Build response
                result = {
                    "answer": answer,
                    "sources": formatted_sources,
                    "citations": [c.__dict__ for c in citations],
                    "is_legal": True,
                    "num_sources": len(source_documents),
                    "usage": {
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost": cb.total_cost
                    }
                }
                
                # Add Phase 1 metadata
                if reranker_metadata:
                    result["reranker_metadata"] = reranker_metadata
                
                if validation_result:
                    result["validation"] = {
                        "passed": validation_result.get("validation_passed", True),
                        "confidence": validation_result.get("confidence", 1.0),
                        "recommendation": validation_result.get("recommendation", "PASS")
                    }
                
                # PHASE 1: EVALUATION - Track quality metrics
                if self.evaluator:
                    try:
                        eval_result = self.evaluator.evaluate_rag_response(
                            question=question,
                            answer=answer,
                            retrieved_docs=source_documents,
                            run_generation_eval=True
                        )
                        result["evaluation"] = eval_result.get("generation_metrics", {})
                        
                        if SAVE_EVALUATION_HISTORY:
                            self.evaluator.save_evaluation_history(EVALUATION_HISTORY_PATH)
                    except Exception as e:
                        print(f"Warning: Evaluation failed: {e}")
                
                # PHASE 1: Cache the response
                if self.cache:
                    try:
                        # Cache a clean version without cache metadata
                        cache_data = {k: v for k, v in result.items() if k != "cache_hit"}
                        self.cache.set(question, cache_data)
                    except Exception as e:
                        print(f"Warning: Cache save failed: {e}")
                
                return result
        
        except Exception as e:
            print(f"Error during query: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "answer": "An error occurred while processing your query.",
                "sources": [],
                "citations": []
            }
    
    def query_with_filter(self,
                         question: str,
                         doc_type: Optional[str] = None,
                         section_number: Optional[str] = None) -> Dict:
        """
        Query with metadata filtering.
        
        Args:
            question: User question
            doc_type: Filter by document type
            section_number: Filter by section number
        
        Returns:
            Query result dictionary
        """
        if not self.is_initialized:
            return {
                "error": "RAG Engine not initialized.",
                "answer": "",
                "sources": [],
                "citations": []
            }
        
        # Create filter dictionary
        filter_dict = {}
        if doc_type:
            filter_dict['doc_type'] = doc_type
        if section_number:
            filter_dict['section_number'] = section_number
        
        # Create filtered retriever
        filtered_retriever = self.vector_store_manager.get_retriever(
            search_type="mmr",
            k=TOP_K_RETRIEVAL,
            filter_dict=filter_dict if filter_dict else None
        )
        
        # Store original retriever and update for this query
        old_retriever = self.retriever
        self.retriever = filtered_retriever
        
        # Recreate QA chain with new retriever
        self._create_qa_chain()
        
        # Run query
        result = self.query(question)
        
        # Restore original retriever and chain
        self.retriever = old_retriever
        self._create_qa_chain()
        
        return result
    
    def _is_harmful_query(self, question: str) -> bool:
        """
        Detect queries about committing crimes, escaping justice, or harmful activities.
        
        Returns:
            True if query appears to be about harmful/illegal activities
        """
        question_lower = question.lower()
        
        # Patterns indicating harmful intent (admitting to or planning crimes)
        harmful_patterns = [
            # Admissions of violence
            ('killed', ['escape', 'hide', 'run', 'away', 'avoid', 'evade']),
            ('murder', ['escape', 'hide', 'run', 'away', 'avoid', 'evade', 'how to']),
            ('shot', ['escape', 'hide', 'run', 'avoid', 'evade']),
            ('stabbed', ['escape', 'hide', 'run', 'avoid', 'evade']),
            ('assault', ['escape', 'hide', 'avoid', 'evade']),
            # Planning crimes
            ('how to kill', []),
            ('how to murder', []),
            ('get away with', ['crime', 'murder', 'killing']),
            ('escape police', []),
            ('escape arrest', []),
            ('hide body', []),
            ('destroy evidence', []),
            ('flee jurisdiction', []),
        ]
        
        for primary, secondary_list in harmful_patterns:
            if primary in question_lower:
                if not secondary_list:  # No secondary required
                    return True
                for secondary in secondary_list:
                    if secondary in question_lower:
                        return True
        
        return False
    
    def _is_legal_query(self, question: str) -> bool:
        """
        Determine if query is a legitimate legal question.
        Returns False for harmful queries or non-legal questions.
        """
        # First check for harmful content
        if self._is_harmful_query(question):
            return False
        
        legal_keywords = [
            'law', 'legal', 'court', 'rule', 'case', 'trial', 'judge', 'jury',
            'evidence', 'procedure', 'criminal', 'civil', 'motion', 'appeal',
            'defendant', 'plaintiff', 'prosecution', 'defense', 'attorney',
            'statute', 'regulation', 'jurisdiction', 'verdict', 'sentence',
            'rights', 'liability', 'contract', 'tort', 'felony', 'misdemeanor',
            'michigan', 'federal', 'mcr', 'frcp', 'frcrp', 'fre',
            'charged', 'accused', 'arrest', 'bail', 'plea', 'hearing'
        ]
        
        question_lower = question.lower()
        
        # If question contains any legal keyword, consider it legal
        for keyword in legal_keywords:
            if keyword in question_lower:
                return True
        
        # If question mark and mentions specific legal terms, likely legal
        if '?' in question and any(term in question_lower for term in ['what', 'how', 'when', 'where', 'who', 'which']):
            return True
        
        return False
    
    def get_relevant_documents(self, question: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents without generating an answer.
        
        Args:
            question: Query
            k: Number of documents to retrieve
        
        Returns:
            List of relevant documents
        """
        if not self.is_initialized:
            return []
        
        # Update retriever k value
        retriever = self.vector_store_manager.get_retriever(search_type="mmr", k=k)
        # Use invoke instead of deprecated get_relevant_documents
        return retriever.invoke(question)
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG engine."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        stats = self.vector_store_manager.get_collection_stats()
        stats['model'] = self.model_name
        stats['embedding_model'] = self.embedding_model
        stats['chunk_size'] = self.chunk_size
        stats['status'] = 'initialized'
        
        # Add Phase 1 statistics
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        if self.evaluator:
            stats['evaluation'] = self.evaluator.get_average_metrics()
        
        if self.reranker:
            stats['reranker_enabled'] = True
        
        if self.validator:
            stats['validation_enabled'] = True
        
        return stats
    
    def clear_cache(self):
        """Clear the semantic cache."""
        if self.cache:
            self.cache.clear()
            print("✓ Cache cleared")
        else:
            print("Cache not enabled")
    
    def save_cache(self):
        """Manually save cache to disk."""
        if self.cache:
            self.cache.save_cache()
        else:
            print("Cache not enabled")
    
    def get_evaluation_summary(self, last_n: Optional[int] = None) -> Dict:
        """
        Get evaluation summary.
        
        Args:
            last_n: Number of recent evaluations to summarize
        
        Returns:
            Evaluation summary
        """
        if self.evaluator:
            return self.evaluator.get_average_metrics(last_n)
        else:
            return {"error": "Evaluator not enabled"}


# Global engine instance
_engine_instance = None


def get_rag_engine(force_reload: bool = False) -> LegalRAGEngine:
    """
    Get or create the global RAG engine instance.
    
    Args:
        force_reload: Force reload of documents
    
    Returns:
        Initialized LegalRAGEngine instance
    """
    global _engine_instance
    
    if _engine_instance is None:
        _engine_instance = LegalRAGEngine()
        _engine_instance.initialize(force_reload=force_reload)
    elif force_reload:
        _engine_instance.initialize(force_reload=True)
    
    return _engine_instance
# this was the rag engine wher ive attached all the properties of the RAG and build the proper flow of the rag