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
            system_prompt = """You are an expert legal assistant specializing in Michigan state law and federal criminal/civil procedure rules.

Your role is to provide accurate legal information based ONLY on the provided context from official legal documents.

CRITICAL RULES:
1. ONLY answer questions related to Michigan law, Federal Rules of Criminal Procedure, Federal Rules of Civil Procedure, Federal Rules of Evidence, and Michigan Court Rules
2. Base your answers STRICTLY on the provided context
3. If the context doesn't contain relevant information, say "I don't have information about that in the available legal documents"
4. For non-legal questions, respond with: "{non_legal_response}"
5. Always cite specific rules, sections, or jury instructions when providing information
6. Be precise and accurate - legal information must be correct
7. Use clear, professional legal language
8. If multiple rules apply, mention all relevant provisions
9. Include the specific citation (e.g., MCR 6.101, Fed. R. Crim. P. 12, etc.)

Context from legal documents:
{{context}}

Question: {{input}}

Provide a comprehensive answer with proper legal citations:"""

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
            
            template = """You are an expert legal assistant specializing in Michigan state law and federal criminal/civil procedure rules.

CONTEXT:
{context}

QUESTION: {question}

Based only on the provided context, answer the legal question. Include specific citations."""
            
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
        Query the RAG engine.
        
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
                # MANUAL RETRIEVAL WITH CONTEXT LIMITING
                # Step 1: Retrieve documents
                source_documents = self.retriever.invoke(question)
                
                # Step 2: Limit context to avoid token overflow
                source_documents = limit_context(source_documents, MAX_CONTEXT)
                
                # Step 3: Format context
                context_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])
                
                # Step 4: Create prompt and get response
                system_prompt = f"""You are an expert legal assistant specializing in Michigan state law and federal criminal/civil procedure rules.

Your role is to provide accurate legal information based ONLY on the provided context from official legal documents.

CRITICAL RULES:
1. ONLY answer questions related to Michigan law, Federal Rules of Criminal Procedure, Federal Rules of Civil Procedure, Federal Rules of Evidence, and Michigan Court Rules
2. Base your answers STRICTLY on the provided context
3. If the context doesn't contain relevant information, say "I don't have information about that in the available legal documents"
4. Always cite specific rules, sections, or jury instructions when providing information
5. Be precise and accurate - legal information must be correct
6. Use clear, professional legal language

Context from legal documents:
{context_text}

Question: {question}

Provide a comprehensive answer with proper legal citations:"""
                
                # Call LLM directly
                response = self.llm.invoke(system_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
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
                
                return {
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
        
        return stats


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