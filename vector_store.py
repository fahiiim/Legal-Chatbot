"""
Vector Store Manager with FAISS
Handles embedding storage, retrieval, and metadata filtering.
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        from langchain_core.retrievers import ContextualCompressionRetriever
        from langchain_core.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        ContextualCompressionRetriever = None
        LLMChainExtractor = None
    
from langchain_openai import ChatOpenAI


class VectorStoreManager:
    """Manages vector storage and retrieval for legal documents using FAISS."""
    
    def __init__(self, 
                 persist_directory: str = "./legal_vectors",
                 collection_name: str = "legal_documents",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.index_path = os.path.join(persist_directory, "faiss_index")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        self.metadata_store = {}
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to embed and store
        
        Returns:
            FAISS vector store instance
        """
        if FAISS is None:
            raise ImportError("FAISS not available. Install faiss-cpu: pip install faiss-cpu")
        
        print(f"Creating vector store with {len(documents)} documents...")
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Store metadata
        for i, doc in enumerate(documents):
            self.metadata_store[i] = doc.metadata.copy()
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # Save to disk
        self.vectorstore.save_local(self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        
        print(f"Vector store created and persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self):
        """
        Load existing vector store from disk.
        
        Returns:
            FAISS vector store instance or None if doesn't exist
        """
        if FAISS is None:
            raise ImportError("FAISS not available. Install faiss-cpu: pip install faiss-cpu")
        
        if not os.path.exists(self.index_path):
            print(f"No existing vector store found at {self.index_path}")
            return None
        
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            
            print(f"Vector store loaded from {self.index_path}")
            return self.vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        self.vectorstore.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
    
    def get_retriever(self, k: int = 5, filter_dict: Optional[Dict] = None, 
                       search_type: str = "similarity", lambda_mult: float = 0.5):
        """
        Get a retriever with specified parameters.
        
        Args:
            k: Number of documents to retrieve
            filter_dict: Metadata filter dictionary (not fully supported in FAISS)
            search_type: Type of search - 'similarity' or 'mmr'
            lambda_mult: Lambda multiplier for MMR search (diversity vs relevance)
        
        Returns:
            Configured retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = {"k": k}
        
        # Add MMR-specific parameters
        if search_type == "mmr":
            search_kwargs["lambda_mult"] = lambda_mult
            search_kwargs["fetch_k"] = k * 4  # Fetch more candidates for MMR
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return self.retriever
    
    def get_contextual_compression_retriever(self,
                                            base_retriever=None,
                                            llm_model: str = "gpt-4o",
                                            k: int = 5):
        """
        Get a contextual compression retriever that extracts only relevant parts.
        
        Args:
            base_retriever: Base retriever to use (if None, creates default)
            llm_model: LLM model for compression
            k: Number of documents to retrieve
        
        Returns:
            Contextual compression retriever or None if not available
        """
        if ContextualCompressionRetriever is None or LLMChainExtractor is None:
            print("Warning: Contextual compression not available. Using standard retrieval.")
            return base_retriever if base_retriever else self.get_retriever(k=k)
        
        if base_retriever is None:
            base_retriever = self.get_retriever(k=k*2)
        
        # Initialize LLM for compression
        llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Create compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters (limited support in FAISS)
        
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Metadata filters
        
        Returns:
            List of tuples (document, score)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def search_by_document_type(self, query: str, doc_type: str, k: int = 5) -> List[Document]:
        """
        Search within a specific document type (simulated).
        
        Args:
            query: Search query
            doc_type: Document type to filter by
            k: Number of results
        
        Returns:
            List of relevant documents
        """
        # FAISS doesn't support metadata filtering directly
        # This is a limitation of the free version
        results = self.similarity_search(query, k=k*2)
        
        # Filter by metadata
        filtered = []
        for doc in results:
            if doc.metadata.get('doc_type') == doc_type:
                filtered.append(doc)
                if len(filtered) >= k:
                    break
        
        return filtered if filtered else results[:k]
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        return {
            "total_documents": len(self.metadata_store),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        print(f"Deleted collection at {self.persist_directory}")


class HybridRetriever:
    """
    Hybrid retriever combining vector search with keyword search.
    """
    
    def __init__(self, vectorstore, alpha: float = 0.7):
        """
        Initialize hybrid retriever.
        
        Args:
            vectorstore: Vector store for semantic search
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        self.vectorstore = vectorstore
        self.alpha = alpha
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
        
        Returns:
            List of documents
        """
        # Get semantic search results
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # For FAISS, we'll just use semantic results with simple keyword matching
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc, score in semantic_results:
            # Simple keyword boost
            content = doc.page_content.lower()
            keyword_matches = sum(1 for term in query_terms if term in content)
            keyword_boost = keyword_matches / len(query_terms) if query_terms else 0
            
            combined_score = self.alpha * score + (1 - self.alpha) * keyword_boost
            scored_docs.append((doc, combined_score))
        
        # Sort by combined score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:k]]
