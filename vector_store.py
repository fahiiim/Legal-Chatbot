"""
Vector Store Manager with ChromaDB
Disk-based storage for memory efficiency with large document sets.
"""

import os
import gc
from typing import List, Dict, Optional, Tuple
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings

# Import ChromaDB
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        Chroma = None

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
    """
    Memory-efficient vector storage using ChromaDB with disk persistence.
    
    Key benefits:
    - Disk-based storage: No memory issues with large document sets
    - Automatic persistence: Data saved automatically to disk
    - Efficient batch processing with incremental updates
    - Built-in metadata filtering support
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
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
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
    
    def _filter_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Filter complex metadata from documents to ensure ChromaDB compatibility.
        ChromaDB only accepts str, int, float, bool, or None values.
        """
        filtered_docs = []
        for doc in documents:
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                # Skip lists, dicts, and other complex types
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered_metadata[key] = value
                elif isinstance(value, list):
                    # Convert list to comma-separated string if items are simple
                    if all(isinstance(v, (str, int, float)) for v in value):
                        filtered_metadata[key] = ', '.join(str(v) for v in value[:5])  # Limit to 5 items
                elif isinstance(value, dict):
                    # Skip complex nested dicts
                    pass
            
            filtered_docs.append(Document(
                page_content=doc.page_content,
                metadata=filtered_metadata
            ))
        return filtered_docs
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store with ChromaDB disk-based storage.
        
        Args:
            documents: List of documents to embed and store
        
        Returns:
            Chroma vector store instance
        """
        if Chroma is None:
            raise ImportError("ChromaDB not available. Install: pip install chromadb langchain-chroma")
        
        # Filter complex metadata for ChromaDB compatibility
        documents = self._filter_metadata(documents)
        
        total_docs = len(documents)
        print(f"Creating vector store with {total_docs} documents...")
        print(f"  Using ChromaDB disk-based storage")
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Use moderate batches - ChromaDB is disk-based so memory is less of a concern
        batch_size = 500
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        print(f"  Processing in {total_batches} batches of {batch_size} documents...")
        
        # Process first batch to create initial vectorstore
        first_batch = documents[:batch_size]
        
        print(f"  Creating initial index with batch 1/{total_batches}...")
        self.vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        # Clear memory
        del first_batch
        gc.collect()
        
        # Process remaining batches
        for batch_start in range(batch_size, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch_num = (batch_start // batch_size) + 1
            
            print(f"  Processing batch {batch_num}/{total_batches} (docs {batch_start}-{batch_end})...")
            
            # Get batch
            batch = documents[batch_start:batch_end]
            
            # Add to vectorstore
            try:
                self.vectorstore.add_documents(batch)
            except Exception as e:
                print(f"  ⚠ Error at batch {batch_num}: {e}")
                print(f"    Trying smaller batch size...")
                
                # Try with smaller batch
                gc.collect()
                mini_batch_size = 100
                for mini_start in range(0, len(batch), mini_batch_size):
                    mini_batch = batch[mini_start:mini_start + mini_batch_size]
                    try:
                        self.vectorstore.add_documents(mini_batch)
                    except Exception as e2:
                        print(f"  ⚠ Skipping {len(mini_batch)} documents: {e2}")
                    gc.collect()
            
            # Clean up after each batch
            del batch
            gc.collect()
        
        # Final cleanup
        gc.collect()
        
        print(f"✓ Vector store created with {total_docs} documents")
        print(f"✓ Persisted to {self.persist_directory}")
        return self.vectorstore
    
    def load_vectorstore(self):
        """
        Load existing vector store from disk.
        
        Returns:
            Chroma vector store instance or None if doesn't exist
        """
        if Chroma is None:
            raise ImportError("ChromaDB not available. Install: pip install chromadb langchain-chroma")
        
        if not os.path.exists(self.persist_directory):
            print(f"No existing vector store found at {self.persist_directory}")
            return None
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Check if collection has documents
            collection_count = self.vectorstore._collection.count()
            if collection_count == 0:
                print(f"Vector store exists but is empty at {self.persist_directory}")
                return None
            
            print(f"✓ Loaded vector store from {self.persist_directory} ({collection_count} documents)")
            return self.vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        # Filter complex metadata for ChromaDB compatibility
        documents = self._filter_metadata(documents)
        
        # Process in small batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            gc.collect()
        
        print(f"Added {len(documents)} documents to vector store")
    
    def get_retriever(self, k: int = 5, filter_dict: Optional[Dict] = None, 
                       search_type: str = "similarity", lambda_mult: float = 0.5):
        """
        Get a retriever with specified parameters.
        
        Args:
            k: Number of documents to retrieve
            filter_dict: Metadata filter dictionary for ChromaDB
            search_type: Type of search - 'similarity' or 'mmr'
            lambda_mult: Lambda multiplier for MMR search (diversity vs relevance)
        
        Returns:
            Configured retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = {"k": k}
        
        # Add filter if provided - ChromaDB supports this natively
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
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
        """
        if ContextualCompressionRetriever is None or LLMChainExtractor is None:
            print("Warning: Contextual compression not available. Using standard retrieval.")
            return base_retriever if base_retriever else self.get_retriever(k=k)
        
        if base_retriever is None:
            base_retriever = self.get_retriever(k=k*2)
        
        llm = ChatOpenAI(model=llm_model, temperature=0)
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        if filter_dict:
            results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def search_by_document_type(self, query: str, doc_type: str, k: int = 5) -> List[Document]:
        """Search within a specific document type using ChromaDB filter."""
        # ChromaDB supports native metadata filtering
        filter_dict = {"doc_type": doc_type}
        try:
            results = self.similarity_search(query, k=k, filter_dict=filter_dict)
            if results:
                return results
        except Exception:
            pass
        
        # Fallback to unfiltered search
        return self.similarity_search(query, k=k)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store collection."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = 0
        
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "backend": "ChromaDB (disk-based)"
        }
    
    def delete_collection(self):
        """Delete the entire collection."""
        import shutil
        
        # Reset ChromaDB client
        if self.vectorstore is not None:
            try:
                self.vectorstore._client.delete_collection(self.collection_name)
            except Exception:
                pass
        
        # Remove files from disk
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        self.vectorstore = None
        print(f"Deleted collection at {self.persist_directory}")


class HybridRetriever:
    """Hybrid retriever combining vector search with keyword search."""
    
    def __init__(self, vectorstore, alpha: float = 0.7):
        self.vectorstore = vectorstore
        self.alpha = alpha
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using hybrid approach."""
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc, score in semantic_results:
            content = doc.page_content.lower()
            keyword_matches = sum(1 for term in query_terms if term in content)
            keyword_boost = keyword_matches / len(query_terms) if query_terms else 0
            
            combined_score = self.alpha * score + (1 - self.alpha) * keyword_boost
            scored_docs.append((doc, combined_score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:k]]
