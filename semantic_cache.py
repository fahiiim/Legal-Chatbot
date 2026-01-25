"""
Semantic Cache
Implements intelligent caching to reduce API costs and latency.
"""

from typing import Optional, Dict, List, Tuple
import hashlib
import json
import pickle
import os
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np

try:
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: OpenAI embeddings not available. Cache will use exact string matching only.")


class SemanticCache:
    """
    Semantic cache using embedding similarity for query matching.
    Saves API costs by returning cached responses for similar queries.
    """
    
    def __init__(self,
                 embedding_model: str = "text-embedding-3-small",
                 similarity_threshold: float = 0.95,
                 max_cache_size: int = 1000,
                 ttl_hours: int = 24,
                 persist_path: Optional[str] = "./cache_data"):
        """
        Initialize semantic cache.
        
        Args:
            embedding_model: OpenAI embedding model
            similarity_threshold: Minimum similarity for cache hit (0-1)
            max_cache_size: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
            persist_path: Path to persist cache data
        """
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        self.persist_path = persist_path
        
        # Initialize embeddings
        self.embeddings_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embeddings_model = OpenAIEmbeddings(model=embedding_model)
            except Exception as e:
                print(f"Warning: Could not initialize embeddings: {e}")
        
        # Cache storage: {query_hash: cache_entry}
        self.cache = OrderedDict()
        
        # Embedding cache for faster similarity search
        self.embedding_cache = {}  # {query_hash: embedding_vector}
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0
        }
        
        # Load existing cache if available
        if persist_path:
            self.load_cache()
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        if not self.embeddings_model:
            return None
        
        try:
            embedding = self.embeddings_model.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        if "timestamp" not in entry:
            return True
        
        timestamp = datetime.fromisoformat(entry["timestamp"])
        return datetime.now() - timestamp > self.ttl
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Get cached response for query.
        
        Args:
            query: User query
        
        Returns:
            Cached response dict or None if not found
        """
        # Check exact match first (fastest)
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            
            # Check if expired
            if self._is_expired(entry):
                del self.cache[query_hash]
                if query_hash in self.embedding_cache:
                    del self.embedding_cache[query_hash]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(query_hash)
            self.stats["hits"] += 1
            
            return {
                **entry["response"],
                "cache_hit": True,
                "cache_type": "exact_match"
            }
        
        # Semantic similarity search
        if self.embeddings_model:
            query_embedding = self._get_embedding(query)
            
            if query_embedding is not None:
                best_match = None
                best_similarity = 0.0
                
                for cached_hash, cached_entry in self.cache.items():
                    # Skip expired entries
                    if self._is_expired(cached_entry):
                        continue
                    
                    # Get or compute embedding
                    if cached_hash not in self.embedding_cache:
                        cached_query = cached_entry.get("query", "")
                        self.embedding_cache[cached_hash] = self._get_embedding(cached_query)
                    
                    cached_embedding = self.embedding_cache.get(cached_hash)
                    
                    if cached_embedding is not None:
                        similarity = self._compute_similarity(query_embedding, cached_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = cached_entry
                
                # Check if best match exceeds threshold
                if best_match and best_similarity >= self.similarity_threshold:
                    self.stats["hits"] += 1
                    
                    return {
                        **best_match["response"],
                        "cache_hit": True,
                        "cache_type": "semantic_match",
                        "similarity_score": round(best_similarity, 4)
                    }
        
        # Cache miss
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, response: Dict):
        """
        Cache a response for a query.
        
        Args:
            query: User query
            response: Response dict to cache
        """
        query_hash = self._hash_query(query)
        
        # Create cache entry
        entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to cache
        self.cache[query_hash] = entry
        
        # Cache embedding for semantic search
        if self.embeddings_model:
            embedding = self._get_embedding(query)
            if embedding is not None:
                self.embedding_cache[query_hash] = embedding
        
        # Move to end (most recent)
        self.cache.move_to_end(query_hash)
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_cache_size:
            oldest_hash = next(iter(self.cache))
            del self.cache[oldest_hash]
            if oldest_hash in self.embedding_cache:
                del self.embedding_cache[oldest_hash]
            self.stats["evictions"] += 1
        
        self.stats["saves"] += 1
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.embedding_cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0
        }
    
    def remove_expired(self):
        """Remove all expired entries."""
        expired_hashes = [
            hash_key for hash_key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        
        for hash_key in expired_hashes:
            del self.cache[hash_key]
            if hash_key in self.embedding_cache:
                del self.embedding_cache[hash_key]
            self.stats["evictions"] += 1
        
        return len(expired_hashes)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 4),
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size
        }
    
    def save_cache(self, filepath: Optional[str] = None):
        """
        Persist cache to disk.
        
        Args:
            filepath: Path to save cache (uses persist_path if None)
        """
        if filepath is None:
            if self.persist_path is None:
                print("No persist path configured")
                return
            filepath = os.path.join(self.persist_path, "semantic_cache.pkl")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Remove expired before saving
        self.remove_expired()
        
        cache_data = {
            "cache": dict(self.cache),
            "embedding_cache": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in self.embedding_cache.items()},
            "stats": self.stats,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_cache_size": self.max_cache_size,
                "ttl_hours": self.ttl.total_seconds() / 3600
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cache saved to {filepath} ({len(self.cache)} entries)")
    
    def load_cache(self, filepath: Optional[str] = None):
        """
        Load cache from disk.
        
        Args:
            filepath: Path to load cache from (uses persist_path if None)
        """
        if filepath is None:
            if self.persist_path is None:
                return
            filepath = os.path.join(self.persist_path, "semantic_cache.pkl")
        
        if not os.path.exists(filepath):
            print(f"No cache file found at {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cache = OrderedDict(cache_data.get("cache", {}))
            
            # Convert embedding lists back to numpy arrays
            self.embedding_cache = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in cache_data.get("embedding_cache", {}).items()
            }
            
            self.stats = cache_data.get("stats", self.stats)
            
            # Remove expired entries
            removed = self.remove_expired()
            
            print(f"Cache loaded from {filepath} ({len(self.cache)} entries, {removed} expired removed)")
        
        except Exception as e:
            print(f"Error loading cache: {e}")


class SimpleLRUCache:
    """
    Simple LRU cache without semantic matching (fallback when embeddings unavailable).
    """
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """
        Initialize simple cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_hours: Time-to-live in hours
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.stats = {"hits": 0, "misses": 0}
    
    def _hash_query(self, query: str) -> str:
        """Hash query string."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached response."""
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            entry = self.cache[query_hash]
            
            # Check expiration
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - timestamp > self.ttl:
                del self.cache[query_hash]
                self.stats["misses"] += 1
                return None
            
            self.cache.move_to_end(query_hash)
            self.stats["hits"] += 1
            return {**entry["response"], "cache_hit": True}
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, response: Dict):
        """Cache a response."""
        query_hash = self._hash_query(query)
        
        self.cache[query_hash] = {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        self.cache.move_to_end(query_hash)
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 4),
            "cache_size": len(self.cache)
        }
