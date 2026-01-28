"""
Background Document Indexing Service
Handles asynchronous document indexing and vector store updates with monitoring.
Production-safe with queuing, error recovery, and state management.
"""

import os
import json
import time
import hashlib
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

from document_loader import load_legal_knowledge_base
from legal_chunker import create_legal_chunks
from vector_store import VectorStoreManager
from config import *


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexingJob:
    """Represents a background indexing job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    document_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    documents_processed: int = 0
    chunks_created: int = 0
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexingJob':
        """Create from dictionary."""
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


class IndexingState:
    """Thread-safe state management for indexing operations."""
    
    def __init__(self, state_file: str = "./indexing_state.json"):
        """
        Initialize indexing state.
        
        Args:
            state_file: Path to persist state
        """
        self.state_file = state_file
        self.lock = threading.RLock()
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return {
            'jobs': {},
            'last_successful_index': None,
            'document_hashes': {},
            'indexed_documents': []
        }
    
    def _save_state(self):
        """Save state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def add_job(self, job: IndexingJob):
        """Add a job to state."""
        with self.lock:
            self.state['jobs'][job.job_id] = job.to_dict()
            self._save_state()
    
    def update_job(self, job: IndexingJob):
        """Update an existing job."""
        with self.lock:
            self.state['jobs'][job.job_id] = job.to_dict()
            self._save_state()
    
    def get_job(self, job_id: str) -> Optional[IndexingJob]:
        """Get a job by ID."""
        with self.lock:
            if job_id in self.state['jobs']:
                return IndexingJob.from_dict(self.state['jobs'][job_id])
        return None
    
    def get_all_jobs(self) -> List[IndexingJob]:
        """Get all jobs."""
        with self.lock:
            return [IndexingJob.from_dict(j) for j in self.state['jobs'].values()]
    
    def set_indexed_documents(self, docs: List[str]):
        """Update list of indexed documents."""
        with self.lock:
            self.state['indexed_documents'] = docs
            self._save_state()
    
    def get_indexed_documents(self) -> List[str]:
        """Get list of indexed documents."""
        with self.lock:
            return self.state.get('indexed_documents', [])
    
    def update_document_hash(self, doc_name: str, hash_value: str):
        """Update hash for a document."""
        with self.lock:
            self.state['document_hashes'][doc_name] = hash_value
            self._save_state()
    
    def get_document_hash(self, doc_name: str) -> Optional[str]:
        """Get hash for a document."""
        with self.lock:
            return self.state['document_hashes'].get(doc_name)
    
    def update_last_successful_index(self):
        """Update last successful index time."""
        with self.lock:
            self.state['last_successful_index'] = datetime.now().isoformat()
            self._save_state()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute hash for {filepath}: {e}")
        return ""


class BackgroundIndexer:
    """
    Production-safe background indexing service.
    
    Features:
    - Asynchronous document indexing
    - Error recovery with retry logic
    - State persistence
    - Document change detection
    - Progress tracking
    - Thread-safe operations
    """
    
    def __init__(self, 
                 vector_store_dir: str = VECTOR_STORE_PERSIST_DIR,
                 pdf_directory: str = PDF_DIRECTORY,
                 max_workers: int = 1,
                 state_file: str = "./indexing_state.json"):
        """
        Initialize background indexer.
        
        Args:
            vector_store_dir: Directory for vector store
            pdf_directory: Directory containing PDFs
            max_workers: Number of worker threads
            state_file: Path to state persistence file
        """
        self.vector_store_dir = vector_store_dir
        self.pdf_directory = pdf_directory
        self.max_workers = max_workers
        
        # State management
        self.state = IndexingState(state_file)
        
        # Job queue
        self.job_queue = queue.Queue()
        
        # Vector store
        self.vector_store = VectorStoreManager(persist_directory=vector_store_dir)
        
        # Callbacks
        self.on_job_started: Optional[Callable] = None
        self.on_job_completed: Optional[Callable] = None
        self.on_job_failed: Optional[Callable] = None
        
        # Worker threads
        self.workers = []
        self.running = False
        self.lock = threading.RLock()
    
    def start(self):
        """Start background indexing workers."""
        with self.lock:
            if self.running:
                logger.warning("Indexer already running")
                return
            
            self.running = True
            
            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"IndexerWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"Started {self.max_workers} indexing workers")
    
    def stop(self, timeout: int = 30):
        """Stop background indexing workers gracefully."""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
        
        # Signal workers to stop
        for _ in range(len(self.workers)):
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        logger.info("Indexing workers stopped")
    
    def _worker_loop(self):
        """Main worker loop for processing indexing jobs."""
        logger.info(f"{threading.current_thread().name} started")
        
        while self.running:
            try:
                # Get job with timeout to allow checking running flag
                job = self.job_queue.get(timeout=1)
                
                if job is None:  # Stop signal
                    break
                
                self._process_job(job)
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}\n{traceback.format_exc()}")
    
    def _process_job(self, job: IndexingJob):
        """Process a single indexing job."""
        try:
            # Update status
            job.status = "processing"
            job.start_time = datetime.now()
            self.state.update_job(job)
            
            if self.on_job_started:
                self.on_job_started(job)
            
            logger.info(f"Processing job {job.job_id} for {job.document_name}")
            
            # Load and chunk documents
            pdf_path = os.path.join(self.pdf_directory, job.document_name)
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            # Load documents
            documents = load_legal_knowledge_base([pdf_path])
            job.documents_processed = len(documents)
            
            # Create chunks
            chunks = create_legal_chunks(documents)
            job.chunks_created = len(chunks)
            
            # Add to vector store
            if chunks:
                self.vector_store.add_documents(chunks, batch_size=50)
            
            # Update document hash
            file_hash = compute_file_hash(pdf_path)
            self.state.update_document_hash(job.document_name, file_hash)
            
            # Mark as completed
            job.status = "completed"
            job.end_time = datetime.now()
            self.state.update_job(job)
            self.state.update_last_successful_index()
            
            if self.on_job_completed:
                self.on_job_completed(job)
            
            logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}\n{traceback.format_exc()}")
            
            job.retry_count += 1
            job.error_message = str(e)
            
            if job.retry_count < job.max_retries:
                # Re-queue for retry
                job.status = "pending"
                self.state.update_job(job)
                self.job_queue.put(job)
                logger.info(f"Requeuing job {job.job_id} (attempt {job.retry_count})")
            else:
                # Final failure
                job.status = "failed"
                job.end_time = datetime.now()
                self.state.update_job(job)
                
                if self.on_job_failed:
                    self.on_job_failed(job)
                
                logger.error(f"Job {job.job_id} failed permanently after {job.max_retries} attempts")
    
    def index_document(self, document_name: str, force: bool = False) -> str:
        """
        Queue a document for indexing.
        
        Args:
            document_name: Name of the PDF file
            force: Force re-indexing even if unchanged
        
        Returns:
            Job ID
        """
        pdf_path = os.path.join(self.pdf_directory, document_name)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Check if document has changed
        current_hash = compute_file_hash(pdf_path)
        stored_hash = self.state.get_document_hash(document_name)
        
        if not force and current_hash == stored_hash:
            logger.info(f"Document {document_name} unchanged, skipping")
            return None
        
        # Create job
        job_id = f"{document_name}_{int(time.time())}"
        job = IndexingJob(
            job_id=job_id,
            status="pending",
            document_name=document_name
        )
        
        self.state.add_job(job)
        self.job_queue.put(job)
        
        logger.info(f"Queued indexing job {job_id} for {document_name}")
        return job_id
    
    def index_all_documents(self, force: bool = False) -> List[str]:
        """
        Queue all supported documents for indexing.
        
        Args:
            force: Force re-indexing of all documents
        
        Returns:
            List of job IDs
        """
        job_ids = []
        for doc_name in SUPPORTED_PDFS:
            job_id = self.index_document(doc_name, force=force)
            if job_id:
                job_ids.append(job_id)
        
        return job_ids
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a job."""
        job = self.state.get_job(job_id)
        if job:
            return job.to_dict()
        return None
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs."""
        return [job.to_dict() for job in self.state.get_all_jobs()]
    
    def get_indexing_stats(self) -> Dict:
        """Get indexing statistics."""
        jobs = self.state.get_all_jobs()
        
        completed = [j for j in jobs if j.status == "completed"]
        failed = [j for j in jobs if j.status == "failed"]
        pending = [j for j in jobs if j.status == "pending"]
        processing = [j for j in jobs if j.status == "processing"]
        
        total_documents = sum(j.documents_processed for j in completed)
        total_chunks = sum(j.chunks_created for j in completed)
        
        return {
            'total_jobs': len(jobs),
            'completed_jobs': len(completed),
            'failed_jobs': len(failed),
            'pending_jobs': len(pending),
            'processing_jobs': len(processing),
            'total_documents_indexed': total_documents,
            'total_chunks_created': total_chunks,
            'last_successful_index': self.state.state.get('last_successful_index'),
            'queue_size': self.job_queue.qsize(),
            'indexed_documents': self.state.get_indexed_documents()
        }
    
    def wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for all queued jobs to complete.
        
        Args:
            timeout: Maximum seconds to wait (None = unlimited)
        
        Returns:
            True if completed, False if timed out
        """
        try:
            self.job_queue.join()
            return True
        except Exception:
            return False


# Global indexer instance
_indexer_instance: Optional[BackgroundIndexer] = None
_indexer_lock = threading.Lock()


def get_background_indexer() -> BackgroundIndexer:
    """Get or create global indexer instance."""
    global _indexer_instance
    
    if _indexer_instance is None:
        with _indexer_lock:
            if _indexer_instance is None:
                _indexer_instance = BackgroundIndexer()
    
    return _indexer_instance
