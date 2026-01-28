#!/usr/bin/env python3
"""
Production Initialization Script
Sets up and validates the Michigan Legal RAG Chatbot for production deployment.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """Check environment variables."""
    logger.info("Checking environment variables...")
    
    required = ['OPENAI_API_KEY']
    missing = []
    
    for var in required:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.error("Please set OPENAI_API_KEY in .env file or environment")
        return False
    
    logger.info("✓ Environment variables valid")
    return True


def check_directories():
    """Check and create required directories."""
    logger.info("Checking directory structure...")
    
    required_dirs = [
        'knowledge_base',
        'chroma_db',
        'logs'
    ]
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if not path.exists():
            logger.info(f"Creating directory: {dir_name}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"✓ Directory exists: {dir_name}")
    
    return True


def check_dependencies():
    """Check Python dependencies."""
    logger.info("Checking Python dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'langchain',
        'chromadb',
        'openai',
        'pydantic'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    return True


def check_pdf_files():
    """Check for required PDF files."""
    logger.info("Checking PDF files...")
    
    from config import SUPPORTED_PDFS, PDF_DIRECTORY
    
    found = []
    missing = []
    
    for pdf in SUPPORTED_PDFS:
        path = Path(PDF_DIRECTORY) / pdf
        if path.exists():
            found.append(pdf)
            logger.info(f"✓ Found: {pdf}")
        else:
            missing.append(pdf)
            logger.warning(f"✗ Missing: {pdf}")
    
    if not found:
        logger.error("No PDF files found in knowledge_base/")
        logger.error("Add PDF files to knowledge_base/ before starting")
        return False
    
    logger.info(f"Found {len(found)}/{len(SUPPORTED_PDFS)} PDF files")
    
    if missing:
        logger.warning(f"Missing: {len(missing)} PDF files")
        logger.info("These can be added later for expanded functionality")
    
    return len(found) > 0


def check_vector_store():
    """Check vector store status."""
    logger.info("Checking vector store...")
    
    chroma_path = Path('chroma_db')
    
    if chroma_path.exists() and list(chroma_path.iterdir()):
        logger.info("✓ Vector store exists (will be used)")
        return True
    else:
        logger.info("Vector store empty - will be created on first index")
        return True


def validate_config():
    """Validate configuration file."""
    logger.info("Validating configuration...")
    
    try:
        from config import (
            OPENAI_MODEL,
            EMBEDDING_MODEL,
            CHUNK_SIZE,
            TOP_K_RETRIEVAL,
            MAX_CONTEXT_TOKENS
        )
        
        logger.info(f"  Model: {OPENAI_MODEL}")
        logger.info(f"  Embedding: {EMBEDDING_MODEL}")
        logger.info(f"  Chunk Size: {CHUNK_SIZE} tokens")
        logger.info(f"  Top K: {TOP_K_RETRIEVAL}")
        logger.info(f"  Max Context: {MAX_CONTEXT_TOKENS} tokens")
        
        return True
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return False


def initialize_state():
    """Initialize indexing state file."""
    logger.info("Initializing state file...")
    
    state_file = Path('indexing_state.json')
    
    if not state_file.exists():
        initial_state = {
            'jobs': {},
            'last_successful_index': None,
            'document_hashes': {},
            'indexed_documents': []
        }
        
        with open(state_file, 'w') as f:
            json.dump(initial_state, f, indent=2)
        
        logger.info("✓ State file created")
    else:
        logger.info("✓ State file exists")
    
    return True


def test_openai_connection():
    """Test OpenAI API connection."""
    logger.info("Testing OpenAI API connection...")
    
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY
        
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set")
            return False
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Test with minimal request
        response = client.models.list()
        logger.info(f"✓ OpenAI connection successful")
        logger.info(f"  Available models: {len(response.data)} models")
        
        return True
    except Exception as e:
        logger.error(f"OpenAI connection failed: {e}")
        return False


def run_all_checks():
    """Run all validation checks."""
    logger.info("=" * 60)
    logger.info("Michigan Legal RAG Chatbot - Production Setup")
    logger.info("=" * 60)
    
    checks = [
        ("Environment Variables", check_environment),
        ("Directory Structure", check_directories),
        ("Python Dependencies", check_dependencies),
        ("Configuration", validate_config),
        ("State Initialization", initialize_state),
        ("PDF Files", check_pdf_files),
        ("Vector Store", check_vector_store),
        ("OpenAI API", test_openai_connection),
    ]
    
    results = []
    
    for name, check_fn in checks:
        logger.info(f"\n{name}:")
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary:")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n✓ All checks passed! System is ready for production.")
        return True
    else:
        logger.error(f"\n✗ {total - passed} checks failed. Please fix before deploying.")
        return False


def main():
    """Main entry point."""
    try:
        success = run_all_checks()
        
        if success:
            logger.info("\nNext steps:")
            logger.info("1. Start background indexing: POST /indexing/start")
            logger.info("2. Monitor health: GET /health")
            logger.info("3. Check detailed status: GET /health/detailed")
            logger.info("\nRun: uvicorn app:app --host 0.0.0.0 --port 8000")
            return 0
        else:
            logger.error("\nPlease fix the above issues before starting the application.")
            return 1
    
    except KeyboardInterrupt:
        logger.info("\nSetup cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
