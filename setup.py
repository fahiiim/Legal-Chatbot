"""
Setup and Initialization Script
Initializes the Michigan Legal RAG Chatbot system.
"""

import os
import sys
from pathlib import Path


def check_pdfs():
    """Check if all required PDFs are present."""
    print("Checking for required PDF files...")
    
    pdf_dir = Path("knowledge_base")
    if not pdf_dir.exists():
        print("ERROR: knowledge_base directory not found!")
        print("Creating knowledge_base directory...")
        pdf_dir.mkdir(exist_ok=True)
        return False
    
    required_pdfs = [
        "federal-rules-of-criminal-procedure-dec-1-2024_0.pdf",
        "federal-rules-of-evidence-dec-1-2024_0.pdf",
        "federal-rules-of-civil-procedure-dec-1-2024_0.pdf",
        "criminal-jury-instructions.pdf",
        "model-civil-jury-instructions.pdf",
        "michigan-court-rules.pdf"
    ]
    
    missing = []
    for pdf in required_pdfs:
        if not (pdf_dir / pdf).exists():
            missing.append(pdf)
    
    if missing:
        print(f"WARNING: Missing {len(missing)} PDF files:")
        for pdf in missing:
            print(f"  - {pdf}")
        return False
    
    print(f"[OK] All {len(required_pdfs)} required PDFs found!")
    return True


def check_environment():
    """Check environment setup."""
    print("\nChecking environment setup...")
    
    # Check for .env file
    if not Path(".env").exists():
        print("WARNING: .env file not found!")
        print("Creating .env template...")
        with open(".env", "w") as f:
            f.write("OPENAI_API_KEY=your-api-key-here\n")
        print("Please edit .env and add your OpenAI API key")
        return False
    
    # Check for OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("WARNING: OPENAI_API_KEY not set in .env file")
        return False
    
    print("[OK] Environment configured")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain_openai",
        "faiss",
        "pdfplumber",
        "tiktoken",
        "tqdm"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"WARNING: Missing {len(missing)} packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    print("[OK] All required packages installed")
    return True


def initialize_system():
    """Initialize the RAG system."""
    print("\n" + "="*60)
    print("MICHIGAN LEGAL RAG CHATBOT - SYSTEM INITIALIZATION")
    print("="*60 + "\n")
    
    checks = {
        "PDFs": check_pdfs(),
        "Environment": check_environment(),
        "Dependencies": check_dependencies()
    }
    
    print("\n" + "="*60)
    print("INITIALIZATION SUMMARY")
    print("="*60)
    
    for check, status in checks.items():
        status_str = "[PASS]" if status else "[FAIL]"
        print(f"{check:20} {status_str}")
    
    if all(checks.values()):
        print("\n[OK] System ready to initialize!")
        print("\nNext steps:")
        print("1. Run: python -m uvicorn app:app --reload")
        print("2. Access API at: http://localhost:8000")
        print("3. View docs at: http://localhost:8000/docs")
        return True
    else:
        print("\n[FAIL] System not ready. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = initialize_system()
    sys.exit(0 if success else 1)
