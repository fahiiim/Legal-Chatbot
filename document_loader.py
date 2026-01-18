"""
Advanced Legal Document Loader
Loads PDFs with structure detection and metadata extraction.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import PyPDF2
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from legal_preprocessor import LegalPreprocessor, LegalSection
from tqdm import tqdm


@dataclass
class LegalDocument:
    """Represents a loaded legal document with metadata."""
    content: str
    filename: str
    doc_type: str
    metadata: Dict
    sections: List[LegalSection]
    page_count: int


class LegalDocumentLoader:
    """Advanced loader for legal PDF documents."""
    
    def __init__(self, pdf_directory: str):
        """Initialize the loader with a directory of PDFs."""
        self.pdf_directory = pdf_directory
        self.preprocessor = LegalPreprocessor()
    
    def load_document(self, filename: str) -> Optional[LegalDocument]:
        """Load a single PDF document with advanced processing."""
        filepath = os.path.join(self.pdf_directory, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return None
        
        try:
            # Extract text using pdfplumber (better for complex layouts)
            text = self._extract_text_pdfplumber(filepath)
            
            # Fallback to PyPDF2 if pdfplumber fails
            if not text or len(text.strip()) < 100:
                text = self._extract_text_pypdf2(filepath)
            
            # Clean and normalize text
            text = self.preprocessor.clean_text(text)
            
            # Detect document type
            doc_type = self.preprocessor.detect_document_type(text, filename)
            
            # Extract metadata
            metadata = self.preprocessor.extract_metadata(text, filename)
            metadata['doc_type'] = doc_type
            
            # Extract sections based on document type
            sections = self.preprocessor.extract_sections(text, doc_type)
            
            # Get page count
            page_count = self._get_page_count(filepath)
            metadata['page_count'] = page_count
            
            return LegalDocument(
                content=text,
                filename=filename,
                doc_type=doc_type,
                metadata=metadata,
                sections=sections,
                page_count=page_count
            )
        
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            return None
    
    def load_all_documents(self, filenames: List[str]) -> List[LegalDocument]:
        """Load all specified PDF documents."""
        documents = []
        
        print(f"Loading {len(filenames)} legal documents...")
        for filename in tqdm(filenames, desc="Loading PDFs"):
            doc = self.load_document(filename)
            if doc:
                documents.append(doc)
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _extract_text_pdfplumber(self, filepath: str) -> str:
        """Extract text using pdfplumber (better for tables and complex layouts)."""
        if pdfplumber is None:
            return ""
        
        text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Add page marker for reference
                        text += f"\n[PAGE {page_num}]\n{page_text}\n"
        except Exception as e:
            print(f"pdfplumber error for {filepath}: {e}")
        
        return text
    
    def _extract_text_pypdf2(self, filepath: str) -> str:
        """Fallback extraction using PyPDF2."""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[PAGE {page_num}]\n{page_text}\n"
        except Exception as e:
            print(f"PyPDF2 error for {filepath}: {e}")
        
        return text
    
    def _get_page_count(self, filepath: str) -> int:
        """Get the number of pages in a PDF."""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except:
            return 0
    
    def documents_to_langchain(self, legal_docs: List[LegalDocument]) -> List[Document]:
        """Convert LegalDocument objects to LangChain Document objects."""
        langchain_docs = []
        
        for legal_doc in legal_docs:
            # Create a document for each section (if sections exist)
            if legal_doc.sections:
                for section in legal_doc.sections:
                    metadata = {
                        'source': legal_doc.filename,
                        'doc_type': legal_doc.doc_type,
                        'section_number': section.section_number,
                        'section_title': section.title,
                        'section_level': section.level,
                        **legal_doc.metadata
                    }
                    
                    # Include section context in content
                    content = f"[{legal_doc.doc_type.upper()} - Section {section.section_number}: {section.title}]\n\n{section.content}"
                    
                    langchain_docs.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
            else:
                # If no sections detected, use full document
                metadata = {
                    'source': legal_doc.filename,
                    'doc_type': legal_doc.doc_type,
                    **legal_doc.metadata
                }
                
                langchain_docs.append(Document(
                    page_content=legal_doc.content,
                    metadata=metadata
                ))
        
        return langchain_docs


def load_legal_knowledge_base(pdf_directory: str, pdf_files: List[str]) -> List[Document]:
    """
    Main function to load legal knowledge base.
    
    Args:
        pdf_directory: Directory containing PDF files
        pdf_files: List of PDF filenames to load
    
    Returns:
        List of LangChain Document objects ready for chunking
    """
    loader = LegalDocumentLoader(pdf_directory)
    legal_docs = loader.load_all_documents(pdf_files)
    langchain_docs = loader.documents_to_langchain(legal_docs)
    
    print(f"Created {len(langchain_docs)} document sections for processing")
    return langchain_docs
