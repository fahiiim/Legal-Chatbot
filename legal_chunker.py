"""
Intelligent Legal Document Chunker
Implements article-by-article chunking with semantic awareness.
"""

import re
from typing import List, Dict, Optional, Tuple
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from legal_preprocessor import LegalPreprocessor
import tiktoken


class LegalChunker:
    """
    Advanced chunking strategy for legal documents.
    Chunks by article/section while respecting semantic boundaries.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = "gpt-4"):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: target size is token for each size
            chunk_overlap: Overlap between chunks in tokens
            model_name: Model name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e2:
                # Fallback to simple word count if tiktoken fails
                print(f"Warning: Could not initialize tiktoken: {e2}")
                self.tokenizer = None
        
        self.preprocessor = LegalPreprocessor()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            # Simple approximation: 1 token ≈ 4 characters
            return len(text) // 4
        return len(self.tokenizer.encode(text))
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using intelligent legal-aware splitting.
        
        Args:
            documents: List of LangChain documents to chunk
        
        Returns:
            List of chunked documents with preserved metadata
        """
        chunked_docs = []
        
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            
            # Choose chunking strategy based on document type
            if 'section_number' in doc.metadata:
                # Document is already section-based from loader
                chunks = self._chunk_section(doc)
            elif 'rules' in doc_type:
                chunks = self._chunk_rules_document(doc)
            elif 'jury' in doc_type:
                chunks = self._chunk_jury_instructions(doc)
            else:
                chunks = self._chunk_generic(doc)
            
            chunked_docs.extend(chunks)
        
        print(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def _chunk_section(self, doc: Document) -> List[Document]:
        """
        Chunk an already-section-based document.
        If section is too large, split it intelligently.
        """
        chunks = []
        content = doc.page_content
        token_count = self.count_tokens(content)
        
        # If section is small enough, keep as-is
        if token_count <= self.chunk_size:
            return [doc]
        
        # Otherwise, split by subsections or paragraphs
        subsection_chunks = self._split_by_subsections(content, doc.metadata)
        
        if subsection_chunks:
            chunks.extend(subsection_chunks)
        else:
            # Fallback to paragraph-based splitting
            chunks.extend(self._split_by_paragraphs(content, doc.metadata))
        
        return chunks
    
    def _split_by_subsections(self, content: str, metadata: Dict) -> List[Document]:
        """Split content by legal subsections (a), (1), (i), etc."""
        chunks = []
        
        # Pattern to match subsections: (a), (1), (i), etc.
        subsection_pattern = re.compile(r'\n\s*\(([a-z]|\d+|[ivxlcdm]+)\)\s+', re.IGNORECASE | re.MULTILINE)
        
        matches = list(subsection_pattern.finditer(content))
        
        if len(matches) < 2:
            return []  # Not enough subsections
        
        # Extract main section header (before first subsection)
        header = content[:matches[0].start()].strip()
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            subsection_content = content[start:end].strip()
            subsection_number = match.group(1)
            
            # Include header context
            full_content = f"{header}\n\n{subsection_content}"
            
            # Check if still too large
            if self.count_tokens(full_content) > self.chunk_size:
                # Further split by sentences
                chunks.extend(self._split_by_sentences(full_content, metadata, subsection_number))
            else:
                chunk_metadata = metadata.copy()
                chunk_metadata['subsection'] = subsection_number
                chunk_metadata['chunk_type'] = 'subsection'
                
                chunks.append(Document(
                    page_content=full_content,
                    metadata=chunk_metadata
                ))
        
        return chunks
    
    def _split_by_paragraphs(self, content: str, metadata: Dict) -> List[Document]:
        """Split content by paragraphs while maintaining context."""
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_type'] = 'paragraph'
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, para]
                    current_tokens = self.count_tokens(overlap_text) + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_type'] = 'paragraph'
            
            chunks.append(Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _split_by_sentences(self, content: str, metadata: Dict, subsection: str = None) -> List[Document]:
        """Split content by sentences (last resort for very long sections)."""
        chunks = []
        
        # Simple sentence splitting (can be enhanced with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)
            
            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_type'] = 'sentence'
                if subsection:
                    chunk_metadata['subsection'] = subsection
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                ))
                
                # Overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, sentence]
                    current_tokens = self.count_tokens(overlap_text) + sent_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
        
        # Add remaining
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_type'] = 'sentence'
            if subsection:
                chunk_metadata['subsection'] = subsection
            
            chunks.append(Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _chunk_rules_document(self, doc: Document) -> List[Document]:
        """Chunk a rules document (Federal Rules, Michigan Court Rules)."""
        # Rules documents are typically already split by the loader
        return self._chunk_section(doc)
    
    def _chunk_jury_instructions(self, doc: Document) -> List[Document]:
        """Chunk jury instructions."""
        # Each jury instruction should be kept together if possible
        return self._chunk_section(doc)
    
    def _chunk_generic(self, doc: Document) -> List[Document]:
        """
        Fallback chunking using langchains RecursiveCharacterTextSplitter
        with legal-aware separators.
        """
        # Define separators in order of preference for legal documents
        separators = [
            "\n\n## ",      # Markdown headers
            "\n\n### ",
            "\n\nSection ",
            "\n\nRule ",
            "\n\nArticle ",
            "\n\n",         # Paragraphs
            "\n",           # Lines
            ". ",           # Sentences
            " ",            # Words
            ""              # Characters
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=self.count_tokens,
        )
        
        chunks = splitter.split_documents([doc])
        
        # Add chunk type to metadata
        for chunk in chunks:
            chunk.metadata['chunk_type'] = 'recursive'
        
        return chunks


class SemanticChunker:
    """
    Advanced semantic chunker that groups related legal content.
    Uses embeddings to create semantically coherent chunks.
    """
    
    def __init__(self, embedding_model, similarity_threshold: float = 0.7):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_model: Embedding model to use for similarity
            similarity_threshold: Threshold for semantic similarity
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
    
    def semantic_chunk(self, documents: List[Document]) -> List[Document]:
        """
        Create semantically coherent chunks by grouping similar content.
        This is an advanced feature for future enhancement.
        """
        # This would use embeddings to group semantically related sections
        # For now, we'll use the standard chunker
        # Future implementation would:
        # 1. Embed each section
        # 2. Calculate similarity between adjacent sections
        # 3. Merge sections below similarity threshold
        # 4. Split sections above token limit
        
        return documents


def create_legal_chunks(documents: List[Document], 
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200) -> List[Document]:
    """
    Main function to create intelligent chunks from legal documents.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents
    """
    chunker = LegalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)
