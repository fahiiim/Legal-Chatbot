"""
Intelligent Legal Document Chunker
Implements article-by-article chunking with semantic awareness.
Enhanced with TOC extraction, hierarchical context, and cross-reference tracking.
"""

import re
from typing import List, Dict, Optional, Tuple, Set
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from legal_preprocessor import LegalPreprocessor
import tiktoken


class TableOfContentsExtractor:
    """
    Extracts and creates separate chunks for Table of Contents/Index pages.
    This allows the RAG system to understand document structure and navigate to relevant sections.
    """
    
    # Patterns to detect TOC/Index pages
    TOC_PATTERNS = [
        re.compile(r'TABLE\s+OF\s+CONTENTS', re.IGNORECASE),
        re.compile(r'CONTENTS\s*\n', re.IGNORECASE),
        re.compile(r'INDEX\s*\n', re.IGNORECASE),
        re.compile(r'TABLE\s+OF\s+RULES', re.IGNORECASE),
    ]
    
    # Patterns to detect TOC entries (rule/section followed by page number)
    TOC_ENTRY_PATTERNS = [
        # "Rule 1. Title ... 5" or "Rule 1.1 Title...........5"
        re.compile(r'(?:Rule|RULE)\s+(\d+(?:\.\d+)?)\s*[.\-—–]?\s*([^.\n]+)\.{2,}\s*(\d+)', re.IGNORECASE),
        # "MCR 6.101 Title...5"
        re.compile(r'(?:MCR|Rule)\s+(\d+\.\d+)\s*[.\-—–]?\s*([^.\n]+)\.{2,}\s*(\d+)', re.IGNORECASE),
        # "Section 1. Title...5" or "§ 1. Title...5"
        re.compile(r'(?:Section|Sec\.|§)\s*(\d+(?:\.\d+)?)\s*[.\-—–]?\s*([^.\n]+)\.{2,}\s*(\d+)', re.IGNORECASE),
        # "Article I. Title...5"
        re.compile(r'(?:Article|ARTICLE)\s+([IVXLCDM]+|\d+)\s*[.\-—–]?\s*([^.\n]+)\.{2,}\s*(\d+)', re.IGNORECASE),
        # "M Civ JI 3.01 Title...5"
        re.compile(r'M\s+(?:Civ|Crim)\s+JI\s+(\d+\.\d+)\s*[:\-—–]?\s*([^.\n]+)\.{2,}\s*(\d+)', re.IGNORECASE),
        # Simple: "Title ... page" or "1.1 Title ... 5"
        re.compile(r'^(\d+(?:\.\d+)?)\s+([^.\n]{5,50})\.{2,}\s*(\d+)\s*$', re.MULTILINE),
    ]
    
    @classmethod
    def is_toc_page(cls, text: str) -> bool:
        """Check if text appears to be a table of contents page."""
        # Check for TOC header
        for pattern in cls.TOC_PATTERNS:
            if pattern.search(text[:500]):  # Check in first 500 chars
                return True
        
        # Check for high density of TOC-like entries (dotted lines with page numbers)
        dotted_line_count = len(re.findall(r'\.{3,}\s*\d+', text))
        lines = text.split('\n')
        if len(lines) > 5 and dotted_line_count / len(lines) > 0.3:
            return True
        
        return False
    
    @classmethod
    def extract_toc_entries(cls, text: str) -> List[Dict]:
        """Extract structured entries from TOC text."""
        entries = []
        
        for pattern in cls.TOC_ENTRY_PATTERNS:
            for match in pattern.finditer(text):
                entry = {
                    'number': match.group(1),
                    'title': match.group(2).strip().rstrip('.'),
                    'page': match.group(3) if len(match.groups()) >= 3 else None
                }
                # Avoid duplicates
                if entry not in entries:
                    entries.append(entry)
        
        return entries
    
    @classmethod
    def create_toc_chunk(cls, text: str, metadata: Dict) -> Optional[Document]:
        """Create a special TOC chunk with structured navigation data."""
        entries = cls.extract_toc_entries(text)
        
        if not entries:
            return None
        
        # Create a structured summary
        toc_summary = "TABLE OF CONTENTS / INDEX\n\n"
        toc_summary += "This document contains the following sections:\n\n"
        
        for entry in entries:
            toc_summary += f"• {entry['number']}: {entry['title']}"
            if entry.get('page'):
                toc_summary += f" (Page {entry['page']})"
            toc_summary += "\n"
        
        toc_metadata = metadata.copy()
        toc_metadata['chunk_type'] = 'table_of_contents'
        toc_metadata['is_navigation'] = True
        toc_metadata['entry_count'] = len(entries)
        toc_metadata['toc_entries'] = entries  # Store structured data
        
        return Document(
            page_content=toc_summary,
            metadata=toc_metadata
        )


class LegalDefinitionExtractor:
    """
    Extracts legal definitions and creates separate searchable chunks.
    Definitions are crucial for legal interpretation.
    """
    
    DEFINITION_PATTERNS = [
        # "X" means / "X" is defined as
        re.compile(r'"([^"]+)"\s+(?:means?|is defined as|shall mean|refers to)\s+([^.]+\.)', re.IGNORECASE),
        # (a) "Term" means...
        re.compile(r'\([a-z]\)\s*"([^"]+)"\s+(?:means?|is defined as)\s+([^.]+\.)', re.IGNORECASE),
        # Definitions section patterns
        re.compile(r'(?:As used in this (?:rule|section|article)[^,]*,\s*)?["\']([^"\']+)["\'](?:\s+or\s+["\'][^"\']+["\'])?\s+means?\s+([^.]+\.)', re.IGNORECASE),
    ]
    
    @classmethod
    def extract_definitions(cls, text: str) -> List[Dict]:
        """Extract term definitions from text."""
        definitions = []
        
        for pattern in cls.DEFINITION_PATTERNS:
            for match in pattern.finditer(text):
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                # Skip very short or very long definitions (likely false positives)
                if 10 < len(definition) < 500 and len(term) > 1:
                    definitions.append({
                        'term': term,
                        'definition': definition,
                        'full_match': match.group(0)
                    })
        
        return definitions
    
    @classmethod
    def create_definition_chunks(cls, definitions: List[Dict], metadata: Dict) -> List[Document]:
        """Create searchable chunks for each definition."""
        chunks = []
        
        for defn in definitions:
            content = f"LEGAL DEFINITION\n\nTerm: {defn['term']}\n\nDefinition: {defn['definition']}"
            
            defn_metadata = metadata.copy()
            defn_metadata['chunk_type'] = 'definition'
            defn_metadata['defined_term'] = defn['term']
            defn_metadata['is_definition'] = True
            
            chunks.append(Document(
                page_content=content,
                metadata=defn_metadata
            ))
        
        return chunks


class CrossReferenceTracker:
    """
    Tracks and enriches chunks with cross-reference information.
    Legal documents heavily cross-reference other sections/rules.
    """
    
    CROSS_REF_PATTERNS = [
        # MCR references
        re.compile(r'(?:see\s+)?MCR\s+(\d+\.\d+(?:\([a-zA-Z0-9]+\))?)', re.IGNORECASE),
        # Federal Rules references
        re.compile(r'(?:see\s+)?(?:Rule|RULE)\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)', re.IGNORECASE),
        re.compile(r'(?:Fed\.\s*R\.\s*(?:Civ|Crim|Evid)\.\s*P\.|FR(?:C|Cr|E)P?)\s*(\d+)', re.IGNORECASE),
        # Section references
        re.compile(r'(?:see\s+)?(?:section|§)\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
        # "under Rule X" or "pursuant to Rule X"
        re.compile(r'(?:under|pursuant to|in accordance with)\s+(?:Rule|MCR)\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
    ]
    
    @classmethod
    def extract_cross_references(cls, text: str) -> List[str]:
        """Extract all cross-references from text."""
        references = set()
        
        for pattern in cls.CROSS_REF_PATTERNS:
            for match in pattern.finditer(text):
                references.add(match.group(1))
        
        return list(references)
    
    @classmethod
    def enrich_metadata(cls, content: str, metadata: Dict) -> Dict:
        """Add cross-reference information to metadata."""
        refs = cls.extract_cross_references(content)
        if refs:
            metadata['cross_references'] = refs
            metadata['has_cross_references'] = True
        return metadata


class LegalChunker:
    """
    Advanced chunking strategy for legal documents.
    Chunks by article/section while respecting semantic boundaries.
    Includes TOC extraction, definition handling, and hierarchical context.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = "gpt-4",
                 extract_toc: bool = True,
                 extract_definitions: bool = True,
                 preserve_hierarchy: bool = True):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: target size in tokens for each chunk
            chunk_overlap: Overlap between chunks in tokens
            model_name: Model name for token counting
            extract_toc: Whether to create separate TOC chunks
            extract_definitions: Whether to create separate definition chunks
            preserve_hierarchy: Whether to include parent context in child chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_toc = extract_toc
        self.extract_definitions = extract_definitions
        self.preserve_hierarchy = preserve_hierarchy
        
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
        self.toc_extractor = TableOfContentsExtractor()
        self.definition_extractor = LegalDefinitionExtractor()
        self.cross_ref_tracker = CrossReferenceTracker()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer is None:
            # Simple approximation: 1 token ≈ 4 characters
            return len(text) // 4
        return len(self.tokenizer.encode(text))
    
    def _extract_page_ranges(self, content: str) -> List[Tuple[str, int, int]]:
        """Extract page-based segments from content with [PAGE X] markers."""
        page_pattern = re.compile(r'\[PAGE\s+(\d+)\]')
        matches = list(page_pattern.finditer(content))
        
        segments = []
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            page_content = content[start:end].strip()
            segments.append((page_content, page_num, page_num))
        
        return segments
    
    def _process_toc_pages(self, doc: Document) -> List[Document]:
        """Process and extract TOC/Index pages as separate chunks."""
        toc_chunks = []
        content = doc.page_content
        
        # Check page by page for TOC content
        page_segments = self._extract_page_ranges(content)
        
        if not page_segments:
            # No page markers, check entire content
            if TableOfContentsExtractor.is_toc_page(content[:3000]):
                toc_chunk = TableOfContentsExtractor.create_toc_chunk(
                    content[:3000], doc.metadata
                )
                if toc_chunk:
                    toc_chunks.append(toc_chunk)
        else:
            # Check each page for TOC content
            for page_content, page_start, page_end in page_segments[:10]:  # TOC usually in first 10 pages
                if TableOfContentsExtractor.is_toc_page(page_content):
                    toc_metadata = doc.metadata.copy()
                    toc_metadata['page_number'] = page_start
                    toc_chunk = TableOfContentsExtractor.create_toc_chunk(
                        page_content, toc_metadata
                    )
                    if toc_chunk:
                        toc_chunks.append(toc_chunk)
        
        return toc_chunks
    
    def _process_definitions(self, doc: Document) -> List[Document]:
        """Extract definitions as separate searchable chunks."""
        definitions = LegalDefinitionExtractor.extract_definitions(doc.page_content)
        return LegalDefinitionExtractor.create_definition_chunks(definitions, doc.metadata)
    
    def _build_hierarchical_context(self, doc: Document) -> str:
        """Build context string from document hierarchy."""
        context_parts = []
        
        # Add document type context
        doc_type = doc.metadata.get('doc_type', '')
        if doc_type:
            doc_type_display = doc_type.replace('_', ' ').title()
            context_parts.append(f"Document: {doc_type_display}")
        
        # Add source
        source = doc.metadata.get('source', '')
        if source:
            context_parts.append(f"Source: {source}")
        
        # Add section hierarchy
        if doc.metadata.get('section_number'):
            section_info = f"Section {doc.metadata['section_number']}"
            if doc.metadata.get('section_title'):
                section_info += f": {doc.metadata['section_title']}"
            context_parts.append(section_info)
        
        # Add parent section if available
        if doc.metadata.get('parent_section'):
            context_parts.append(f"Part of: {doc.metadata['parent_section']}")
        
        return " | ".join(context_parts)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using intelligent legal-aware splitting.
        
        Args:
            documents: List of LangChain documents to chunk
        
        Returns:
            List of chunked documents with preserved metadata
        """
        chunked_docs = []
        toc_docs = []
        definition_docs = []
        
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_tokens = self.count_tokens(doc.page_content)
            
            print(f"Processing: {doc.metadata.get('source', 'unknown')} ({doc_tokens} tokens)")
            
            # Step 1: Extract TOC/Index if enabled (do this first, before other processing)
            if self.extract_toc:
                toc_chunks = self._process_toc_pages(doc)
                if toc_chunks:
                    print(f"  Extracted {len(toc_chunks)} TOC/Index chunk(s)")
                    toc_docs.extend(toc_chunks)
            
            # Step 2: Extract definitions if enabled
            if self.extract_definitions:
                defn_chunks = self._process_definitions(doc)
                if defn_chunks:
                    print(f"  Extracted {len(defn_chunks)} definition chunk(s)")
                    definition_docs.extend(defn_chunks)
            
            # Step 3: Add hierarchical context prefix if enabled
            if self.preserve_hierarchy:
                hierarchy_context = self._build_hierarchical_context(doc)
                if hierarchy_context and not doc.page_content.startswith('['):
                    # Prepend context to content
                    doc = Document(
                        page_content=f"[{hierarchy_context}]\n\n{doc.page_content}",
                        metadata=doc.metadata
                    )
            
            # Step 4: Enrich metadata with cross-references
            doc.metadata = CrossReferenceTracker.enrich_metadata(
                doc.page_content, doc.metadata
            )
            
            # Step 5: Choose chunking strategy based on document characteristics
            # ALWAYS chunk if document is too large, regardless of type
            if doc_tokens > self.chunk_size * 2:
                print(f"  Document too large ({doc_tokens} tokens), using generic chunker")
                chunks = self._chunk_generic(doc)
            # Choose chunking strategy based on document type
            elif 'section_number' in doc.metadata:
                # Document is already section-based from loader
                chunks = self._chunk_section(doc)
            elif 'rules' in doc_type:
                chunks = self._chunk_rules_document(doc)
            elif 'jury' in doc_type:
                chunks = self._chunk_jury_instructions(doc)
            else:
                chunks = self._chunk_generic(doc)
            
            # Step 6: Post-process chunks to add cross-references and enhance metadata
            for chunk in chunks:
                chunk.metadata = CrossReferenceTracker.enrich_metadata(
                    chunk.page_content, chunk.metadata
                )
                # Add chunk statistics
                chunk.metadata['token_count'] = self.count_tokens(chunk.page_content)
            
            chunked_docs.extend(chunks)
        
        # Combine all chunks: TOC first (for navigation), then definitions, then content
        all_docs = toc_docs + definition_docs + chunked_docs
        
        print(f"\nChunking Summary:")
        print(f"  - TOC/Index chunks: {len(toc_docs)}")
        print(f"  - Definition chunks: {len(definition_docs)}")
        print(f"  - Content chunks: {len(chunked_docs)}")
        print(f"  - Total chunks: {len(all_docs)}")
        
        return all_docs
    
    def _chunk_section(self, doc: Document) -> List[Document]:
        """
        Chunk an already-section-based document.
        If section is too large, split it intelligently while preserving context.
        """
        chunks = []
        content = doc.page_content
        token_count = self.count_tokens(content)
        
        # If section is small enough, keep as-is
        if token_count <= self.chunk_size:
            return [doc]
        
        # Extract the section header for context preservation
        section_header = self._extract_section_header(content, doc.metadata)
        
        # Otherwise, split by subsections or paragraphs
        subsection_chunks = self._split_by_subsections(content, doc.metadata, section_header)
        
        if subsection_chunks:
            chunks.extend(subsection_chunks)
        else:
            # Fallback to paragraph-based splitting
            chunks.extend(self._split_by_paragraphs(content, doc.metadata, section_header))
        
        return chunks
    
    def _extract_section_header(self, content: str, metadata: Dict) -> str:
        """Extract the section header/title for context in child chunks."""
        header_parts = []
        
        # Use metadata if available
        if metadata.get('section_number'):
            header_parts.append(f"Section {metadata['section_number']}")
        if metadata.get('section_title'):
            header_parts.append(metadata['section_title'])
        
        if header_parts:
            return " - ".join(header_parts)
        
        # Try to extract from content
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable header length
                # Check if it looks like a header
                if re.match(r'^(?:Rule|RULE|Section|MCR|Article|M\s+(?:Civ|Crim)\s+JI)', line, re.IGNORECASE):
                    return line
        
        return ""
    
    def _split_by_subsections(self, content: str, metadata: Dict, section_header: str = "") -> List[Document]:
        """Split content by legal subsections (a), (1), (i), etc., preserving parent context."""
        chunks = []
        
        # Pattern to match subsections: (a), (1), (i), etc.
        subsection_pattern = re.compile(r'\n\s*\(([a-z]|\d+|[ivxlcdm]+)\)\s+', re.IGNORECASE | re.MULTILINE)
        
        matches = list(subsection_pattern.finditer(content))
        
        if len(matches) < 2:
            return []  # Not enough subsections
        
        # Extract main section header (before first subsection)
        header = content[:matches[0].start()].strip()
        
        # Build parent context string
        parent_context = ""
        if section_header:
            parent_context = f"[Parent: {section_header}]\n\n"
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            subsection_content = content[start:end].strip()
            subsection_number = match.group(1)
            
            # Include header context and parent context
            full_content = f"{parent_context}{header}\n\n{subsection_content}"
            
            # Check if still too large
            if self.count_tokens(full_content) > self.chunk_size:
                # Further split by sentences
                chunks.extend(self._split_by_sentences(full_content, metadata, subsection_number))
            else:
                chunk_metadata = metadata.copy()
                chunk_metadata['subsection'] = subsection_number
                chunk_metadata['chunk_type'] = 'subsection'
                chunk_metadata['parent_section'] = section_header
                
                chunks.append(Document(
                    page_content=full_content,
                    metadata=chunk_metadata
                ))
        
        return chunks
    
    def _split_by_paragraphs(self, content: str, metadata: Dict, section_header: str = "") -> List[Document]:
        """Split content by paragraphs while maintaining context."""
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Build parent context
        context_prefix = ""
        if section_header:
            context_prefix = f"[Context: {section_header}]\n\n"
        
        current_chunk = []
        current_tokens = self.count_tokens(context_prefix)
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = context_prefix + '\n\n'.join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_type'] = 'paragraph'
                chunk_metadata['parent_section'] = section_header
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                ))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    current_chunk = [overlap_text, para]
                    current_tokens = self.count_tokens(context_prefix + overlap_text) + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = self.count_tokens(context_prefix) + para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = context_prefix + '\n\n'.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_type'] = 'paragraph'
            chunk_metadata['parent_section'] = section_header
            
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
        """
        Chunk jury instructions with special handling.
        Jury instructions should be kept together when possible as they are 
        self-contained units meant to be read to juries.
        """
        content = doc.page_content
        token_count = self.count_tokens(content)
        
        # If small enough, keep as single chunk
        if token_count <= self.chunk_size:
            doc.metadata['chunk_type'] = 'jury_instruction'
            return [doc]
        
        # For larger instructions, try to split by numbered points or paragraphs
        # but always include the instruction header
        
        # Extract instruction header (usually the first few lines)
        lines = content.split('\n')
        header_lines = []
        content_start = 0
        
        for i, line in enumerate(lines):
            # Look for the end of the header (usually instruction number + title)
            if i < 5 and (re.match(r'^M\s+(?:Civ|Crim)\s+JI', line) or 
                         re.match(r'^\[\w+', line) or
                         line.strip() and not re.match(r'^\d+\.?\s', line)):
                header_lines.append(line)
                content_start = i + 1
            else:
                break
        
        header = '\n'.join(header_lines).strip()
        remaining_content = '\n'.join(lines[content_start:]).strip()
        
        # Now chunk the remaining content, always prepending the header
        chunks = []
        
        # Try to split by numbered points first
        numbered_pattern = re.compile(r'\n\s*(\d+)\.\s+', re.MULTILINE)
        matches = list(numbered_pattern.finditer(remaining_content))
        
        if len(matches) >= 2:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(remaining_content)
                
                point_content = remaining_content[start:end].strip()
                full_content = f"{header}\n\n{point_content}"
                
                if self.count_tokens(full_content) > self.chunk_size:
                    # Split further by sentences
                    sub_chunks = self._split_by_sentences(full_content, doc.metadata, f"point_{match.group(1)}")
                    chunks.extend(sub_chunks)
                else:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata['chunk_type'] = 'jury_instruction_point'
                    chunk_metadata['instruction_point'] = match.group(1)
                    
                    chunks.append(Document(
                        page_content=full_content,
                        metadata=chunk_metadata
                    ))
        else:
            # Fallback to paragraph splitting with header preservation
            chunks = self._split_by_paragraphs(remaining_content, doc.metadata, header)
        
        return chunks if chunks else [doc]
    
    def _chunk_generic(self, doc: Document) -> List[Document]:
        """
        Fallback chunking using LangChain's RecursiveCharacterTextSplitter
        with legal-aware separators and enhanced context preservation.
        """
        # Build context header from metadata
        context_header = self._build_context_header(doc.metadata)
        
        # Define separators in order of preference for legal documents
        separators = [
            "\n\n## ",      # Markdown headers
            "\n\n### ",
            "\n\nRule ",    # Legal rule starts
            "\n\nRULE ",
            "\n\nSection ", # Section starts
            "\n\nSECTION ",
            "\n\nArticle ",
            "\n\nARTICLE ",
            "\n[PAGE ",     # Page markers
            "\nRule ",      # Rules without double newline
            "\n\n(a) ",     # Subsection starts
            "\n\n(1) ",
            "\n\n(i) ",
            "\n\n",         # Paragraphs
            "\n",           # Lines
            ". ",           # Sentences
            " ",            # Words
            ""              # Characters
        ]
        
        # Use character-based splitting with approximate token conversion
        # 1 token ≈ 4 characters, so multiply by 4 for character-based splitting
        char_chunk_size = self.chunk_size * 4
        char_chunk_overlap = self.chunk_overlap * 4
        
        # Account for context header in chunk size
        if context_header:
            header_chars = len(context_header)
            char_chunk_size = max(char_chunk_size - header_chars - 10, 500)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_chunk_size,
            chunk_overlap=char_chunk_overlap,
            separators=separators,
            length_function=len,  # Use character count for reliability
        )
        
        chunks = splitter.split_documents([doc])
        
        # Add context header and enhance metadata for each chunk
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Prepend context header if available
            if context_header:
                enhanced_content = f"{context_header}\n\n{chunk.page_content}"
            else:
                enhanced_content = chunk.page_content
            
            chunk.metadata['chunk_type'] = 'recursive'
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
            
            enhanced_chunks.append(Document(
                page_content=enhanced_content,
                metadata=chunk.metadata
            ))
        
        print(f"  Split document into {len(enhanced_chunks)} chunks")
        return enhanced_chunks
    
    def _build_context_header(self, metadata: Dict) -> str:
        """Build a context header from metadata to prepend to chunks."""
        parts = []
        
        # Document source and type
        source = metadata.get('source', '')
        doc_type = metadata.get('doc_type', '')
        
        if source:
            parts.append(f"Source: {source}")
        if doc_type:
            parts.append(f"Type: {doc_type.replace('_', ' ').title()}")
        
        # Section information
        section_num = metadata.get('section_number', '')
        section_title = metadata.get('section_title', '')
        
        if section_num:
            section_info = f"Section: {section_num}"
            if section_title:
                section_info += f" - {section_title}"
            parts.append(section_info)
        
        if parts:
            return "[" + " | ".join(parts) + "]"
        return ""


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


class ParentDocumentChunker:
    """
    Creates both large parent chunks and small child chunks.
    Child chunks are used for retrieval, parent chunks provide context.
    This helps get more relevant context when answering questions.
    """
    
    def __init__(self,
                 parent_chunk_size: int = 2000,
                 child_chunk_size: int = 400,
                 child_overlap: int = 50):
        """
        Initialize parent-child chunker.
        
        Args:
            parent_chunk_size: Size of parent chunks (larger, for context)
            child_chunk_size: Size of child chunks (smaller, for retrieval)
            child_overlap: Overlap between child chunks
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        
        self.parent_chunker = LegalChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            extract_toc=False,
            extract_definitions=False,
            preserve_hierarchy=True
        )
        
        self.child_chunker = LegalChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            extract_toc=False,
            extract_definitions=False,
            preserve_hierarchy=False
        )
    
    def create_parent_child_chunks(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, Document]]:
        """
        Create parent and child chunks with linking.
        
        Returns:
            Tuple of (child_chunks for indexing, parent_lookup dict)
        """
        import uuid
        
        # First create parent chunks
        parent_chunks = self.parent_chunker.chunk_documents(documents)
        
        # Create mapping from parent ID to parent chunk
        parent_lookup = {}
        all_child_chunks = []
        
        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())
            parent.metadata['parent_id'] = parent_id
            parent_lookup[parent_id] = parent
            
            # Create child chunks from this parent
            child_docs = self.child_chunker.chunk_documents([parent])
            
            for child in child_docs:
                child.metadata['parent_id'] = parent_id
                child.metadata['is_child_chunk'] = True
                all_child_chunks.append(child)
        
        print(f"Created {len(parent_chunks)} parent chunks and {len(all_child_chunks)} child chunks")
        return all_child_chunks, parent_lookup


def create_legal_chunks(documents: List[Document], 
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       extract_toc: bool = True,
                       extract_definitions: bool = True,
                       preserve_hierarchy: bool = True) -> List[Document]:
    """
    Main function to create intelligent chunks from legal documents.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        extract_toc: Whether to create separate TOC/Index chunks
        extract_definitions: Whether to create separate definition chunks
        preserve_hierarchy: Whether to preserve hierarchical context
    
    Returns:
        List of chunked documents
    """
    chunker = LegalChunker(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        extract_toc=extract_toc,
        extract_definitions=extract_definitions,
        preserve_hierarchy=preserve_hierarchy
    )
    return chunker.chunk_documents(documents)


def create_parent_child_chunks(documents: List[Document],
                               parent_size: int = 2000,
                               child_size: int = 400) -> Tuple[List[Document], Dict]:
    """
    Create parent-child chunk structure for enhanced retrieval.
    
    Small child chunks are indexed for precise retrieval.
    Large parent chunks are returned for better context.
    
    Args:
        documents: List of documents to chunk
        parent_size: Size of parent chunks (for context)
        child_size: Size of child chunks (for retrieval)
    
    Returns:
        Tuple of (child_chunks, parent_lookup_dict)
    """
    chunker = ParentDocumentChunker(
        parent_chunk_size=parent_size,
        child_chunk_size=child_size
    )
    return chunker.create_parent_child_chunks(documents)
