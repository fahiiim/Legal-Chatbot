"""
Citation Extractor and Validator
Extracts and validates legal citations from responses.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """Represents a legal citation."""
    citation_text: str
    source: str
    rule_number: Optional[str] = None
    section_number: Optional[str] = None
    page_number: Optional[int] = None
    confidence: float = 1.0
    
    def to_report_dict(self) -> Dict:
        """
        Convert citation to report-friendly dictionary format.
        
        Returns:
            Dictionary with 'name', 'reference', and 'description' keys
        """
        # Map source types to readable names
        source_display_map = {
            'mcr': 'Michigan Court Rules',
            'frcp': 'Federal Rules of Civil Procedure',
            'frcrp': 'Federal Rules of Criminal Procedure',
            'fre': 'Federal Rules of Evidence',
            'm_civ_ji': 'Michigan Civil Jury Instructions',
            'm_crim_ji': 'Michigan Criminal Jury Instructions',
            'michigan_court_rules': 'Michigan Court Rules',
            'federal_criminal_rules': 'Federal Rules of Criminal Procedure',
            'federal_civil_rules': 'Federal Rules of Civil Procedure',
            'federal_evidence_rules': 'Federal Rules of Evidence',
            'criminal_jury_instructions': 'Michigan Criminal Jury Instructions',
            'civil_jury_instructions': 'Michigan Civil Jury Instructions',
        }
        
        source_display = source_display_map.get(self.source, self.source.replace('_', ' ').title())
        
        return {
            'name': self.citation_text,
            'reference': source_display,
            'description': f"Rule/Section {self.rule_number}" if self.rule_number else "",
        }


class CitationExtractor:
    """Extracts and formats legal citations."""
    
    # Citation patterns
    CITATION_PATTERNS = {
        'mcr': re.compile(r'(MCR\s+\d+\.\d+(?:\(\w+\))?)', re.IGNORECASE),
        'frcp': re.compile(r'((?:Fed\.\s*R\.\s*Civ\.\s*P\.|FRCP)\s*\d+(?:\([a-z]\))?)', re.IGNORECASE),
        'frcrp': re.compile(r'((?:Fed\.\s*R\.\s*Crim\.\s*P\.|FRCrP)\s*\d+(?:\([a-z]\))?)', re.IGNORECASE),
        'fre': re.compile(r'((?:Fed\.\s*R\.\s*Evid\.|FRE)\s*\d+(?:\([a-z]\))?)', re.IGNORECASE),
        'm_civ_ji': re.compile(r'(M\s+Civ\s+JI\s+\d+\.\d+)', re.IGNORECASE),
        'm_crim_ji': re.compile(r'(M\s+Crim\s+JI\s+\d+\.\d+)', re.IGNORECASE),
    }
    
    @classmethod
    def extract_citations(cls, text: str) -> List[Citation]:
        """
        Extract all citations from text.
        
        Args:
            text: Text to extract citations from
        
        Returns:
            List of Citation objects
        """
        citations = []
        
        for source_type, pattern in cls.CITATION_PATTERNS.items():
            matches = pattern.findall(text)
            for match in matches:
                citation = Citation(
                    citation_text=match,
                    source=source_type,
                    rule_number=cls._extract_rule_number(match)
                )
                citations.append(citation)
        
        # Remove duplicates
        unique_citations = []
        seen = set()
        for citation in citations:
            if citation.citation_text not in seen:
                seen.add(citation.citation_text)
                unique_citations.append(citation)
        
        return unique_citations
    
    @staticmethod
    def _extract_rule_number(citation_text: str) -> Optional[str]:
        """Extract rule/section number from citation."""
        number_pattern = re.compile(r'\d+(?:\.\d+)?(?:\([a-z]\))?')
        match = number_pattern.search(citation_text)
        return match.group(0) if match else None
    
    @classmethod
    def format_citation(cls, citation: Citation) -> str:
        """Format citation in standard legal format."""
        # This can be customized based on citation style (Bluebook, etc.)
        return citation.citation_text
    
    @classmethod
    def validate_citation(cls, citation: Citation, source_documents: List) -> bool:
        """
        Validate that citation exists in source documents.
        
        Args:
            citation: Citation to validate
            source_documents: List of source documents to check against
        
        Returns:
            True if citation is valid
        """
        citation_text = citation.citation_text.lower()
        
        for doc in source_documents:
            if hasattr(doc, 'page_content'):
                content = doc.page_content.lower()
            else:
                content = str(doc).lower()
            
            if citation_text in content:
                return True
        
        return False
    
    @classmethod
    def create_citation_list(cls, citations: List[Citation]) -> str:
        """
        Create a formatted citation list for response.
        
        Args:
            citations: List of citations
        
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        citation_lines = []
        for i, citation in enumerate(citations, 1):
            citation_lines.append(f"{i}. {cls.format_citation(citation)}")
        
        return "\n".join(citation_lines)
    
    @classmethod
    def enrich_response_with_citations(cls, response: str, source_documents: List) -> Tuple[str, List[Citation]]:
        """
        Add citation markers to response and extract citation list.
        
        Args:
            response: Generated response text
            source_documents: Source documents used
        
        Returns:
            Tuple of (enriched response, citations list)
        """
        citations = cls.extract_citations(response)
        
        # Also extract citations from source documents
        doc_citations = []
        for doc in source_documents:
            if hasattr(doc, 'metadata'):
                # Extract from metadata
                if 'section_number' in doc.metadata:
                    section = doc.metadata['section_number']
                    doc_type = doc.metadata.get('doc_type', '')
                    
                    # Create appropriate citation
                    if 'michigan_court_rules' in doc_type:
                        citation_text = f"MCR {section}"
                    elif 'federal_criminal_rules' in doc_type:
                        citation_text = f"Fed. R. Crim. P. {section}"
                    elif 'federal_civil_rules' in doc_type:
                        citation_text = f"Fed. R. Civ. P. {section}"
                    elif 'federal_evidence_rules' in doc_type:
                        citation_text = f"Fed. R. Evid. {section}"
                    else:
                        citation_text = f"Section {section}"
                    
                    citation = Citation(
                        citation_text=citation_text,
                        source=doc_type,
                        section_number=section
                    )
                    doc_citations.append(citation)
        
        # Combine and deduplicate
        all_citations = citations + doc_citations
        unique_citations = []
        seen = set()
        for citation in all_citations:
            if citation.citation_text not in seen:
                seen.add(citation.citation_text)
                unique_citations.append(citation)
        
        return response, unique_citations


class SourceTracker:
    """Tracks and formats source attribution for responses."""
    
    @staticmethod
    def format_sources(source_documents: List) -> str:
        """
        Format source documents for display.
        
        Args:
            source_documents: List of source documents
        
        Returns:
            Formatted source string
        """
        if not source_documents:
            return "No sources available."
        
        sources = []
        seen_sources = set()
        
        for doc in source_documents:
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                source = metadata.get('source', 'Unknown')
                section = metadata.get('section_number', '')
                title = metadata.get('section_title', '')
                
                # Create source identifier
                source_id = f"{source}_{section}"
                
                if source_id not in seen_sources:
                    seen_sources.add(source_id)
                    
                    if section and title:
                        source_str = f"• {source}: Section {section} - {title}"
                    elif section:
                        source_str = f"• {source}: Section {section}"
                    else:
                        source_str = f"• {source}"
                    
                    sources.append(source_str)
        
        return "\n".join(sources)
    
    @staticmethod
    def get_source_metadata(source_documents: List) -> List[Dict]:
        """
        Extract metadata from source documents and format for report generator.
        
        Args:
            source_documents: List of source documents
        
        Returns:
            List of metadata dictionaries with standardized keys for reporting
        """
        metadata_list = []
        seen_sources = set()
        
        for doc in source_documents:
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                
                # Build a unique identifier to avoid duplicates
                source_id = f"{meta.get('source', '')}_{meta.get('section_number', '')}_{meta.get('section_title', '')}"
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                
                # Map document metadata to report-friendly format
                # Determine document type display name
                doc_type = meta.get('doc_type', 'legal_document')
                type_display_map = {
                    'federal_criminal_rules': 'Federal Rules of Criminal Procedure',
                    'federal_civil_rules': 'Federal Rules of Civil Procedure',
                    'federal_evidence_rules': 'Federal Rules of Evidence',
                    'michigan_court_rules': 'Michigan Court Rules',
                    'criminal_jury_instructions': 'Michigan Criminal Jury Instructions',
                    'civil_jury_instructions': 'Michigan Civil Jury Instructions',
                }
                type_display = type_display_map.get(doc_type, doc_type.replace('_', ' ').title())
                
                # Build title from available metadata
                source_file = meta.get('source', 'Unknown')
                section_num = meta.get('section_number', '')
                section_title = meta.get('section_title', '')
                
                # Create a descriptive title
                if section_title and section_num:
                    title = f"{type_display} - {section_num}: {section_title}"
                elif section_num:
                    title = f"{type_display} - Section {section_num}"
                elif section_title:
                    title = f"{type_display} - {section_title}"
                else:
                    # Fall back to filename
                    title = source_file.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
                
                # Get page info
                page = meta.get('page_number', meta.get('page', ''))
                if page:
                    page = f"Page {page}"
                
                formatted_meta = {
                    'title': title,
                    'type': type_display,
                    'page': page,
                    # Preserve original metadata for reference
                    'source_file': source_file,
                    'section_number': section_num,
                    'section_title': section_title,
                    'doc_type': doc_type,
                }
                metadata_list.append(formatted_meta)
        
        return metadata_list
