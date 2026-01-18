"""
Legal Document Preprocessor
Handles cleaning, normalization, and structure detection for legal documents.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LegalSection:
    """Represents a section of a legal document."""
    section_number: str
    title: str
    content: str
    level: int  # 1 for Article, 2 for Section, 3 for Subsection, etc.
    parent_section: Optional[str] = None
    page_number: Optional[int] = None


class LegalPreprocessor:
    """Preprocessor for legal documents with structure detection."""
    
    # Regex patterns for different legal document structures
    PATTERNS = {
        # Federal Rules patterns
        'federal_rule': re.compile(r'(?:Rule|RULE)\s+(\d+(?:\.\d+)?)\s*[.\-—–]?\s*([^\n]+)', re.IGNORECASE),
        
        # Michigan Court Rules (e.g., MCR 6.101)
        'mcr_rule': re.compile(r'(?:MCR|Rule)\s+(\d+\.\d+(?:\(\w+\))?)\s*[.\-—–]?\s*([^\n]+)', re.IGNORECASE),
        
        # Article/Section patterns (e.g., Article I, Section 1)
        'article': re.compile(r'(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+)\s*[.\-—–]?\s*([^\n]+)', re.IGNORECASE),
        'section': re.compile(r'(?:SECTION|Section|Sec\.|§)\s+(\d+(?:\.\d+)*)\s*[.\-—–]?\s*([^\n]+)', re.IGNORECASE),
        
        # Subsections (a), (1), (i), etc.
        'subsection_alpha': re.compile(r'^\s*\(([a-z])\)\s+(.+)', re.MULTILINE),
        'subsection_numeric': re.compile(r'^\s*\((\d+)\)\s+(.+)', re.MULTILINE),
        'subsection_roman': re.compile(r'^\s*\(([ivxlcdm]+)\)\s+(.+)', re.MULTILINE | re.IGNORECASE),
        
        # Citation patterns
        'citation_mcr': re.compile(r'MCR\s+\d+\.\d+(?:\(\w+\))?'),
        'citation_frcp': re.compile(r'(?:Fed\.\s*R\.\s*Civ\.\s*P\.|FRCP)\s*\d+'),
        'citation_frcrp': re.compile(r'(?:Fed\.\s*R\.\s*Crim\.\s*P\.|FRCrP)\s*\d+'),
        'citation_fre': re.compile(r'(?:Fed\.\s*R\.\s*Evid\.|FRE)\s*\d+'),
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize legal text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize dashes and hyphens
        text = text.replace('—', '-').replace('–', '-')
        
        # Fix common OCR errors in legal documents
        text = text.replace('§', 'Section')
        text = text.replace('¶', 'Paragraph')
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def detect_document_type(text: str, filename: str) -> str:
        """Detect the type of legal document."""
        filename_lower = filename.lower()
        
        if 'criminal-procedure' in filename_lower or 'frcrp' in filename_lower:
            return 'federal_criminal_rules'
        elif 'civil-procedure' in filename_lower or 'frcp' in filename_lower:
            return 'federal_civil_rules'
        elif 'evidence' in filename_lower or 'fre' in filename_lower:
            return 'federal_evidence_rules'
        elif 'michigan-court-rules' in filename_lower or 'mcr' in filename_lower:
            return 'michigan_court_rules'
        elif 'criminal-jury' in filename_lower:
            return 'michigan_criminal_jury_instructions'
        elif 'civil-jury' in filename_lower or 'model-civil' in filename_lower:
            return 'michigan_civil_jury_instructions'
        else:
            return 'unknown'
    
    @classmethod
    def extract_sections(cls, text: str, doc_type: str) -> List[LegalSection]:
        """Extract structured sections from legal document based on type."""
        sections = []
        
        if 'federal' in doc_type and 'rules' in doc_type:
            sections = cls._extract_federal_rules(text)
        elif 'michigan_court_rules' in doc_type:
            sections = cls._extract_mcr_rules(text)
        elif 'jury' in doc_type:
            sections = cls._extract_jury_instructions(text)
        else:
            # Generic article/section extraction
            sections = cls._extract_generic_sections(text)
        
        return sections
    
    @classmethod
    def _extract_federal_rules(cls, text: str) -> List[LegalSection]:
        """Extract Federal Rules (FRCP, FRCrP, FRE)."""
        sections = []
        lines = text.split('\n')
        current_rule = None
        current_content = []
        
        for i, line in enumerate(lines):
            rule_match = cls.PATTERNS['federal_rule'].search(line)
            
            if rule_match:
                # Save previous rule if exists
                if current_rule:
                    sections.append(LegalSection(
                        section_number=current_rule[0],
                        title=current_rule[1].strip(),
                        content='\n'.join(current_content).strip(),
                        level=1
                    ))
                
                # Start new rule
                current_rule = (rule_match.group(1), rule_match.group(2))
                current_content = [line]
            elif current_rule:
                current_content.append(line)
        
        # Add last rule
        if current_rule:
            sections.append(LegalSection(
                section_number=current_rule[0],
                title=current_rule[1].strip(),
                content='\n'.join(current_content).strip(),
                level=1
            ))
        
        return sections
    
    @classmethod
    def _extract_mcr_rules(cls, text: str) -> List[LegalSection]:
        """Extract Michigan Court Rules."""
        sections = []
        lines = text.split('\n')
        current_rule = None
        current_content = []
        
        for line in lines:
            rule_match = cls.PATTERNS['mcr_rule'].search(line)
            
            if rule_match:
                # Save previous rule
                if current_rule:
                    sections.append(LegalSection(
                        section_number=current_rule[0],
                        title=current_rule[1].strip(),
                        content='\n'.join(current_content).strip(),
                        level=1
                    ))
                
                # Start new rule
                current_rule = (rule_match.group(1), rule_match.group(2))
                current_content = [line]
            elif current_rule:
                current_content.append(line)
        
        # Add last rule
        if current_rule:
            sections.append(LegalSection(
                section_number=current_rule[0],
                title=current_rule[1].strip(),
                content='\n'.join(current_content).strip(),
                level=1
            ))
        
        return sections
    
    @classmethod
    def _extract_jury_instructions(cls, text: str) -> List[LegalSection]:
        """Extract jury instructions."""
        sections = []
        
        # Jury instructions often numbered like "M Civ JI 3.01" or "M Crim JI 10.01"
        pattern = re.compile(r'M\s+(?:Civ|Crim)\s+JI\s+(\d+\.\d+)\s*[:\-—–]?\s*([^\n]+)', re.IGNORECASE)
        
        lines = text.split('\n')
        current_instruction = None
        current_content = []
        
        for line in lines:
            instr_match = pattern.search(line)
            
            if instr_match:
                # Save previous instruction
                if current_instruction:
                    sections.append(LegalSection(
                        section_number=current_instruction[0],
                        title=current_instruction[1].strip(),
                        content='\n'.join(current_content).strip(),
                        level=1
                    ))
                
                # Start new instruction
                current_instruction = (instr_match.group(1), instr_match.group(2))
                current_content = [line]
            elif current_instruction:
                current_content.append(line)
        
        # Add last instruction
        if current_instruction:
            sections.append(LegalSection(
                section_number=current_instruction[0],
                title=current_instruction[1].strip(),
                content='\n'.join(current_content).strip(),
                level=1
            ))
        
        return sections
    
    @classmethod
    def _extract_generic_sections(cls, text: str) -> List[LegalSection]:
        """Extract sections using generic article/section patterns."""
        sections = []
        
        # Try article pattern
        article_matches = list(cls.PATTERNS['article'].finditer(text))
        if article_matches:
            for i, match in enumerate(article_matches):
                start = match.start()
                end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
                content = text[start:end]
                
                sections.append(LegalSection(
                    section_number=match.group(1),
                    title=match.group(2).strip(),
                    content=content.strip(),
                    level=1
                ))
        
        # Try section pattern
        section_matches = list(cls.PATTERNS['section'].finditer(text))
        if section_matches:
            for i, match in enumerate(section_matches):
                start = match.start()
                end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
                content = text[start:end]
                
                sections.append(LegalSection(
                    section_number=match.group(1),
                    title=match.group(2).strip(),
                    content=content.strip(),
                    level=2
                ))
        
        return sections
    
    @classmethod
    def extract_citations(cls, text: str) -> List[str]:
        """Extract legal citations from text."""
        citations = []
        
        for pattern_name, pattern in cls.PATTERNS.items():
            if pattern_name.startswith('citation_'):
                citations.extend(pattern.findall(text))
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def extract_metadata(text: str, filename: str) -> Dict:
        """Extract metadata from legal document."""
        metadata = {
            'filename': filename,
            'word_count': len(text.split()),
            'char_count': len(text),
        }
        
        # Extract year from filename if present
        year_match = re.search(r'(19|20)\d{2}', filename)
        if year_match:
            metadata['year'] = year_match.group(0)
        
        # Detect effective date
        date_pattern = re.compile(r'(?:Effective|Amended|Revised)\s+(?:Date:?\s+)?(\w+\s+\d{1,2},?\s+\d{4})', re.IGNORECASE)
        date_match = date_pattern.search(text[:5000])  # Search in first 5000 chars
        if date_match:
            metadata['effective_date'] = date_match.group(1)
        
        return metadata
