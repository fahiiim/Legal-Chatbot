"""
Response Validator
Validates generated answers against source documents to prevent hallucinations.
"""

from typing import List, Dict, Optional, Tuple
import re
import json

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_openai import ChatOpenAI


class ResponseValidator:
    """
    Validates that generated responses are grounded in source documents.
    Prevents hallucinations and ensures accuracy in legal context.
    """
    
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini",
                 strict_mode: bool = True):
        """
        Initialize response validator.
        
        Args:
            llm_model: LLM model for validation (use cheaper model)
            strict_mode: If True, flags any unsupported claims
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.strict_mode = strict_mode
    
    def validate_response(self, 
                         answer: str, 
                         sources: List[Document],
                         question: Optional[str] = None) -> Dict:
        """
        Validate that answer is grounded in source documents.
        
        Args:
            answer: Generated answer to validate
            sources: Source documents used for generation
            question: Original question (optional, for context)
        
        Returns:
            Validation result dictionary
        """
        if not sources:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "validation_passed": False,
                "reason": "No source documents provided",
                "unsupported_claims": ["All claims - no sources"],
                "supported_claims": []
            }
        
        # Extract source content (limit to avoid token overflow)
        sources_text = self._format_sources(sources)
        
        # Create validation prompt
        validation_prompt = self._create_validation_prompt(answer, sources_text, question)
        
        try:
            response = self.llm.invoke(validation_prompt)
            content = response.content
            
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(content)
            
            # Add overall validation flag
            validation_result["validation_passed"] = (
                validation_result.get("is_grounded", False) and
                validation_result.get("confidence", 0.0) >= 0.7
            )
            
            return validation_result
        
        except Exception as e:
            print(f"Error during validation: {e}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "validation_passed": False,
                "reason": f"Validation error: {str(e)}",
                "unsupported_claims": [],
                "supported_claims": []
            }
    
    def _format_sources(self, sources: List[Document], max_chars: int = 4000) -> str:
        """Format source documents for validation."""
        formatted = []
        total_chars = 0
        
        for i, doc in enumerate(sources):
            source_name = doc.metadata.get('source', f'Document {i+1}')
            section = doc.metadata.get('section_number', '')
            
            header = f"\n--- Source {i+1}: {source_name}"
            if section:
                header += f" (Section {section})"
            header += " ---\n"
            
            content = doc.page_content[:1000]  # Limit each source
            
            if total_chars + len(header) + len(content) > max_chars:
                break
            
            formatted.append(header + content)
            total_chars += len(header) + len(content)
        
        return "\n".join(formatted)
    
    def _create_validation_prompt(self, 
                                  answer: str, 
                                  sources_text: str,
                                  question: Optional[str] = None) -> str:
        """Create validation prompt for LLM."""
        question_context = f"\nOriginal Question: {question}\n" if question else ""
        
        prompt = f"""You are a strict validator for a legal RAG system. Your job is to verify that the answer is completely grounded in the provided source documents.

{question_context}
ANSWER TO VALIDATE:
{answer}

SOURCE DOCUMENTS:
{sources_text}

VALIDATION TASK:
Carefully analyze if EVERY claim in the answer is supported by the source documents.

For legal accuracy:
- Legal citations must exactly match sources
- Rules and procedures must be accurately stated from sources
- No information should be added that isn't in sources
- Paraphrasing is OK if meaning is preserved

Return a JSON object with:
{{
    "is_grounded": <true if ALL claims supported, false otherwise>,
    "confidence": <0.0 to 1.0, how confident in this validation>,
    "supported_claims": ["<claim 1 from answer>", "<claim 2>", ...],
    "unsupported_claims": ["<claim from answer not in sources>", ...],
    "hallucinated_info": ["<any made-up information>", ...],
    "citation_errors": ["<incorrect citations>", ...],
    "overall_assessment": "<brief summary>",
    "recommendation": "<PASS, FLAG_FOR_REVIEW, or REJECT>"
}}

Be strict: Even minor unsupported details should be flagged for legal content."""
        
        return prompt
    
    def quick_validate(self, answer: str, sources: List[Document]) -> bool:
        """
        Quick validation check (faster, less thorough).
        
        Args:
            answer: Generated answer
            sources: Source documents
        
        Returns:
            True if answer appears valid
        """
        # Check if answer contains legal citations
        has_citations = self._check_citations(answer)
        
        # Check if answer references sources
        references_sources = self._check_source_references(answer, sources)
        
        # Simple keyword overlap check
        keyword_overlap = self._calculate_keyword_overlap(answer, sources)
        
        # Quick validation passes if:
        # 1. Has citations OR references sources
        # 2. Reasonable keyword overlap (>0.3)
        return (has_citations or references_sources) and keyword_overlap > 0.3
    
    def _check_citations(self, text: str) -> bool:
        """Check if text contains legal citations."""
        citation_patterns = [
            r'MCR\s+\d+\.\d+',
            r'Fed\.\s*R\.\s*(?:Civ\.|Crim\.)\s*P\.\s*\d+',
            r'Fed\.\s*R\.\s*Evid\.\s*\d+',
            r'M\s+(?:Civ|Crim)\s+JI\s+\d+\.\d+',
            r'Rule\s+\d+',
            r'Section\s+\d+'
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _check_source_references(self, answer: str, sources: List[Document]) -> bool:
        """Check if answer references source documents."""
        answer_lower = answer.lower()
        
        # Check for source names
        for doc in sources:
            source_name = doc.metadata.get('source', '').lower()
            if source_name and source_name in answer_lower:
                return True
        
        # Check for generic references like "according to" or "states that"
        reference_phrases = ['according to', 'states that', 'provides that', 'specifies']
        return any(phrase in answer_lower for phrase in reference_phrases)
    
    def _calculate_keyword_overlap(self, answer: str, sources: List[Document]) -> float:
        """Calculate keyword overlap between answer and sources."""
        # Extract keywords (simple word tokenization)
        def extract_keywords(text: str) -> set:
            # Remove common legal stopwords
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'be', 'are', 'have'}
            words = re.findall(r'\b\w+\b', text.lower())
            return {w for w in words if len(w) > 3 and w not in stopwords}
        
        answer_keywords = extract_keywords(answer)
        
        if not answer_keywords:
            return 0.0
        
        # Combine all source keywords
        source_keywords = set()
        for doc in sources:
            source_keywords.update(extract_keywords(doc.page_content))
        
        if not source_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = answer_keywords & source_keywords
        union = answer_keywords | source_keywords
        
        return len(intersection) / len(union) if union else 0.0
    
    def validate_with_fallback(self, 
                              answer: str, 
                              sources: List[Document],
                              question: Optional[str] = None) -> Dict:
        """
        Validate with fallback to quick validation if LLM validation fails.
        
        Args:
            answer: Generated answer
            sources: Source documents
            question: Original question
        
        Returns:
            Validation result
        """
        # Try full validation first
        result = self.validate_response(answer, sources, question)
        
        # If validation failed or low confidence, try quick check
        if not result.get("validation_passed", False):
            quick_check = self.quick_validate(answer, sources)
            result["quick_validation"] = quick_check
            
            # If quick validation passes but full validation failed, flag for review
            if quick_check:
                result["recommendation"] = "FLAG_FOR_REVIEW"
                result["note"] = "Quick validation passed but full validation raised concerns"
        
        return result
    
    def handle_validation_failure(self, 
                                  answer: str,
                                  validation_result: Dict,
                                  sources: List[Document]) -> Dict:
        """
        Handle validation failure by suggesting corrections.
        
        Args:
            answer: Original answer
            validation_result: Validation result
            sources: Source documents
        
        Returns:
            Dictionary with suggestions and actions
        """
        unsupported_claims = validation_result.get("unsupported_claims", [])
        
        if not unsupported_claims:
            return {
                "action": "ACCEPT",
                "corrected_answer": answer,
                "note": "Validation passed"
            }
        
        # Create correction prompt
        sources_text = self._format_sources(sources)
        
        correction_prompt = f"""The following answer contains unsupported claims. Please revise it to only include information from the sources.

Original Answer:
{answer}

Unsupported Claims to Remove/Revise:
{json.dumps(unsupported_claims, indent=2)}

Available Sources:
{sources_text}

Return ONLY the corrected answer that includes only information from the sources. Maintain legal accuracy and citation format."""
        
        try:
            response = self.llm.invoke(correction_prompt)
            corrected_answer = response.content
            
            return {
                "action": "CORRECTED",
                "original_answer": answer,
                "corrected_answer": corrected_answer,
                "unsupported_claims": unsupported_claims,
                "note": "Answer was automatically corrected to remove unsupported claims"
            }
        
        except Exception as e:
            return {
                "action": "FLAG",
                "corrected_answer": answer,
                "error": str(e),
                "note": "Could not automatically correct. Manual review required."
            }


class LegalResponseValidator(ResponseValidator):
    """
    Legal-specific response validator with additional checks.
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini", strict_mode: bool = True):
        """Initialize legal validator."""
        super().__init__(llm_model, strict_mode)
        
        # Legal citation patterns
        self.citation_patterns = {
            'mcr': re.compile(r'MCR\s+\d+\.\d+(?:\(\w+\))?', re.IGNORECASE),
            'frcp': re.compile(r'(?:Fed\.\s*R\.\s*Civ\.\s*P\.|FRCP)\s*\d+', re.IGNORECASE),
            'frcrp': re.compile(r'(?:Fed\.\s*R\.\s*Crim\.\s*P\.|FRCrP)\s*\d+', re.IGNORECASE),
            'fre': re.compile(r'(?:Fed\.\s*R\.\s*Evid\.|FRE)\s*\d+', re.IGNORECASE),
        }
    
    def validate_legal_citations(self, answer: str, sources: List[Document]) -> Dict:
        """
        Validate that legal citations in answer exist in sources.
        
        Args:
            answer: Generated answer with citations
            sources: Source documents
        
        Returns:
            Citation validation results
        """
        # Extract citations from answer
        answer_citations = self._extract_citations(answer)
        
        # Extract citations from sources
        source_citations = set()
        for doc in sources:
            source_citations.update(self._extract_citations(doc.page_content))
        
        # Check which citations are supported
        valid_citations = []
        invalid_citations = []
        
        for citation in answer_citations:
            if citation in source_citations:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        return {
            "total_citations": len(answer_citations),
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_accuracy": len(valid_citations) / len(answer_citations) if answer_citations else 1.0,
            "all_citations_valid": len(invalid_citations) == 0
        }
    
    def _extract_citations(self, text: str) -> set:
        """Extract all legal citations from text."""
        citations = set()
        
        for pattern in self.citation_patterns.values():
            matches = pattern.findall(text)
            citations.update(matches)
        
        return citations
