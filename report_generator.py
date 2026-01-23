"""
Legal Report Generator
Formats RAG responses into professional legal reports for attorney review.
"""

from typing import List, Dict, Optional
from datetime import datetime
from tier_router import TierRouter


class LegalReportGenerator:
    """
    Generates professional legal reports from RAG query results.
    """
    
    @staticmethod
    def generate_report(
        query: str,
        answer: str,
        tier: int,
        tier_description: str,
        tier_reasoning: str,
        tier_recommendation: str,
        citations: List[Dict] = None,
        sources: List[Dict] = None,
        usage: Optional[Dict] = None
    ) -> str:
        """
        Generate a formatted legal report.
        
        Args:
            query: Original user query
            answer: RAG-generated answer
            tier: Complexity tier (1-4)
            tier_description: Tier description
            tier_reasoning: Why this tier was assigned
            tier_recommendation: Tier-based recommendation
            citations: List of legal citations
            sources: List of source documents
            usage: Token usage information
        
        Returns:
            Formatted legal report string
        """
        
        if citations is None:
            citations = []
        if sources is None:
            sources = []
        
        report_parts = []
        
        # Header
        report_parts.append("=" * 80)
        report_parts.append("LEGAL RESEARCH REPORT")
        report_parts.append("Michigan Legal RAG Chatbot - Attorney Review")
        report_parts.append("=" * 80)
        report_parts.append(f"\nGenerated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        report_parts.append("")
        
        # Case Classification Section
        report_parts.append("-" * 80)
        report_parts.append("CASE CLASSIFICATION & URGENCY ASSESSMENT")
        report_parts.append("-" * 80)
        report_parts.append(f"\nComplexity Tier: {tier}")
        report_parts.append(f"Classification: {tier_description}")
        report_parts.append(f"\nTier Determination Basis:")
        report_parts.append(f"  {tier_reasoning}")
        report_parts.append(f"\nRecommendation:")
        report_parts.append(f"  {tier_recommendation}")
        report_parts.append("")
        
        # Client Query Section
        report_parts.append("-" * 80)
        report_parts.append("CLIENT STATEMENT / CASE FACTS")
        report_parts.append("-" * 80)
        report_parts.append(f"\n{query}")
        report_parts.append("")
        
        # Legal Analysis Section
        report_parts.append("-" * 80)
        report_parts.append("LEGAL ANALYSIS & FINDINGS")
        report_parts.append("-" * 80)
        report_parts.append("")
        report_parts.append(answer)
        report_parts.append("")
        
        # Citations Section
        if citations:
            report_parts.append("-" * 80)
            report_parts.append("APPLICABLE STATUTES, RULES & JURY INSTRUCTIONS")
            report_parts.append("-" * 80)
            report_parts.append("")
            for idx, citation in enumerate(citations, 1):
                citation_name = citation.get("name", "Unknown Citation")
                citation_reference = citation.get("reference", "")
                citation_description = citation.get("description", "")
                
                report_parts.append(f"{idx}. {citation_name}")
                if citation_reference:
                    report_parts.append(f"   Reference: {citation_reference}")
                if citation_description:
                    report_parts.append(f"   Description: {citation_description}")
                report_parts.append("")
            report_parts.append("")
        
        # Sources Section
        if sources:
            report_parts.append("-" * 80)
            report_parts.append("RESEARCH SOURCES")
            report_parts.append("-" * 80)
            report_parts.append(f"\nTotal Sources Consulted: {len(sources)}")
            report_parts.append("")
            for idx, source in enumerate(sources, 1):
                source_title = source.get("title", "Unknown Source")
                source_type = source.get("type", "Document")
                source_page = source.get("page", "")
                
                report_parts.append(f"{idx}. {source_title}")
                report_parts.append(f"   Type: {source_type}")
                if source_page:
                    report_parts.append(f"   Page/Section: {source_page}")
                report_parts.append("")
            report_parts.append("")
        
        # Tier Information Detail
        report_parts.append("-" * 80)
        report_parts.append("TIER CLASSIFICATION DETAILS")
        report_parts.append("-" * 80)
        report_parts.append("")
        report_parts.append(LegalReportGenerator._get_tier_details(tier))
        report_parts.append("")
        
        # Footer
        report_parts.append("=" * 80)
        report_parts.append("DISCLAIMER")
        report_parts.append("=" * 80)
        report_parts.append("""
This report is generated using Retrieval-Augmented Generation (RAG) technology
and should be reviewed and verified by a qualified attorney. While sourced from
Michigan Model Criminal Jury Instructions, Federal Rules of Evidence, and Federal
Rules of Criminal Procedure, this information is provided for educational purposes
and does not constitute legal advice. An attorney should conduct independent legal
research and provide personalized counsel based on the specific facts of the case.
""")
        report_parts.append("=" * 80)
        
        return "\n".join(report_parts)
    
    @staticmethod
    def _get_tier_details(tier: int) -> str:
        """Get detailed explanation of tier classification."""
        
        tier_details = {
            1: """
TIER 1: ROUTINE / LOW-RISK
Typical Cases: Traffic violations, civil infractions, name changes, small claims

Characteristics:
  • Simple procedural matters
  • Limited legal complexity
  • Lower stakes (minimal financial/personal impact)
  • Straightforward resolution paths
  • Can often be handled pro se or with limited counsel

Recommended Action:
  • Self-help resources may be sufficient
  • Consider consulting with attorney for guidance only
  • Plan for lower legal fees
""",
            2: """
TIER 2: MODERATE / LITIGATION
Typical Cases: Contested matters, felony charges, custody disputes, probation violations

Characteristics:
  • Moderate legal complexity
  • Multi-party involvement or significant court proceedings
  • Moderate stakes (meaningful financial/personal consequences)
  • Requires strategic planning
  • Need for professional legal representation

Recommended Action:
  • Consult with qualified attorney soon
  • Prepare documentation and timeline
  • Budget for reasonable legal fees
  • Plan for court appearances and discovery
""",
            3: """
TIER 3: HIGH-STAKES / SERIOUS FELONY
Typical Cases: Violent crimes, federal offenses, constitutional violations, serious property crimes

Characteristics:
  • Significant legal complexity
  • Serious criminal exposure or major civil consequences
  • High stakes (potential incarceration, major fines, loss of rights)
  • Requires specialized expertise
  • Multiple strategic options need evaluation

Recommended Action:
  • Engage experienced criminal defense or specialized counsel IMMEDIATELY
  • Do not discuss with law enforcement without attorney present
  • Preserve all evidence and documentation
  • Plan for extended legal process and significant investment
  • Consider expert witnesses or specialized services
""",
            4: """
TIER 4: COMPLEX / APPELLATE
Typical Cases: Supreme Court matters, capital cases, precedent-setting litigation

Characteristics:
  • Complex legal analysis and novel issues
  • Highest stakes possible (life/death, major constitutional issues, business-critical)
  • Requires appellate or specialized expertise
  • Multiple potential legal theories
  • Significant impact on other matters or stakeholders

Recommended Action:
  • Engage top-tier specialized legal counsel IMMEDIATELY
  • Consider appellate specialists and experts in the specific practice area
  • Expect lengthy and expensive legal process
  • Coordinate with other parties/stakeholders as needed
  • Prepare for potential policy or precedent implications
"""
        }
        
        return tier_details.get(tier, "Tier information unavailable.")
    
    @staticmethod
    def generate_summary_report(
        query: str,
        answer: str,
        tier: int,
        tier_description: str,
        tier_reasoning: str
    ) -> str:
        """
        Generate a shorter summary report (useful for quick review).
        
        Args:
            query: Original user query
            answer: RAG-generated answer
            tier: Complexity tier (1-4)
            tier_description: Tier description
            tier_reasoning: Why this tier was assigned
        
        Returns:
            Formatted summary report string
        """
        
        report_parts = []
        
        report_parts.append("LEGAL RESEARCH SUMMARY")
        report_parts.append("=" * 60)
        report_parts.append(f"\nTier: {tier} - {tier_description}")
        report_parts.append(f"Basis: {tier_reasoning}")
        report_parts.append(f"\nQuery: {query[:150]}{'...' if len(query) > 150 else ''}")
        report_parts.append("\n" + "-" * 60)
        report_parts.append("FINDINGS:\n")
        report_parts.append(answer)
        report_parts.append("\n" + "=" * 60)
        
        return "\n".join(report_parts)
