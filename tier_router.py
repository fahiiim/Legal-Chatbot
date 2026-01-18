"""
Tier Classification Router
Classifies legal queries into complexity tiers.
"""

import re
from typing import Dict, Tuple
from config import TIER_KEYWORDS


class TierRouter:
    """
    Classifies legal queries into 4 tiers based on complexity.
    """
    
    TIER_DESCRIPTIONS = {
        1: "Routine / Low-Risk (traffic tickets, civil infractions, name changes, small claims)",
        2: "Moderate / Litigation (felony charges, contested hearings, motion practice)",
        3: "High-Stakes / Serious Felony (violent crimes, federal cases, constitutional issues)",
        4: "Complex / Appellate (Supreme Court, capital cases, precedent-setting)"
    }
    
    @classmethod
    def classify_tier(cls, query: str) -> Tuple[int, str, str]:
        """
        Classify query into a tier.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (tier_number, tier_description, reasoning)
        """
        query_lower = query.lower()
        scores = {1: 0, 2: 0, 3: 0, 4: 0}
        matched_keywords = {1: [], 2: [], 3: [], 4: []}
        
        # Score based on keyword matches
        for tier, keywords in TIER_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    scores[tier] += 1
                    matched_keywords[tier].append(keyword)
        
        # Determine tier (highest score wins)
        tier = max(scores, key=scores.get)
        
        # If no keywords matched, default to Tier 2 (moderate)
        if all(score == 0 for score in scores.values()):
            tier = 2
            reasoning = "No specific tier indicators found; defaulting to Tier 2 (moderate complexity)"
        else:
            # Create reasoning
            reasoning = f"Query matched "
            matched = []
            for t, kws in matched_keywords.items():
                if kws:
                    matched.append(f"Tier {t}: {', '.join(kws)}")
            reasoning += "; ".join(matched)
        
        return tier, cls.TIER_DESCRIPTIONS[tier], reasoning
    
    @classmethod
    def get_tier_recommendation(cls, tier: int) -> str:
        """
        Get recommendation based on tier.
        
        Args:
            tier: Tier number (1-4)
        
        Returns:
            Recommendation string
        """
        recommendations = {
            1: "This appears to be a routine matter. You may be able to handle this yourself or with limited legal assistance.",
            2: "This matter involves moderate complexity. Consider consulting with an attorney to understand your options.",
            3: "This is a serious matter with significant consequences. Strong legal representation is recommended.",
            4: "This is a complex legal matter requiring specialized expertise. Seek experienced legal counsel immediately."
        }
        
        return recommendations.get(tier, "Consult with a legal professional for guidance.")
