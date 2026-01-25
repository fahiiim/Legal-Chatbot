"""
RAG Evaluation Framework
Implements comprehensive metrics for retrieval and generation quality.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from langchain_openai import ChatOpenAI


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard IR metrics."""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[Document], 
                       relevant_doc_ids: List[str], 
                       k: Optional[int] = None) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs (ground truth)
            k: Top K to consider (None = all)
        
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        if k:
            retrieved_docs = retrieved_docs[:k]
        
        retrieved_ids = [doc.metadata.get('source', str(i)) for i, doc in enumerate(retrieved_docs)]
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        
        return relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[Document],
                    relevant_doc_ids: List[str],
                    k: Optional[int] = None) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs (ground truth)
            k: Top K to consider
        
        Returns:
            Recall score (0-1)
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        if k:
            retrieved_docs = retrieved_docs[:k]
        
        retrieved_ids = [doc.metadata.get('source', str(i)) for i, doc in enumerate(retrieved_docs)]
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        
        return relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[Document],
                            relevant_doc_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
        
        Returns:
            MRR score
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        retrieved_ids = [doc.metadata.get('source', str(i)) for i, doc in enumerate(retrieved_docs)]
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[Document],
                  relevance_scores: Dict[str, float],
                  k: Optional[int] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved_docs: List of retrieved documents
            relevance_scores: Dict mapping doc_id to relevance score (0-1 or 0-5)
            k: Top K to consider
        
        Returns:
            NDCG score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        if k:
            retrieved_docs = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.metadata.get('source', str(i))
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 1)
        
        # Calculate Ideal DCG
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:len(retrieved_docs)]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @classmethod
    def evaluate_retrieval(cls,
                          retrieved_docs: List[Document],
                          relevant_doc_ids: List[str],
                          relevance_scores: Optional[Dict[str, float]] = None,
                          k: int = 5) -> Dict:
        """
        Comprehensive retrieval evaluation.
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: Ground truth relevant doc IDs
            relevance_scores: Optional relevance scores for NDCG
            k: Top K for metrics
        
        Returns:
            Dictionary of metrics
        """
        precision = cls.precision_at_k(retrieved_docs, relevant_doc_ids, k)
        recall = cls.recall_at_k(retrieved_docs, relevant_doc_ids, k)
        f1 = cls.f1_score(precision, recall)
        mrr = cls.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids)
        
        metrics = {
            f"precision@{k}": round(precision, 4),
            f"recall@{k}": round(recall, 4),
            f"f1@{k}": round(f1, 4),
            "mrr": round(mrr, 4),
            "num_retrieved": len(retrieved_docs[:k]),
            "num_relevant": len(relevant_doc_ids)
        }
        
        if relevance_scores:
            ndcg = cls.ndcg_at_k(retrieved_docs, relevance_scores, k)
            metrics[f"ndcg@{k}"] = round(ndcg, 4)
        
        return metrics


class GenerationEvaluator:
    """Evaluates generation quality using LLM-as-judge."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """
        Initialize generation evaluator.
        
        Args:
            llm_model: LLM model for evaluation (use cheaper model)
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
    
    def evaluate_faithfulness(self, answer: str, sources: List[Document]) -> Dict:
        """
        Evaluate if answer is faithful to source documents.
        
        Args:
            answer: Generated answer
            sources: Source documents used
        
        Returns:
            Dictionary with faithfulness score and analysis
        """
        sources_text = "\n\n".join([f"Source {i+1}: {doc.page_content[:500]}" 
                                    for i, doc in enumerate(sources[:3])])
        
        prompt = f"""Evaluate if the answer is faithful to the provided sources.

Sources:
{sources_text}

Answer:
{answer}

Rate the faithfulness on a scale of 1-5:
1 - Completely unfaithful, contradicts sources or invents information
2 - Mostly unfaithful with some accurate elements
3 - Partially faithful, mix of supported and unsupported claims
4 - Mostly faithful with minor unsupported details
5 - Completely faithful, all claims supported by sources

Return a JSON object with:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>",
    "unsupported_claims": ["<claim1>", "<claim2>"],
    "is_faithful": <true/false>
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            # Extract JSON from response
            content = response.content
            
            # Try to parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            result['is_faithful'] = result.get('score', 0) >= 4
            return result
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return {
                "score": 0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "unsupported_claims": [],
                "is_faithful": False
            }
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> Dict:
        """
        Evaluate if answer is relevant to the question.
        
        Args:
            question: User question
            answer: Generated answer
        
        Returns:
            Dictionary with relevancy score and analysis
        """
        prompt = f"""Evaluate if the answer is relevant to the question.

Question:
{question}

Answer:
{answer}

Rate the relevancy on a scale of 1-5:
1 - Completely irrelevant, doesn't address the question
2 - Mostly irrelevant, only tangentially related
3 - Partially relevant, addresses some aspects
4 - Mostly relevant, addresses main question with minor gaps
5 - Completely relevant, fully addresses the question

Return a JSON object with:
{{
    "score": <1-5>,
    "reasoning": "<brief explanation>",
    "is_relevant": <true/false>
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            result['is_relevant'] = result.get('score', 0) >= 4
            return result
        except Exception as e:
            print(f"Error evaluating relevancy: {e}")
            return {
                "score": 0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "is_relevant": False
            }
    
    def evaluate_completeness(self, question: str, answer: str, sources: List[Document]) -> Dict:
        """
        Evaluate if answer is complete given available sources.
        
        Args:
            question: User question
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Dictionary with completeness analysis
        """
        sources_text = "\n\n".join([f"Source {i+1}: {doc.page_content[:500]}" 
                                    for i, doc in enumerate(sources[:3])])
        
        prompt = f"""Evaluate if the answer is complete given the question and available sources.

Question:
{question}

Available Sources:
{sources_text}

Answer:
{answer}

Rate completeness on a scale of 1-5:
1 - Highly incomplete, missing critical information available in sources
2 - Mostly incomplete, several important gaps
3 - Partially complete, addresses main points but missing details
4 - Mostly complete, minor details missing
5 - Complete, fully utilizes available source information

Return JSON:
{{
    "score": <1-5>,
    "reasoning": "<explanation>",
    "missing_information": ["<what's missing>"],
    "is_complete": <true/false>
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            result['is_complete'] = result.get('score', 0) >= 4
            return result
        except Exception as e:
            print(f"Error evaluating completeness: {e}")
            return {
                "score": 0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "missing_information": [],
                "is_complete": False
            }
    
    def comprehensive_evaluation(self, 
                                 question: str, 
                                 answer: str, 
                                 sources: List[Document]) -> Dict:
        """
        Run all generation evaluations.
        
        Args:
            question: User question
            answer: Generated answer
            sources: Source documents
        
        Returns:
            Comprehensive evaluation results
        """
        faithfulness = self.evaluate_faithfulness(answer, sources)
        relevancy = self.evaluate_answer_relevancy(question, answer)
        completeness = self.evaluate_completeness(question, answer, sources)
        
        # Calculate overall score
        overall_score = (
            faithfulness.get('score', 0) * 0.4 +
            relevancy.get('score', 0) * 0.3 +
            completeness.get('score', 0) * 0.3
        )
        
        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "completeness": completeness,
            "overall_score": round(overall_score, 2),
            "overall_quality": "excellent" if overall_score >= 4.5 else
                             "good" if overall_score >= 3.5 else
                             "acceptable" if overall_score >= 2.5 else "poor",
            "timestamp": datetime.now().isoformat()
        }


class RAGEvaluator:
    """Combined RAG evaluation for end-to-end assessment."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """Initialize RAG evaluator."""
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator(llm_model)
        self.evaluation_history = []
    
    def evaluate_rag_response(self,
                             question: str,
                             answer: str,
                             retrieved_docs: List[Document],
                             relevant_doc_ids: Optional[List[str]] = None,
                             run_generation_eval: bool = True) -> Dict:
        """
        Comprehensive RAG evaluation.
        
        Args:
            question: User question
            answer: Generated answer
            retrieved_docs: Retrieved documents
            relevant_doc_ids: Ground truth relevant docs (if available)
            run_generation_eval: Whether to run LLM-based generation eval
        
        Returns:
            Complete evaluation results
        """
        results = {
            "question": question,
            "timestamp": datetime.now().isoformat()
        }
        
        # Retrieval evaluation (if ground truth available)
        if relevant_doc_ids:
            results["retrieval_metrics"] = self.retrieval_eval.evaluate_retrieval(
                retrieved_docs, relevant_doc_ids, k=5
            )
        
        # Generation evaluation
        if run_generation_eval:
            results["generation_metrics"] = self.generation_eval.comprehensive_evaluation(
                question, answer, retrieved_docs
            )
        
        # Store in history
        self.evaluation_history.append(results)
        
        return results
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict:
        """
        Calculate average metrics across evaluations.
        
        Args:
            last_n: Number of recent evaluations to average (None = all)
        
        Returns:
            Average metrics
        """
        if not self.evaluation_history:
            return {}
        
        history = self.evaluation_history[-last_n:] if last_n else self.evaluation_history
        
        # Average retrieval metrics
        retrieval_metrics = defaultdict(list)
        for eval_result in history:
            if "retrieval_metrics" in eval_result:
                for key, value in eval_result["retrieval_metrics"].items():
                    if isinstance(value, (int, float)):
                        retrieval_metrics[key].append(value)
        
        avg_retrieval = {key: round(np.mean(values), 4) 
                        for key, values in retrieval_metrics.items()}
        
        # Average generation metrics
        generation_scores = []
        for eval_result in history:
            if "generation_metrics" in eval_result:
                score = eval_result["generation_metrics"].get("overall_score", 0)
                generation_scores.append(score)
        
        avg_generation = round(np.mean(generation_scores), 2) if generation_scores else 0.0
        
        return {
            "retrieval_metrics": avg_retrieval,
            "average_generation_score": avg_generation,
            "num_evaluations": len(history)
        }
    
    def save_evaluation_history(self, filepath: str = "evaluation_history.json"):
        """Save evaluation history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        print(f"Saved {len(self.evaluation_history)} evaluations to {filepath}")
    
    def load_evaluation_history(self, filepath: str = "evaluation_history.json"):
        """Load evaluation history from file."""
        try:
            with open(filepath, 'r') as f:
                self.evaluation_history = json.load(f)
            print(f"Loaded {len(self.evaluation_history)} evaluations from {filepath}")
        except FileNotFoundError:
            print(f"No evaluation history found at {filepath}")
