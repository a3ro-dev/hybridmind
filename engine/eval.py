import numpy as np
from typing import List, Set

class EvalMetrics:
    """
    Standard information retrieval evaluation metrics.
    """
    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        if not retrieved or not ground_truth:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant = len([doc for doc in retrieved_k if doc in ground_truth])
        return relevant / len(retrieved_k) if retrieved_k else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        if not ground_truth:
            return 0.0
        if not retrieved:
            return 0.0
        retrieved_k = retrieved[:k]
        relevant = len([doc for doc in retrieved_k if doc in ground_truth])
        return relevant / len(ground_truth)

    @staticmethod
    def mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
        if not retrieved or not ground_truth:
            return 0.0
        for i, doc in enumerate(retrieved):
            if doc in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain (binary relevance).
        """
        if not retrieved or not ground_truth:
            return 0.0
            
        retrieved_k = retrieved[:k]
        dcg = 0.0
        for i, doc in enumerate(retrieved_k):
            if doc in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed and log2(1) is 0
                
        # Calculate IDCG (Ideal DCG)
        idcg = 0.0
        num_ideal = min(len(ground_truth), k)
        for i in range(num_ideal):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
