import os
import math
import pickle
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

class BM25Index:
    """
    In-memory and disk-backed Okapi BM25 Index.
    Implemented in pure Python to eliminate dependency issues.
    """
    def __init__(self, index_path: str = None, k1: float = 1.5, b: float = 0.75):
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_lengths = {}
        self.doc_freqs = {}      # term -> num docs containing term
        self.term_freqs = {}     # doc_id -> { term -> freq }
        self.idf = {}            # term -> idf score
        
        self.doc_ids = []
        
    def _init_nltk(self):
        """Initialize NLTK stemmer if not already done."""
        if not hasattr(self, '_stemmer'):
            import nltk
            try:
                self._stemmer = nltk.stem.PorterStemmer()
            except LookupError:
                nltk.download('punkt')
                nltk.download('punkt_tab')
                self._stemmer = nltk.stem.PorterStemmer()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize using regex filtering and NLTK PorterStemmer."""
        self._init_nltk()
        import re
        # Strip all non-alphanumeric characters, lowercase, split
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return [self._stemmer.stem(t) for t in tokens if len(t) > 1]

    def add(self, node_id: str, text: str):
        if node_id in self.doc_lengths:
            # Handle update by removing first (naive approach)
            pass
            
        tokens = self.tokenize(text)
        if not tokens:
            return
            
        self.doc_count += 1
        self.doc_lengths[node_id] = len(tokens)
        self.doc_ids.append(node_id)
        
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
            
        self.term_freqs[node_id] = counts
        for t in counts:
            self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
            
        # Recompute avgdl
        total_len = sum(self.doc_lengths.values())
        self.avgdl = total_len / self.doc_count
        
        # We can lazy compute idf during search or precompute here.
        # Since addition happens one by one, we will recompute lazily during search.

    def add_batch(self, batch: List[Tuple[str, str]]):
        for node_id, text in batch:
            self.add(node_id, text)

    def _compute_idf(self, term: str) -> float:
        if term in self.idf:
            return self.idf[term]
        
        n_q = self.doc_freqs.get(term, 0)
        # Standard BM25 IDF formula
        idf = math.log(((self.doc_count - n_q + 0.5) / (n_q + 0.5)) + 1.0)
        self.idf[term] = idf
        return idf

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.doc_count == 0:
            return []
            
        tokens = self.tokenize(query_text)
        if not tokens:
            return []
            
        # Precompute IDF for query terms
        for t in tokens:
            self._compute_idf(t)
            
        scores = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        for t in tokens:
            idf = self.idf.get(t, 0.0)
            if idf <= 0:
                continue
                
            for doc_id in self.doc_ids:
                tf = self.term_freqs[doc_id].get(t, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[doc_id]
                    # Score computation
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_id] += idf * (numerator / denominator)
                    
        # Sort by score desc
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [res for res in sorted_docs[:top_k] if res[1] > 0.0]

    def clear(self):
        self.doc_count = 0
        self.avgdl = 0.0
        self.doc_lengths = {}
        self.doc_freqs = {}
        self.term_freqs = {}
        self.idf = {}
        self.doc_ids = []
        if self.index_path and os.path.exists(self.index_path):
            os.remove(self.index_path)

    def save(self):
        if not self.index_path:
            return
        logger.info(f"Saving BM25 index to {self.index_path}")
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'doc_count': self.doc_count,
                'avgdl': self.avgdl,
                'doc_lengths': self.doc_lengths,
                'doc_freqs': self.doc_freqs,
                'term_freqs': self.term_freqs,
                'doc_ids': self.doc_ids
            }, f)

    def load(self):
        if not self.index_path or not os.path.exists(self.index_path):
            return
        logger.info(f"Loading BM25 index from {self.index_path}")
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.doc_count = data['doc_count']
            self.avgdl = data['avgdl']
            self.doc_lengths = data['doc_lengths']
            self.doc_freqs = data['doc_freqs']
            self.term_freqs = data['term_freqs']
            self.doc_ids = data['doc_ids']
            
    def rebuild_from_nodes(self, nodes: List[Dict[str, Any]]):
        self.clear()
        for node in nodes:
            self.add(node["id"], node["text"])

    @property
    def size(self) -> int:
        return self.doc_count
