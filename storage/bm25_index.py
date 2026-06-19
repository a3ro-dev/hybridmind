import os
import math
import pickle
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


class PorterStemmer:
    """Pure-Python Porter stemmer (Porter, 1980). No external dependencies."""

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        w = word.lower()
        w = self._step_1a(w)
        w = self._step_1b(w)
        w = self._step_1c(w)
        w = self._step_2(w)
        w = self._step_3(w)
        w = self._step_4(w)
        w = self._step_5a(w)
        w = self._step_5b(w)
        return w

    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        c = word[i]
        if c in 'aeiou':
            return False
        if c == 'y':
            if i == 0:
                return True
            return not PorterStemmer._is_consonant(word, i - 1)
        return True

    @staticmethod
    def _measure(word: str) -> int:
        """Return the VC (vowel-consonant) measure m."""
        n = 0
        i = 0
        L = len(word)
        while i < L and PorterStemmer._is_consonant(word, i):
            i += 1
        if i >= L:
            return 0
        while i < L:
            while i < L and not PorterStemmer._is_consonant(word, i):
                i += 1
            if i >= L:
                break
            while i < L and PorterStemmer._is_consonant(word, i):
                i += 1
            n += 1
        return n

    @staticmethod
    def _has_vowel(word: str) -> bool:
        for i, c in enumerate(word):
            if c in 'aeiou':
                return True
            if c == 'y' and i > 0 and PorterStemmer._is_consonant(word, i - 1):
                return True
        return False

    @staticmethod
    def _ends_with_double_consonant(word: str) -> bool:
        if len(word) < 2:
            return False
        c1, c2 = word[-2], word[-1]
        return PorterStemmer._is_consonant(word, -2) and PorterStemmer._is_consonant(word, -1) and c1 == c2

    @staticmethod
    def _ends_with_cvc(word: str) -> bool:
        if len(word) < 3:
            return False
        c1, v, c2 = word[-3], word[-2], word[-1]
        if not PorterStemmer._is_consonant(word, -3):
            return False
        if PorterStemmer._is_consonant(word, -2):
            return False
        if not PorterStemmer._is_consonant(word, -1):
            return False
        return c2 not in 'wxy'

    @staticmethod
    def _replace(word: str, suffix: str, replacement: str = '') -> tuple:
        if word.endswith(suffix):
            return word[:len(word) - len(suffix)] + replacement, True
        return word, False

    @staticmethod
    def _replace_m(word: str, suffix: str, replacement: str, min_m: int) -> tuple:
        stem, ok = PorterStemmer._replace(word, suffix, replacement)
        if ok and PorterStemmer._measure(stem) > min_m:
            return stem, True
        return word, False

    @staticmethod
    def _replace_m0(word: str, suffix: str, replacement: str = '') -> tuple:
        return PorterStemmer._replace_m(word, suffix, replacement, 0)

    @staticmethod
    def _replace_m1(word: str, suffix: str, replacement: str = '') -> tuple:
        return PorterStemmer._replace_m(word, suffix, replacement, 1)

    def _step_1a(self, w: str) -> str:
        if w.endswith('sses'):
            return w[:-2]
        if w.endswith('ies'):
            return w[:-2]
        if w.endswith('ss'):
            return w
        if w.endswith('s') and not w.endswith('us') and not w.endswith('ss'):
            return w[:-1]
        return w

    def _step_1b(self, w: str) -> str:
        w, ok = self._replace_m0(w, 'eed', 'ee')
        if ok:
            return w
        if w.endswith('ed'):
            stem = w[:-2]
            if self._has_vowel(stem):
                return self._step_1b_final(stem)
            return w
        if w.endswith('ing'):
            stem = w[:-3]
            if self._has_vowel(stem):
                return self._step_1b_final(stem)
            return w
        return w

    def _step_1b_final(self, w: str) -> str:
        if w.endswith('at') or w.endswith('bl') or w.endswith('iz'):
            return w + 'e'
        if self._ends_with_double_consonant(w) and w[-1] not in 'lsz':
            return w[:-1]
        if self._ends_with_cvc(w) and self._measure(w) == 1:
            return w + 'e'
        return w

    def _step_1c(self, w: str) -> str:
        if w.endswith('y') and len(w) > 2 and self._has_vowel(w[:-1]):
            return w[:-1] + 'i'
        return w

    def _step_2(self, w: str) -> str:
        repls = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
            ('izer', 'ize'), ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'),
            ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
        ]
        for suff, repl in repls:
            w, ok = self._replace_m0(w, suff, repl)
            if ok:
                return w
        return w

    def _step_3(self, w: str) -> str:
        repls = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'), ('iciti', 'ic'),
            ('ical', 'ic'), ('ful', ''), ('ness', ''),
        ]
        for suff, repl in repls:
            w, ok = self._replace_m0(w, suff, repl)
            if ok:
                return w
        return w

    def _step_4(self, w: str) -> str:
        suffixes = ['ement', 'ment', 'ent', 'ance', 'ence', 'able', 'ible', 'ant',
                     'ism', 'ate', 'iti', 'ous', 'ive', 'ize', 'al', 'er', 'ic']
        if len(w) >= 4 and w[-4] == 't' and w[-3:] == 'ion':
            stem = w[:-3]
            if self._measure(stem) > 1:
                return stem
        if len(w) >= 3 and w[-3] == 's' and w[-2:] == 'ion':
            stem = w[:-3]
            if self._measure(stem) > 1:
                return stem
        for suff in suffixes:
            w, ok = self._replace_m1(w, suff)
            if ok:
                return w
        return w

    def _step_5a(self, w: str) -> str:
        w, ok = self._replace_m1(w, 'e')
        if ok:
            return w
        if w.endswith('e') and not self._ends_with_cvc(w):
            m = self._measure(w[:-1])
            if m == 1:
                return w[:-1]
        return w

    def _step_5b(self, w: str) -> str:
        if self._measure(w) > 1 and self._ends_with_double_consonant(w) and w[-1] == 'l':
            return w[:-1]
        return w


# English stop words (common words with little informational value)
_STOP_WORDS: set = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'by', 'with', 'from', 'up', 'about', 'into', 'over', 'after',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'can', 'could', 'shall', 'should',
    'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'am', 'so', 'if', 'no', 'not', 'nor',
    'as', 'very', 'just', 'than', 'then', 'now', 'also', 'well', 'here', 'there',
    'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'own', 'same', 'too', 'any', 'really', 'much', 'many',
}


class BM25Index:
    """
    In-memory and disk-backed Okapi BM25 Index.
    Implemented in pure Python to eliminate dependency issues.

    Features:
    - Pure-Python Porter stemming for morphological normalization
    - Stop word filtering to reduce noise
    - Pickle-based persistence
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

        # Lazy-init stemmer on first use
        self._stemmer_obj = None

    def _stem(self, word: str) -> str:
        """Stem a single word using the Porter stemmer."""
        if self._stemmer_obj is None:
            self._stemmer_obj = PorterStemmer()
        return self._stemmer_obj.stem(word)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize with stemming and stop word removal."""
        import re
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        # Filter single-char tokens (usually not meaningful after stemming)
        tokens = [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]
        # Apply Porter stemming to normalize morphological variants
        tokens = [self._stem(t) for t in tokens]
        # Filter tokens that collapsed to < 2 chars after stemming
        tokens = [t for t in tokens if len(t) > 1]
        return tokens

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
