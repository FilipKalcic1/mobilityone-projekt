"""
BM25 Index — exact term matching complement to FAISS semantic search.

FAISS uses dense embeddings (cosine similarity in 1536-dim space).
BM25 uses sparse term frequencies (exact word matching).

When a user says an exact term from documentation (e.g., "VehicleTypes"),
FAISS might rank it lower than a semantically similar but wrong tool.
BM25 catches these exact matches.

Score merge in unified_search.py:
    final_score = faiss_cosine + bm25_boost
    where bm25_boost = BM25_WEIGHT * normalized_bm25_score

No external dependencies — uses a simple TF-IDF based BM25 implementation.
"""
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# BM25 parameters
_K1 = 1.5   # Term frequency saturation
_B = 0.75   # Length normalization


def _tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, split on non-alphanumeric, remove short tokens."""
    text = text.lower()
    # Remove Croatian diacritics for matching
    diacritics = str.maketrans({
        'č': 'c', 'ć': 'c', 'ž': 'z', 'š': 's', 'đ': 'd',
    })
    text = text.translate(diacritics)
    tokens = re.split(r'[^a-z0-9]+', text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """
    Lightweight BM25 index for tool documentation.

    Built from tool_documentation.json fields:
    - purpose
    - when_to_use
    - example_queries_hr
    - synonyms_hr
    - tool_id itself (split on underscores)
    """

    def __init__(self):
        self._tool_ids: List[str] = []
        self._tool_id_to_idx: Dict[str, int] = {}
        self._doc_tokens: List[List[str]] = []  # tokenized docs
        self._doc_freq: Dict[str, int] = {}     # document frequency per term
        self._avg_dl: float = 0.0               # average document length
        self._n_docs: int = 0
        self._built = False

    def build(self, tool_docs: dict):
        """Build BM25 index from tool_documentation.json."""
        self._tool_ids = []
        self._tool_id_to_idx = {}
        self._doc_tokens = []
        self._doc_freq = {}

        for tool_id, doc in tool_docs.items():
            idx = len(self._tool_ids)
            self._tool_ids.append(tool_id)
            self._tool_id_to_idx[tool_id.lower()] = idx

            # Build document text from all relevant fields
            parts = []

            # Tool ID itself (split on underscores for matching)
            parts.append(tool_id.replace("_", " "))

            # Purpose
            purpose = doc.get("purpose", "")
            if purpose:
                parts.append(purpose)

            # When to use
            when_to_use = doc.get("when_to_use", [])
            if when_to_use:
                parts.extend(when_to_use)

            # Example queries (Croatian)
            examples = doc.get("example_queries_hr", [])
            if examples:
                parts.extend(examples)

            # Synonyms (Croatian)
            synonyms = doc.get("synonyms_hr", [])
            if synonyms:
                parts.extend(synonyms)

            text = " ".join(parts)
            tokens = _tokenize(text)
            self._doc_tokens.append(tokens)

            # Track document frequency (unique terms per doc)
            seen = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
                    seen.add(token)

        self._n_docs = len(self._tool_ids)
        total_tokens = sum(len(d) for d in self._doc_tokens)
        self._avg_dl = total_tokens / max(self._n_docs, 1)
        self._built = True

        logger.info(
            f"BM25Index: Built index for {self._n_docs} tools, "
            f"vocab={len(self._doc_freq)}, avg_dl={self._avg_dl:.1f}"
        )

    def _bm25_term_score(self, token: str, idx: int) -> float:
        """Calculate BM25 score for a single term in a document."""
        df = self._doc_freq.get(token, 0)
        if df == 0:
            return 0.0

        doc_tokens = self._doc_tokens[idx]
        tf = doc_tokens.count(token)
        if tf == 0:
            return 0.0

        dl = len(doc_tokens)
        idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf * (_K1 + 1)
        denominator = tf + _K1 * (1 - _B + _B * dl / self._avg_dl)
        return idf * numerator / denominator

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search for tools matching query terms.

        Returns:
            List of (tool_id, bm25_score) sorted by score descending.
        """
        if not self._built:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = {}

        for token in query_tokens:
            if self._doc_freq.get(token, 0) == 0:
                continue

            for idx in range(len(self._doc_tokens)):
                term_score = self._bm25_term_score(token, idx)
                if term_score > 0:
                    scores[idx] = scores.get(idx, 0.0) + term_score

        # Sort by score and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [(self._tool_ids[idx], score) for idx, score in ranked]

    def get_score(self, query: str, tool_id: str) -> float:
        """Get BM25 score for a specific tool given a query."""
        if not self._built:
            return 0.0

        idx = self._tool_id_to_idx.get(tool_id.lower())
        if idx is None:
            return 0.0

        query_tokens = _tokenize(query)
        if not query_tokens:
            return 0.0

        score = 0.0
        for token in query_tokens:
            score += self._bm25_term_score(token, idx)

        return score

    def get_scores_batch(self, query: str, tool_ids: List[str]) -> Dict[str, float]:
        """Get BM25 scores for a batch of tools (used during boost pipeline)."""
        if not self._built:
            return {}

        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        result = {}
        for tool_id in tool_ids:
            idx = self._tool_id_to_idx.get(tool_id.lower())
            if idx is None:
                continue

            score = 0.0
            for token in query_tokens:
                score += self._bm25_term_score(token, idx)

            if score > 0:
                result[tool_id] = score

        return result

    @property
    def is_built(self) -> bool:
        return self._built


# Singleton
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Get singleton BM25Index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
