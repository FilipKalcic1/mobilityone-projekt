"""
Query Type Classifier - Types and data structures only.

WARNING: The regex-based classification logic has been REMOVED.
Use services.intent_classifier.classify_query_type_ml() instead.

The ML-based classifier in intent_classifier.py:
- Has 100% accuracy vs ~67% regex
- Uses TF-IDF + LogisticRegression
- Training data in data/training/query_type.jsonl
- Automatically trained on first use

This file contains ONLY:
1. QueryType enum (used by unified_search.py)
2. QueryTypeResult dataclass (used by unified_search.py)
3. Backwards-compatible functions that delegate to ML

Removed 350+ lines of regex patterns - now ML-only.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Tipovi upita za suffix filtering."""
    SINGLE_ENTITY = "single_entity"      # Dohvati jednu stavku po ID-u
    DOCUMENTS = "documents"               # Rad s dokumentima/prilozima
    METADATA = "metadata"                 # Metapodaci, struktura, shema
    LIST = "list"                         # Lista svih stavki
    AGGREGATION = "aggregation"           # Agregacije, statistike
    TREE = "tree"                         # Hijerarhijska struktura
    DELETE_CRITERIA = "delete_criteria"   # Delete by criteria
    BULK_UPDATE = "bulk_update"           # Bulk/batch update
    DEFAULT_SET = "default_set"           # Postavljanje kao zadano
    THUMBNAIL = "thumbnail"               # Thumbnails, preview
    PROJECTION = "projection"             # Projection of specific fields
    UNKNOWN = "unknown"                   # Nije prepoznato


@dataclass
class QueryTypeResult:
    """Rezultat klasifikacije upita."""
    query_type: QueryType
    confidence: float
    matched_pattern: str
    preferred_suffixes: List[str]      # Preferred suffixes
    excluded_suffixes: List[str]       # Excluded suffixes


# Backwards-compatible interface - delegates to ML classifier
class QueryTypeClassifier:
    """
    DEPRECATED: Use classify_query_type_ml() from intent_classifier instead.
    This class exists only for backwards compatibility.
    """

    def __init__(self):
        """Initialize - does nothing, ML classifier is lazy-loaded."""
        pass

    def classify(self, query: str) -> QueryTypeResult:
        """
        Classify query type using ML.
        DEPRECATED: Use classify_query_type_ml() directly.
        """
        from services.intent_classifier import classify_query_type_ml

        ml_result = classify_query_type_ml(query)

        # Convert ML result to QueryTypeResult
        try:
            query_type = QueryType[ml_result.query_type]
        except KeyError:
            query_type = QueryType.UNKNOWN

        return QueryTypeResult(
            query_type=query_type,
            confidence=ml_result.confidence,
            matched_pattern="ML",
            preferred_suffixes=ml_result.preferred_suffixes,
            excluded_suffixes=ml_result.excluded_suffixes
        )


# Singleton instance
_classifier: Optional[QueryTypeClassifier] = None


def get_query_type_classifier() -> QueryTypeClassifier:
    """Get singleton instance - returns ML-backed classifier."""
    global _classifier
    if _classifier is None:
        _classifier = QueryTypeClassifier()
    return _classifier


def classify_query_type(query: str) -> QueryTypeResult:
    """
    Classify query type using ML.

    DEPRECATED: Use classify_query_type_ml() from intent_classifier instead.
    This function exists for backwards compatibility.
    """
    classifier = get_query_type_classifier()
    return classifier.classify(query)
