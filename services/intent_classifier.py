"""
Intent Classifier - ML-based intent classification.

REPLACES:
- action_intent_detector.py (414 lines of regex)
- query_router.py routing logic (660 lines of regex)

Uses trained ML model for intent classification.
Accuracy updated after each retrain — see training output for current metrics.
"""

import hashlib
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from services.dynamic_threshold import ClassificationSignal, NO_SIGNAL, PredictionSet
from services.errors import ClassificationError, ErrorCode, SecurityError
from services.tracing import get_tracer, trace_span

# Text normalization — canonical source is text_normalizer.py.
# Re-exported here for backward compatibility (dozens of importers).
from services.text_normalizer import (  # noqa: F401 — re-export
    DIACRITIC_MAP,
    SYNONYM_MAP,
    normalize_diacritics,
    normalize_query,
    normalize_synonyms,
)

logger = logging.getLogger(__name__)
_tracer = get_tracer("intent_classifier")



# Action intent detection — canonical source is action_intent_detector.py.
# Re-exported here for backward compatibility.
from services.action_intent_detector import (  # noqa: F401 — re-export
    ActionIntent,
    IntentDetectionResult,
    get_allowed_methods,
)

# Default paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "intent"
TRAINING_DATA_PATH = Path(__file__).parent.parent / "data" / "training" / "intent_full.jsonl"


@dataclass
class IntentPrediction:
    """Result of intent prediction.

    signal: ClassificationSignal computed from the full probability vector.
    When constructed from predict methods, this is exact (from_probabilities).
    When constructed without probs (e.g. fallback), estimated from alternatives.
    """
    intent: str
    action: str
    tool: Optional[str]
    confidence: float
    alternatives: Optional[List[Tuple[str, float]]] = None
    signal: Optional[ClassificationSignal] = None
    prediction_set: Optional[PredictionSet] = None

    # Number of intent classes — read from model metadata at load time,
    # fallback to this default for signal estimation without a loaded model.
    _N_CLASSES_DEFAULT: int = 29

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.signal is None:
            # Fallback: estimate signal from sparse alternatives
            self.signal = ClassificationSignal.from_alternatives(
                self.confidence, self.alternatives,
                n_classes=self._N_CLASSES_DEFAULT,
            )


class IntentClassifier:
    """
    ML-based intent classifier.

    Supports multiple algorithms:
    - 'tfidf_lr': TF-IDF + Logistic Regression (fast, word patterns only)
    - 'azure_embedding': Azure OpenAI embeddings (SEMANTIC understanding, best)
    - 'sbert_lr': Sentence-BERT + Logistic Regression (offline semantic)
    - 'fasttext': FastText classifier (good balance)

    RECOMMENDATION: Use 'azure_embedding' for production - it understands MEANING,
    not just word patterns. Handles typos, novel phrasings, and generalizes
    to queries never seen in training.
    """

    def __init__(self, algorithm: str = "tfidf_lr", model_path: Optional[Path] = None) -> None:
        """
        Initialize classifier.

        Args:
            algorithm: One of 'tfidf_lr', 'sbert_lr', 'fasttext'
            model_path: Path to trained model directory
        """
        self.algorithm = algorithm
        self.model_path = model_path or MODEL_DIR
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.intent_to_metadata = {}  # Maps intent -> (action, tool)
        self._loaded = False
        self._q_hat = None
        self._cp_coverage = None

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _verify_model_integrity(model_file: Path) -> bool:
        """Verify model file integrity via SHA-256 sidecar.

        Returns True if hash matches or no sidecar exists (first load).
        Raises SecurityError if hash mismatch (tampering detected).
        """
        hash_file = model_file.with_suffix(model_file.suffix + ".sha256")
        if not hash_file.exists():
            logger.warning(
                f"No integrity hash for {model_file.name} — "
                f"run training to generate .sha256 sidecar"
            )
            # Generate hash for existing model (bootstrap)
            actual_hash = IntentClassifier._compute_file_hash(model_file)
            hash_file.write_text(actual_hash, encoding="utf-8")
            logger.info(f"Generated integrity hash for {model_file.name}")
            return True

        expected_hash = hash_file.read_text(encoding="utf-8").strip()
        actual_hash = IntentClassifier._compute_file_hash(model_file)
        if actual_hash != expected_hash:
            raise SecurityError(
                ErrorCode.MODEL_INTEGRITY_FAILED,
                f"Model integrity check FAILED for {model_file.name}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}... "
                f"— file may have been tampered with"
            )
        return True

    def load(self) -> bool:
        """Load trained model from disk with integrity verification."""
        model_file = None
        try:
            model_file = self.model_path / f"{self.algorithm}_model.pkl"
            meta_file = self.model_path / "metadata.json"

            if not model_file.exists():
                logger.warning(f"Model file not found: {model_file}")
                return False

            # SECURITY: Verify SHA-256 integrity before deserializing pickle
            self._verify_model_integrity(model_file)

            with open(model_file, "rb") as f:
                saved = pickle.load(f)
                self.model = saved["model"]
                self.vectorizer = saved.get("vectorizer")
                self.label_encoder = saved["label_encoder"]

            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.intent_to_metadata = json.load(f)

            # Load conformal prediction calibration if available
            cp_file = self.model_path / "cp_calibration.json"
            if cp_file.exists():
                with open(cp_file, "r", encoding="utf-8") as f:
                    cp_data = json.load(f)
                self._q_hat = cp_data["q_hat"]
                self._cp_coverage = cp_data.get("coverage_target", 0.70)
                logger.info(
                    f"Loaded CP calibration: q_hat={self._q_hat:.4f}, "
                    f"coverage={self._cp_coverage}"
                )
            else:
                self._q_hat = None
                self._cp_coverage = None

            self._loaded = True
            logger.info(f"Loaded {self.algorithm} model from {model_file}")
            return True

        except SecurityError:
            logger.critical(f"MODEL INTEGRITY FAILURE — refusing to load {model_file}")
            raise
        except Exception as e:
            path_str = str(model_file) if model_file else str(self.model_path)
            err = ClassificationError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"Failed to load {self.algorithm} model: {e}",
                metadata={"model_path": path_str},
                cause=e,
            )
            logger.error(str(err))
            return False

    def train(self, training_data_path: Optional[Path] = None) -> Dict[str, float]:
        """Train the classifier. Delegates to services.intent_training."""
        from services.intent_training import (
            load_training_data, train_tfidf_lr, train_sbert_lr,
            train_fasttext, train_azure_embedding, save_metadata,
        )
        data_path = training_data_path or TRAINING_DATA_PATH
        texts, labels, metadata = load_training_data(data_path)

        dispatch = {
            "tfidf_lr": train_tfidf_lr,
            "sbert_lr": train_sbert_lr,
            "fasttext": train_fasttext,
            "azure_embedding": train_azure_embedding,
        }
        train_fn = dispatch.get(self.algorithm)
        if train_fn is None:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        metrics = train_fn(self, texts, labels)
        self.intent_to_metadata = metadata
        save_metadata(self)
        self._loaded = True
        return metrics

    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent for a text query.

        Args:
            text: User query

        Returns:
            IntentPrediction with intent, action, tool, and confidence
        """
        with trace_span(_tracer, "classifier.predict", {"algorithm": self.algorithm, "query.length": len(text)}) as span:
            if not self._loaded:
                if not self.load():
                    return IntentPrediction(
                        intent="UNKNOWN",
                        action="NONE",
                        tool=None,
                        confidence=0.0
                    )

            # Apply normalization: lowercase, diacritics, synonyms
            text_clean = normalize_query(text)

            if self.algorithm == "tfidf_lr":
                result = self._predict_tfidf_lr(text_clean)
            elif self.algorithm == "sbert_lr":
                result = self._predict_sbert_lr(text_clean)
            elif self.algorithm == "fasttext":
                result = self._predict_fasttext(text_clean)
            elif self.algorithm == "azure_embedding":
                result = self._predict_azure_embedding(text_clean)
            else:
                result = IntentPrediction(
                    intent="UNKNOWN",
                    action="NONE",
                    tool=None,
                    confidence=0.0
                )

            span.set_attribute("classifier.intent", result.intent)
            span.set_attribute("classifier.confidence", result.confidence)
            return result

    def _predict_tfidf_lr(self, text: str) -> IntentPrediction:
        """Predict using TF-IDF + LR model."""
        with trace_span(_tracer, "classifier.predict_tfidf", {"query.length": len(text)}) as span:
            X = self.vectorizer.transform([text])
            probs = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])

            # Cache label names once (avoids repeated inverse_transform calls)
            label_names = self.label_encoder.inverse_transform(
                range(len(probs))
            ).tolist()

            intent = label_names[pred_idx]
            meta = self.intent_to_metadata.get(intent, {})

            # Get top 3 alternatives
            top_indices = np.argsort(probs)[-3:][::-1]
            alternatives = [
                (label_names[idx], float(probs[idx]))
                for idx in top_indices[1:]
            ]

            # Full signal from complete probability vector (exact, not estimated)
            signal = ClassificationSignal.from_probabilities(probs.tolist())

            # Conformal prediction set (if calibrated)
            prediction_set = None
            if self._q_hat is not None:
                prediction_set = PredictionSet.from_probabilities(
                    probs.tolist(), label_names, self._q_hat, self._cp_coverage if self._cp_coverage is not None else 0.70
                )

            result = IntentPrediction(
                intent=intent,
                action=meta.get("action", "NONE"),
                tool=meta.get("tool"),
                confidence=confidence,
                alternatives=alternatives,
                signal=signal,
                prediction_set=prediction_set,
            )
            span.set_attribute("classifier.intent", result.intent)
            span.set_attribute("classifier.confidence", result.confidence)
            return result

    def _predict_sbert_lr(self, text: str) -> IntentPrediction:
        """Predict using SBERT + LR model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return IntentPrediction(
                intent="UNKNOWN",
                action="NONE",
                tool=None,
                confidence=0.0
            )

        # Load SBERT if not loaded (may download on first use)
        if self.vectorizer is None:
            logger.warning(
                "SentenceTransformer not loaded from model file — "
                "downloading paraphrase-multilingual-MiniLM-L12-v2 (first-time latency)"
            )
            self.vectorizer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        X = self.vectorizer.encode([text])
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        # Cache label names once (avoids repeated inverse_transform calls)
        label_names = self.label_encoder.inverse_transform(
            range(len(probs))
        ).tolist()

        intent = label_names[pred_idx]
        meta = self.intent_to_metadata.get(intent, {})

        # Get top 3 alternatives
        top_indices = np.argsort(probs)[-3:][::-1]
        alternatives = [
            (label_names[idx], float(probs[idx]))
            for idx in top_indices[1:]
        ]

        # Full signal from complete probability vector (exact, not estimated)
        signal = ClassificationSignal.from_probabilities(probs.tolist())

        # Conformal prediction set (if calibrated)
        prediction_set = None
        if self._q_hat is not None:
            prediction_set = PredictionSet.from_probabilities(
                probs.tolist(), label_names, self._q_hat, self._cp_coverage if self._cp_coverage is not None else 0.70
            )

        return IntentPrediction(
            intent=intent,
            action=meta.get("action", "NONE"),
            tool=meta.get("tool"),
            confidence=confidence,
            alternatives=alternatives,
            signal=signal,
            prediction_set=prediction_set,
        )

    def _predict_fasttext(self, text: str) -> IntentPrediction:
        """Predict using FastText model."""
        labels, probs = self.model.predict(text, k=3)

        intent = labels[0].replace("__label__", "")
        confidence = float(probs[0])

        meta = self.intent_to_metadata.get(intent, {})

        alternatives = [
            (label.replace("__label__", ""), float(prob))
            for label, prob in zip(labels[1:], probs[1:])
        ]

        return IntentPrediction(
            intent=intent,
            action=meta.get("action", "NONE"),
            tool=meta.get("tool"),
            confidence=confidence,
            alternatives=alternatives
        )

    def _predict_azure_embedding(self, text: str) -> IntentPrediction:
        """Predict using Azure OpenAI embeddings - SEMANTIC matching."""
        from config import get_settings
        settings = get_settings()

        # Cache sync client as instance variable (avoid creating per call)
        if not hasattr(self, '_sync_embedding_client') or self._sync_embedding_client is None:
            from openai import AzureOpenAI
            self._sync_embedding_client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )

        # Get query embedding (sync)
        response = self._sync_embedding_client.embeddings.create(
            input=[text],
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        embedding = response.data[0].embedding
        X = np.array([embedding])

        # Predict
        probs = self.model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        # Cache label names once (avoids repeated inverse_transform calls)
        label_names = self.label_encoder.inverse_transform(
            range(len(probs))
        ).tolist()

        intent = label_names[pred_idx]
        meta = self.intent_to_metadata.get(intent, {})

        # Get top 3 alternatives
        top_indices = np.argsort(probs)[-3:][::-1]
        alternatives = [
            (label_names[idx], float(probs[idx]))
            for idx in top_indices[1:]
        ]

        # Full signal from complete probability vector (exact, not estimated)
        signal = ClassificationSignal.from_probabilities(probs.tolist())

        # Conformal prediction set (if calibrated)
        prediction_set = None
        if self._q_hat is not None:
            prediction_set = PredictionSet.from_probabilities(
                probs.tolist(), label_names, self._q_hat, self._cp_coverage if self._cp_coverage is not None else 0.70
            )

        return IntentPrediction(
            intent=intent,
            action=meta.get("action", "NONE"),
            tool=meta.get("tool"),
            confidence=confidence,
            alternatives=alternatives,
            signal=signal,
            prediction_set=prediction_set,
        )



# Singleton instance
_classifier: Optional[IntentClassifier] = None
_classifier_lock = threading.Lock()


def get_intent_classifier(algorithm: str = "tfidf_lr") -> IntentClassifier:
    """Get or create singleton classifier instance (thread-safe)."""
    global _classifier
    if _classifier is not None and _classifier.algorithm == algorithm:
        return _classifier
    with _classifier_lock:
        if _classifier is None or _classifier.algorithm != algorithm:
            _classifier = IntentClassifier(algorithm=algorithm)
            _classifier.load()
        return _classifier


# ---
# ENSEMBLE CLASSIFIER - Best of both worlds
# ---

# Cache for semantic classifier (only load when needed)
_semantic_classifier: Optional[IntentClassifier] = None
_semantic_model_unavailable_until: float = 0.0  # Timestamp: retry after this time
_SEMANTIC_RETRY_SECONDS: float = 300.0  # Retry loading semantic model after 5 min
_semantic_lock = threading.Lock()

# Ensemble fallback: use semantic if TF-IDF isn't confident enough.
# Delegates to DecisionEngine.ML_FAST_PATH boundary (0.85 at α=0.0).
ENSEMBLE_FALLBACK_THRESHOLD = 0.75  # backward-compat re-export for tests
from services.dynamic_threshold import get_engine as _get_engine, DecisionEngine


def _get_semantic_classifier() -> IntentClassifier:
    """Get semantic classifier (lazy loaded, retries after 5 min cooldown)."""
    global _semantic_classifier, _semantic_model_unavailable_until

    # Skip if we recently failed — but retry after cooldown period
    now = time.monotonic()
    if _semantic_model_unavailable_until > now:
        raise FileNotFoundError("Azure embedding model not available (cooldown)")

    if _semantic_classifier is not None:
        return _semantic_classifier

    with _semantic_lock:
        # Double-check after acquiring lock
        if _semantic_classifier is not None:
            return _semantic_classifier

        # Check if model file exists BEFORE creating classifier
        model_file = MODEL_DIR / "azure_embedding_model.pkl"
        if not model_file.exists():
            _semantic_model_unavailable_until = now + _SEMANTIC_RETRY_SECONDS
            logger.info("Semantic fallback disabled - azure_embedding model not found (retry in 5min)")
            raise FileNotFoundError(f"Azure embedding model not available: {model_file}")

        _semantic_classifier = IntentClassifier(algorithm="azure_embedding")
        if not _semantic_classifier.load():
            _semantic_model_unavailable_until = now + _SEMANTIC_RETRY_SECONDS
            _semantic_classifier = None
            raise RuntimeError("Failed to load azure_embedding model")
    return _semantic_classifier


def predict_with_ensemble(query: str) -> IntentPrediction:
    """
    Smart ensemble: TF-IDF first, semantic fallback.

    1. Use TF-IDF (fast, no API calls)
    2. If confidence < 85%, use semantic embeddings (understands meaning)
    3. Return the more confident prediction

    This gives speed + generalization.
    """
    with trace_span(_tracer, "predict_with_ensemble", {
        "query_length": len(query),
    }) as span:
        return _predict_with_ensemble_inner(query, span)


def _predict_with_ensemble_inner(query: str, span) -> IntentPrediction:
    """Inner ensemble logic wrapped by trace span."""
    # Try TF-IDF first (fast)
    tfidf = get_intent_classifier("tfidf_lr")
    tfidf_pred = tfidf.predict(query)

    # If confident (via DecisionEngine), return immediately
    _engine = _get_engine()
    if _engine.decide(tfidf_pred.signal, DecisionEngine.ML_FAST_PATH).is_accept:
        span.set_attribute("ml.intent", tfidf_pred.intent)
        span.set_attribute("ml.confidence", tfidf_pred.confidence)
        span.set_attribute("ml.source", "tfidf_fast_path")
        return tfidf_pred

    # Skip semantic fallback if we already know it's unavailable (prevents log spam)
    if _semantic_model_unavailable_until > time.monotonic():
        span.set_attribute("ml.intent", tfidf_pred.intent)
        span.set_attribute("ml.confidence", tfidf_pred.confidence)
        span.set_attribute("ml.source", "tfidf_no_semantic")
        return tfidf_pred

    # Low confidence - use semantic for better understanding
    try:
        semantic = _get_semantic_classifier()
        sem_pred = semantic.predict(query)

        # Return the more confident prediction
        if sem_pred.confidence > tfidf_pred.confidence:
            logger.info(
                f"Ensemble: TF-IDF {tfidf_pred.confidence:.1%} < threshold, "
                f"using semantic {sem_pred.confidence:.1%}"
            )
            span.set_attribute("ml.intent", sem_pred.intent)
            span.set_attribute("ml.confidence", sem_pred.confidence)
            span.set_attribute("ml.source", "semantic")
            return sem_pred
    except Exception as e:
        if _semantic_model_unavailable_until <= time.monotonic():
            err = ClassificationError(
                ErrorCode.ENSEMBLE_ALL_FAILED,
                f"Semantic fallback failed: {e}",
                cause=e,
            )
            logger.warning(str(err))

    span.set_attribute("ml.intent", tfidf_pred.intent)
    span.set_attribute("ml.confidence", tfidf_pred.confidence)
    span.set_attribute("ml.source", "tfidf_fallback")
    return tfidf_pred


# Action intent detection and tool filtering — canonical source is
# action_intent_detector.py. Re-exported here for backward compatibility.
from services.action_intent_detector import (  # noqa: F401 — re-export
    detect_action_intent,
    filter_tools_by_intent,
)


# ---
# QUERY TYPE CLASSIFIER (ML-based)
# Replaces regex patterns in query_type_classifier.py
# ---

QUERY_TYPE_MODEL_DIR = Path(__file__).parent.parent / "models" / "query_type"
QUERY_TYPE_TRAINING_PATH = Path(__file__).parent.parent / "data" / "training" / "query_type.jsonl"


@dataclass
class QueryTypePrediction:
    """Result of query type prediction."""
    query_type: str
    confidence: float
    preferred_suffixes: List[str]
    excluded_suffixes: List[str]
    alternatives: List[Tuple[str, float]] = None
    signal: ClassificationSignal = None
    prediction_set: Optional[PredictionSet] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.signal is None:
            self.signal = ClassificationSignal.from_alternatives(
                self.confidence, self.alternatives, n_classes=12,
            )


# Suffix rules for each query type
QUERY_TYPE_SUFFIX_RULES = {
    "DOCUMENTS": {
        "preferred": ["_id_documents_documentId", "_id_documents", "_documents"],
        "excluded": ["_metadata", "_Agg", "_GroupBy", "_tree"]
    },
    "THUMBNAIL": {
        "preferred": ["_thumb", "_id_documents_documentId_thumb"],
        "excluded": ["_metadata", "_Agg"]
    },
    "METADATA": {
        "preferred": ["_id_metadata", "_metadata", "_Metadata"],
        "excluded": ["_documents", "_thumb", "_Agg"]
    },
    "AGGREGATION": {
        "preferred": ["_Agg", "_GroupBy", "_Aggregation"],
        "excluded": ["_id", "_documents", "_metadata"]
    },
    "TREE": {
        "preferred": ["_tree"],
        "excluded": ["_documents", "_metadata", "_Agg"]
    },
    "DELETE_CRITERIA": {
        "preferred": ["_DeleteByCriteria"],
        "excluded": ["_id", "_documents"]
    },
    "BULK_UPDATE": {
        "preferred": ["_multipatch", "_bulk"],
        "excluded": ["_id", "_documents"]
    },
    "DEFAULT_SET": {
        "preferred": ["_SetAsDefault", "_id_documents_documentId_SetAsDefault"],
        "excluded": ["_thumb", "_Agg"]
    },
    "PROJECTION": {
        "preferred": ["_ProjectTo"],
        "excluded": ["_documents", "_metadata"]
    },
    "LIST": {
        "preferred": [],
        "excluded": ["_id", "_id_documents", "_id_metadata", "_Agg", "_tree"]
    },
    "SINGLE_ENTITY": {
        "preferred": ["_id"],
        "excluded": ["_documents", "_metadata", "_Agg", "_tree", "_thumb"]
    },
    "UNKNOWN": {
        "preferred": [],
        "excluded": []
    }
}


class QueryTypeClassifierML:
    """
    ML-based query type classifier.
    Replaces regex patterns in query_type_classifier.py (91 patterns).
    Uses same TF-IDF + LogisticRegression approach as IntentClassifier.
    """

    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None
        self._loaded = False
        self._q_hat = None
        self._cp_coverage = None

    def load(self) -> bool:
        """Load trained model from disk."""
        try:
            model_file = QUERY_TYPE_MODEL_DIR / "tfidf_model.pkl"
            if not model_file.exists():
                logger.warning("QueryType model not found, training...")
                return self.train()

            # SECURITY: Verify SHA-256 integrity before deserializing pickle
            IntentClassifier._verify_model_integrity(model_file)

            with open(model_file, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
                self.model = data['model']
            self._loaded = True

            # Load CP calibration if available
            cp_file = QUERY_TYPE_MODEL_DIR / "cp_calibration.json"
            if cp_file.exists():
                with open(cp_file, "r", encoding="utf-8") as f:
                    cp_data = json.load(f)
                self._q_hat = cp_data["q_hat"]
                self._cp_coverage = cp_data.get("coverage_target", 0.70)
                logger.info(f"QueryType CP calibration loaded: q_hat={self._q_hat:.4f}")
            else:
                self._q_hat = None
                self._cp_coverage = None

            return True
        except Exception as e:
            err = ClassificationError(
                ErrorCode.MODEL_LOAD_FAILED,
                f"Failed to load QueryType model: {e}",
                cause=e,
            )
            logger.error(str(err))
            return False

    def train(self) -> bool:
        """Train the model. Delegates to services.intent_training."""
        from services.intent_training import train_query_type
        return train_query_type(self, QUERY_TYPE_TRAINING_PATH, QUERY_TYPE_MODEL_DIR)

    def predict(self, text: str) -> QueryTypePrediction:
        """Predict query type for text."""
        if not self._loaded:
            self.load()

        if not self._loaded or self.model is None:
            return QueryTypePrediction(
                query_type="UNKNOWN",
                confidence=0.0,
                preferred_suffixes=[],
                excluded_suffixes=[]
            )

        try:
            # Apply normalization for consistency with IntentClassifier
            text_normalized = normalize_query(text)
            X = self.vectorizer.transform([text_normalized])
            probs = self.model.predict_proba(X)[0]
            predicted_idx = np.argmax(probs)
            predicted_type = self.model.classes_[predicted_idx]
            confidence = probs[predicted_idx]

            # Extract top-3 alternatives (preserves probability info for margin decisions)
            top_indices = np.argsort(probs)[-3:][::-1]
            alternatives = [
                (str(self.model.classes_[idx]), float(probs[idx]))
                for idx in top_indices[1:]
            ]

            # Full signal from complete probability vector (exact, not estimated)
            signal = ClassificationSignal.from_probabilities(probs.tolist())

            # Conformal prediction set (if calibrated)
            prediction_set = None
            if self._q_hat is not None:
                label_names = self.model.classes_.tolist()
                prediction_set = PredictionSet.from_probabilities(
                    probs.tolist(), label_names, self._q_hat, self._cp_coverage if self._cp_coverage is not None else 0.70
                )

            # Get suffix rules
            rules = QUERY_TYPE_SUFFIX_RULES.get(predicted_type, {"preferred": [], "excluded": []})

            return QueryTypePrediction(
                query_type=predicted_type,
                confidence=float(confidence),
                preferred_suffixes=rules["preferred"],
                excluded_suffixes=rules["excluded"],
                alternatives=alternatives,
                signal=signal,
                prediction_set=prediction_set,
            )

        except Exception as e:
            err = ClassificationError(
                ErrorCode.PREDICTION_FAILED,
                f"QueryType prediction failed: {e}",
                cause=e,
            )
            logger.error(str(err))
            return QueryTypePrediction(
                query_type="UNKNOWN",
                confidence=0.0,
                preferred_suffixes=[],
                excluded_suffixes=[]
            )


# Singleton for QueryType classifier
_query_type_classifier: Optional[QueryTypeClassifierML] = None
_qt_singleton_lock = threading.Lock()


def get_query_type_classifier_ml() -> QueryTypeClassifierML:
    """Get singleton QueryType classifier."""
    global _query_type_classifier
    if _query_type_classifier is None:
        with _qt_singleton_lock:
            if _query_type_classifier is None:
                _query_type_classifier = QueryTypeClassifierML()
                _query_type_classifier.load()
    return _query_type_classifier


def classify_query_type_ml(query: str) -> QueryTypePrediction:
    """
    Classify query type using ML.
    REPLACES: query_type_classifier.py (91 regex patterns)
    """
    classifier = get_query_type_classifier_ml()
    return classifier.predict(query)


if __name__ == "__main__":
    # Train and test the classifier
    import sys

    algorithm = sys.argv[1] if len(sys.argv) > 1 else "tfidf_lr"

    print(f"Training {algorithm} classifier...")
    classifier = IntentClassifier(algorithm=algorithm)
    metrics = classifier.train()

    print("\n=== Training Results ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test predictions
    test_queries = [
        "koliko imam kilometara",
        "unesi kilometrazu",
        "rezerviraj auto za sutra",
        "moje rezervacije",
        "prijavi stetu",
        "bok",
        "hvala",
    ]

    print("\n=== Test Predictions ===")
    for query in test_queries:
        pred = classifier.predict(query)
        print(f"  '{query}'")
        print(f"    Intent: {pred.intent} ({pred.confidence:.2%})")
        print(f"    Action: {pred.action}, Tool: {pred.tool}")
        if pred.alternatives:
            print(f"    Alternatives: {pred.alternatives}")
