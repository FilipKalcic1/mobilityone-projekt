"""
Intent Classifier Training — offline model training.

Extracted from intent_classifier.py to separate runtime prediction (<1ms)
from offline training (minutes-to-hours). Only imported during `python -m`
training runs, never on the hot path.

All training methods operate on an IntentClassifier instance passed as `self`.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from services.text_normalizer import normalize_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentClassifier training methods
# ---------------------------------------------------------------------------

def load_training_data(path: Path) -> Tuple[List[str], List[str], Dict]:
    """Load training data from JSONL file."""
    texts = []
    labels = []
    metadata = {}

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSONL line {i}: {e}")
                continue
            normalized_text = normalize_query(item["text"])
            texts.append(normalized_text)
            labels.append(item["intent"])

            intent = item["intent"]
            if intent not in metadata:
                metadata[intent] = {
                    "action": item["action"],
                    "tool": item["tool"]
                }

    return texts, labels, metadata


def train_tfidf_lr(classifier, texts: List[str], labels: List[str]) -> Dict[str, float]:
    """Train TF-IDF + Logistic Regression model with Platt scaling calibration.

    Uses FeatureUnion of word n-grams + char_wb n-grams:
    - Word n-grams: exact word matching for standard queries
    - Char_wb n-grams: character-level matching for typo resilience

    Calibration: CalibratedClassifierCV (Platt scaling) ensures that
    predict_proba returns well-calibrated probabilities.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import brier_score_loss

    classifier.vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=10000,
            min_df=2,
        )),
    ])
    X = classifier.vectorizer.fit_transform(texts)

    classifier.label_encoder = LabelEncoder()
    y = classifier.label_encoder.fit_transform(labels)

    base_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=10.0
    )
    classifier.model = CalibratedClassifierCV(
        estimator=base_model,
        cv=5,
        method="sigmoid"
    )
    classifier.model.fit(X, y)

    cv_scores = cross_val_score(classifier.model, X, y, cv=5, scoring="accuracy")

    y_prob = classifier.model.predict_proba(X)
    classes = classifier.model.classes_
    y_bin = label_binarize(y, classes=classes)
    brier = float(np.mean([
        brier_score_loss(y_bin[:, i], y_prob[:, i])
        for i in range(len(classes))
    ]))

    save_model(classifier)

    return {
        "accuracy": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "brier_score": brier,
        "calibrated": True
    }


def train_sbert_lr(classifier, texts: List[str], labels: List[str]) -> Dict[str, float]:
    """Train Sentence-BERT + Logistic Regression model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed.")
        return {"error": "sentence-transformers not installed"}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Encoding texts with SBERT...")
    X = sbert_model.encode(texts, show_progress_bar=True)

    classifier.vectorizer = sbert_model

    classifier.label_encoder = LabelEncoder()
    y = classifier.label_encoder.fit_transform(labels)

    classifier.model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=10.0
    )
    classifier.model.fit(X, y)

    cv_scores = cross_val_score(classifier.model, X, y, cv=5, scoring="accuracy")

    save_model(classifier, include_vectorizer=False)

    return {
        "accuracy": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist()
    }


def train_fasttext(classifier, texts: List[str], labels: List[str]) -> Dict[str, float]:
    """Train FastText model."""
    try:
        import fasttext
    except ImportError:
        logger.error("fasttext not installed.")
        return {"error": "fasttext not installed"}

    from sklearn.preprocessing import LabelEncoder

    classifier.label_encoder = LabelEncoder()
    classifier.label_encoder.fit(labels)

    train_file = classifier.model_path / "fasttext_train.txt"
    classifier.model_path.mkdir(parents=True, exist_ok=True)

    with open(train_file, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            f.write(f"__label__{label} {text}\n")

    classifier.model = fasttext.train_supervised(
        str(train_file),
        epoch=50,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        loss="softmax"
    )

    test_result = classifier.model.test(str(train_file))

    model_file = classifier.model_path / "fasttext_model.bin"
    classifier.model.save_model(str(model_file))

    return {
        "accuracy": test_result[1],
        "precision": test_result[1],
        "recall": test_result[2]
    }


def train_azure_embedding(classifier, texts: List[str], labels: List[str]) -> Dict[str, float]:
    """Train using Azure OpenAI embeddings — SEMANTIC understanding."""
    import asyncio
    from config import get_settings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    from services.openai_client import get_embedding_client

    settings = get_settings()
    client = get_embedding_client()

    async def get_embeddings(batch: List[str]) -> List[List[float]]:
        response = await client.embeddings.create(
            input=batch,
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        return [item.embedding for item in response.data]

    async def embed_all() -> np.ndarray:
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await get_embeddings(batch)
            embeddings.extend(batch_embeddings)
            logger.info(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
        return np.array(embeddings)

    logger.info("Generating Azure OpenAI embeddings for training data...")
    X = asyncio.run(embed_all())

    classifier.label_encoder = LabelEncoder()
    y = classifier.label_encoder.fit_transform(labels)

    classifier.model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=10.0
    )
    classifier.model.fit(X, y)

    cv_scores = cross_val_score(classifier.model, X, y, cv=5, scoring="accuracy")

    classifier.vectorizer = {
        "type": "azure_embedding",
        "deployment": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    }

    save_model(classifier)

    return {
        "accuracy": float(cv_scores.mean()),
        "accuracy_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "embedding_dim": X.shape[1]
    }


def save_model(classifier, include_vectorizer: bool = True) -> None:
    """Save model to disk."""
    classifier.model_path.mkdir(parents=True, exist_ok=True)
    model_file = classifier.model_path / f"{classifier.algorithm}_model.pkl"

    save_data = {
        "model": classifier.model,
        "label_encoder": classifier.label_encoder
    }
    if include_vectorizer and classifier.vectorizer is not None:
        save_data["vectorizer"] = classifier.vectorizer

    with open(model_file, "wb") as f:
        pickle.dump(save_data, f)

    logger.info(f"Saved model to {model_file}")


def save_metadata(classifier) -> None:
    """Save metadata to disk."""
    classifier.model_path.mkdir(parents=True, exist_ok=True)
    meta_file = classifier.model_path / "metadata.json"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(classifier.intent_to_metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# QueryTypeClassifierML training
# ---------------------------------------------------------------------------

def train_query_type(classifier, training_path: Path, model_dir: Path) -> bool:
    """Train the QueryType classifier from JSONL training data.

    Uses FeatureUnion of word n-grams + char_wb n-grams.
    Uses raw LogisticRegression (no CalibratedClassifierCV).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from collections import Counter as _Counter

        texts, labels = [], []
        with open(training_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item['text'])
                    labels.append(item['query_type'])

        if not texts:
            logger.error("No training data found")
            return False

        classifier.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=8000,
                min_df=1,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=15000,
                min_df=1,
            )),
        ])
        X = classifier.vectorizer.fit_transform(texts)

        base_model = LogisticRegression(
            max_iter=1000,
            C=10.0,
            class_weight='balanced'
        )

        label_counts = _Counter(labels)
        min_class_count = min(label_counts.values())
        n_classes = len(label_counts)

        classifier.model = base_model
        classifier.model.fit(X, labels)

        cv_report = min(5, min_class_count, n_classes)
        if cv_report >= 2:
            scores = cross_val_score(base_model, X, labels, cv=cv_report)
            accuracy = np.mean(scores)
        else:
            accuracy = 0.0
        logger.info(f"QueryType classifier trained: {accuracy:.1%} accuracy, "
                    f"{len(texts)} examples, {n_classes} classes, "
                    f"features={X.shape[1]}")

        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "tfidf_model.pkl", 'wb') as f:
            pickle.dump({'vectorizer': classifier.vectorizer, 'model': classifier.model}, f)

        classifier._loaded = True
        return True

    except Exception as e:
        logger.error(f"Failed to train QueryType model: {e}")
        return False
