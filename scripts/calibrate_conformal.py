"""
Calibrate Conformal Prediction thresholds for intent and query_type classifiers.

Uses Adaptive Prediction Sets (APS) algorithm on held-out calibration data
to compute q_hat — the nonconformity threshold that guarantees coverage.

Coverage guarantee: P(true_label ∈ prediction_set) >= 1 - alpha.

Usage:
    python scripts/calibrate_conformal.py

Requires: trained models in models/ directory
Output:
    models/intent/cp_calibration.json
    models/query_type/cp_calibration.json
"""
import json
import math
import sys
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np

# Resolve paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

INTENT_TEST_PATH = PROJECT_DIR / "data" / "training" / "intent_test.jsonl"
QUERY_TYPE_PATH = PROJECT_DIR / "data" / "training" / "query_type.jsonl"
INTENT_MODEL_DIR = PROJECT_DIR / "models" / "intent"
QUERY_TYPE_MODEL_DIR = PROJECT_DIR / "models" / "query_type"

# Alpha = 0.30 chosen empirically: gives mean set size ~4-5 (ideal for mediation)
# while maintaining empirical coverage > 98% on both classifiers.
# Formal guarantee is coverage >= 70%, but the peaked distributions of
# well-trained LogisticRegression models yield empirical coverage of 98-100%.
DEFAULT_ALPHA = 0.30


def load_jsonl(path, text_field="text", label_field="intent"):
    """Load JSONL file as (text, label) pairs."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append((item[text_field], item[label_field]))
    return examples


def compute_aps_scores(predict_proba_fn, label_names, calibration_data):
    """Compute APS nonconformity scores for each calibration example.

    APS score = cumulative probability mass needed to include the true label,
    when probabilities are sorted descending.

    Args:
        predict_proba_fn: function(text) -> np.array of probabilities
        label_names: list of class names (same order as probs)
        calibration_data: list of (text, true_label) tuples

    Returns:
        list of (aps_score, true_label, predicted_label, is_correct) tuples
    """
    scores = []
    for text, true_label in calibration_data:
        probs = predict_proba_fn(text)

        # Sort (label, prob) descending
        pairs = sorted(
            zip(label_names, probs), key=lambda x: x[1], reverse=True
        )

        # APS score: cumulative prob until true label is included
        cumulative = 0.0
        for label, prob in pairs:
            cumulative += prob
            if label == true_label:
                break

        predicted_label = pairs[0][0]
        scores.append((cumulative, true_label, predicted_label,
                        predicted_label == true_label))

    return scores


def compute_q_hat(scores, alpha=DEFAULT_ALPHA):
    """Compute q_hat from APS scores using finite-sample correction.

    q_hat = quantile(scores, ceil((n+1)*(1-alpha))/n)

    This guarantees marginal coverage >= 1 - alpha on exchangeable data.
    """
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    aps_values = [s[0] for s in scores]
    q_hat = float(np.quantile(aps_values, level))
    return q_hat


def evaluate_prediction_sets(scores, q_hat):
    """Evaluate prediction set sizes and coverage using calibrated q_hat.

    Returns dict with coverage, mean/median set size, and distribution.
    """
    sizes = []
    covered = 0

    for aps_score, true_label, predicted_label, is_correct in scores:
        # The true label is in the prediction set iff its APS score <= q_hat
        if aps_score <= q_hat + 1e-9:
            covered += 1

        # Compute set size: how many labels included before threshold
        # For simplicity, approximate from APS score structure
        # A more accurate way: we'd need the full probs, but APS score
        # directly tells us the cumulative mass. We need to count.
        # We'll compute this separately.
        pass

    coverage = covered / len(scores) if scores else 0.0
    return coverage


def compute_set_sizes(predict_proba_fn, label_names, data, q_hat):
    """Compute actual prediction set sizes for each example."""
    sizes = []
    covered = 0

    for text, true_label in data:
        probs = predict_proba_fn(text)

        # Sort descending
        pairs = sorted(
            zip(label_names, probs), key=lambda x: x[1], reverse=True
        )

        # APS: include labels while cumulative <= q_hat, plus boundary
        cumulative = 0.0
        set_labels = []
        for label, prob in pairs:
            cumulative += prob
            set_labels.append(label)
            if cumulative > q_hat:
                break

        sizes.append(len(set_labels))
        if true_label in set_labels:
            covered += 1

    return {
        "sizes": sizes,
        "coverage": covered / len(data) if data else 0.0,
        "mean_set_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_set_size": int(np.median(sizes)) if sizes else 0,
        "set_size_distribution": dict(Counter(sizes)),
    }


def model_file_hash(path):
    """Compute short hash of model file for reproducibility tracking."""
    if not path.exists():
        return "missing"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def calibrate_intent(alpha=DEFAULT_ALPHA):
    """Calibrate conformal prediction for intent classifier."""
    print("\n=== Intent Classifier Calibration ===")

    # Load test data
    if not INTENT_TEST_PATH.exists():
        print(f"ERROR: {INTENT_TEST_PATH} not found")
        return None

    data = load_jsonl(INTENT_TEST_PATH, text_field="text", label_field="intent")
    print(f"Loaded {len(data)} calibration examples from intent_test.jsonl")

    # Load model
    import pickle
    model_path = INTENT_MODEL_DIR / "tfidf_lr_model.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        return None

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    vectorizer = saved["vectorizer"]
    label_encoder = saved["label_encoder"]
    label_names = label_encoder.classes_.tolist()

    print(f"Model classes ({len(label_names)}): {label_names}")

    # Define predict_proba function
    def predict_proba(text):
        from services.intent_classifier import normalize_query
        text_norm = normalize_query(text)
        X = vectorizer.transform([text_norm])
        return model.predict_proba(X)[0]

    # Compute APS scores
    print("Computing APS nonconformity scores...")
    scores = compute_aps_scores(predict_proba, label_names, data)

    # Accuracy check
    correct = sum(1 for s in scores if s[3])
    accuracy = correct / len(scores)
    print(f"Base accuracy: {accuracy:.1%} ({correct}/{len(scores)})")

    # Compute q_hat
    q_hat = compute_q_hat(scores, alpha)
    print(f"q_hat = {q_hat:.6f} (alpha={alpha}, coverage target={1-alpha:.0%})")

    # Evaluate prediction sets
    eval_result = compute_set_sizes(predict_proba, label_names, data, q_hat)
    print(f"Empirical coverage: {eval_result['coverage']:.1%}")
    print(f"Mean set size: {eval_result['mean_set_size']:.2f}")
    print(f"Median set size: {eval_result['median_set_size']}")
    print(f"Set size distribution: {eval_result['set_size_distribution']}")

    # Save calibration
    result = {
        "algorithm": "APS",
        "alpha": alpha,
        "coverage_target": 1 - alpha,
        "q_hat": q_hat,
        "n_calibration": len(data),
        "n_classes": len(label_names),
        "class_names": label_names,
        "empirical_coverage": eval_result["coverage"],
        "mean_set_size": eval_result["mean_set_size"],
        "median_set_size": eval_result["median_set_size"],
        "set_size_distribution": {
            str(k): v for k, v in eval_result["set_size_distribution"].items()
        },
        "base_accuracy": accuracy,
        "calibration_date": datetime.now().isoformat(),
        "model_hash": model_file_hash(model_path),
    }

    output_path = INTENT_MODEL_DIR / "cp_calibration.json"
    INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    return result


def calibrate_query_type(alpha=DEFAULT_ALPHA):
    """Calibrate conformal prediction for query type classifier."""
    print("\n=== Query Type Classifier Calibration ===")

    # Load data
    if not QUERY_TYPE_PATH.exists():
        print(f"ERROR: {QUERY_TYPE_PATH} not found")
        return None

    all_data = load_jsonl(QUERY_TYPE_PATH, text_field="text", label_field="query_type")
    print(f"Loaded {len(all_data)} total examples from query_type.jsonl")

    # 80/20 split (deterministic shuffle)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_data))
    split_idx = int(len(all_data) * 0.8)
    cal_indices = indices[split_idx:]
    cal_data = [all_data[i] for i in cal_indices]
    print(f"Using {len(cal_data)} examples for calibration (20% holdout)")

    # Load model
    import pickle
    model_path = QUERY_TYPE_MODEL_DIR / "tfidf_model.pkl"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        return None

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    vectorizer = saved["vectorizer"]
    label_names = model.classes_.tolist()

    print(f"Model classes ({len(label_names)}): {label_names}")

    # Define predict_proba function
    def predict_proba(text):
        from services.intent_classifier import normalize_query
        text_norm = normalize_query(text)
        X = vectorizer.transform([text_norm])
        return model.predict_proba(X)[0]

    # Compute APS scores
    print("Computing APS nonconformity scores...")
    scores = compute_aps_scores(predict_proba, label_names, cal_data)

    # Accuracy check
    correct = sum(1 for s in scores if s[3])
    accuracy = correct / len(scores)
    print(f"Base accuracy: {accuracy:.1%} ({correct}/{len(scores)})")

    # Compute q_hat
    q_hat = compute_q_hat(scores, alpha)
    print(f"q_hat = {q_hat:.6f} (alpha={alpha}, coverage target={1-alpha:.0%})")

    # Evaluate prediction sets
    eval_result = compute_set_sizes(predict_proba, label_names, cal_data, q_hat)
    print(f"Empirical coverage: {eval_result['coverage']:.1%}")
    print(f"Mean set size: {eval_result['mean_set_size']:.2f}")
    print(f"Median set size: {eval_result['median_set_size']}")
    print(f"Set size distribution: {eval_result['set_size_distribution']}")

    # Save calibration
    result = {
        "algorithm": "APS",
        "alpha": alpha,
        "coverage_target": 1 - alpha,
        "q_hat": q_hat,
        "n_calibration": len(cal_data),
        "n_classes": len(label_names),
        "class_names": label_names,
        "empirical_coverage": eval_result["coverage"],
        "mean_set_size": eval_result["mean_set_size"],
        "median_set_size": eval_result["median_set_size"],
        "set_size_distribution": {
            str(k): v for k, v in eval_result["set_size_distribution"].items()
        },
        "base_accuracy": accuracy,
        "calibration_date": datetime.now().isoformat(),
        "model_hash": model_file_hash(model_path),
    }

    output_path = QUERY_TYPE_MODEL_DIR / "cp_calibration.json"
    QUERY_TYPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    return result


def main():
    print("=== Conformal Prediction Calibration ===")
    print(f"Alpha = {DEFAULT_ALPHA}, Coverage target = {1 - DEFAULT_ALPHA:.0%}")

    intent_result = calibrate_intent()
    qt_result = calibrate_query_type()

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    if intent_result:
        print(f"\nIntent classifier:")
        print(f"  q_hat = {intent_result['q_hat']:.6f}")
        print(f"  Coverage = {intent_result['empirical_coverage']:.1%}")
        print(f"  Mean set size = {intent_result['mean_set_size']:.2f}")
        print(f"  Output: models/intent/cp_calibration.json")

    if qt_result:
        print(f"\nQuery type classifier:")
        print(f"  q_hat = {qt_result['q_hat']:.6f}")
        print(f"  Coverage = {qt_result['empirical_coverage']:.1%}")
        print(f"  Mean set size = {qt_result['mean_set_size']:.2f}")
        print(f"  Output: models/query_type/cp_calibration.json")

    if not intent_result and not qt_result:
        print("\nERROR: No calibrations completed.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
