"""
Evaluate intent classifier accuracy on held-out test set.

Uses data/training/intent_test.jsonl (never seen during training).
Reports real-world accuracy, per-intent breakdown, and confusion matrix.

Usage:
    python scripts/evaluate_accuracy.py
    python scripts/evaluate_accuracy.py --retrain   # Retrain first, then evaluate
"""

import json
import sys
import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
for name in ["services", "openai", "httpx", "httpcore", "sklearn"]:
    logging.getLogger(name).setLevel(logging.ERROR)

DATA_DIR = Path(__file__).parent.parent / "data" / "training"
TRAIN_FILE = DATA_DIR / "intent_full.jsonl"
TEST_FILE = DATA_DIR / "intent_test.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrain_model(algorithm="tfidf_lr"):
    """Retrain model on clean training data."""
    from services.intent_classifier import IntentClassifier

    print(f"\n{'='*60}")
    print(f"RETRAINING: {algorithm}")
    print(f"{'='*60}")

    clf = IntentClassifier(algorithm=algorithm)
    metrics = clf.train(training_data_path=TRAIN_FILE)

    print(f"Cross-val accuracy: {metrics['accuracy']:.2%}")
    if "accuracy_std" in metrics:
        print(f"Cross-val std: {metrics['accuracy_std']:.4f}")
    return clf


def evaluate(algorithm="tfidf_lr"):
    """Evaluate on held-out test set."""
    from services.intent_classifier import IntentClassifier

    print(f"\n{'='*60}")
    print(f"EVALUATING: {algorithm} on held-out test set")
    print(f"{'='*60}")

    clf = IntentClassifier(algorithm=algorithm)
    if not clf.load():
        print(f"ERROR: Could not load {algorithm} model")
        return None

    test_data = load_jsonl(TEST_FILE)
    print(f"Test samples: {len(test_data)}")

    correct = 0
    total = 0
    per_intent_correct = defaultdict(int)
    per_intent_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    errors = []

    for sample in test_data:
        text = sample["text"]
        expected = sample["intent"]
        pred = clf.predict(text)

        per_intent_total[expected] += 1
        total += 1

        if pred.intent == expected:
            correct += 1
            per_intent_correct[expected] += 1
        else:
            confusion[expected][pred.intent] += 1
            errors.append({
                "text": text,
                "expected": expected,
                "predicted": pred.intent,
                "confidence": pred.confidence,
            })

    accuracy = correct / total if total > 0 else 0

    print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.2%}")

    # Per-intent breakdown
    print(f"\n{'INTENT':30s} {'ACC':>7s} {'CORRECT':>7s} {'TOTAL':>5s}")
    print("-" * 55)
    for intent in sorted(per_intent_total.keys()):
        c = per_intent_correct[intent]
        t = per_intent_total[intent]
        acc = c / t if t > 0 else 0
        marker = " <-- LOW" if acc < 0.80 else ""
        print(f"{intent:30s} {acc:6.1%} {c:7d} {t:5d}{marker}")

    # Confusion matrix (only errors)
    if confusion:
        print(f"\nCONFUSION MATRIX (errors only):")
        print(f"{'EXPECTED':30s} {'PREDICTED':30s} {'COUNT':>5s}")
        print("-" * 67)
        for expected in sorted(confusion.keys()):
            for predicted, count in sorted(confusion[expected].items(), key=lambda x: -x[1]):
                print(f"{expected:30s} {predicted:30s} {count:5d}")

    # Worst errors
    if errors:
        print(f"\nTOP ERRORS (by confidence):")
        for e in sorted(errors, key=lambda x: -x["confidence"])[:10]:
            print(f"  '{e['text'][:40]}' -> {e['predicted']} ({e['confidence']:.1%}) expected {e['expected']}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_intent": {k: per_intent_correct[k] / per_intent_total[k]
                       for k in per_intent_total},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Retrain before evaluating")
    args = parser.parse_args()

    if not TEST_FILE.exists():
        print(f"ERROR: Test file not found: {TEST_FILE}")
        print("Run scripts/clean_training_data.py first to create train/test split.")
        sys.exit(1)

    if args.retrain:
        retrain_model("tfidf_lr")

    results = evaluate("tfidf_lr")

    if results:
        if results["accuracy"] >= 0.85:
            print(f"\nPASS: Accuracy {results['accuracy']:.2%} >= 85% threshold")
        else:
            print(f"\nFAIL: Accuracy {results['accuracy']:.2%} < 85% threshold")


if __name__ == "__main__":
    main()
