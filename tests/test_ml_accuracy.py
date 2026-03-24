"""
ML Accuracy Test — Tests intent classifier against ALL ground-truth data.

Loads every example from intent_test.jsonl (397 examples) and measures:
- Overall intent accuracy %
- Per-intent precision/recall
- Tool prediction accuracy
- Action prediction accuracy
- Misclassified examples (for debugging)

Run: pytest tests/test_ml_accuracy.py -v
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.intent_classifier import (
    IntentClassifier,
    IntentPrediction,
    get_intent_classifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DATA_PATH = PROJECT_ROOT / "data" / "training" / "intent_test.jsonl"
FULL_DATA_PATH = PROJECT_ROOT / "data" / "training" / "intent_full.jsonl"


def _load_examples(path: Path) -> list[dict]:
    """Load ground-truth examples from JSONL."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


@pytest.fixture(scope="module")
def classifier() -> IntentClassifier:
    """Load the TF-IDF classifier once for the whole module."""
    clf = get_intent_classifier("tfidf_lr")
    assert clf._loaded, "Model failed to load — check models/intent/tfidf_lr_model.pkl"
    return clf


@pytest.fixture(scope="module")
def test_examples() -> list[dict]:
    """Load the held-out test set (397 examples)."""
    assert TEST_DATA_PATH.exists(), f"Test data not found: {TEST_DATA_PATH}"
    return _load_examples(TEST_DATA_PATH)


@pytest.fixture(scope="module")
def full_examples() -> list[dict]:
    """Load the full training set (1650 examples)."""
    assert FULL_DATA_PATH.exists(), f"Full data not found: {FULL_DATA_PATH}"
    return _load_examples(FULL_DATA_PATH)


@pytest.fixture(scope="module")
def test_predictions(classifier, test_examples) -> list[tuple[dict, IntentPrediction]]:
    """Run classifier on every test example. Returns (example, prediction) pairs."""
    results = []
    for ex in test_examples:
        pred = classifier.predict(ex["text"])
        results.append((ex, pred))
    return results


# ---------------------------------------------------------------------------
# Accuracy report helper
# ---------------------------------------------------------------------------

def _compute_metrics(examples_and_preds: list[tuple[dict, IntentPrediction]]):
    """Compute accuracy metrics from (example, prediction) pairs."""
    total = len(examples_and_preds)
    intent_correct = 0
    tool_correct = 0
    action_correct = 0

    per_intent_tp = defaultdict(int)   # true positives
    per_intent_fn = defaultdict(int)   # false negatives (missed)
    per_intent_fp = defaultdict(int)   # false positives (wrong prediction)
    per_intent_total = defaultdict(int)

    misclassified = []

    for ex, pred in examples_and_preds:
        gt_intent = ex["intent"]
        gt_tool = ex.get("tool")
        gt_action = ex.get("action", "")

        per_intent_total[gt_intent] += 1

        if pred.intent == gt_intent:
            intent_correct += 1
            per_intent_tp[gt_intent] += 1
        else:
            per_intent_fn[gt_intent] += 1
            per_intent_fp[pred.intent] += 1
            misclassified.append({
                "text": ex["text"],
                "expected": gt_intent,
                "predicted": pred.intent,
                "confidence": pred.confidence,
            })

        if pred.tool == gt_tool:
            tool_correct += 1

        if pred.action == gt_action:
            action_correct += 1

    return {
        "total": total,
        "intent_correct": intent_correct,
        "intent_accuracy": intent_correct / total if total else 0,
        "tool_correct": tool_correct,
        "tool_accuracy": tool_correct / total if total else 0,
        "action_correct": action_correct,
        "action_accuracy": action_correct / total if total else 0,
        "per_intent_tp": dict(per_intent_tp),
        "per_intent_fn": dict(per_intent_fn),
        "per_intent_fp": dict(per_intent_fp),
        "per_intent_total": dict(per_intent_total),
        "misclassified": misclassified,
    }


def _print_report(metrics: dict):
    """Print a human-readable accuracy report."""
    print("\n" + "=" * 70)
    print("ML INTENT CLASSIFIER — ACCURACY REPORT")
    print("=" * 70)
    print(f"Total examples:     {metrics['total']}")
    print(f"Intent accuracy:    {metrics['intent_accuracy']:.1%}  ({metrics['intent_correct']}/{metrics['total']})")
    print(f"Tool accuracy:      {metrics['tool_accuracy']:.1%}  ({metrics['tool_correct']}/{metrics['total']})")
    print(f"Action accuracy:    {metrics['action_accuracy']:.1%}  ({metrics['action_correct']}/{metrics['total']})")

    print("\n--- Per-Intent Breakdown ---")
    print(f"{'Intent':<28} {'Recall':>8} {'Correct':>8} {'Total':>8}")
    print("-" * 56)

    for intent in sorted(metrics["per_intent_total"].keys()):
        total = metrics["per_intent_total"][intent]
        tp = metrics["per_intent_tp"].get(intent, 0)
        recall = tp / total if total else 0
        marker = " !" if recall < 0.80 else ""
        print(f"{intent:<28} {recall:>7.1%} {tp:>8} {total:>8}{marker}")

    if metrics["misclassified"]:
        print(f"\n--- Misclassified ({len(metrics['misclassified'])}) ---")
        for m in metrics["misclassified"][:30]:  # cap output
            print(f"  [{m['confidence']:.0%}] \"{m['text']}\" -> {m['predicted']} (expected {m['expected']})")
        if len(metrics["misclassified"]) > 30:
            print(f"  ... and {len(metrics['misclassified']) - 30} more")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLAccuracyOnTestSet:
    """Accuracy tests on the held-out test set (intent_test.jsonl, 397 examples)."""

    def test_model_loads(self, classifier):
        """Model file exists and loads successfully."""
        assert classifier._loaded

    def test_minimum_test_examples(self, test_examples):
        """Test set has enough examples for meaningful metrics."""
        assert len(test_examples) >= 100, f"Only {len(test_examples)} test examples"

    def test_all_intents_covered(self, test_examples):
        """Every intent in the test set has at least 2 examples."""
        from collections import Counter
        intent_counts = Counter(ex["intent"] for ex in test_examples)
        for intent, count in intent_counts.items():
            assert count >= 2, f"Intent {intent} has only {count} test example(s)"

    def test_overall_intent_accuracy_above_90(self, test_predictions):
        """Overall intent accuracy must be >= 90% on held-out test set."""
        metrics = _compute_metrics(test_predictions)
        _print_report(metrics)
        assert metrics["intent_accuracy"] >= 0.90, (
            f"Intent accuracy {metrics['intent_accuracy']:.1%} below 90% threshold"
        )

    def test_overall_tool_accuracy_above_85(self, test_predictions):
        """Tool prediction accuracy must be >= 85%."""
        metrics = _compute_metrics(test_predictions)
        assert metrics["tool_accuracy"] >= 0.85, (
            f"Tool accuracy {metrics['tool_accuracy']:.1%} below 85% threshold"
        )

    def test_overall_action_accuracy_above_90(self, test_predictions):
        """Action (GET/POST/DELETE) accuracy must be >= 90%."""
        metrics = _compute_metrics(test_predictions)
        assert metrics["action_accuracy"] >= 0.90, (
            f"Action accuracy {metrics['action_accuracy']:.1%} below 90% threshold"
        )

    def test_no_intent_below_70_recall(self, test_predictions):
        """No single intent should have recall below 70%."""
        metrics = _compute_metrics(test_predictions)
        for intent, total in metrics["per_intent_total"].items():
            tp = metrics["per_intent_tp"].get(intent, 0)
            recall = tp / total if total else 0
            assert recall >= 0.70, (
                f"Intent {intent} has only {recall:.1%} recall ({tp}/{total})"
            )

    def test_critical_intents_above_90(self, test_predictions):
        """High-impact intents (booking, damage, mileage) must be >= 90% recall."""
        metrics = _compute_metrics(test_predictions)
        critical = ["BOOK_VEHICLE", "REPORT_DAMAGE", "INPUT_MILEAGE", "CANCEL_RESERVATION"]
        for intent in critical:
            total = metrics["per_intent_total"].get(intent, 0)
            if total == 0:
                continue
            tp = metrics["per_intent_tp"].get(intent, 0)
            recall = tp / total
            assert recall >= 0.90, (
                f"Critical intent {intent} has only {recall:.1%} recall ({tp}/{total})"
            )

    def test_greeting_not_confused_with_action(self, test_predictions):
        """GREETING should never be misclassified as an action intent."""
        metrics = _compute_metrics(test_predictions)
        for m in metrics["misclassified"]:
            if m["expected"] == "GREETING":
                assert m["predicted"] in ("GREETING", "UNKNOWN", "HELP"), (
                    f"Greeting \"{m['text']}\" misclassified as action: {m['predicted']}"
                )


class TestMLAccuracyOnFullSet:
    """Accuracy on the full training set — should be very high (model saw this data)."""

    @pytest.fixture(scope="class")
    def full_predictions(self, classifier, full_examples):
        results = []
        for ex in full_examples:
            pred = classifier.predict(ex["text"])
            results.append((ex, pred))
        return results

    def test_training_accuracy_above_95(self, full_predictions):
        """Training set accuracy should be >= 95% (sanity check for model fit)."""
        metrics = _compute_metrics(full_predictions)
        _print_report(metrics)
        assert metrics["intent_accuracy"] >= 0.95, (
            f"Training accuracy {metrics['intent_accuracy']:.1%} — model may be underfitting"
        )


class TestConfidenceDistribution:
    """Tests about prediction confidence levels."""

    def test_correct_predictions_have_high_confidence(self, test_predictions):
        """Correctly classified examples should mostly have confidence > 80%."""
        correct_confs = [
            pred.confidence
            for ex, pred in test_predictions
            if pred.intent == ex["intent"]
        ]
        if not correct_confs:
            pytest.skip("No correct predictions")
        high_conf_ratio = sum(1 for c in correct_confs if c >= 0.80) / len(correct_confs)
        assert high_conf_ratio >= 0.80, (
            f"Only {high_conf_ratio:.1%} of correct predictions have confidence >= 80%"
        )

    def test_misclassified_have_lower_confidence(self, test_predictions):
        """Wrong predictions should on average have lower confidence than correct ones."""
        correct_confs = [
            pred.confidence for ex, pred in test_predictions if pred.intent == ex["intent"]
        ]
        wrong_confs = [
            pred.confidence for ex, pred in test_predictions if pred.intent != ex["intent"]
        ]
        if not wrong_confs or not correct_confs:
            pytest.skip("Need both correct and incorrect predictions")

        avg_correct = sum(correct_confs) / len(correct_confs)
        avg_wrong = sum(wrong_confs) / len(wrong_confs)
        assert avg_wrong < avg_correct, (
            f"Wrong predictions avg confidence ({avg_wrong:.2f}) >= correct ({avg_correct:.2f})"
        )


class TestIntentCoverage:
    """Verify all configured intents are testable."""

    def test_all_configured_intents_in_test_data(self, test_examples):
        """Every intent in INTENT_CONFIG should have test examples."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from tool_routing import INTENT_CONFIG

        test_intents = {ex["intent"] for ex in test_examples}
        config_intents = set(INTENT_CONFIG.keys())

        missing = config_intents - test_intents
        # Filter out intents that are just aliases or special
        special = {"UNKNOWN", "HELP", "FALLBACK"}
        missing = missing - special

        assert len(missing) == 0, (
            f"Intents in INTENT_CONFIG but missing from test data: {missing}"
        )
