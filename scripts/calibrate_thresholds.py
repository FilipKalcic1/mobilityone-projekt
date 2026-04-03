"""
Calibrate DecisionEngine α parameter.

Grid search over α ∈ [0.0, 0.1, ..., 1.0] to find the optimal
entropy penalty for threshold decisions.

This script analyzes the ML classifier's probability distributions
on training/validation data to determine:
1. Optimal α (entropy weight)
2. Per-context threshold values
3. Decision quality metrics

Usage:
    python scripts/calibrate_thresholds.py

Requires: trained intent classifier model (data/models/)
"""
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# Resolve paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

TOOL_DOCS_PATH = PROJECT_DIR / "config" / "tool_documentation.json"
TRAINING_DATA_PATH = PROJECT_DIR / "data" / "training" / "query_type.jsonl"


def load_training_data():
    """Load training examples with their labels."""
    examples = []
    if TRAINING_DATA_PATH.exists():
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    examples.append((item.get("text", ""), item.get("query_type", "")))
    return examples


def simulate_decisions(alpha_values, confidence_margin_pairs, context="general"):
    """
    Simulate threshold decisions across α values.

    Args:
        alpha_values: List of α to test
        confidence_margin_pairs: List of (confidence, margin, is_correct) tuples
        context: Threshold context

    Returns:
        Dict of α → {accept_correct, accept_wrong, defer_correct, defer_wrong, ...}
    """
    from services.dynamic_threshold import (
        get_engine, ClassificationSignal, DecisionAction, DecisionEngine,
    )

    _CONTEXT_TO_BOUNDARY = {
        "intent_filter": DecisionEngine.INTENT_FILTER,
        "query_type": DecisionEngine.QUERY_TYPE,
        "possessive": DecisionEngine.POSSESSIVE,
        "ml_fast_path": DecisionEngine.ML_FAST_PATH,
        "mutation": DecisionEngine.MUTATION,
        "general": DecisionEngine.GENERAL,
    }

    results = {}

    for alpha in alpha_values:
        engine = get_engine(alpha=alpha)
        boundary = _CONTEXT_TO_BOUNDARY.get(context, DecisionEngine.GENERAL)
        stats = {
            "accept_correct": 0,
            "accept_wrong": 0,
            "boost_correct": 0,
            "boost_wrong": 0,
            "defer_correct": 0,
            "defer_wrong": 0,
        }

        for confidence, margin, is_correct in confidence_margin_pairs:
            # Reconstruct minimal probability vector from confidence and margin
            p2 = confidence - margin
            probs = [confidence, max(p2, 0.0)]
            signal = ClassificationSignal.from_probabilities(probs)
            decision = engine.decide(signal, boundary)

            action_name = decision.action.name.lower()
            key = f"{action_name}_{'correct' if is_correct else 'wrong'}"
            if key in stats:
                stats[key] += 1

        # Derived metrics
        total = len(confidence_margin_pairs)
        accepts = stats["accept_correct"] + stats["accept_wrong"]
        defers = stats["defer_correct"] + stats["defer_wrong"]

        stats["total"] = total
        stats["accept_rate"] = accepts / total if total else 0
        stats["defer_rate"] = defers / total if total else 0

        # Precision: of all accepted, how many were correct?
        stats["accept_precision"] = (
            stats["accept_correct"] / accepts if accepts else 0
        )
        # Safety: of all wrong predictions, how many did we defer?
        wrong_total = stats["accept_wrong"] + stats["boost_wrong"] + stats["defer_wrong"]
        stats["wrong_deferred_rate"] = (
            stats["defer_wrong"] / wrong_total if wrong_total else 1.0
        )

        results[alpha] = stats

    return results


def main():
    print("=== DecisionEngine Calibration ===\n")

    # Load training data
    examples = load_training_data()
    print(f"Loaded {len(examples)} training examples")

    if not examples:
        print("ERROR: No training data found. Run generate_query_type_training.py first.")
        sys.exit(1)

    # Try to load and run the classifier
    try:
        from services.intent_classifier import classify_query_type_ml
    except Exception as e:
        print(f"Cannot load classifier: {e}")
        print("Running in offline mode with synthetic data...")
        run_offline_calibration()
        return

    # Collect (confidence, margin, is_correct) for each example
    print("Collecting classifier predictions...")
    pairs = []
    correct = 0

    for text, label in examples:
        result = classify_query_type_ml(text)
        confidence = result.confidence

        # Get margin from alternatives if available
        alts = getattr(result, 'alternatives', [])
        if len(alts) >= 2:
            margin = alts[0][1] - alts[1][1]
        else:
            margin = confidence  # If no alternatives, margin = confidence

        is_correct = (result.query_type == label)
        if is_correct:
            correct += 1
        pairs.append((confidence, margin, is_correct))

    accuracy = correct / len(examples) if examples else 0
    print(f"Base accuracy: {accuracy:.1%} ({correct}/{len(examples)})")

    # Grid search
    alpha_values = [round(x * 0.1, 1) for x in range(11)]
    print(f"\nGrid search over α = {alpha_values}\n")

    results = simulate_decisions(alpha_values, pairs, context="general")

    # Print results table
    print(f"{'α':>4} | {'Accept%':>8} | {'Defer%':>8} | {'Precision':>9} | {'Wrong→Defer':>11}")
    print("-" * 55)

    best_alpha = 0.0
    best_score = 0.0

    for alpha in alpha_values:
        s = results[alpha]
        score = s["accept_precision"] * 0.7 + s["wrong_deferred_rate"] * 0.3
        marker = ""
        if score > best_score:
            best_score = score
            best_alpha = alpha
            marker = " <- best"

        print(
            f"{alpha:>4.1f} | "
            f"{s['accept_rate']:>7.1%} | "
            f"{s['defer_rate']:>7.1%} | "
            f"{s['accept_precision']:>8.1%} | "
            f"{s['wrong_deferred_rate']:>10.1%}"
            f"{marker}"
        )

    print(f"\nOptimal α = {best_alpha} (score = {best_score:.4f})")
    print(f"\nTo apply: update services/dynamic_threshold.py -> get_engine(alpha={best_alpha})")

    # Per-context analysis
    print("\n=== Per-Context Analysis ===")
    for context in ["intent_filter", "query_type", "possessive", "ml_fast_path", "mutation"]:
        ctx_results = simulate_decisions([best_alpha], pairs, context=context)
        s = ctx_results[best_alpha]
        print(
            f"  {context:20s}: accept={s['accept_rate']:.1%}, "
            f"defer={s['defer_rate']:.1%}, "
            f"precision={s['accept_precision']:.1%}"
        )


def run_offline_calibration():
    """Run calibration with synthetic probability distributions."""
    print("\n--- Offline Calibration (synthetic data) ---\n")
    from services.dynamic_threshold import (
        get_engine, ClassificationSignal, DecisionAction,
    )

    # Test scenarios that demonstrate entropy-awareness
    scenarios = [
        # (description, confidence, margin, should_be)
        ("Sure correct",    0.95, 0.85, DecisionAction.ACCEPT),
        ("Sure wrong",      0.95, 0.80, DecisionAction.ACCEPT),
        ("Ambiguous high",  0.60, 0.05, DecisionAction.BOOST),
        ("Ambiguous low",   0.45, 0.03, DecisionAction.DEFER),
        ("Clear moderate",  0.70, 0.50, DecisionAction.ACCEPT),
        ("Low but clear",   0.50, 0.35, DecisionAction.BOOST),
    ]

    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"{'Scenario':20s} | ", end="")
    for a in alpha_values:
        print(f"a={a:.1f}    ", end="| ")
    print()
    print("-" * (22 + 12 * len(alpha_values)))

    for desc, conf, margin, expected in scenarios:
        print(f"{desc:20s} | ", end="")
        p2 = conf - margin
        probs = [conf, max(p2, 0.0)]
        signal = ClassificationSignal.from_probabilities(probs)

        for alpha in alpha_values:
            engine = get_engine(alpha=alpha)
            decision = engine.decide(signal, engine.GENERAL)
            action_short = decision.action.name[:6]
            marker = "v" if decision.action is expected else " "
            print(f"{action_short:6s} {marker} ", end="| ")
        print()

    print(f"\nRecommendation: Start with a=0.0 (no change), "
          f"increase to 0.2-0.4 after collecting production data.")


if __name__ == "__main__":
    main()
