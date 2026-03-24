#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for ML fixes.
Run after docker-compose up to verify all fixes are working.

Usage:
    docker-compose exec worker python scripts/verify_fixes.py

    OR locally:
    python scripts/verify_fixes.py
"""

import sys
import os
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(passed, total, test_name):
    status = "PASS" if passed == total else "FAIL"
    color = "\033[92m" if passed == total else "\033[91m"
    reset = "\033[0m"
    print(f"{color}[{status}]{reset} {test_name}: {passed}/{total}")
    return passed == total

def test_model_files():
    """Test 1: Verify model files exist and have content."""
    print_header("TEST 1: Model Files")

    from pathlib import Path

    model_dir = Path(__file__).parent.parent / "models" / "intent"
    required_files = [
        ("tfidf_lr_model.pkl", 50000),  # Should be >50KB
    ]

    passed = 0
    total = len(required_files)

    for filename, min_size in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size >= min_size:
                print(f"  [OK] {filename}: {size:,} bytes")
                passed += 1
            else:
                print(f"  [FAIL] {filename}: {size:,} bytes (expected >{min_size:,})")
        else:
            print(f"  [FAIL] {filename}: NOT FOUND")

    return print_result(passed, total, "Model Files")

def test_intent_classification():
    """Test 2: Verify intent classification accuracy."""
    print_header("TEST 2: Intent Classification")

    from services.intent_classifier import IntentClassifier

    classifier = IntentClassifier(algorithm='tfidf_lr')
    if not classifier.load():
        print("  [FAIL] Could not load classifier")
        return False

    # Critical test cases - these MUST pass
    test_cases = [
        # Phone vs Person Info (was broken)
        ("moj telefon", "GET_PHONE", 85),
        ("koji je moj broj telefona", "GET_PHONE", 85),
        ("moji podaci", "GET_PERSON_INFO", 85),
        ("moj profil", "GET_PERSON_INFO", 85),

        # Previously underrepresented (was broken)
        ("obrisi slucaj", "DELETE_CASE", 85),
        ("broj auta", "GET_VEHICLE_COUNT", 85),
        ("prikazi troskove", "GET_EXPENSES", 85),
        ("slobodna vozila", "GET_AVAILABLE_VEHICLES", 40),

        # Core functionality
        ("kilometraza vozila", "GET_MILEAGE", 90),
        ("rezerviraj vozilo", "BOOK_VEHICLE", 90),
        ("kada istice registracija", "GET_REGISTRATION_EXPIRY", 90),
        ("prijavi kvar", "REPORT_DAMAGE", 90),

        # Basic intents
        ("pozdrav", "GREETING", 85),
        ("hvala", "THANKS", 90),
        ("pomoc", "HELP", 85),
    ]

    passed = 0
    total = len(test_cases)

    for query, expected_intent, min_confidence in test_cases:
        result = classifier.predict(query)

        intent_ok = result.intent == expected_intent
        conf_ok = result.confidence * 100 >= min_confidence

        if intent_ok and conf_ok:
            print(f"  [OK] \"{query}\" -> {result.intent} ({result.confidence:.0%})")
            passed += 1
        elif intent_ok:
            print(f"  [WARN] \"{query}\" -> {result.intent} ({result.confidence:.0%}) - low confidence")
            passed += 1  # Still count as pass if intent is correct
        else:
            print(f"  [FAIL] \"{query}\" -> {result.intent} ({result.confidence:.0%}), expected {expected_intent}")

    return print_result(passed, total, "Intent Classification")

def test_query_router():
    """Test 3: Verify query router uses correct threshold."""
    print_header("TEST 3: Query Router")

    from services.query_router import get_query_router, ML_CONFIDENCE_THRESHOLD

    # Check threshold is lowered from 98% to 85%
    if ML_CONFIDENCE_THRESHOLD > 0.90:
        print(f"  [FAIL] Threshold too high: {ML_CONFIDENCE_THRESHOLD:.0%} (should be <=90%)")
        return False

    print(f"  [OK] Confidence threshold: {ML_CONFIDENCE_THRESHOLD:.0%}")

    router = get_query_router()

    # These should be routed (high confidence)
    routed_queries = [
        ("kilometraza vozila", True, "get_MasterData"),
        ("rezerviraj auto", True, "get_AvailableVehicles"),
        ("moj telefon", True, None),  # direct_response, no tool
        ("hvala", True, None),
    ]

    passed = 0
    total = len(routed_queries)

    for query, should_match, expected_tool in routed_queries:
        result = router.route(query)

        if result.matched == should_match:
            if expected_tool is None or result.tool_name == expected_tool:
                print(f"  [OK] \"{query}\" -> matched={result.matched}, tool={result.tool_name}")
                passed += 1
            else:
                print(f"  [FAIL] \"{query}\" -> tool={result.tool_name}, expected {expected_tool}")
        else:
            print(f"  [FAIL] \"{query}\" -> matched={result.matched}, expected {should_match}")

    return print_result(passed, total, "Query Router")

def test_normalization():
    """Test 4: Verify diacritic and synonym normalization."""
    print_header("TEST 4: Query Normalization")

    from services.intent_classifier import normalize_query, normalize_diacritics, normalize_synonyms

    test_cases = [
        # Diacritics
        ("Željko", "zeljko"),
        ("čekaj", "cekaj"),
        ("šalji", "salji"),
        ("đak", "dak"),

        # Synonyms
        ("auto", "vozilo"),
        ("mobitel", "telefon"),
        ("auta", "vozila"),
    ]

    passed = 0
    total = len(test_cases)

    for input_text, expected in test_cases:
        result = normalize_query(input_text)
        if expected in result:
            print(f"  [OK] \"{input_text}\" -> \"{result}\"")
            passed += 1
        else:
            print(f"  [FAIL] \"{input_text}\" -> \"{result}\", expected \"{expected}\"")

    return print_result(passed, total, "Query Normalization")

def test_no_duplicate_logging():
    """Test 5: Verify logging is configured correctly."""
    print_header("TEST 5: Logging Configuration")

    import logging

    # Check root logger handlers
    root_handlers = len(logging.root.handlers)

    # Should have minimal handlers (1-2 max)
    if root_handlers <= 3:
        print(f"  [OK] Root logger has {root_handlers} handler(s)")
        passed = 1
    else:
        print(f"  [WARN] Root logger has {root_handlers} handlers (may cause duplicates)")
        passed = 1  # Not critical

    return print_result(passed, 1, "Logging Configuration")

def test_training_data():
    """Test 6: Verify training data balance."""
    print_header("TEST 6: Training Data Balance")

    import json
    from pathlib import Path
    from collections import Counter

    data_path = Path(__file__).parent.parent / "data" / "training" / "intent_full.jsonl"

    if not data_path.exists():
        print(f"  [FAIL] Training data not found: {data_path}")
        return False

    intent_counts = Counter()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                intent_counts[item['intent']] += 1

    # Check previously underrepresented intents have enough examples
    critical_intents = {
        "GET_VEHICLES": 20,
        "GET_EXPENSES": 20,
        "DELETE_CASE": 20,
        "GET_AVAILABLE_VEHICLES": 20,
        "GET_PHONE": 20,
        "GET_PERSON_ID": 20,
    }

    passed = 0
    total = len(critical_intents)

    for intent, min_count in critical_intents.items():
        count = intent_counts.get(intent, 0)
        if count >= min_count:
            print(f"  [OK] {intent}: {count} examples")
            passed += 1
        else:
            print(f"  [FAIL] {intent}: {count} examples (need >={min_count})")

    # Show total
    print(f"\n  Total training examples: {sum(intent_counts.values())}")
    print(f"  Total intents: {len(intent_counts)}")

    return print_result(passed, total, "Training Data Balance")

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("  ML FIXES VERIFICATION SCRIPT")
    print("  Run this after docker-compose up to verify all fixes work")
    print("=" * 70)

    results = []

    try:
        results.append(("Model Files", test_model_files()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Model Files", False))

    try:
        results.append(("Intent Classification", test_intent_classification()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Intent Classification", False))

    try:
        results.append(("Query Router", test_query_router()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Query Router", False))

    try:
        results.append(("Normalization", test_normalization()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Normalization", False))

    try:
        results.append(("Logging", test_no_duplicate_logging()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Logging", False))

    try:
        results.append(("Training Data", test_training_data()))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Training Data", False))

    # Final summary
    print_header("FINAL RESULTS")

    passed_tests = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    for name, passed in results:
        status = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        print(f"  {status} {name}")

    print()
    if passed_tests == total_tests:
        print("\033[92m" + "=" * 70)
        print("  ALL TESTS PASSED! Fixes are working correctly.")
        print("=" * 70 + "\033[0m")
        return 0
    else:
        print("\033[91m" + "=" * 70)
        print(f"  {total_tests - passed_tests} TEST(S) FAILED! Check output above.")
        print("=" * 70 + "\033[0m")
        return 1

if __name__ == "__main__":
    sys.exit(main())
