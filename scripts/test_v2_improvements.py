"""
Test V2.0 Improvements - Query Type Classifier & Training-Free Selection

This script tests the improvements made in v2.0:
1. Query Type Classifier - detects suffix type before FAISS
2. Training-free LLM selection - uses tool_documentation.json only
3. Improved suffix differentiation

Run: python scripts/test_v2_improvements.py
"""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_type_classifier import (
    QueryTypeClassifier,
    classify_query_type,
    QueryType
)


def test_query_type_classifier():
    """Test Query Type Classifier with various queries."""

    print("=" * 70)
    print("TEST 1: Query Type Classifier")
    print("=" * 70)

    # Test cases: (query, expected_type, description)
    test_cases = [
        # DOCUMENTS
        ("dohvati dokumente kompanije", QueryType.DOCUMENTS, "Documents query"),
        ("upload prilog za vozilo", QueryType.DOCUMENTS, "Upload attachment"),
        ("preuzmi PDF dokument", QueryType.DOCUMENTS, "Download PDF"),

        # THUMBNAIL
        ("prikaži sličicu dokumenta", QueryType.THUMBNAIL, "Thumbnail request"),
        ("daj mi preview slike", QueryType.THUMBNAIL, "Preview request"),

        # METADATA
        ("koja je struktura podataka za vozila", QueryType.METADATA, "Metadata/structure"),
        ("prikaži shemu za expenses", QueryType.METADATA, "Schema request"),
        ("metapodaci za kompaniju", QueryType.METADATA, "Metadata request"),

        # AGGREGATION
        ("koliko ima ukupno troškova", QueryType.AGGREGATION, "Aggregation - total"),
        ("prikaži statistiku vozila", QueryType.AGGREGATION, "Statistics"),
        ("prosjek kilometraže po mjesecu", QueryType.AGGREGATION, "Average"),
        ("grupiraj troškove po tipu", QueryType.AGGREGATION, "Group by"),

        # TREE
        ("hijerarhija organizacijskih jedinica", QueryType.TREE, "Hierarchy"),
        ("prikaži stablo odjela", QueryType.TREE, "Tree structure"),

        # DELETE_CRITERIA
        ("obriši sve stare zapise", QueryType.DELETE_CRITERIA, "Bulk delete"),
        ("izbriši sve po kriteriju", QueryType.DELETE_CRITERIA, "Delete by criteria"),

        # BULK_UPDATE
        ("ažuriraj sve vozila", QueryType.BULK_UPDATE, "Bulk update"),
        ("masovno ažuriranje", QueryType.BULK_UPDATE, "Mass update"),

        # DEFAULT_SET
        ("postavi kao zadani dokument", QueryType.DEFAULT_SET, "Set as default"),
        ("označi kao primarni", QueryType.DEFAULT_SET, "Set as primary"),

        # LIST
        ("sve kompanije", QueryType.LIST, "List all"),
        ("popis svih vozila", QueryType.LIST, "List vehicles"),
        ("prikaži sve zaposlenike", QueryType.LIST, "Show all employees"),

        # SINGLE_ENTITY
        ("dohvati kompaniju 123", QueryType.SINGLE_ENTITY, "Single entity by ID"),
        ("prikaži detalje vozila", QueryType.SINGLE_ENTITY, "Entity details"),
        ("info o jednom zaposleniku", QueryType.SINGLE_ENTITY, "Single employee info"),
    ]

    classifier = QueryTypeClassifier()

    correct = 0
    total = len(test_cases)

    print(f"\n{'Query':<45} | {'Expected':<15} | {'Got':<15} | {'Conf':<6} | {'Status'}")
    print("-" * 100)

    for query, expected_type, description in test_cases:
        result = classifier.classify(query)

        is_correct = result.query_type == expected_type
        if is_correct:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"

        # Truncate query for display
        display_query = query[:42] + "..." if len(query) > 45 else query

        print(
            f"{display_query:<45} | "
            f"{expected_type.value:<15} | "
            f"{result.query_type.value:<15} | "
            f"{result.confidence:.2f}   | "
            f"{status}"
        )

    accuracy = correct / total * 100
    print("-" * 100)
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    return accuracy


def test_suffix_filtering():
    """Test suffix filtering with Query Type Classifier."""

    print("\n" + "=" * 70)
    print("TEST 2: Suffix Filtering")
    print("=" * 70)

    classifier = QueryTypeClassifier()

    # Simulate FAISS results - tools with similar scores
    mock_faiss_results = [
        "get_Companies_id",
        "get_Companies_id_documents",
        "get_Companies_id_documents_documentId",
        "get_Companies_id_documents_documentId_thumb",
        "get_Companies_id_metadata",
        "get_Companies",
        "get_Companies_Agg",
        "get_Companies_tree",
    ]

    test_queries = [
        ("dohvati dokumente kompanije", ["get_Companies_id_documents", "get_Companies_id_documents_documentId"]),
        ("prikaži sličicu dokumenta kompanije", ["get_Companies_id_documents_documentId_thumb"]),
        ("metapodaci kompanije", ["get_Companies_id_metadata"]),
        ("sve kompanije", ["get_Companies"]),
        ("hijerarhija kompanija", ["get_Companies_tree"]),
        ("ukupno kompanija", ["get_Companies_Agg"]),
        ("dohvati kompaniju po ID-u", ["get_Companies_id"]),
    ]

    print(f"\n{'Query':<40} | {'Expected Top Tools':<50} | {'Got':<50}")
    print("-" * 145)

    for query, expected_top in test_queries:
        result = classifier.classify(query)
        filtered = classifier.filter_tools_by_type(mock_faiss_results, result)

        # Get top results
        top_filtered = filtered[:2]

        display_query = query[:37] + "..." if len(query) > 40 else query
        expected_str = ", ".join(expected_top)[:47]
        got_str = ", ".join(top_filtered)[:47]

        # Check if expected tools are in top results
        match = any(exp in top_filtered for exp in expected_top)
        status = "OK" if match else "FAIL"

        print(f"{display_query:<40} | {expected_str:<50} | {got_str:<47} {status}")

    print()


def test_problematic_cases():
    """Test cases that were problematic before v2.0."""

    print("\n" + "=" * 70)
    print("TEST 3: Previously Problematic Cases")
    print("=" * 70)

    classifier = QueryTypeClassifier()

    # These are cases from confusion_report.json that had 0% accuracy
    problematic_cases = [
        {
            "query": "dohvati kompaniju",
            "confusion": "get_Companies_id vs get_Companies_id_documents_documentId",
            "correct_suffix": "_id",
            "wrong_suffix": "_id_documents_documentId",
        },
        {
            "query": "dokumenti kompanije",
            "confusion": "get_Companies_id_documents vs get_Companies_id",
            "correct_suffix": "_id_documents",
            "wrong_suffix": "_id",
        },
        {
            "query": "metapodaci vozila",
            "confusion": "get_Vehicles_id_metadata vs get_Vehicles_id",
            "correct_suffix": "_id_metadata",
            "wrong_suffix": "_id",
        },
        {
            "query": "obriši sve stare unose",
            "confusion": "delete_X_DeleteByCriteria vs delete_X_id",
            "correct_suffix": "_DeleteByCriteria",
            "wrong_suffix": "_id",
        },
    ]

    print(f"\n{'Query':<30} | {'Confusion':<50} | {'V2.0 Solution'}")
    print("-" * 120)

    for case in problematic_cases:
        result = classifier.classify(case["query"])

        # Check if query type would help
        preferred = result.preferred_suffixes
        excluded = result.excluded_suffixes

        would_help = (
            any(case["correct_suffix"] in s for s in preferred) or
            any(case["wrong_suffix"] in s for s in excluded)
        )

        solution = f"Type: {result.query_type.value}, Pref: {preferred[:2]}"
        status = "OK FIXED" if would_help else "? CHECK"

        print(f"{case['query']:<30} | {case['confusion']:<50} | {solution} {status}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("       V2.0 IMPROVEMENTS TEST SUITE")
    print("       Query Type Classifier & Training-Free Selection")
    print("=" * 70)

    # Test 1: Query Type Classifier accuracy
    accuracy = test_query_type_classifier()

    # Test 2: Suffix filtering
    test_suffix_filtering()

    # Test 3: Problematic cases
    test_problematic_cases()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
V2.0 Improvements:
1. Query Type Classifier: {accuracy:.1f}% accuracy on test cases
2. Removed training_queries.json from LLM few-shot
3. Added suffix-based filtering and boosting
4. Added 30% penalty for excluded suffixes

Expected Impact:
- Top-1 accuracy: +15-20% for suffix-confused tools
- Reduced confusion between:
  - get_X_id vs get_X_id_documents
  - get_X_id vs get_X_id_metadata
  - delete_X_id vs delete_X_DeleteByCriteria

To fully test, run the unified search with real queries.
""")


if __name__ == "__main__":
    main()
