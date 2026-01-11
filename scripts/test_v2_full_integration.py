"""
Full Integration Test V2.0 - Tests with real FAISS and tool documentation.

This tests the complete pipeline:
1. Query Type Classification
2. Action Intent Detection
3. FAISS Semantic Search
4. Boosting with query type
5. Final ranking

Run: python scripts/test_v2_full_integration.py
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from typing import List, Tuple

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.query_type_classifier import classify_query_type, QueryType
from services.action_intent_detector import detect_action_intent, ActionIntent


# Test cases: (query, expected_tool, description)
# These are based on confusion_report.json problematic cases
TEST_CASES = [
    # === BASIC CRUD ===
    ("dohvati sve kompanije", "get_Companies", "List all companies"),
    ("dohvati kompaniju po ID-u", "get_Companies_id", "Get single company"),
    ("dodaj novu kompaniju", "post_Companies", "Create company"),
    ("azuriraj kompaniju", "put_Companies_id", "Update company"),
    ("obrisi kompaniju", "delete_Companies_id", "Delete company"),

    # === DOCUMENTS (previously confused) ===
    ("dohvati dokumente kompanije", "get_Companies_id_documents", "Company documents"),
    ("dodaj dokument kompaniji", "post_Companies_id_documents", "Add company document"),
    ("preuzmi dokument kompanije", "get_Companies_id_documents_documentId", "Get specific document"),
    ("obrisi dokument kompanije", "delete_Companies_id_documents_documentId", "Delete document"),

    # === METADATA (previously confused) ===
    ("metapodaci kompanije", "get_Companies_id_metadata", "Company metadata"),
    ("struktura podataka za vozila", "get_Vehicles_id_metadata", "Vehicle metadata"),

    # === THUMBNAILS ===
    ("prikazi slicicu dokumenta", "get_Companies_id_documents_documentId_thumb", "Document thumbnail"),

    # === AGGREGATIONS ===
    ("ukupno troskova", "get_Expenses_Agg", "Expenses aggregation"),
    ("statistika vozila", "get_Vehicles_Agg", "Vehicle stats"),
    ("grupiraj troskove po tipu", "get_Expenses_GroupBy", "Group expenses"),

    # === TREE/HIERARCHY ===
    ("hijerarhija organizacijskih jedinica", "get_OrgUnits_tree", "Org units tree"),

    # === BULK OPERATIONS ===
    ("obrisi sve stare zapise", "delete_Companies_DeleteByCriteria", "Bulk delete"),
    ("azuriraj vise vozila odjednom", "patch_Vehicles_multipatch", "Bulk update"),

    # === SET AS DEFAULT ===
    ("postavi dokument kao zadani", "put_Companies_id_documents_documentId_SetAsDefault", "Set default"),

    # === VEHICLES ===
    ("dohvati sva vozila", "get_Vehicles", "List vehicles"),
    ("dohvati vozilo po ID-u", "get_Vehicles_id", "Get single vehicle"),
    ("kalendar vozila", "get_VehicleCalendar", "Vehicle calendar"),
    ("dodaj rezervaciju vozila", "post_VehicleCalendar", "Add reservation"),

    # === PERSONS ===
    ("dohvati sve zaposlenike", "get_Persons", "List persons"),
    ("dohvati zaposlenika", "get_Persons_id", "Get single person"),
    ("dokumenti zaposlenika", "get_Persons_id_documents", "Person documents"),

    # === EXPENSES ===
    ("dohvati sve troskove", "get_Expenses", "List expenses"),
    ("dodaj novi trosak", "post_Expenses", "Add expense"),

    # === CASES (damage reports) ===
    ("prijavi stetu na vozilu", "post_AddCase", "Report damage"),
    ("imam kvar na autu", "post_AddCase", "Report malfunction"),

    # === MILEAGE ===
    ("unesi kilometrazu", "post_AddMileage", "Add mileage"),
    ("koliko km ima vozilo", "get_MasterData", "Get vehicle mileage"),

    # === EDGE CASES ===
    ("projekcija podataka kompanije", "get_Companies_ProjectTo", "Projection"),
]


async def run_integration_test():
    """Run full integration test with real FAISS."""

    print("=" * 80)
    print("       V2.0 FULL INTEGRATION TEST")
    print("       Testing with real FAISS and tool documentation")
    print("=" * 80)

    # Load tool documentation
    doc_path = Path(__file__).parent.parent / "config" / "tool_documentation.json"
    if not doc_path.exists():
        print(f"ERROR: Tool documentation not found: {doc_path}")
        return

    with open(doc_path, 'r', encoding='utf-8') as f:
        tool_documentation = json.load(f)

    print(f"\nLoaded {len(tool_documentation)} tools from documentation")

    # Initialize FAISS
    print("Initializing FAISS vector store...")
    from services.faiss_vector_store import initialize_faiss_store

    try:
        faiss_store = await initialize_faiss_store(tool_documentation)
        print(f"FAISS initialized: {faiss_store.get_stats()}")
    except Exception as e:
        print(f"ERROR initializing FAISS: {e}")
        print("\nRunning without FAISS - testing classification only...")
        await run_classification_only_test()
        return

    # Initialize unified search
    from services.unified_search import initialize_unified_search
    unified_search = await initialize_unified_search()

    print(f"\nUnified Search stats: {unified_search.get_stats()}")

    # Run tests
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    results = {
        "top_1": 0,
        "top_3": 0,
        "top_5": 0,
        "top_10": 0,
        "total": len(TEST_CASES),
        "failures": []
    }

    print(f"\n{'Query':<40} | {'Expected':<35} | {'Got':<35} | {'Rank':<5}")
    print("-" * 120)

    for query, expected_tool, description in TEST_CASES:
        try:
            # Run unified search
            response = await unified_search.search(query, top_k=10)

            # Find rank of expected tool
            rank = -1
            got_tool = ""
            for i, result in enumerate(response.results):
                if i == 0:
                    got_tool = result.tool_id
                if result.tool_id == expected_tool:
                    rank = i + 1
                    break

            # Update metrics
            if rank == 1:
                results["top_1"] += 1
                status = "TOP1"
            elif rank <= 3:
                results["top_3"] += 1
                status = f"TOP{rank}"
            elif rank <= 5:
                results["top_5"] += 1
                status = f"TOP{rank}"
            elif rank <= 10:
                results["top_10"] += 1
                status = f"TOP{rank}"
            else:
                status = "MISS"
                results["failures"].append({
                    "query": query,
                    "expected": expected_tool,
                    "got": got_tool,
                    "query_type": response.query_type.value,
                    "intent": response.intent.value
                })

            # Also count top-3 cumulative
            if rank <= 3 and rank > 0:
                results["top_3"] += 0  # Already counted in top_1

            # Truncate for display
            display_query = query[:37] + "..." if len(query) > 40 else query
            display_expected = expected_tool[:32] + "..." if len(expected_tool) > 35 else expected_tool
            display_got = got_tool[:32] + "..." if len(got_tool) > 35 else got_tool

            print(f"{display_query:<40} | {display_expected:<35} | {display_got:<35} | {status:<5}")

        except Exception as e:
            print(f"{query[:40]:<40} | ERROR: {str(e)[:60]}")
            results["failures"].append({
                "query": query,
                "expected": expected_tool,
                "error": str(e)
            })

    # Calculate cumulative metrics
    top_1_acc = results["top_1"] / results["total"] * 100
    top_3_acc = (results["top_1"] + results["top_3"]) / results["total"] * 100
    top_5_acc = (results["top_1"] + results["top_3"] + results["top_5"]) / results["total"] * 100
    top_10_acc = (results["top_1"] + results["top_3"] + results["top_5"] + results["top_10"]) / results["total"] * 100

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"""
Accuracy Metrics:
-----------------
Top-1:  {results['top_1']}/{results['total']} ({top_1_acc:.1f}%)
Top-3:  {results['top_1'] + results['top_3']}/{results['total']} ({top_3_acc:.1f}%)
Top-5:  {results['top_1'] + results['top_3'] + results['top_5']}/{results['total']} ({top_5_acc:.1f}%)
Top-10: {results['top_1'] + results['top_3'] + results['top_5'] + results['top_10']}/{results['total']} ({top_10_acc:.1f}%)

Comparison with V1.0:
---------------------
V1.0 Top-1: ~67% (baseline)
V2.0 Top-1: {top_1_acc:.1f}% {'(+' + str(round(top_1_acc - 67, 1)) + '%)' if top_1_acc > 67 else ''}

V1.0 Top-3: ~93% (baseline)
V2.0 Top-3: {top_3_acc:.1f}% {'(+' + str(round(top_3_acc - 93, 1)) + '%)' if top_3_acc > 93 else ''}
""")

    if results["failures"]:
        print("\nFailed Cases:")
        print("-" * 80)
        for failure in results["failures"][:10]:  # Show first 10 failures
            print(f"  Query: {failure['query']}")
            print(f"  Expected: {failure['expected']}")
            print(f"  Got: {failure.get('got', 'N/A')}")
            if 'query_type' in failure:
                print(f"  QueryType: {failure['query_type']}, Intent: {failure['intent']}")
            print()

    return results


async def run_classification_only_test():
    """Run test without FAISS - just classification."""

    print("\n" + "=" * 80)
    print("CLASSIFICATION-ONLY TEST (No FAISS)")
    print("=" * 80)

    print(f"\n{'Query':<45} | {'Intent':<10} | {'QueryType':<15} | {'Preferred Suffix'}")
    print("-" * 100)

    for query, expected_tool, description in TEST_CASES[:20]:  # First 20
        intent_result = detect_action_intent(query)
        query_type_result = classify_query_type(query)

        display_query = query[:42] + "..." if len(query) > 45 else query
        preferred = query_type_result.preferred_suffixes[:2] if query_type_result.preferred_suffixes else []

        print(
            f"{display_query:<45} | "
            f"{intent_result.intent.value:<10} | "
            f"{query_type_result.query_type.value:<15} | "
            f"{preferred}"
        )


def main():
    """Main entry point."""
    asyncio.run(run_integration_test())


if __name__ == "__main__":
    main()
