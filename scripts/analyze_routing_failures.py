"""
Analyze Tool Routing Failures - Create categorized report of generic/overlapping issues.
"""

import asyncio
import json
import sys
import os
from collections import defaultdict

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    # Load tool documentation
    with open('config/tool_documentation.json', 'r', encoding='utf-8') as f:
        tool_docs = json.load(f)

    print("=" * 70)
    print("ANALYSIS: Generic & Overlapping Query Patterns")
    print("=" * 70)

    # =========================================================================
    # 1. GENERIC QUERY PATTERNS (match many tools)
    # =========================================================================
    generic_patterns = defaultdict(list)

    # Find queries that are too generic
    generic_keywords = [
        ("prosječnu vrijednost", "_Agg"),
        ("maksimalnu vrijednost", "_Agg"),
        ("minimalnu vrijednost", "_Agg"),
        ("agregirane podatke", "_Agg"),
        ("grupirane podatke", "_GroupBy"),
        ("grupiraj prema", "_GroupBy"),
        ("s kolonama", "_ProjectTo"),
        ("filtriraj", "_ProjectTo"),
        ("sortiraj", "_ProjectTo"),
        ("metapodatke", "_metadata"),
        ("dokument s ID", "_documents"),
        ("informacije o", "get_"),
        ("podatke o", "get_"),
        ("dohvati", "get_"),
        ("prikaži", "get_"),
        ("ažuriraj", "put_"),
        ("izbriši", "delete_"),
        ("obriši", "delete_"),
    ]

    for op_id, doc in tool_docs.items():
        queries = doc.get('example_queries_hr', [])
        for q in queries:
            q_lower = q.lower()
            for pattern, suffix in generic_keywords:
                if pattern in q_lower:
                    # Check if query is too generic (doesn't mention entity)
                    entity_mentioned = False
                    # Extract entity from operation_id
                    parts = op_id.split('_')
                    if len(parts) > 1:
                        entity = parts[1].lower()
                        if entity in q_lower or entity[:-1] in q_lower:  # handle plurals
                            entity_mentioned = True

                    if not entity_mentioned and suffix in op_id:
                        generic_patterns[pattern].append({
                            'tool': op_id,
                            'query': q[:60]
                        })

    print("\n" + "=" * 70)
    print("1. GENERIC QUERY PATTERNS (too vague, match many tools)")
    print("=" * 70)

    for pattern, items in sorted(generic_patterns.items(), key=lambda x: -len(x[1])):
        if len(items) >= 3:
            print(f"\n  Pattern: '{pattern}' - {len(items)} tools affected")
            for item in items[:5]:
                print(f"    - {item['tool']}: '{item['query']}...'")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")

    # =========================================================================
    # 2. OVERLAPPING TOOL SUFFIXES
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. OVERLAPPING TOOL CATEGORIES (similar operations)")
    print("=" * 70)

    suffix_groups = defaultdict(list)
    for op_id in tool_docs.keys():
        # Group by suffix pattern
        if "_Agg" in op_id:
            suffix_groups["*_Agg (Aggregations)"].append(op_id)
        elif "_GroupBy" in op_id:
            suffix_groups["*_GroupBy (Grouping)"].append(op_id)
        elif "_ProjectTo" in op_id:
            suffix_groups["*_ProjectTo (Projections)"].append(op_id)
        elif "_metadata" in op_id:
            suffix_groups["*_metadata (Metadata)"].append(op_id)
        elif "_DeleteByCriteria" in op_id:
            suffix_groups["*_DeleteByCriteria (Bulk Delete)"].append(op_id)
        elif "_multipatch" in op_id:
            suffix_groups["*_multipatch (Bulk Update)"].append(op_id)
        elif "_documents_documentId" in op_id:
            suffix_groups["*_documents_documentId (Document by ID)"].append(op_id)
        elif "_documents" in op_id and "_documentId" not in op_id:
            suffix_groups["*_documents (Document list)"].append(op_id)

    for suffix, tools in sorted(suffix_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  {suffix}: {len(tools)} tools")
        # Show sample
        for t in tools[:3]:
            print(f"    - {t}")
        if len(tools) > 3:
            print(f"    ... and {len(tools) - 3} more")

    # =========================================================================
    # 3. ENTITY CONFUSION (similar entities)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. ENTITY CONFUSION (similar entity names)")
    print("=" * 70)

    # Find entities with similar names
    entities = defaultdict(list)
    for op_id in tool_docs.keys():
        parts = op_id.split('_')
        if len(parts) > 1:
            entity = parts[1]
            entities[entity].append(op_id)

    # Find similar entities
    confusing_pairs = [
        ("Companies", "CostCenters", "kompanije vs troškovna mjesta"),
        ("Vehicles", "VehicleTypes", "vozila vs tipovi vozila"),
        ("Persons", "PersonTypes", "osobe vs tipovi osoba"),
        ("Cases", "CaseTypes", "slučajevi vs tipovi slučajeva"),
        ("Documents", "DocumentTypes", "dokumenti vs tipovi dokumenata"),
        ("Equipment", "EquipmentTypes", "oprema vs tipovi opreme"),
        ("Trips", "TripTypes", "putovanja vs tipovi putovanja"),
        ("Expenses", "ExpenseTypes", "troškovi vs tipovi troškova"),
        ("Calendar", "VehicleCalendar", "kalendar vs kalendar vozila"),
        ("Activities", "PeriodicActivities", "aktivnosti vs periodične aktivnosti"),
    ]

    for e1, e2, desc in confusing_pairs:
        count1 = len([t for t in tool_docs.keys() if f"_{e1}" in t or t.startswith(f"get_{e1}") or t.startswith(f"post_{e1}")])
        count2 = len([t for t in tool_docs.keys() if f"_{e2}" in t or t.startswith(f"get_{e2}") or t.startswith(f"post_{e2}")])
        if count1 > 0 and count2 > 0:
            print(f"\n  {desc}:")
            print(f"    - {e1}: {count1} tools")
            print(f"    - {e2}: {count2} tools")

    # =========================================================================
    # 4. CRUD OVERLAP (get/put/delete for same entity)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. CRUD OVERLAP (queries don't specify action clearly)")
    print("=" * 70)

    # Find tools where queries could match multiple CRUD operations
    crud_confusion = []
    for op_id, doc in tool_docs.items():
        queries = doc.get('example_queries_hr', [])
        for q in queries:
            q_lower = q.lower()
            # Check for ambiguous queries
            ambiguous_words = ["podatke", "informacije", "detalje", "stavku"]
            if any(w in q_lower for w in ambiguous_words):
                # Could match get, put, or patch
                if not any(w in q_lower for w in ["dohvati", "prikaži", "pokaži", "lista"]):
                    if not any(w in q_lower for w in ["ažuriraj", "promijeni", "izmijeni", "update"]):
                        if not any(w in q_lower for w in ["obriši", "izbriši", "ukloni", "makni"]):
                            crud_confusion.append({
                                'tool': op_id,
                                'query': q[:60]
                            })

    if crud_confusion:
        print(f"\n  Ambiguous CRUD queries: {len(crud_confusion)} found")
        for item in crud_confusion[:10]:
            print(f"    - {item['tool']}: '{item['query']}...'")
        if len(crud_confusion) > 10:
            print(f"    ... and {len(crud_confusion) - 10} more")

    # =========================================================================
    # 5. RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. RECOMMENDATIONS")
    print("=" * 70)

    print("""
  A) GENERIC QUERY FIX:
     - Example queries should ALWAYS include entity name
     - BAD:  "Daj mi prosječnu vrijednost za polje x"
     - GOOD: "Daj mi prosječnu vrijednost kilometraže za VOZILA"

  B) LLM AS PRIMARY ROUTER:
     - QueryRouter: Use for FAST shortcuts (common phrases)
     - FAISS: Use for semantic similarity search
     - LLM (UnifiedRouter): FINAL DECISION MAKER

     Flow: Query → QueryRouter (hints) → FAISS (candidates) → LLM (decision)

  C) SUFFIX DISAMBIGUATION:
     - _Agg: "prosječno", "suma", "maksimum", "minimum" + ENTITY
     - _GroupBy: "grupiraj po", "po kategorijama" + ENTITY
     - _ProjectTo: "samo stupce", "filtrirana lista" + ENTITY
     - _metadata: "metapodaci za" + ENTITY

  D) ENTITY SPECIFICITY:
     - Queries should mention SPECIFIC entity, not just operation
     - "Dohvati vozilo" → get_Vehicles_id
     - "Dohvati tip vozila" → get_VehicleTypes_id
""")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
