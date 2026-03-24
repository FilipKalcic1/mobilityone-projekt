"""
Auto-improve Tool Documentation - Automatski poboljšava example_queries_hr

Logika:
1. Čita confusion_report.json da vidi koji alati imaju problema
2. Za svaki problematični alat generira SPECIFIČNIJE upite
3. Koristi razliku između alata da napravi distinktivne upite

Strategije poboljšanja:
- Dodaje ime entiteta u upit (npr. "kompanija", "vozilo", "tim")
- Dodaje akciju specifičnu za HTTP metodu
- Koristi parametre alata za kontekst
- Razlikuje od sličnih alata eksplicitno
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: Dict):
    """Save JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Entity mappings - Croatian names for common entities
ENTITY_NAMES = {
    "companies": "kompanija",
    "company": "kompanija",
    "vehicles": "vozilo",
    "vehicle": "vozilo",
    "persons": "osoba",
    "person": "osoba",
    "teams": "tim",
    "team": "tim",
    "cases": "predmet",
    "case": "predmet",
    "expenses": "trosak",
    "expense": "trosak",
    "mileage": "kilometraza",
    "calendar": "kalendar",
    "reservation": "rezervacija",
    "booking": "rezervacija",
    "documents": "dokument",
    "document": "dokument",
    "equipment": "oprema",
    "partners": "partner",
    "partner": "partner",
    "tenants": "najam",
    "tenant": "najam",
    "orgunit": "organizacijska jedinica",
    "orgunits": "organizacijska jedinica",
    "metadata": "metapodaci",
    "settings": "postavke",
    "permissions": "dozvole",
    "roles": "uloge",
    "costcenters": "troskovni centar",
    "costcenter": "troskovni centar",
    "schedulingmodels": "model rasporeda",
    "periodicactivities": "periodicna aktivnost",
    "trips": "putovanje",
}

# Action verbs by HTTP method
ACTION_VERBS = {
    "get": ["dohvati", "prikazi", "pronadi", "pogledaj", "pokazi mi", "daj mi"],
    "post": ["dodaj", "kreiraj", "napravi", "unesi", "stvori", "zapisi"],
    "put": ["azuriraj", "promijeni", "izmijeni", "postavi"],
    "patch": ["azuriraj", "promijeni djelomicno", "modificiraj"],
    "delete": ["obrisi", "ukloni", "izbriši", "makni"],
}

# Specific suffixes and their meanings
SUFFIX_MEANINGS = {
    "_id": "pojedinacni",  # Single item by ID
    "_documents": "dokumenti",
    "_documents_documentId": "pojedinacni dokument",
    "_thumb": "slicica",
    "_metadata": "metapodaci",
    "_Agg": "agregacija",
    "_GroupBy": "grupiranje",
    "_ProjectTo": "projekcija",
    "_DeleteByCriteria": "brisanje po kriterijima",
    "_multipatch": "visestruko azuriranje",
    "_SetAsDefault": "postavi kao zadano",
    "_tree": "stablo",
}


def extract_entity_from_tool_id(tool_id: str) -> str:
    """Extract entity name from tool_id."""
    # Remove method prefix
    name = re.sub(r'^(get|post|put|patch|delete)_', '', tool_id, flags=re.IGNORECASE)

    # Remove common suffixes
    for suffix in sorted(SUFFIX_MEANINGS.keys(), key=len, reverse=True):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # Remove _id suffix
    name = re.sub(r'_id$', '', name, flags=re.IGNORECASE)

    # Convert to lowercase for lookup
    name_lower = name.lower()

    # Try to find entity
    for key, value in ENTITY_NAMES.items():
        if key in name_lower:
            return value

    return name


def get_method_from_tool_id(tool_id: str) -> str:
    """Extract HTTP method from tool_id."""
    match = re.match(r'^(get|post|put|patch|delete)_', tool_id, flags=re.IGNORECASE)
    return match.group(1).lower() if match else "get"


def get_suffix_type(tool_id: str) -> str:
    """Get the suffix type of the tool."""
    for suffix, meaning in sorted(SUFFIX_MEANINGS.items(), key=lambda x: len(x[0]), reverse=True):
        if tool_id.endswith(suffix):
            return suffix
    return ""


def generate_distinctive_queries(
    tool_id: str,
    confused_with: List[str],
    existing_queries: List[str],
    tool_doc: Dict
) -> List[str]:
    """
    Generate queries that distinguish this tool from those it's confused with.
    """
    method = get_method_from_tool_id(tool_id)
    entity = extract_entity_from_tool_id(tool_id)
    suffix = get_suffix_type(tool_id)

    new_queries = []
    verbs = ACTION_VERBS.get(method, ["dohvati"])

    # Strategy 1: Add entity + suffix-specific queries
    if suffix == "_id":
        new_queries.extend([
            f"{verbs[0]} {entity} s ID-om",
            f"{verbs[0]} tocno jedan {entity}",
            f"prikazi detalje {entity} broj",
        ])
    elif suffix == "_documents":
        new_queries.extend([
            f"prikazi sve dokumente za {entity}",
            f"lista dokumenata {entity}",
            f"koji dokumenti su prilozeni {entity}",
        ])
    elif suffix == "_documents_documentId":
        new_queries.extend([
            f"{verbs[0]} jedan specifican dokument za {entity}",
            f"prikazi tocno taj dokument {entity}",
            f"dohvati dokument s ID-om za {entity}",
        ])
    elif suffix == "_thumb":
        new_queries.extend([
            f"prikazi slicicu dokumenta za {entity}",
            f"thumbnail dokumenta {entity}",
            f"mala slika dokumenta",
        ])
    elif suffix == "_metadata":
        new_queries.extend([
            f"metapodaci za {entity}",
            f"dodatne informacije o {entity}",
            f"meta info {entity}",
        ])
    elif suffix == "_Agg":
        new_queries.extend([
            f"agregirana statistika za {entity}",
            f"suma/prosjek/max {entity}",
            f"zbirni podaci {entity}",
        ])
    elif suffix == "_GroupBy":
        new_queries.extend([
            f"grupiraj {entity} po kategoriji",
            f"grupirani podaci za {entity}",
            f"grupiranje {entity} prema",
        ])
    elif suffix == "_ProjectTo":
        new_queries.extend([
            f"samo odredene kolone {entity}",
            f"projekcija polja za {entity}",
            f"dohvati samo id i naziv {entity}",
        ])
    elif suffix == "_DeleteByCriteria":
        new_queries.extend([
            f"obrisi sve {entity} koji zadovoljavaju uvjet",
            f"masovno brisanje {entity} prema filteru",
            f"ukloni {entity} po kriterijima",
        ])
    elif suffix == "_multipatch":
        new_queries.extend([
            f"azuriraj vise {entity} odjednom",
            f"masovno azuriranje {entity}",
            f"promijeni visestruke {entity} s ID-evima",
        ])
    elif suffix == "_SetAsDefault":
        new_queries.extend([
            f"postavi kao zadani dokument za {entity}",
            f"oznaci dokument kao default",
            f"zadani dokument {entity}",
        ])

    # Strategy 2: Method-specific distinctive queries
    if method == "post" and not suffix:
        new_queries.extend([
            f"kreiraj novi {entity}",
            f"dodaj novi {entity} u sustav",
            f"napravi {entity}",
        ])
    elif method == "put" and "_id" in tool_id.lower():
        new_queries.extend([
            f"potpuno azuriraj {entity}",
            f"zamijeni sve podatke {entity}",
            f"puna izmjena {entity}",
        ])
    elif method == "patch":
        new_queries.extend([
            f"djelomicno azuriraj {entity}",
            f"promijeni samo neka polja {entity}",
            f"parcijalna izmjena {entity}",
        ])
    elif method == "delete" and "_id" in tool_id.lower():
        new_queries.extend([
            f"obrisi {entity} s odredenim ID-om",
            f"ukloni jednu stavku {entity}",
            f"izbriši {entity} broj",
        ])

    # Strategy 3: If confused with specific tools, add differentiating queries
    for confused_tool in confused_with[:3]:
        confused_entity = extract_entity_from_tool_id(confused_tool)
        confused_suffix = get_suffix_type(confused_tool)

        if confused_entity != entity:
            new_queries.append(f"{verbs[0]} {entity} (ne {confused_entity})")

        if confused_suffix != suffix:
            if suffix and not confused_suffix:
                suffix_meaning = SUFFIX_MEANINGS.get(suffix, "")
                if suffix_meaning:
                    new_queries.append(f"{verbs[0]} {suffix_meaning} za {entity}")

    # Strategy 4: Use tool parameters if available
    params = tool_doc.get("parameters", [])
    if params and isinstance(params, list):
        for param in params[:3]:
            if isinstance(param, dict):
                param_name = param.get("name", "")
                if param_name and param_name not in ["tenantId", "personId"]:
                    # Convert camelCase to readable
                    readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', param_name).lower()
                    new_queries.append(f"{verbs[0]} {entity} filtrirano po {readable}")

    # Filter out duplicates and existing queries
    existing_lower = {q.lower() for q in existing_queries}
    unique_queries = []
    seen = set()

    for q in new_queries:
        q_lower = q.lower()
        if q_lower not in existing_lower and q_lower not in seen:
            unique_queries.append(q)
            seen.add(q_lower)

    return unique_queries[:5]  # Return max 5 new queries


def improve_stealer_queries(
    tool_id: str,
    stolen_from: List[Tuple[str, int]],
    existing_queries: List[str],
    tool_doc: Dict
) -> List[str]:
    """
    Make queries for 'stealer' tools more specific so they don't match everything.
    """
    method = get_method_from_tool_id(tool_id)
    entity = extract_entity_from_tool_id(tool_id)
    suffix = get_suffix_type(tool_id)

    # Add very specific queries that won't match other tools
    specific_queries = []

    # Add the exact entity name multiple times
    if entity:
        specific_queries.extend([
            f"samo za {entity}",
            f"konkretno {entity}",
            f"specificno {entity} podaci",
        ])

    # Add suffix-specific restrictive queries
    if suffix:
        suffix_meaning = SUFFIX_MEANINGS.get(suffix, "")
        if suffix_meaning:
            specific_queries.extend([
                f"iskljucivo {suffix_meaning} za {entity}",
                f"samo {suffix_meaning}",
            ])

    # Filter existing
    existing_lower = {q.lower() for q in existing_queries}
    unique = [q for q in specific_queries if q.lower() not in existing_lower]

    return unique[:3]


def run_auto_improvement():
    """Main function to auto-improve documentation."""
    print("=" * 70)
    print("AUTO-IMPROVE TOOL DOCUMENTATION")
    print("=" * 70)
    print()

    # Load files
    print("[1/4] Loading files...")

    confusion_path = project_root / "confusion_report.json"
    if not confusion_path.exists():
        print("ERROR: confusion_report.json not found!")
        print("Run test_confusion_analysis.py first.")
        return

    confusion = load_json(confusion_path)

    doc_path = project_root / "config" / "tool_documentation.json"
    tool_docs = load_json(doc_path)

    print(f"      Loaded {len(tool_docs)} tool docs")
    print(f"      Loaded confusion report")
    print()

    # Get tools needing work
    tools_needing_work = confusion.get("tools_needing_work", [])
    top_confusions = confusion.get("top_confusions", [])

    print(f"[2/4] Analyzing {len(tools_needing_work)} tools with <50% Top-1...")

    # Build confusion map: tool -> [confused_with tools]
    confusion_map = defaultdict(list)
    for conf in top_confusions:
        expected = conf["expected"]
        got = conf["got"]
        confusion_map[expected].append(got)

    # Track improvements
    improved_count = 0
    queries_added = 0

    # Process each problematic tool
    print("[3/4] Generating improved queries...")

    for tool_info in tools_needing_work:
        tool_id = tool_info["tool"]
        top1_acc = tool_info["top1_acc"]
        confused_with = tool_info.get("confused_with", [])

        if tool_id not in tool_docs:
            continue

        doc = tool_docs[tool_id]
        existing_queries = doc.get("example_queries_hr", [])

        # Generate new queries
        new_queries = generate_distinctive_queries(
            tool_id, confused_with, existing_queries, doc
        )

        if new_queries:
            # Add to existing queries
            doc["example_queries_hr"] = existing_queries + new_queries
            queries_added += len(new_queries)
            improved_count += 1

    # Also improve "stealer" tools
    print("      Improving 'stealer' tools...")

    # Build stealer map from confusions
    stealer_map = defaultdict(list)
    for conf in top_confusions:
        got = conf["got"]
        expected = conf["expected"]
        count = conf["count"]
        stealer_map[got].append((expected, count))

    # Get top stealers
    top_stealers = sorted(
        stealer_map.items(),
        key=lambda x: sum(c for _, c in x[1]),
        reverse=True
    )[:20]

    for stealer_tool, stolen_from in top_stealers:
        if stealer_tool not in tool_docs:
            continue

        doc = tool_docs[stealer_tool]
        existing_queries = doc.get("example_queries_hr", [])

        # Make queries more specific
        specific_queries = improve_stealer_queries(
            stealer_tool, stolen_from, existing_queries, doc
        )

        if specific_queries:
            doc["example_queries_hr"] = existing_queries + specific_queries
            queries_added += len(specific_queries)

    # Save updated documentation
    print("[4/4] Saving improved documentation...")
    save_json(doc_path, tool_docs)

    print()
    print("=" * 70)
    print("REZULTATI")
    print("=" * 70)
    print(f"   Alata poboljsano: {improved_count}")
    print(f"   Novih upita dodano: {queries_added}")
    print()
    print("SLJEDECI KORACI:")
    print("   1. Regeneriraj embeddings:")
    print("      python scripts/generate_tool_embeddings.py --force-regenerate")
    print()
    print("   2. Testiraj ponovo:")
    print("      python scripts/test_search_accuracy_v2.py")
    print("=" * 70)


if __name__ == "__main__":
    run_auto_improvement()
