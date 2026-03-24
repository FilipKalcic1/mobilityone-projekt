"""
Generate query type training data from tool_documentation.json.

Maps tool_id suffixes to query types, then uses each tool's example_queries_hr
as labeled training examples. Also adds template augmentations.

Usage:
    python scripts/generate_query_type_training.py

Output:
    data/training/query_type.jsonl  (overwrites existing file)
"""
import json
import os
import sys
import re
from collections import Counter
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TOOL_DOCS_PATH = PROJECT_DIR / "config" / "tool_documentation.json"
EXISTING_TRAINING_PATH = PROJECT_DIR / "data" / "training" / "query_type.jsonl"
OUTPUT_PATH = EXISTING_TRAINING_PATH  # overwrite

# Suffix → query type mapping (mirrors tool_family_index.py _SUFFIX_MAP)
_SUFFIX_TO_QUERY_TYPE = {
    "": "LIST",                          # get_Vehicles → LIST
    "id": "SINGLE_ENTITY",              # get_Vehicles_id → SINGLE_ENTITY
    "GroupBy": "AGGREGATION",            # get_Vehicles_GroupBy → AGGREGATION
    "Agg": "AGGREGATION",               # get_Vehicles_Agg → AGGREGATION
    "id_documents": "DOCUMENTS",         # get_Vehicles_id_documents → DOCUMENTS
    "id_documents_documentId": "DOCUMENTS",
    "documents": "DOCUMENTS",
    "id_metadata": "METADATA",           # get_Vehicles_id_metadata → METADATA
    "metadata": "METADATA",
    "tree": "TREE",                      # get_OrgUnits_tree → TREE
    "DeleteByCriteria": "DELETE_CRITERIA",
    "multipatch": "BULK_UPDATE",
    "id_documents_documentId_SetAsDefault": "DEFAULT_SET",
    "SetAsDefault": "DEFAULT_SET",
    "id_documents_documentId_thumb": "THUMBNAIL",
    "thumb": "THUMBNAIL",
    "ProjectTo": "PROJECTION",           # get_Vehicles_ProjectTo → PROJECTION
    "filter": "LIST",                    # filter variants are still list queries
    "on": "LIST",                        # get_X_on → list variant
    "on_Agg": "AGGREGATION",
    "on_GroupBy": "AGGREGATION",
    "on_ProjectTo": "PROJECTION",
    "FileIds": "DOCUMENTS",
    "from_to": "LIST",
    "from_to_Agg": "AGGREGATION",
    "from_to_GroupBy": "AGGREGATION",
}

# Method-specific overrides: delete base → DELETE_CRITERIA, post/put base → LIST
_METHOD_QUERY_TYPE_OVERRIDE = {
    ("delete", ""): "DELETE_CRITERIA",      # delete_Companies → DELETE_CRITERIA
    ("delete", "id"): "SINGLE_ENTITY",      # delete_Companies_id → SINGLE_ENTITY (single delete)
}

# Template augmentations per query type
# These add natural language variations Croatian users might type
_AUGMENTATION_TEMPLATES = {
    "LIST": [
        "prikaži sve {entity_hr}",
        "dohvati sve {entity_hr}",
        "popis {entity_hr}",
        "lista {entity_hr}",
        "koja {entity_hr} imamo",
        "koje {entity_hr} postoje",
        "svi {entity_hr}",
        "sva {entity_hr}",
        "daj mi {entity_hr}",
        "pokaži {entity_hr}",
        "pregledaj {entity_hr}",
        "imam li {entity_hr}",
        "koliko {entity_hr} ima u sustavu",
        "mogu li vidjeti {entity_hr}",
        "trebam listu {entity_hr}",
    ],
    "SINGLE_ENTITY": [
        "detalji o {entity_hr}",
        "info o {entity_hr}",
        "podaci o {entity_hr}",
        "dohvati {entity_hr} po id",
        "prikaži {entity_hr} s id",
        "jedno {entity_hr}",
        "konkretno {entity_hr}",
        "moj {entity_hr}",
        "moje {entity_hr}",
        "moja {entity_hr}",
        "koji je moj {entity_hr}",
        "koje je moje {entity_hr}",
        "status {entity_hr}",
        "pojedinosti {entity_hr}",
        "informacije o {entity_hr}",
    ],
    "AGGREGATION": [
        "grupiraj {entity_hr}",
        "statistika {entity_hr}",
        "ukupno {entity_hr}",
        "koliko ima {entity_hr}",
        "broj {entity_hr}",
        "suma {entity_hr}",
        "prosjek {entity_hr}",
        "grupiraj {entity_hr} po mjesecu",
        "grupiraj {entity_hr} po godini",
        "izvještaj {entity_hr}",
        "agregirani podaci za {entity_hr}",
        "prebrojati {entity_hr}",
    ],
    "DOCUMENTS": [
        "dokumenti za {entity_hr}",
        "prilozi {entity_hr}",
        "datoteke {entity_hr}",
        "pdf {entity_hr}",
        "dodaj dokument za {entity_hr}",
        "preuzmi dokument {entity_hr}",
        "prikaži priloge {entity_hr}",
        "dokumentacija {entity_hr}",
    ],
    "METADATA": [
        "metapodaci {entity_hr}",
        "struktura {entity_hr}",
        "polja {entity_hr}",
        "shema {entity_hr}",
        "atributi {entity_hr}",
        "koja polja ima {entity_hr}",
    ],
    "TREE": [
        "hijerarhija {entity_hr}",
        "stablo {entity_hr}",
        "parent {entity_hr}",
        "podređeni {entity_hr}",
        "nadređeni {entity_hr}",
    ],
    "DELETE_CRITERIA": [
        "obriši sve {entity_hr}",
        "masovno obriši {entity_hr}",
        "izbriši {entity_hr} po kriteriju",
        "obriši stare {entity_hr}",
        "obriši neaktivne {entity_hr}",
    ],
    "BULK_UPDATE": [
        "ažuriraj sve {entity_hr}",
        "masovno ažuriraj {entity_hr}",
        "promijeni sve {entity_hr}",
        "bulk update {entity_hr}",
    ],
    "DEFAULT_SET": [
        "postavi zadano za {entity_hr}",
        "označi kao zadano {entity_hr}",
        "postavi default {entity_hr}",
    ],
    "THUMBNAIL": [
        "sličica {entity_hr}",
        "preview {entity_hr}",
        "thumb {entity_hr}",
        "mali pregled {entity_hr}",
    ],
    "PROJECTION": [
        "samo ime i id za {entity_hr}",
        "samo naziv {entity_hr}",
        "određena polja {entity_hr}",
        "projekcija {entity_hr}",
    ],
}

# Entity → Croatian noun forms (for augmentation templates)
_ENTITY_HR = {
    "Companies": ["kompanija", "tvrtki", "firmi"],
    "Vehicles": ["vozila", "automobila", "auta"],
    "Persons": ["osoba", "zaposlenika", "korisnika"],
    "Expenses": ["troškova", "troškove", "rashoda"],
    "Trips": ["putovanja", "vožnji"],
    "Cases": ["slučajeva", "šteta", "kvarova"],
    "Equipment": ["opreme", "uređaja"],
    "Partners": ["partnera", "dobavljača"],
    "Teams": ["timova", "ekipa"],
    "OrgUnits": ["organizacijskih jedinica", "odjela"],
    "Roles": ["uloga", "rola"],
    "Tags": ["oznaka", "tagova"],
    "Pools": ["poolova", "bazena vozila"],
    "Tenants": ["tenanta"],
    "Documents": ["dokumenata"],
    "CostCenters": ["troškovnih centara", "mjesta troška"],
    "VehicleTypes": ["tipova vozila"],
    "EquipmentTypes": ["tipova opreme"],
    "CaseTypes": ["tipova slučajeva"],
    "ExpenseTypes": ["tipova troškova"],
    "ExpenseGroups": ["grupa troškova"],
    "DocumentTypes": ["tipova dokumenata"],
    "VehicleContracts": ["ugovora vozila"],
    "VehicleCalendar": ["rezervacija vozila", "kalendara vozila"],
    "EquipmentCalendar": ["kalendara opreme"],
    "MileageReports": ["kilometraže", "prijeđenog puta"],
    "PeriodicActivities": ["periodičnih aktivnosti", "servisa"],
    "SchedulingModels": ["modela rasporeda"],
    "VehicleAssignments": ["dodjela vozila"],
    "TeamMembers": ["članova tima"],
    "PersonOrgUnits": ["organizacijskih jedinica osobe"],
    "PersonPeriodicActivities": ["aktivnosti osobe"],
    "TenantPermissions": ["dozvola tenanta"],
}


def parse_tool_id(tool_id: str):
    """Parse tool_id into (method, entity, suffix)."""
    parts = tool_id.split("_")
    if len(parts) < 2:
        return None, None, None

    method = parts[0].lower()
    if method not in ("get", "post", "put", "patch", "delete"):
        return None, None, None

    entity = parts[1]
    suffix = "_".join(parts[2:]) if len(parts) > 2 else ""

    return method, entity, suffix


def suffix_to_query_type(method: str, suffix: str) -> str:
    """Map a tool_id suffix to a query type string."""
    # Check method-specific overrides first
    key = (method, suffix)
    if key in _METHOD_QUERY_TYPE_OVERRIDE:
        return _METHOD_QUERY_TYPE_OVERRIDE[key]

    # Direct suffix match
    if suffix in _SUFFIX_TO_QUERY_TYPE:
        return _SUFFIX_TO_QUERY_TYPE[suffix]

    # Partial match: check if suffix ends with a known suffix
    for known_suffix, qtype in sorted(_SUFFIX_TO_QUERY_TYPE.items(), key=lambda x: -len(x[0])):
        if not known_suffix:
            continue
        if suffix.endswith(known_suffix) or suffix.lower().endswith(known_suffix.lower()):
            return qtype

    return None


def load_existing_training():
    """Load existing hand-crafted training data."""
    examples = []
    if EXISTING_TRAINING_PATH.exists():
        with open(EXISTING_TRAINING_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def generate_from_tool_docs():
    """Generate training examples from tool_documentation.json."""
    if not TOOL_DOCS_PATH.exists():
        print(f"ERROR: {TOOL_DOCS_PATH} not found")
        sys.exit(1)

    with open(TOOL_DOCS_PATH, 'r', encoding='utf-8') as f:
        tool_docs = json.load(f)

    examples = []
    skipped = 0
    entity_set = set()

    for tool_id, doc in tool_docs.items():
        method, entity, suffix = parse_tool_id(tool_id)
        if not method or not entity:
            skipped += 1
            continue

        query_type = suffix_to_query_type(method, suffix)
        if not query_type:
            skipped += 1
            continue

        entity_set.add(entity)

        # Extract example_queries_hr as training examples
        for example in doc.get("example_queries_hr", []):
            example = example.strip()
            if len(example) >= 3:
                examples.append({"text": example, "query_type": query_type})

    print(f"Extracted {len(examples)} examples from {len(entity_set)} entities ({skipped} tools skipped)")
    return examples, entity_set


def generate_augmentations(entity_set):
    """Generate augmented training examples from templates."""
    augmented = []
    entities_with_hr = set()

    for entity in entity_set:
        hr_forms = _ENTITY_HR.get(entity, [])
        if not hr_forms:
            continue
        entities_with_hr.add(entity)

        for query_type, templates in _AUGMENTATION_TEMPLATES.items():
            for template in templates:
                for hr_form in hr_forms[:2]:  # Use up to 2 Croatian forms per entity
                    text = template.format(entity_hr=hr_form)
                    augmented.append({"text": text, "query_type": query_type})

    print(f"Generated {len(augmented)} augmented examples for {len(entities_with_hr)} entities")
    return augmented


def deduplicate(examples):
    """Deduplicate by normalized text, keeping first occurrence."""
    seen = set()
    unique = []
    for ex in examples:
        # Normalize: lowercase, strip, collapse spaces
        key = re.sub(r'\s+', ' ', ex["text"].lower().strip())
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def main():
    print("=" * 60)
    print("Query Type Training Data Generator")
    print("=" * 60)

    # Step 1: Load existing hand-crafted data
    existing = load_existing_training()
    print(f"\n1. Existing hand-crafted examples: {len(existing)}")

    # Step 2: Generate from tool_documentation.json
    from_docs, entity_set = generate_from_tool_docs()
    print(f"2. From tool_documentation.json: {len(from_docs)}")

    # Step 3: Generate augmentations
    augmented = generate_augmentations(entity_set)
    print(f"3. Template augmentations: {len(augmented)}")

    # Step 4: Merge (existing first = higher priority in dedup)
    all_examples = existing + from_docs + augmented
    print(f"\n   Total before dedup: {len(all_examples)}")

    # Step 5: Deduplicate
    unique = deduplicate(all_examples)
    print(f"   Total after dedup: {len(unique)}")

    # Step 6: Show distribution
    dist = Counter(ex["query_type"] for ex in unique)
    print(f"\n   Distribution ({len(dist)} classes):")
    for qtype, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"     {qtype:20s}: {count:4d}")

    # Step 7: Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for ex in unique:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n   Written to: {OUTPUT_PATH}")
    print(f"   Total examples: {len(unique)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
