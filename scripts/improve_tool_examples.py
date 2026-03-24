"""
Improve Tool Documentation Example Queries

Automatically enhances example queries to be more distinctive
based on tool suffix patterns.

Goal: Improve Top-1 accuracy from 67% to 80%+
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Entity translations (Croatian)
ENTITY_TRANSLATIONS = {
    "Companies": "kompanija",
    "Vehicles": "vozila",
    "Persons": "osoba",
    "Expenses": "troškova",
    "Trips": "putovanja",
    "Cases": "slučajeva",
    "Equipment": "opreme",
    "Partners": "partnera",
    "Teams": "timova",
    "OrgUnits": "organizacijskih jedinica",
    "CostCenters": "troškovnih centara",
    "DocumentTypes": "tipova dokumenata",
    "VehicleCalendar": "rezervacija",
    "MasterData": "matičnih podataka",
    "Lookup": "šifrarnika",
    "Metadata": "metapodataka",
    "PersonTypes": "tipova osoba",
    "PersonActivityTypes": "tipova aktivnosti",
    "TenantPermissions": "dopuštenja",
    "Tenants": "tenanta",
    "TeamMembers": "članova tima",
    "PersonOrgUnits": "pripadnosti osobama",
    "PersonPeriodicActivities": "periodičnih aktivnosti",
}

# Suffix-specific query templates
SUFFIX_TEMPLATES = {
    # Base entity (LIST)
    "": [
        "Dohvati sve {entity}",
        "Prikaži listu svih {entity}",
        "Popis svih {entity} u sustavu",
    ],
    # Single entity by ID
    "_id": [
        "Dohvati {entity_single} po ID-u",
        "Prikaži detalje {entity_single} s ID-em {id}",
        "Podaci o jednoj {entity_single}",
    ],
    # Documents
    "_id_documents": [
        "Dohvati dokumente za {entity_single}",
        "Prikaži priloge {entity_single}",
        "Dokumenti vezani uz {entity_single}",
    ],
    "_id_documents_documentId": [
        "Dohvati specifični dokument za {entity_single}",
        "Prikaži pojedini dokument {entity_single}",
    ],
    "_documents": [
        "Dohvati sve dokumente {entity}",
        "Lista dokumenata za {entity}",
    ],
    # Metadata
    "_id_metadata": [
        "Dohvati metapodatke za {entity_single}",
        "Prikaži strukturu {entity_single}",
        "Metadata za {entity_single}",
    ],
    "_metadata": [
        "Dohvati metapodatke {entity}",
        "Struktura podataka {entity}",
    ],
    # Aggregations
    "_Agg": [
        "Statistika {entity}",
        "Agregacija {entity}",
        "Ukupno {entity}",
        "Koliko ima {entity}",
    ],
    "_GroupBy": [
        "Grupiraj {entity} po kriteriju",
        "Grupirana statistika {entity}",
    ],
    # Special operations
    "_ProjectTo": [
        "Dohvati {entity} s odabranim poljima",
        "Projekcija {entity}",
        "Samo određeni stupci za {entity}",
    ],
    "_tree": [
        "Hijerarhija {entity}",
        "Stablo {entity}",
        "Parent-child struktura {entity}",
    ],
    "_thumb": [
        "Sličica dokumenta za {entity_single}",
        "Preview dokumenta {entity_single}",
        "Thumbnail {entity_single}",
    ],
    "_SetAsDefault": [
        "Postavi zadani dokument za {entity_single}",
        "Označi kao glavni dokument {entity_single}",
    ],
    "_DeleteByCriteria": [
        "Masovno obriši {entity}",
        "Obriši {entity} po kriterijima",
        "Bulk delete {entity}",
    ],
    "_multipatch": [
        "Masovno ažuriraj {entity}",
        "Bulk update {entity}",
        "Ažuriraj više {entity} odjednom",
    ],
}

# HTTP method specific prefixes
METHOD_PREFIXES = {
    "get": "",
    "post": "Kreiraj ",
    "put": "Ažuriraj ",
    "patch": "Djelomično ažuriraj ",
    "delete": "Obriši ",
}


def get_entity_from_tool(tool_id: str) -> Tuple[str, str]:
    """Extract entity name from tool ID."""
    parts = tool_id.split("_")
    if len(parts) < 2:
        return "", ""

    method = parts[0].lower()
    entity = parts[1]

    return method, entity


def get_suffix(tool_id: str) -> str:
    """Extract suffix from tool ID."""
    parts = tool_id.split("_")
    if len(parts) <= 2:
        return ""

    # Reconstruct suffix
    suffix_parts = parts[2:]

    # Check known suffixes
    for suffix in sorted(SUFFIX_TEMPLATES.keys(), key=len, reverse=True):
        if suffix and tool_id.endswith(suffix.replace("_", "")):
            return suffix

    # Try to match partial
    suffix = "_" + "_".join(suffix_parts)

    # Simplify to known patterns
    if "_documents_documentId_thumb" in suffix:
        return "_thumb"
    if "_documents_documentId_SetAsDefault" in suffix:
        return "_SetAsDefault"
    if "_documents_documentId" in suffix:
        return "_id_documents_documentId"
    if "_documents" in suffix:
        return "_id_documents"
    if "_metadata" in suffix:
        return "_id_metadata"
    if "Agg" in suffix:
        return "_Agg"
    if "GroupBy" in suffix:
        return "_GroupBy"
    if "ProjectTo" in suffix:
        return "_ProjectTo"
    if "tree" in suffix.lower():
        return "_tree"
    if "DeleteByCriteria" in suffix:
        return "_DeleteByCriteria"
    if "multipatch" in suffix.lower():
        return "_multipatch"
    if "_id" in suffix and len(suffix_parts) == 1:
        return "_id"

    return ""


def generate_examples(tool_id: str, method: str, entity: str, suffix: str) -> List[str]:
    """Generate improved example queries for a tool."""
    # Get entity translation
    entity_hr = ENTITY_TRANSLATIONS.get(entity, entity.lower())

    # Handle singular form
    entity_single = entity_hr
    if entity_single.endswith("a"):
        entity_single = entity_single[:-1] + "e"  # kompanija -> kompanije
    elif entity_single.endswith("i"):
        entity_single = entity_single[:-1] + "a"  # vozila -> vozila

    # Get templates for this suffix
    templates = SUFFIX_TEMPLATES.get(suffix, SUFFIX_TEMPLATES.get("", []))

    # Get method prefix
    method_prefix = METHOD_PREFIXES.get(method, "")

    examples = []
    for template in templates[:3]:  # Max 3 examples
        example = template.format(
            entity=entity_hr,
            entity_single=entity_single,
            id="123"
        )

        # Add method prefix for non-GET operations
        if method != "get" and not example.lower().startswith(method_prefix.lower().strip()):
            example = method_prefix + example.lower()

        examples.append(example)

    return examples


def improve_documentation():
    """Main function to improve tool documentation."""
    doc_path = Path("config/tool_documentation.json")

    with open(doc_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    improved_count = 0

    for tool_id, doc in docs.items():
        method, entity = get_entity_from_tool(tool_id)
        suffix = get_suffix(tool_id)

        if not entity:
            continue

        # Generate new examples
        new_examples = generate_examples(tool_id, method, entity, suffix)

        if new_examples:
            # Merge with existing, keeping unique
            existing = doc.get("example_queries_hr", [])
            merged = list(dict.fromkeys(new_examples + existing))[:5]  # Max 5

            if merged != existing:
                doc["example_queries_hr"] = merged
                improved_count += 1

    # Save updated documentation
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Improved {improved_count} tools")
    return improved_count


if __name__ == "__main__":
    improve_documentation()
