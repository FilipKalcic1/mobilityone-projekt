"""
Deduplicate synonyms_hr between get_X / get_X_id pairs and enrich
action-specific synonyms for delete/post/put tools.

This ensures FAISS embeddings can discriminate between:
- get_Vehicles (list all) vs get_Vehicles_id (get one by ID)
- delete_Expenses_id vs get_Expenses (different action)

Run: python scripts/deduplicate_synonyms.py
"""
import json
import os
import sys

# Ensure UTF-8 output
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

TOOL_DOC_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'tool_documentation.json')

# Words that indicate LIST intent — should be in get_X but NOT in get_X_id
LIST_INDICATORS = [
    'dohvati', 'prikaži', 'pokaži', 'koji su', 'koliko',
    'lista', 'popis', 'sve', 'svi', 'sva', 'pregledaj',
]

# Words that indicate SINGLE ENTITY intent — should be in get_X_id but NOT in get_X
ID_INDICATORS = [
    'po ID', 'specifičn', 'konkret', 'detalj', 'jednog', 'jednu', 'jedno',
    'informacije o', 'podatke o',
]

# Action verbs per HTTP method — for enriching write tool synonyms
ACTION_VERBS = {
    'delete': ['obriši', 'ukloni', 'makni', 'izbriši', 'brisanje'],
    'post': ['dodaj', 'kreiraj', 'novi', 'nova', 'novo', 'unesi', 'napravi'],
    'put': ['ažuriraj', 'promijeni', 'update', 'izmijeni', 'osvježi'],
    'patch': ['djelomično ažuriraj', 'parcijalno promijeni'],
}

# Generic verbs that should be REMOVED from write tools (they belong to GET)
GET_VERBS = ['dohvati', 'prikaži', 'pokaži', 'koji su', 'koliko', 'pregledaj']

# Entity name extraction from tool documentation
def extract_entity_name(doc):
    """Get the Croatian entity name from tool documentation."""
    # Use the first noun-like synonym as entity name
    synonyms = doc.get('synonyms_hr', [])
    for syn in synonyms:
        # Find short entity nouns (1-2 words, no verbs)
        words = syn.split()
        if len(words) <= 2 and not any(v in syn.lower() for v in ['dohvati', 'prikaži', 'pokaži', 'obriši', 'dodaj', 'ažuriraj', 'po id']):
            return syn
    return None


def has_indicator(synonym, indicators):
    """Check if a synonym contains any of the indicator words."""
    syn_lower = synonym.lower()
    return any(ind.lower() in syn_lower for ind in indicators)


def deduplicate_pair(base_doc, id_doc, base_id, id_tool_id):
    """Deduplicate synonyms between a get_X / get_X_id pair."""
    base_syns = list(base_doc.get('synonyms_hr', []))
    id_syns = list(id_doc.get('synonyms_hr', []))

    if not base_syns and not id_syns:
        return base_syns, id_syns, 0

    changes = 0

    # Step 1: Remove LIST indicators from ID tool
    new_id_syns = []
    for syn in id_syns:
        if has_indicator(syn, LIST_INDICATORS) and not has_indicator(syn, ID_INDICATORS):
            changes += 1
        else:
            new_id_syns.append(syn)

    # Step 2: Remove ID indicators from base list tool
    new_base_syns = []
    for syn in base_syns:
        if has_indicator(syn, ID_INDICATORS) and not has_indicator(syn, LIST_INDICATORS):
            changes += 1
        else:
            new_base_syns.append(syn)

    # Step 3: Add list-specific synonyms to base if missing
    entity_name = extract_entity_name(base_doc) or extract_entity_name(id_doc)
    if entity_name:
        list_additions = [
            f"lista {entity_name}",
            f"popis {entity_name}",
            f"sve {entity_name}",
        ]
        for addition in list_additions:
            if addition not in new_base_syns and len(new_base_syns) < 10:
                new_base_syns.append(addition)
                changes += 1

        # Add ID-specific synonyms to _id tool if missing
        id_additions = [
            f"detalji {entity_name}",
            f"jedan {entity_name}",
        ]
        for addition in id_additions:
            if addition not in new_id_syns and len(new_id_syns) < 10:
                new_id_syns.append(addition)
                changes += 1

    # Ensure minimum 5 synonyms each
    while len(new_base_syns) < 5 and base_syns:
        # Re-add from original but with "sve" prefix
        for syn in base_syns:
            modified = f"sve {syn}" if not syn.startswith('sve ') else syn
            if modified not in new_base_syns:
                new_base_syns.append(modified)
                break
        else:
            break

    while len(new_id_syns) < 5 and id_syns:
        for syn in id_syns:
            modified = f"jedan {syn}" if not syn.startswith('jedan ') else syn
            if modified not in new_id_syns:
                new_id_syns.append(modified)
                break
        else:
            break

    return new_base_syns, new_id_syns, changes


def enrich_write_tool(doc, tool_id):
    """Enrich delete/post/put tool synonyms with action-specific verbs."""
    method = tool_id.split('_')[0].lower()
    if method not in ACTION_VERBS:
        return doc.get('synonyms_hr', []), 0

    syns = list(doc.get('synonyms_hr', []))
    changes = 0

    # Remove GET-specific verbs from write tools
    new_syns = []
    for syn in syns:
        if any(gv in syn.lower() for gv in GET_VERBS):
            changes += 1
        else:
            new_syns.append(syn)

    # Add action-specific synonyms if not enough
    entity_name = extract_entity_name(doc)
    if entity_name and method in ACTION_VERBS:
        for verb in ACTION_VERBS[method][:3]:
            action_syn = f"{verb} {entity_name}"
            if action_syn not in new_syns and len(new_syns) < 10:
                new_syns.append(action_syn)
                changes += 1

    # Ensure minimum 5
    if len(new_syns) < 5:
        new_syns.extend(syns[:5 - len(new_syns)])

    return new_syns, changes


def main():
    with open(TOOL_DOC_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)

    total_changes = 0
    pair_count = 0
    write_count = 0

    # Step 1: Deduplicate get_X / get_X_id pairs
    print("=== DEDUPLICATING get_X / get_X_id PAIRS ===\n")
    processed_ids = set()

    for tool_id in sorted(docs.keys()):
        if tool_id.startswith('get_') and tool_id.endswith('_id') and '_id_' not in tool_id:
            base = tool_id[:-3]
            if base in docs and base not in processed_ids:
                base_syns, id_syns, changes = deduplicate_pair(
                    docs[base], docs[tool_id], base, tool_id
                )
                if changes > 0:
                    print(f"  {base}: {docs[base].get('synonyms_hr', [])} -> {base_syns}")
                    print(f"  {tool_id}: {docs[tool_id].get('synonyms_hr', [])} -> {id_syns}")
                    print()
                    docs[base]['synonyms_hr'] = base_syns
                    docs[tool_id]['synonyms_hr'] = id_syns
                    total_changes += changes
                    pair_count += 1
                processed_ids.add(base)
                processed_ids.add(tool_id)

    # Step 2: Enrich write tools
    print("\n=== ENRICHING WRITE TOOL SYNONYMS ===\n")
    for tool_id in sorted(docs.keys()):
        if tool_id.startswith(('delete_', 'post_', 'put_', 'patch_')):
            new_syns, changes = enrich_write_tool(docs[tool_id], tool_id)
            if changes > 0:
                old_syns = docs[tool_id].get('synonyms_hr', [])
                docs[tool_id]['synonyms_hr'] = new_syns
                total_changes += changes
                write_count += 1

    # Step 3: Verify no pair still has >50% overlap
    print("\n=== VERIFICATION ===\n")
    high_overlap = 0
    for tool_id in docs:
        if tool_id.startswith('get_') and tool_id.endswith('_id') and '_id_' not in tool_id:
            base = tool_id[:-3]
            if base in docs:
                base_syns = set(docs[base].get('synonyms_hr', []))
                id_syns = set(docs[tool_id].get('synonyms_hr', []))
                if base_syns and id_syns:
                    overlap = base_syns & id_syns
                    overlap_pct = len(overlap) / max(len(base_syns), len(id_syns))
                    if overlap_pct > 0.5:
                        high_overlap += 1
                        print(f"  WARNING: {base} / {tool_id} still has {overlap_pct:.0%} overlap: {overlap}")

    print(f"\nSummary:")
    print(f"  Pairs deduplicated: {pair_count}")
    print(f"  Write tools enriched: {write_count}")
    print(f"  Total synonym changes: {total_changes}")
    print(f"  High-overlap pairs remaining: {high_overlap}")

    # Save
    with open(TOOL_DOC_PATH, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {TOOL_DOC_PATH}")


if __name__ == '__main__':
    main()
