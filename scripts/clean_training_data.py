"""
Clean and balance training data for intent classifier.

Steps:
1. Remove mechanical suffix augmentations (brzo, hitno, odmah)
2. Balance distribution (cap at 100 per intent)
3. Add natural Croatian variations for under-represented intents
4. Split into train (80%) and test (20%) sets
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data" / "training"
INPUT_FILE = DATA_DIR / "intent_full.jsonl"
TRAIN_FILE = DATA_DIR / "intent_train.jsonl"
TEST_FILE = DATA_DIR / "intent_test.jsonl"
CLEAN_FILE = DATA_DIR / "intent_full_clean.jsonl"

MAX_PER_INTENT = 100

# Natural Croatian variations for under-represented intents
AUGMENTATIONS = {
    "GET_VEHICLE_DOCUMENTS": [
        {"text": "imam li neke dokumente za vozilo", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "pokazi mi dokumente od auta", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "trebam vidjeti papire vozila", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "dokumenti od mog vozila", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "koji papiri postoje za auto", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "daj mi dokumentaciju vozila", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "vozacka dokumentacija", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "provjeri dokumente za moj auto", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "gdje su papiri od auta", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
        {"text": "pokazi certifikate vozila", "intent": "GET_VEHICLE_DOCUMENTS", "action": "GET", "tool": "get_Vehicles_id_documents"},
    ],
    "GET_VEHICLE_COUNT": [
        {"text": "koliko auta imamo", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "ukupan broj automobila", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "koliko je auta u nasoj floti", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "daj mi broj svih vozila", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "koliko vozila ukupno imamo", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "broj auta u floti", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "koliko automobila ima u sustavu", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "prebrojaj vozila", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "statistika broja vozila", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
        {"text": "kolko ima auta", "intent": "GET_VEHICLE_COUNT", "action": "GET", "tool": "get_Vehicles_Agg"},
    ],
    "DELETE_TRIP": [
        {"text": "zelim obrisati putovanje", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "makni trip", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "ponisti putovanje", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "brisanje putovanja", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "ukloni to putovanje", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "ne treba mi vise taj trip", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "obrisi zadnje putovanje", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "storniraj putovanje", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "zelim ukloniti trip iz sustava", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
        {"text": "ukloni putovanje iz liste", "intent": "DELETE_TRIP", "action": "DELETE", "tool": "delete_Trips_id"},
    ],
    "GET_TENANT_ID": [
        {"text": "koji mi je tenant", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "id moje organizacije", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "pod kojim tenantom sam", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "daj mi tenant identifikator", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "organizacijski id", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "koji je moj tenant u sustavu", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "moj organizacijski identifikator", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "pokazi mi tenant id", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "id tenanta", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
        {"text": "na kojem sam tenantu", "intent": "GET_TENANT_ID", "action": "GET", "tool": None},
    ],
    "GET_LEASING": [
        {"text": "koji mi je lizing", "intent": "GET_LEASING", "action": "GET", "tool": "get_MasterData"},
        {"text": "informacije o leasingu", "intent": "GET_LEASING", "action": "GET", "tool": "get_MasterData"},
        {"text": "tko je lizing davatelj", "intent": "GET_LEASING", "action": "GET", "tool": "get_MasterData"},
        {"text": "detalji o leasingu", "intent": "GET_LEASING", "action": "GET", "tool": "get_MasterData"},
        {"text": "lizing kuca za moj auto", "intent": "GET_LEASING", "action": "GET", "tool": "get_MasterData"},
    ],
    "GET_PLATE": [
        {"text": "koja je moja tablica", "intent": "GET_PLATE", "action": "GET", "tool": "get_MasterData"},
        {"text": "registracijska oznaka auta", "intent": "GET_PLATE", "action": "GET", "tool": "get_MasterData"},
        {"text": "tablice od mog vozila", "intent": "GET_PLATE", "action": "GET", "tool": "get_MasterData"},
        {"text": "reg tablica", "intent": "GET_PLATE", "action": "GET", "tool": "get_MasterData"},
        {"text": "pokazi tablice", "intent": "GET_PLATE", "action": "GET", "tool": "get_MasterData"},
    ],
}


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def remove_suffix_noise(data):
    """Remove mechanical brzo/hitno/odmah suffix augmentations."""
    suffixes = [" brzo", " hitno", " odmah"]

    # Collect all base texts
    base_texts = set()
    for d in data:
        if not any(d["text"].endswith(s) for s in suffixes):
            base_texts.add(d["text"].lower())

    cleaned = []
    added_bases = set()

    for d in data:
        text = d["text"]
        is_suffixed = False
        base = None

        for s in suffixes:
            if text.endswith(s):
                is_suffixed = True
                base = text[:-len(s)].strip()
                break

        if not is_suffixed:
            cleaned.append(d)
        elif base and base.lower() not in base_texts and base.lower() not in added_bases:
            new_entry = d.copy()
            new_entry["text"] = base
            cleaned.append(new_entry)
            added_bases.add(base.lower())

    return cleaned


def balance_distribution(data, max_per_intent):
    """Cap over-represented intents."""
    by_intent = defaultdict(list)
    for d in data:
        by_intent[d["intent"]].append(d)

    balanced = []
    for intent, samples in by_intent.items():
        if len(samples) > max_per_intent:
            # Keep diverse samples - prefer shorter, more distinct ones
            samples_sorted = sorted(samples, key=lambda x: len(x["text"]))
            balanced.extend(samples_sorted[:max_per_intent])
        else:
            balanced.extend(samples)

    return balanced


def add_augmentations(data, augmentations):
    """Add natural variations for under-represented intents."""
    existing = set(d["text"].lower() for d in data)
    count = 0

    for intent, new_samples in augmentations.items():
        for s in new_samples:
            if s["text"].lower() not in existing:
                data.append(s)
                existing.add(s["text"].lower())
                count += 1

    return data, count


def split_train_test(data, test_ratio=0.2):
    """Stratified train/test split."""
    by_intent = defaultdict(list)
    for d in data:
        by_intent[d["intent"]].append(d)

    train = []
    test = []

    for intent, samples in by_intent.items():
        random.shuffle(samples)
        n_test = max(2, int(len(samples) * test_ratio))  # At least 2 test samples
        test.extend(samples[:n_test])
        train.extend(samples[n_test:])

    random.shuffle(train)
    random.shuffle(test)

    return train, test


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("TRAINING DATA CLEANUP")
    print("=" * 60)

    # Load
    data = load_data()
    print(f"\n1. Loaded: {len(data)} samples")

    # Remove suffix noise
    data = remove_suffix_noise(data)
    print(f"2. After suffix cleanup: {len(data)} samples")

    # Balance
    data = balance_distribution(data, MAX_PER_INTENT)
    print(f"3. After balancing (max {MAX_PER_INTENT}): {len(data)} samples")

    # Augment
    data, aug_count = add_augmentations(data, AUGMENTATIONS)
    print(f"4. After augmentation: {len(data)} samples (+{aug_count} new)")

    # Distribution
    counts = Counter(d["intent"] for d in data)
    print(f"\n{'INTENT':30s} {'COUNT':>5s}")
    print("-" * 37)
    for intent, count in counts.most_common():
        print(f"{intent:30s} {count:5d}")
    print(f"{'TOTAL':30s} {len(data):5d}")
    print(f"Min: {min(counts.values())}, Max: {max(counts.values())}, "
          f"Ratio: {max(counts.values())/min(counts.values()):.1f}x")

    # Write clean full dataset
    random.shuffle(data)
    write_jsonl(CLEAN_FILE, data)
    print(f"\n5. Written clean data: {CLEAN_FILE}")

    # Split train/test
    train, test = split_train_test(data, test_ratio=0.2)
    write_jsonl(TRAIN_FILE, train)
    write_jsonl(TEST_FILE, test)

    train_counts = Counter(d["intent"] for d in train)
    test_counts = Counter(d["intent"] for d in test)

    print(f"6. Train: {len(train)} samples, Test: {len(test)} samples")
    print(f"\n{'INTENT':30s} {'TRAIN':>5s} {'TEST':>5s}")
    print("-" * 42)
    for intent in sorted(counts.keys()):
        print(f"{intent:30s} {train_counts.get(intent, 0):5d} {test_counts.get(intent, 0):5d}")

    print("\nDone!")


if __name__ == "__main__":
    main()
