#!/usr/bin/env python3
"""
Regenerate Embeddings Cache Script

Run this after updating the embedding_engine.py to regenerate
all tool embeddings with the new v3.1 enhanced text + synonyms.

Usage:
    python scripts/regenerate_embeddings.py

This will:
1. Delete the existing embedding cache
2. Print instructions for restarting the application
"""

import os
from pathlib import Path

CACHE_PATHS = [
    ".cache/tool_embeddings.json",
    "nova-verzija/.cache/tool_embeddings.json",
]

def main():
    print("="*60)
    print("EMBEDDING CACHE REGENERATION")
    print("="*60)

    deleted = []
    not_found = []

    for cache_path in CACHE_PATHS:
        full_path = Path(cache_path)
        if full_path.exists():
            try:
                size_kb = full_path.stat().st_size / 1024
                os.remove(full_path)
                deleted.append(f"{cache_path} ({size_kb:.1f} KB)")
                print(f"[OK] Deleted: {cache_path}")
            except Exception as e:
                print(f"[FAIL] Could not delete {cache_path}: {e}")
        else:
            not_found.append(cache_path)

    print("\n" + "-"*60)

    if deleted:
        print(f"\nDeleted {len(deleted)} cache file(s):")
        for d in deleted:
            print(f"  - {d}")

    if not_found:
        print(f"\nNot found (already clean):")
        for n in not_found:
            print(f"  - {n}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Restart the application (main.py or worker.py)
2. The system will automatically regenerate embeddings
   with the new v3.1 enhanced text + synonyms
3. This process may take a few minutes for 950 tools

Example:
    docker-compose restart
    # or
    python main.py
""")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
