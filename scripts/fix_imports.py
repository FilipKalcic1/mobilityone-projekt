"""One-shot script to remove verified unused imports."""
import re
import os

REMOVALS = {
    "services/ambiguity_detector.py": ["re", "Tuple"],
    "services/api_capabilities.py": ["List", "Set"],
    "services/api_gateway.py": ["json"],
    "services/cache_service.py": ["List"],
    "services/dependency_resolver.py": ["field"],
    "services/error_learning.py": ["os", "timedelta", "defaultdict"],
    "services/error_parser.py": ["Dict", "Optional"],
    "services/error_translator.py": ["Dict", "Any", "Tuple"],
    "services/faiss_vector_store.py": ["os", "Tuple", "Set"],
    "services/feedback_learning_service.py": ["asyncio", "timedelta", "Set", "func", "update"],
    "services/flow_phrases.py": ["Set"],
    "services/parameter_manager.py": ["ParameterDefinition"],
    "services/quality_tracker.py": ["field"],
    "services/response_extractor.py": ["List"],
    "services/response_formatter.py": ["Union"],
    "services/sanitizer.py": ["List", "Union"],
    "services/tenant_service.py": ["List"],
    "services/tool_evaluator.py": ["List", "timedelta", "defaultdict"],
    "services/user_service.py": ["or_", "TenantService"],
    "services/engine/flow_handler.py": ["ConfirmationDialog"],
    "services/engine/hallucination_handler.py": ["List"],
    "services/engine/user_handler.py": ["Tuple"],
    "services/registry/cache_manager.py": ["Optional"],
    "services/registry/embedding_engine.py": ["DependencySource"],
    "services/registry/embedding_evaluator.py": ["Tuple"],
    "main.py": ["aioredis", "UserMapping", "Conversation", "Message", "ToolExecution", "AuditLog"],
    "admin_api.py": ["re"],
}

def process_file(filepath, names_to_remove):
    with open(filepath) as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    removed = 0
    names_set = set(names_to_remove)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip lines without import or without any target name
        if 'import' not in stripped or not any(n in stripped for n in names_set):
            new_lines.append(line)
            i += 1
            continue

        # Handle multi-line import (with parentheses)
        if '(' in stripped and ')' not in stripped:
            # Gather all lines of the import
            block = [line]
            j = i + 1
            while j < len(lines) and ')' not in lines[j]:
                block.append(lines[j])
                j += 1
            if j < len(lines):
                block.append(lines[j])

            full = ''.join(block)
            # Extract the "from X import" prefix
            m = re.match(r'(\s*from\s+\S+\s+import\s*)\(', full)
            if not m:
                new_lines.extend(block)
                i = j + 1
                continue

            prefix = m.group(1).strip()
            indent = len(line) - len(line.lstrip())
            ind = ' ' * indent

            # Extract all names between parens
            paren_content = full[full.index('(') + 1:full.rindex(')')]
            items = [x.strip().rstrip(',') for x in re.split(r'[,\n]', paren_content) if x.strip().rstrip(',')]
            # Filter out comments
            items = [x for x in items if not x.startswith('#')]

            kept = []
            for item in items:
                # Get the imported name (handle "X as Y")
                name = item.split(' as ')[-1].strip() if ' as ' in item else item.strip()
                if name not in names_set:
                    kept.append(item)
                else:
                    removed += 1

            if not kept:
                # Delete entire import
                i = j + 1
                continue
            elif len(kept) <= 3:
                new_lines.append(f"{ind}{prefix}{', '.join(kept)}\n")
            else:
                new_lines.append(f"{ind}{prefix}(\n")
                for k, item in enumerate(kept):
                    comma = ',' if k < len(kept) - 1 else ''
                    new_lines.append(f"{ind}    {item}{comma}\n")
                new_lines.append(f"{ind})\n")

            i = j + 1
            continue

        # Single-line import
        if stripped.startswith('import ') and 'from' not in stripped:
            # "import X" or "import X as Y"
            parts = stripped.replace('import ', '').split(',')
            kept = []
            for part in parts:
                name = part.strip().split(' as ')[-1].strip()
                if name in names_set:
                    removed += 1
                else:
                    kept.append(part.strip())
            if not kept:
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + 'import ' + ', '.join(kept) + '\n')
            i += 1
            continue

        # "from X import Y, Z" (single line)
        m = re.match(r'(\s*from\s+\S+\s+import\s+)(.*)', line.rstrip())
        if m:
            prefix = m.group(1)
            imports_str = m.group(2).strip()
            items = [x.strip() for x in imports_str.split(',') if x.strip()]
            kept = []
            for item in items:
                name = item.split(' as ')[-1].strip()
                if name in names_set:
                    removed += 1
                else:
                    kept.append(item)
            if not kept:
                i += 1
                continue
            new_lines.append(f"{prefix}{', '.join(kept)}\n")
            i += 1
            continue

        new_lines.append(line)
        i += 1

    if removed > 0:
        # Clean double blank lines
        final = []
        prev_blank = False
        for line in new_lines:
            is_blank = line.strip() == ''
            if is_blank and prev_blank:
                continue
            final.append(line)
            prev_blank = is_blank

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(final)

    return removed


total = 0
for rel_path, names in sorted(REMOVALS.items()):
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
    if not os.path.exists(filepath):
        # Try /app/ prefix for Docker
        filepath = f"/app/{rel_path}"
    if not os.path.exists(filepath):
        print(f"  SKIP: {rel_path}")
        continue

    count = process_file(filepath, names)
    if count:
        print(f"  {rel_path}: removed {count}")
        total += count

print(f"\nDone: {total} unused imports removed")
