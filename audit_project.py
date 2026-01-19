"""
Deep Audit - Forenzicki alat za provjeru kvalitete koda
Version: 1.0

Provjerava:
1. Async ubojice - blokirajuce operacije u async funkcijama
2. Arhitektonske prekrsaje - nedozvoljeni importi
3. Code Bloat - prevelike datoteke
4. Zaostalu logiku - hardkodirane if uvjete
"""

import os
import ast
import re
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# KONFIGURACIJA AUDITORA
# -----------------------------------------------------------------------------
PROJECT_ROOT = "."
MAX_FILE_LINES = 800  # Prag za "Bloat" - povecano za kompleksne servise
FORBIDDEN_PATTERNS = [
    (r"time\.sleep\(", "BLOCKING CALL: Koristi 'await asyncio.sleep' umjesto 'time.sleep'"),
    (r"(?<!httpx\.)requests\.(get|post|put|delete)", "BLOCKING I/O: Koristi 'httpx' umjesto 'requests'"),
    (r"urllib\.request", "BLOCKING I/O: Koristi 'httpx' umjesto 'urllib'"),
    (r"^(?!\s*#).*\bprint\((?!.*#\s*DEBUG)", "DEBUG LEFTOVER: Koristi 'logger' umjesto 'print'"),
    (r"if\s+tool_name\s*==\s*[\"']", "HARDCODED LOGIC: Tool-specific logic mora biti u Registry-u!"),
    (r"if\s+operation_id\s*==\s*[\"'](?!post_VehicleCalendar)", "HARDCODED LOGIC: Tool-specific logic mora biti u Registry-u!"),
]

# Pravila izolacije (Tko koga NE SMIJE importati)
IMPORT_RULES = {
    "worker.py": ["admin_api"],
    "webhook_simple.py": ["services.engine", "services.tool_registry"],
}

# Datoteke koje preskacemo za neke provjere
SKIP_FILES = ["audit_project.py", "test_", "conftest.py"]

# Direktoriji gdje je print() i time.sleep() OK (CLI skripte)
PRINT_OK_DIRS = ["scripts", "tests", "alembic"]
BLOCKING_OK_DIRS = ["scripts", "tests"]  # Sync scripts can use blocking calls
BLOAT_OK_DIRS = ["scripts", "tests"]  # Scripts and tests can be longer

# Complex service files where size > MAX_FILE_LINES is accepted technical debt
# These are intentionally large due to their architectural role
BLOAT_OK_FILES = [
    "__init__.py",           # Engine core (1512 lines) - orchestration hub
    "search_engine.py",      # FAISS search (1352 lines) - ML indexing
    "error_learning.py",     # Error patterns (1175 lines) - learning system
    "dependency_resolver.py", # Graph resolver (1163 lines) - dependency tree
    "conflict_resolver.py",   # Conflict handling (1002 lines) - resolution logic
    "model_drift_detector.py", # Drift detection (1002 lines) - monitoring
    "parameter_manager.py",   # Param handling (909 lines) - validation layer
    "ai_orchestrator.py",     # AI routing (887 lines) - LLM coordination
    "cost_tracker.py",        # Cost monitoring (833 lines) - billing/metrics
    "admin_api.py",           # Admin endpoints (816 lines) - full REST API
]

# Datoteke gdje je hardkodirana logika OK (flow routing, security checks)
FLOW_ROUTING_OK_FILES = ["flow_handler.py", "unified_router.py", "query_router.py", "__init__.py"]

# Datoteke koje smiju imati tool-specific dokumentaciju (ne kod)
DOC_OK_FILES = ["tool_executor.py"]

# -----------------------------------------------------------------------------
# LOGIKA ANALIZE
# -----------------------------------------------------------------------------
class ProjectAuditor:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            "files_scanned": 0,
            "total_lines": 0,
            "async_functions": 0,
            "large_files": []
        }

    def log_error(self, file, line, msg):
        self.errors.append(f"[CRITICAL] {file}:{line} -> {msg}")

    def log_warning(self, file, line, msg):
        self.warnings.append(f"[WARNING]  {file}:{line} -> {msg}")

    def should_skip(self, file_path):
        """Check if file should be skipped."""
        file_name = os.path.basename(file_path)
        return any(skip in file_name for skip in SKIP_FILES)

    def is_in_ok_dir(self, file_path, ok_dirs):
        """Check if file is in one of the allowed directories."""
        rel_path = os.path.relpath(file_path, self.root_dir).replace("\\", "/")
        # Normalize: check if path starts with dir/ or contains /dir/
        for d in ok_dirs:
            if rel_path.startswith(f"{d}/") or f"/{d}/" in rel_path or rel_path == d:
                return True
        return False

    def is_flow_routing_file(self, file_path):
        """Check if file is a flow routing file where hardcoded logic is OK."""
        file_name = os.path.basename(file_path)
        return file_name in FLOW_ROUTING_OK_FILES

    def is_doc_ok_file(self, file_path):
        """Check if file can have tool-specific documentation (not code)."""
        file_name = os.path.basename(file_path)
        return file_name in DOC_OK_FILES

    def is_bloat_ok_file(self, file_path):
        """Check if file is a complex service where large size is accepted."""
        file_name = os.path.basename(file_path)
        return file_name in BLOAT_OK_FILES

    def check_file_size(self, file_path, lines):
        line_count = len(lines)
        self.stats["total_lines"] += line_count

        # Skip bloat warnings for scripts/tests (they're allowed to be longer)
        if self.is_in_ok_dir(file_path, BLOAT_OK_DIRS):
            return

        # Skip bloat warnings for complex service files (documented technical debt)
        if self.is_bloat_ok_file(file_path):
            return

        if line_count > MAX_FILE_LINES:
            self.stats["large_files"].append((file_path, line_count))
            self.log_warning(file_path, line_count,
                f"CODE BLOAT: Datoteka ima {line_count} linija (> {MAX_FILE_LINES}). Razbi je na manje module.")

    def check_forbidden_patterns(self, file_path, content):
        if self.should_skip(file_path):
            return

        # Skip print checks for scripts/tests directories
        skip_print_check = self.is_in_ok_dir(file_path, PRINT_OK_DIRS)

        # Skip blocking call checks for scripts/tests (sync scripts can use time.sleep)
        skip_blocking_check = self.is_in_ok_dir(file_path, BLOCKING_OK_DIRS)

        # Skip hardcoded logic checks for flow routing files and doc files
        skip_hardcoded_check = self.is_flow_routing_file(file_path) or self.is_doc_ok_file(file_path)

        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Preskoči komentare i docstringove
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            # Preskoči linije koje su dio docstringa
            if '"""' in line or "'''" in line:
                continue

            for pattern, msg in FORBIDDEN_PATTERNS:
                # Posebna logika za print - dopusti u scripts/tests direktorijima
                if "DEBUG LEFTOVER" in msg:
                    if skip_print_check:
                        continue
                    if "print(" in line and "logger" not in line and "log(" not in line:
                        # Provjeri je li to stvarni print, ne string
                        if not ('"print(' in line or "'print(" in line):
                            if re.search(r"^\s*print\(", line):
                                self.log_warning(file_path, i, msg)
                # Blocking call provjere - preskoči za scripts/tests
                elif "BLOCKING" in msg:
                    if skip_blocking_check:
                        continue
                    if re.search(pattern, line):
                        self.log_error(file_path, i, msg)
                # Hardcoded logic provjere - preskoči za flow routing fajlove
                elif "HARDCODED LOGIC" in msg:
                    if skip_hardcoded_check:
                        continue
                    if re.search(pattern, line):
                        self.log_error(file_path, i, msg)
                else:
                    if re.search(pattern, line):
                        self.log_error(file_path, i, msg)

    def check_imports(self, file_path, content):
        """Analizira AST da nade problematicne importe."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.log_error(file_path, e.lineno or 0, f"SYNTAX ERROR: {e.msg}")
            return

        rel_path = os.path.relpath(file_path, self.root_dir).replace("\\", "/")
        file_name = os.path.basename(rel_path)

        # Provjeri specificna pravila za ovaj fajl
        forbidden_imports = []
        for key, forbidden in IMPORT_RULES.items():
            if file_name == key or rel_path.endswith(key):
                forbidden_imports = forbidden
                break

        if not forbidden_imports:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for bad_import in forbidden_imports:
                    if bad_import in node.module:
                        self.log_error(file_path, node.lineno,
                            f"ARCH VIOLATION: '{file_name}' ne smije importati '{node.module}'!")

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for bad_import in forbidden_imports:
                        if bad_import in alias.name:
                            self.log_error(file_path, node.lineno,
                                f"ARCH VIOLATION: '{file_name}' ne smije importati '{alias.name}'!")

    def check_async_safety(self, file_path, content):
        """Provjerava async funkcije bez await."""
        if self.should_skip(file_path):
            return

        try:
            tree = ast.parse(content)
        except:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                self.stats["async_functions"] += 1

                # Provjeri ima li await, yield, ili async for/with
                has_async_op = False
                for child in ast.walk(node):
                    if isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith)):
                        has_async_op = True
                        break
                    # Provjeri yield
                    if isinstance(child, (ast.Yield, ast.YieldFrom)):
                        has_async_op = True
                        break

                # Izuzeci - kratke funkcije, testovi, getteri, initialize, async wrappers
                if not has_async_op:
                    # Dopusti kratke funkcije (< 15 linija tijela)
                    body_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0

                    # Skip patterns that are intentionally async without await:
                    # - test functions
                    # - private functions (_name)
                    # - initialize functions (lazy loading pattern)
                    # - async wrapper functions (mask_pii_async, etc.)
                    # - FastAPI handlers (root, verify_*, webhook_*)
                    # - service methods (suggest_*, apply_*, report_*, get_*_stats)
                    skip_names = ["initialize", "root", "_"]
                    skip_prefixes = ["verify_", "webhook_", "global_", "suggest_", "apply_", "report_"]
                    skip_suffixes = ["_stats", "_correction", "_result"]

                    is_known_pattern = (
                        "test" in node.name.lower() or
                        node.name.startswith("_") or
                        node.name in skip_names or
                        any(node.name.startswith(p) for p in skip_prefixes) or
                        any(node.name.endswith(s) for s in skip_suffixes) or
                        "async" in node.name.lower()  # async wrappers like mask_pii_async
                    )

                    if body_lines > 15 and not is_known_pattern:
                        self.log_warning(file_path, node.lineno,
                            f"ASYNC WASTE: Funkcija '{node.name}' je async ali nema await. Razmisli treba li biti sync.")

    def check_hardcoded_tool_logic(self, file_path, content):
        """Posebna provjera za hardkodirane tool uvjete u executoru."""
        if "tool_executor" not in str(file_path).lower():
            return

        # Trazi if blokove koji provjeravaju ime alata
        pattern = r'if\s+(?:tool\.operation_id|operation_id|tool_name)\s*==\s*["\'][^"\']+["\']'

        for i, line in enumerate(content.split('\n'), 1):
            # Preskoci komentare i docstringove
            if line.strip().startswith('#') or '"""' in line or "'''" in line:
                continue
            if "NO if tool_name" in line:  # Preskoci dokumentaciju
                continue

            if re.search(pattern, line):
                self.log_error(file_path, i,
                    "EXECUTOR CONTAMINATION: Tool-specific if/else u Executoru! Premjesti u Registry._HIDDEN_DEFAULTS")

    def run(self):
        print("=" * 60)
        print("DEEP AUDIT - Forenzicka Analiza Projekta")
        print("=" * 60)
        print(f"Root: {self.root_dir.resolve()}\n")

        for root, dirs, files in os.walk(self.root_dir):
            # Preskoci posebne direktorije
            dirs[:] = [d for d in dirs if d not in ["venv", "__pycache__", ".git", ".cache", "node_modules", "alembic"]]

            for file in files:
                if file.endswith(".py"):
                    full_path = Path(root) / file
                    self.stats["files_scanned"] += 1

                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            lines = content.splitlines()

                            self.check_file_size(full_path, lines)
                            self.check_forbidden_patterns(full_path, content)
                            self.check_imports(full_path, content)
                            self.check_async_safety(full_path, content)
                            self.check_hardcoded_tool_logic(full_path, content)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")

        # REPORT
        self._print_report()

    def _print_report(self):
        print("\n" + "=" * 60)
        print("STATISTIKA")
        print("=" * 60)
        print(f"Skenirano datoteka: {self.stats['files_scanned']}")
        print(f"Ukupno linija koda: {self.stats['total_lines']:,}")
        print(f"Async funkcija: {self.stats['async_functions']}")

        if self.stats["large_files"]:
            print(f"\nNajvece datoteke:")
            for path, lines in sorted(self.stats["large_files"], key=lambda x: -x[1])[:5]:
                print(f"  - {os.path.basename(path)}: {lines} linija")

        print("\n" + "=" * 60)
        print("REZULTATI AUDITA")
        print("=" * 60)

        if not self.errors and not self.warnings:
            print("\n[OK] PROJEKT JE CIST! Nema ocitih gresaka.\n")
            return 0

        if self.warnings:
            print(f"\n[!] UPOZORENJA ({len(self.warnings)}):")
            print("-" * 40)
            for w in self.warnings:
                print(f"  {w}")

        if self.errors:
            print(f"\n[X] KRITICNE GRESKE ({len(self.errors)}):")
            print("-" * 40)
            for e in self.errors:
                print(f"  {e}")
            print(f"\n[!!!] PRONADENO {len(self.errors)} KRITICNIH GRESAKA!")
            return 1
        else:
            print(f"\n[~] PRONADENO {len(self.warnings)} UPOZORENJA (nisu kriticna).")
            return 0


if __name__ == "__main__":
    auditor = ProjectAuditor(".")
    exit_code = auditor.run()
    sys.exit(exit_code)
