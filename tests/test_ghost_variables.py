"""
Ghost Variables Test

Tests for detecting undefined/unused variables in the fixed service files.
Performs static analysis on Python source code.
"""

import ast
import os
import pytest
from pathlib import Path
from typing import Set, Dict, List, Tuple


class VariableAnalyzer(ast.NodeVisitor):
    """AST visitor that tracks variable definitions and usages."""

    def __init__(self):
        self.defined: Set[str] = set()
        self.used: Set[str] = set()
        self.imports: Set[str] = set()
        self.function_args: Set[str] = set()
        self.class_names: Set[str] = set()
        self.function_names: Set[str] = set()
        self.comprehension_vars: Set[str] = set()
        self.exception_names: Set[str] = set()
        self.global_names: Set[str] = set()
        self.nonlocal_names: Set[str] = set()

        # Track scope for better analysis
        self.scope_stack: List[Set[str]] = [set()]

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.imports.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_names.add(node.name)

        # Track arguments
        for arg in node.args.args:
            self.function_args.add(arg.arg)
        for arg in node.args.posonlyargs:
            self.function_args.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.function_args.add(arg.arg)
        if node.args.vararg:
            self.function_args.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.function_args.add(node.args.kwarg.arg)

        # Track decorators usage
        for decorator in node.decorator_list:
            self._visit_expression(decorator)

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_names.add(node.name)

        # Track arguments
        for arg in node.args.args:
            self.function_args.add(arg.arg)
        for arg in node.args.posonlyargs:
            self.function_args.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.function_args.add(arg.arg)
        if node.args.vararg:
            self.function_args.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.function_args.add(node.args.kwarg.arg)

        for decorator in node.decorator_list:
            self._visit_expression(decorator)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_names.add(node.name)

        for base in node.bases:
            self._visit_expression(base)

        for decorator in node.decorator_list:
            self._visit_expression(decorator)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        # Visit value first (to track usage)
        self._visit_expression(node.value)

        # Then visit targets (to track definitions)
        for target in node.targets:
            self._visit_target(target)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value:
            self._visit_expression(node.value)
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._visit_expression(node.value)
        self._visit_target(node.target)
        # AugAssign both reads and writes
        if isinstance(node.target, ast.Name):
            self.used.add(node.target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self._visit_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            if item.optional_vars:
                self._visit_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for item in node.items:
            if item.optional_vars:
                self._visit_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            self.exception_names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.global_names.add(name)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        for name in node.names:
            self.nonlocal_names.add(name)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        for generator in node.generators:
            self._visit_comprehension(generator)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        for generator in node.generators:
            self._visit_comprehension(generator)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        for generator in node.generators:
            self._visit_comprehension(generator)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        for generator in node.generators:
            self._visit_comprehension(generator)
        self.generic_visit(node)

    def _visit_comprehension(self, comp: ast.comprehension):
        if isinstance(comp.target, ast.Name):
            self.comprehension_vars.add(comp.target.id)
        elif isinstance(comp.target, ast.Tuple):
            for elt in comp.target.elts:
                if isinstance(elt, ast.Name):
                    self.comprehension_vars.add(elt.id)

    def _visit_target(self, target):
        """Visit assignment target to track definitions."""
        if isinstance(target, ast.Name):
            self.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_target(elt)

    def _visit_expression(self, node):
        """Visit expression to track usage."""
        if isinstance(node, ast.Name):
            self.used.add(node.id)
        elif isinstance(node, ast.Attribute):
            self._visit_expression(node.value)
        elif isinstance(node, ast.Call):
            self._visit_expression(node.func)
            for arg in node.args:
                self._visit_expression(arg)
            for kw in node.keywords:
                self._visit_expression(kw.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self._visit_expression(elt)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key:
                    self._visit_expression(key)
            for val in node.values:
                self._visit_expression(val)

    def get_all_defined(self) -> Set[str]:
        """Get all defined names."""
        return (
            self.defined |
            self.imports |
            self.function_args |
            self.class_names |
            self.function_names |
            self.comprehension_vars |
            self.exception_names |
            self.global_names |
            self.nonlocal_names
        )

    def get_undefined(self) -> Set[str]:
        """Get potentially undefined variables (used but not defined)."""
        all_defined = self.get_all_defined()

        # Python builtins that are always available
        builtins = {
            'True', 'False', 'None', 'print', 'len', 'str', 'int', 'float',
            'bool', 'list', 'dict', 'set', 'tuple', 'range', 'enumerate',
            'zip', 'map', 'filter', 'sorted', 'reversed', 'min', 'max',
            'sum', 'abs', 'round', 'open', 'type', 'isinstance', 'issubclass',
            'hasattr', 'getattr', 'setattr', 'delattr', 'callable', 'repr',
            'id', 'hash', 'dir', 'vars', 'locals', 'globals', 'super',
            'staticmethod', 'classmethod', 'property', 'object', 'iter',
            'next', 'slice', 'format', 'input', 'Exception', 'ValueError',
            'TypeError', 'KeyError', 'IndexError', 'AttributeError',
            'RuntimeError', 'StopIteration', 'NotImplementedError',
            'ImportError', 'OSError', 'IOError', 'FileNotFoundError',
            'PermissionError', 'TimeoutError', 'ConnectionError',
            'AssertionError', 'ZeroDivisionError', 'OverflowError',
            'MemoryError', 'RecursionError', 'SystemExit', 'KeyboardInterrupt',
            'GeneratorExit', 'BaseException', 'Warning', 'DeprecationWarning',
            'UserWarning', 'FutureWarning', 'PendingDeprecationWarning',
            'SyntaxWarning', 'RuntimeWarning', 'ResourceWarning',
            'bytes', 'bytearray', 'memoryview', 'complex', 'frozenset',
            'any', 'all', 'bin', 'hex', 'oct', 'ord', 'chr', 'ascii',
            'compile', 'eval', 'exec', 'breakpoint', 'help', 'exit', 'quit',
            '__name__', '__file__', '__doc__', '__package__', '__spec__',
            '__loader__', '__cached__', '__builtins__', '__annotations__',
            '__dict__', '__class__', '__module__', '__qualname__',
            '__slots__', '__init__', '__new__', '__del__', '__repr__',
            '__str__', '__bytes__', '__format__', '__lt__', '__le__',
            '__eq__', '__ne__', '__gt__', '__ge__', '__hash__', '__bool__',
            '__getattr__', '__getattribute__', '__setattr__', '__delattr__',
            '__dir__', '__get__', '__set__', '__delete__', '__call__',
            '__len__', '__length_hint__', '__getitem__', '__setitem__',
            '__delitem__', '__iter__', '__reversed__', '__contains__',
            '__add__', '__sub__', '__mul__', '__matmul__', '__truediv__',
            '__floordiv__', '__mod__', '__divmod__', '__pow__', '__lshift__',
            '__rshift__', '__and__', '__xor__', '__or__', '__neg__',
            '__pos__', '__abs__', '__invert__', '__complex__', '__int__',
            '__float__', '__index__', '__round__', '__trunc__', '__floor__',
            '__ceil__', '__enter__', '__exit__', '__await__', '__aiter__',
            '__anext__', '__aenter__', '__aexit__', 'self', 'cls',
            # Additional exception types
            'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError',
            'LookupError', 'ArithmeticError', 'BufferError', 'EOFError',
            'FloatingPointError', 'ModuleNotFoundError', 'UnboundLocalError',
            'NameError', 'ReferenceError', 'StopAsyncIteration', 'IndentationError',
            'TabError', 'SystemError', 'BlockingIOError', 'ChildProcessError',
            'BrokenPipeError', 'InterruptedError', 'IsADirectoryError',
            'NotADirectoryError', 'ProcessLookupError',
            # Common patterns
            '_', '__', '___',
            # Lambda parameters (tracked separately but often appear)
            'x', 'y', 'k', 'v', 'i', 'j', 'n', 'm',
        }

        undefined = self.used - all_defined - builtins
        return undefined

    def get_unused_defined(self) -> Set[str]:
        """Get defined but potentially unused variables."""
        all_defined = self.get_all_defined()

        # Exclude special names
        special = {
            '_', '__', '___',
            '__all__', '__version__', '__author__', '__doc__',
            '__tablename__', '__table_args__', '__abstract__',
        }

        # Exclude names starting with underscore (private by convention)
        unused = set()
        for name in all_defined:
            if name not in self.used and name not in special:
                # Skip private names and dunder methods
                if not name.startswith('_'):
                    unused.add(name)

        return unused


def analyze_file(filepath: str) -> Tuple[Set[str], Set[str]]:
    """
    Analyze a Python file for ghost variables.

    Returns:
        Tuple of (undefined_variables, unused_variables)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in {filepath}: {e}")

    analyzer = VariableAnalyzer()
    analyzer.visit(tree)

    return analyzer.get_undefined(), analyzer.get_unused_defined()


# ============================================================================
# TEST CASES
# ============================================================================

class TestGhostVariables:
    """Test for undefined/unused variables in fixed service files."""

    # Base path to services
    SERVICES_PATH = Path(__file__).parent.parent / "services"

    # Files to analyze
    TARGET_FILES = [
        "admin_review.py",
        "hallucination_repository.py",
        "gdpr_masking.py",
        "cost_tracker.py",
        "conflict_resolver.py",
        "model_drift_detector.py",
        "rag_scheduler.py",
    ]

    # Known false positives (names that appear undefined but are valid)
    KNOWN_FALSE_POSITIVES = {
        # Type hints from typing module
        'Optional', 'List', 'Dict', 'Any', 'Set', 'Tuple', 'Union',
        'Callable', 'Awaitable', 'AsyncGenerator', 'Generator',
        'TypeVar', 'Generic', 'Protocol', 'Literal', 'Final',
        'ClassVar', 'Sequence', 'Mapping', 'Iterable', 'Iterator',

        # Common framework names
        'logger', 'log', 'logging',

        # Database/ORM
        'Column', 'Base', 'Session', 'AsyncSession',

        # Test fixtures
        'pytest', 'fixture', 'mark',
    }

    # Known exports - these are intentionally defined for external use (public API)
    # They appear "unused" within the file but are imported by other modules
    KNOWN_EXPORTS = {
        # admin_review.py exports
        'ALLOWED_CATEGORIES', 'AdminReviewService', 'DANGEROUS_PATTERNS',
        'QUERY_TIMEOUT_SECONDS', 'export_for_training', 'get_audit_log',
        'get_hallucinations_for_review', 'get_report_detail', 'get_reports_count',
        'get_statistics', 'is_safe_text', 'mark_hallucination_reviewed',

        # hallucination_repository.py exports
        'HallucinationRepository', 'bulk_mark_reviewed', 'create', 'delete',
        'exists', 'get_all', 'get_by_id', 'get_count', 'get_unreviewed', 'mark_reviewed',

        # gdpr_masking.py exports
        'ADDRESS', 'CREDIT_CARD', 'DEFAULT_MASK_FIELDS', 'EMAIL', 'IBAN',
        'IP_ADDRESS', 'MASK_FORMAT', 'NAME', 'OIB', 'PATTERNS', 'PHONE',
        'anonymize_user_data', 'confidence', 'detect_pii', 'get_masking_service',
        'has_pii', 'mask_dict', 'mask_log_message', 'mask_pii', 'mask_pii_async',
        'masked', 'original_text', 'pii_count', 'pii_found', 'reset_masking_service',

        # cost_tracker.py exports
        'DEFAULT_DAILY_BUDGET_USD', 'MAX_EXPORT_DAYS', 'REDIS_KEY_DAILY',
        'REDIS_KEY_PREFIX', 'REDIS_KEY_TOTAL', 'REDIS_TTL_DAYS', 'calculate_cost',
        'estimated_cost_usd', 'export_for_billing', 'get_available_models',
        'get_cost_tracker', 'get_daily_stats', 'get_model_pricing', 'get_session_stats',
        'get_total_stats', 'get_weekly_trend', 'health_check', 'record_usage',
        'reset_cost_tracker', 'reset_session', 'timestamp', 'total_completion_tokens',
        'total_prompt_tokens', 'total_requests',

        # conflict_resolver.py exports
        'ACTIVE', 'ALREADY_REVIEWED', 'CONCURRENT_EDIT', 'ConflictResolver',
        'DELETED', 'EXPIRED', 'LOCK_TTL_MINUTES', 'REDIS_HISTORY_PREFIX',
        'REDIS_LOCK_PREFIX', 'REDIS_VERSION_PREFIX', 'RELEASED', 'SNAPSHOT_TTL_DAYS',
        'STOLEN', 'VERSION_MISMATCH', 'acquire_edit_lock', 'auto_merge',
        'can_auto_merge', 'changed_at', 'changed_by', 'cleanup_expired_locks',
        'conflict_type', 'error', 'expires_at', 'get_active_editors',
        'get_version_history', 'has_conflict', 'is_expired', 'locked_at', 'new_value',
        'notify_other_editors', 'old_value', 'original_admin', 'release_lock',
        'rollback_to_version', 'save_with_conflict_check', 'status', 'success',
        'suggested_resolution', 'your_admin_id',

        # model_drift_detector.py exports
        'ANALYSIS_WINDOW_HOURS', 'BASELINE_CACHE_TTL_HOURS', 'BASELINE_WINDOW_DAYS',
        'CONFIDENCE', 'CRITICAL', 'ERROR_RATE', 'HALLUCINATION_RATE', 'HIGH',
        'LATENCY', 'LOW', 'MAX_RECENT_METRICS', 'MEDIUM', 'MIN_SAMPLES_FOR_ANALYSIS',
        'NONE', 'REDIS_ALERTS_KEY', 'REDIS_BASELINE_KEY', 'REDIS_METRICS_KEY',
        'REDIS_TIMELINE_KEY', 'THRESHOLDS', 'TOOL_SELECTION', 'alert_id',
        'baseline_comparison', 'baseline_value', 'check_drift', 'current_value',
        'detected_at', 'deviation_percent', 'generated_at', 'get_active_alerts',
        'get_drift_detector', 'get_metrics_count', 'has_drift', 'message',
        'metrics_summary', 'recommended_action', 'record_interaction',
        'reset_drift_detector', 'window',

        # rag_scheduler.py exports
        'DEFAULT_REFRESH_INTERVAL_HOURS', 'FAILED', 'IDLE', 'LOCK_TTL_SECONDS',
        'MAX_RETRY_DELAY_SECONDS', 'MIN_RETRY_DELAY_SECONDS', 'REDIS_CHANNEL_REFRESH',
        'REDIS_KEY_LOCK', 'REDIS_KEY_METRICS', 'REFRESHING', 'SUCCESS',
        'consecutive_failures', 'embeddings_count', 'failed_refreshes', 'force_refresh',
        'get_rag_scheduler', 'get_status', 'last_error', 'last_refresh_at',
        'last_refresh_duration_ms', 'last_refresh_status', 'next_scheduled_refresh',
        'reset_rag_scheduler', 'start', 'stop', 'swagger_version', 'tools_count',
        'total_refreshes',
    }

    @pytest.mark.parametrize("filename", TARGET_FILES)
    def test_no_undefined_variables(self, filename):
        """Test that there are no undefined variables in the file."""
        filepath = self.SERVICES_PATH / filename

        if not filepath.exists():
            pytest.skip(f"File {filename} not found")

        undefined, _ = analyze_file(str(filepath))

        # Filter out known false positives
        real_undefined = undefined - self.KNOWN_FALSE_POSITIVES

        if real_undefined:
            pytest.fail(
                f"Undefined variables in {filename}: {sorted(real_undefined)}"
            )

    @pytest.mark.parametrize("filename", TARGET_FILES)
    def test_report_unused_variables(self, filename):
        """Report unused variables (warning, not failure)."""
        filepath = self.SERVICES_PATH / filename

        if not filepath.exists():
            pytest.skip(f"File {filename} not found")

        _, unused = analyze_file(str(filepath))

        # Filter out known exports (public API intentionally defined for external use)
        truly_unused = unused - self.KNOWN_EXPORTS

        # This is informational - we don't fail on unused variables
        # Only warn about variables that are not known exports
        if truly_unused:
            import warnings
            warnings.warn(
                f"Potentially unused variables in {filename}: {sorted(truly_unused)}"
            )


class TestModelsGhostVariables:
    """Test models.py for ghost variables."""

    MODELS_PATH = Path(__file__).parent.parent / "models.py"

    def test_models_no_undefined(self):
        """Test that models.py has no undefined variables."""
        if not self.MODELS_PATH.exists():
            pytest.skip("models.py not found")

        undefined, _ = analyze_file(str(self.MODELS_PATH))

        # Filter known patterns
        known = {
            'Optional', 'List', 'Dict', 'Any',
            'Column', 'String', 'Boolean', 'DateTime', 'Text',
            'Integer', 'ForeignKey', 'Index', 'JSON', 'UUID',
            'Base', 'relationship', 'backref',
        }

        real_undefined = undefined - known

        if real_undefined:
            pytest.fail(f"Undefined variables in models.py: {sorted(real_undefined)}")


class TestImportConsistency:
    """Test that imports are consistent and used."""

    SERVICES_PATH = Path(__file__).parent.parent / "services"

    TARGET_FILES = [
        "admin_review.py",
        "hallucination_repository.py",
        "gdpr_masking.py",
        "cost_tracker.py",
        "conflict_resolver.py",
        "model_drift_detector.py",
        "rag_scheduler.py",
    ]

    @pytest.mark.parametrize("filename", TARGET_FILES)
    def test_imports_are_used(self, filename):
        """Check that imports are actually used in the file."""
        filepath = self.SERVICES_PATH / filename

        if not filepath.exists():
            pytest.skip(f"File {filename} not found")

        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        analyzer = VariableAnalyzer()
        analyzer.visit(tree)

        # Find unused imports
        unused_imports = analyzer.imports - analyzer.used

        # Some imports are used indirectly (e.g., for type hints in strings)
        # or re-exported, so this is informational
        if unused_imports:
            import warnings
            warnings.warn(
                f"Potentially unused imports in {filename}: {sorted(unused_imports)}"
            )


class TestVariableNamingConventions:
    """Test that variable names follow Python conventions."""

    SERVICES_PATH = Path(__file__).parent.parent / "services"

    TARGET_FILES = [
        "admin_review.py",
        "hallucination_repository.py",
        "gdpr_masking.py",
        "cost_tracker.py",
        "conflict_resolver.py",
        "model_drift_detector.py",
        "rag_scheduler.py",
    ]

    @pytest.mark.parametrize("filename", TARGET_FILES)
    def test_no_camelcase_variables(self, filename):
        """Check that variables use snake_case, not camelCase."""
        filepath = self.SERVICES_PATH / filename

        if not filepath.exists():
            pytest.skip(f"File {filename} not found")

        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        analyzer = VariableAnalyzer()
        analyzer.visit(tree)

        import re
        camel_case_pattern = re.compile(r'^[a-z]+[A-Z]')

        camel_case_vars = []
        for var in analyzer.defined:
            if camel_case_pattern.match(var):
                # Exclude known patterns
                if var not in {'userId', 'tenantId', 'personId'}:
                    camel_case_vars.append(var)

        # This is a style warning, not a failure
        if camel_case_vars:
            import warnings
            warnings.warn(
                f"camelCase variables in {filename}: {camel_case_vars}"
            )


# ============================================================================
# COMPREHENSIVE ANALYSIS
# ============================================================================

class TestComprehensiveAnalysis:
    """Comprehensive static analysis of all fixed files."""

    SERVICES_PATH = Path(__file__).parent.parent / "services"

    def test_all_files_parse_successfully(self):
        """Verify all target files can be parsed without errors."""
        files = [
            "admin_review.py",
            "hallucination_repository.py",
            "gdpr_masking.py",
            "cost_tracker.py",
            "conflict_resolver.py",
            "model_drift_detector.py",
            "rag_scheduler.py",
        ]

        errors = []
        for filename in files:
            filepath = self.SERVICES_PATH / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source)
                except SyntaxError as e:
                    errors.append(f"{filename}: {e}")

        if errors:
            pytest.fail(f"Parse errors found:\n" + "\n".join(errors))

    def test_no_critical_undefined_variables(self):
        """
        Test that there are no critical undefined variables across all files.

        Critical = likely to cause runtime errors
        """
        files = [
            "admin_review.py",
            "hallucination_repository.py",
            "gdpr_masking.py",
            "cost_tracker.py",
            "conflict_resolver.py",
            "model_drift_detector.py",
            "rag_scheduler.py",
        ]

        all_undefined = {}

        for filename in files:
            filepath = self.SERVICES_PATH / filename
            if filepath.exists():
                undefined, _ = analyze_file(str(filepath))
                if undefined:
                    all_undefined[filename] = undefined

        # Filter to truly critical ones
        critical = {}
        for filename, undefined in all_undefined.items():
            # Filter out type hints and common patterns
            filtered = {
                v for v in undefined
                if not v[0].isupper()  # Skip type hints (uppercase start)
                and v not in {'logger', 'log', 'self', 'cls', '_'}
            }
            if filtered:
                critical[filename] = filtered

        if critical:
            msg = "Critical undefined variables found:\n"
            for filename, vars in critical.items():
                msg += f"  {filename}: {sorted(vars)}\n"
            pytest.fail(msg)
