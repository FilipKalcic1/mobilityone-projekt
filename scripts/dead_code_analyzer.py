"""
Dead Code Analyzer for MobilityOne Bot

FEATURES:
1. Vulture-style dead code detection
2. Unused import detection
3. Dependency audit (large packages for single functions)
4. Function call graph analysis

USAGE:
    python scripts/dead_code_analyzer.py
    python scripts/dead_code_analyzer.py --verbose
    python scripts/dead_code_analyzer.py --output report.json

REQUIREMENTS:
    pip install vulture ast-grep
"""

import os
import sys
import ast
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CodeAnalysis:
    """Analysis results."""
    file_path: str
    defined_functions: List[str] = field(default_factory=list)
    defined_classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)
    unused_functions: List[str] = field(default_factory=list)


class DeadCodeAnalyzer:
    """Analyze Python codebase for dead code."""

    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.files: List[Path] = []
        self.all_definitions: Dict[str, str] = {}  # name -> file
        self.all_calls: Set[str] = set()
        self.all_imports: Dict[str, List[str]] = defaultdict(list)  # file -> imports
        self.import_usage: Dict[str, Set[str]] = defaultdict(set)  # file -> used names

        # Known entry points (not dead code even if not called internally)
        self.entry_points = {
            "main", "app", "lifespan", "__init__",
            "health_check", "webhook", "process_message",
            "get_db", "get_admin_service", "verify_admin_token",
            "_handle_signal", "log"
        }

        # FastAPI route decorators mark functions as entry points
        self.route_decorators = {"get", "post", "put", "delete", "patch", "options", "head"}

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        self.files = []
        exclude_dirs = {"__pycache__", ".git", "venv", "env", ".venv", "node_modules", "alembic"}

        for path in self.root.rglob("*.py"):
            if not any(excl in path.parts for excl in exclude_dirs):
                self.files.append(path)

        return self.files

    def analyze_file(self, file_path: Path) -> CodeAnalysis:
        """Analyze a single Python file."""
        analysis = CodeAnalysis(file_path=str(file_path))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        analysis.imports.append(name)
                        self.all_imports[str(file_path)].append(name)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        name = alias.asname or alias.name
                        full_name = f"{module}.{name}" if module else name
                        analysis.imports.append(name)
                        self.all_imports[str(file_path)].append(name)

            # Analyze definitions and calls
            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    func_name = node.name
                    analysis.defined_functions.append(func_name)

                    # Check if it's a route (decorated with @app.get, etc.)
                    is_route = False
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr in self.route_decorators:
                                    is_route = True
                        elif isinstance(decorator, ast.Attribute):
                            if decorator.attr in self.route_decorators:
                                is_route = True

                    if not is_route and not func_name.startswith('_'):
                        self.all_definitions[func_name] = str(file_path)

                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    analysis.defined_classes.append(node.name)
                    self.all_definitions[node.name] = str(file_path)

                # Function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        analysis.function_calls.append(node.func.id)
                        self.all_calls.add(node.func.id)
                        self.import_usage[str(file_path)].add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        analysis.function_calls.append(node.func.attr)
                        self.all_calls.add(node.func.attr)
                        # Track attribute access for import usage
                        if isinstance(node.func.value, ast.Name):
                            self.import_usage[str(file_path)].add(node.func.value.id)

                # Name access (for import usage)
                elif isinstance(node, ast.Name):
                    self.import_usage[str(file_path)].add(node.id)

                # Attribute access
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        self.import_usage[str(file_path)].add(node.value.id)

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

        return analysis

    def find_unused_functions(self) -> List[Tuple[str, str]]:
        """Find functions that are never called."""
        unused = []

        for func_name, file_path in self.all_definitions.items():
            # Skip entry points and magic methods
            if func_name in self.entry_points:
                continue
            if func_name.startswith('__') and func_name.endswith('__'):
                continue
            if func_name.startswith('test_'):  # Test functions
                continue

            if func_name not in self.all_calls:
                unused.append((func_name, file_path))

        return unused

    def find_unused_imports(self) -> Dict[str, List[str]]:
        """Find imports that are never used."""
        unused = defaultdict(list)

        for file_path, imports in self.all_imports.items():
            used = self.import_usage.get(file_path, set())

            for imp in imports:
                # Check if the import or its submodules are used
                base_name = imp.split('.')[0]
                if imp not in used and base_name not in used:
                    unused[file_path].append(imp)

        return dict(unused)

    def analyze_dependencies(self) -> Dict[str, Dict]:
        """Analyze if large packages are imported for minimal use."""
        heavy_packages = {
            "pandas": {"size_mb": 150, "alternatives": ["polars (faster)", "csv module (lightweight)"]},
            "numpy": {"size_mb": 50, "alternatives": ["array module", "math module"]},
            "tensorflow": {"size_mb": 500, "alternatives": ["onnxruntime", "tflite"]},
            "torch": {"size_mb": 800, "alternatives": ["onnxruntime"]},
            "scipy": {"size_mb": 100, "alternatives": ["statsmodels", "pure python"]},
            "matplotlib": {"size_mb": 60, "alternatives": ["plotly (if needed)"]},
        }

        issues = {}

        for file_path, imports in self.all_imports.items():
            for imp in imports:
                base = imp.split('.')[0]
                if base in heavy_packages:
                    usage_count = sum(1 for call in self.import_usage.get(file_path, set())
                                     if base in str(call))
                    if usage_count < 5:  # Using heavy package for few operations
                        issues[file_path] = {
                            "package": base,
                            "usage_count": usage_count,
                            "size_mb": heavy_packages[base]["size_mb"],
                            "alternatives": heavy_packages[base]["alternatives"]
                        }

        return issues

    def run_analysis(self, verbose: bool = False) -> Dict:
        """Run complete analysis."""
        print("Scanning for Python files...")
        self.find_python_files()
        print(f"Found {len(self.files)} Python files")

        print("\nAnalyzing code...")
        analyses = []
        for file_path in self.files:
            analysis = self.analyze_file(file_path)
            analyses.append(analysis)

        print("\nFinding dead code...")
        unused_functions = self.find_unused_functions()
        unused_imports = self.find_unused_imports()
        dependency_issues = self.analyze_dependencies()

        # Generate report
        report = {
            "summary": {
                "files_analyzed": len(self.files),
                "total_functions": len(self.all_definitions),
                "total_calls": len(self.all_calls),
                "unused_functions": len(unused_functions),
                "files_with_unused_imports": len(unused_imports)
            },
            "unused_functions": [
                {"name": name, "file": file}
                for name, file in unused_functions
            ],
            "unused_imports": {
                str(Path(k).relative_to(self.root)): v
                for k, v in unused_imports.items() if v
            },
            "dependency_audit": dependency_issues
        }

        # Print results
        self._print_report(report, verbose)

        return report

    def _print_report(self, report: Dict, verbose: bool):
        """Print analysis report."""
        print("\n" + "=" * 60)
        print("DEAD CODE ANALYSIS REPORT")
        print("=" * 60)

        summary = report["summary"]
        print(f"\nFiles analyzed: {summary['files_analyzed']}")
        print(f"Functions defined: {summary['total_functions']}")
        print(f"Unique function calls: {summary['total_calls']}")

        # Unused functions
        print(f"\n--- POTENTIALLY UNUSED FUNCTIONS ({summary['unused_functions']}) ---")
        if report["unused_functions"]:
            for item in report["unused_functions"][:20]:  # Show top 20
                rel_path = Path(item["file"]).relative_to(self.root)
                print(f"  - {item['name']} in {rel_path}")
            if len(report["unused_functions"]) > 20:
                print(f"  ... and {len(report['unused_functions']) - 20} more")
        else:
            print("  No obviously unused functions found!")

        # Unused imports
        print(f"\n--- UNUSED IMPORTS ({summary['files_with_unused_imports']} files) ---")
        if report["unused_imports"]:
            for file, imports in list(report["unused_imports"].items())[:10]:
                print(f"  {file}:")
                for imp in imports[:5]:
                    print(f"    - {imp}")
        else:
            print("  No unused imports detected!")

        # Dependency audit
        print("\n--- HEAVY DEPENDENCY AUDIT ---")
        if report["dependency_audit"]:
            for file, info in report["dependency_audit"].items():
                rel_path = Path(file).relative_to(self.root)
                print(f"  {rel_path}: {info['package']} (~{info['size_mb']}MB)")
                print(f"    Used only {info['usage_count']} times")
                print(f"    Alternatives: {', '.join(info['alternatives'])}")
        else:
            print("  No heavy dependencies with minimal usage!")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Dead Code Analyzer")
    parser.add_argument("--path", default=".", help="Project root path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    # Get project root
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Path not found: {root}")
        sys.exit(1)

    analyzer = DeadCodeAnalyzer(str(root))
    report = analyzer.run_analysis(verbose=args.verbose)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
