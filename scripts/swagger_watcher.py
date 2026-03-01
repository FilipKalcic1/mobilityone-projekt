"""
Swagger Watcher - Automatic API Change Detection & Documentation Regeneration

Detects changes in Swagger specifications and automatically triggers
documentation regeneration when APIs change.

Features:
1. Hash-based change detection (compares current vs stored hash)
2. Webhook support for real-time notifications
3. Delta mode - only regenerates changed tools (Pillar 9)
4. Scheduled polling option

Usage:
    # Check for changes and regenerate if needed
    python -m scripts.swagger_watcher

    # Check only (no regeneration)
    python -m scripts.swagger_watcher --check-only

    # Force regeneration
    python -m scripts.swagger_watcher --force

    # Start as background watcher (polls every N minutes)
    python -m scripts.swagger_watcher --watch --interval 30
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"
HASH_FILE = CONFIG_DIR / "swagger_hashes.json"


class SwaggerWatcher:
    """
    Monitors Swagger specifications for changes.

    When changes are detected:
    1. Identifies which tools changed (new, modified, deleted)
    2. Triggers incremental documentation generation
    3. Updates embeddings only for changed tools
    4. Saves new hash for future comparisons
    """

    def __init__(self):
        """Initialize watcher."""
        self.stored_hashes: Dict[str, str] = {}
        self.current_hashes: Dict[str, str] = {}
        self.changes: Dict[str, List[str]] = {
            "new": [],       # New operations
            "modified": [],  # Changed operations
            "deleted": []    # Removed operations
        }
        self._load_stored_hashes()

    def _load_stored_hashes(self) -> None:
        """Load previously stored Swagger hashes."""
        if HASH_FILE.exists():
            try:
                with open(HASH_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stored_hashes = data.get("hashes", {})
                    logger.info(f"Loaded hashes for {len(self.stored_hashes)} Swagger sources")
            except Exception as e:
                logger.warning(f"Could not load stored hashes: {e}")
                self.stored_hashes = {}
        else:
            logger.info("No stored hashes found - first run")

    def _save_hashes(self) -> None:
        """Save current hashes for future comparisons."""
        CONFIG_DIR.mkdir(exist_ok=True)

        data = {
            "hashes": self.current_hashes,
            "last_check": datetime.now().isoformat(),
            "changes_detected": bool(self.changes["new"] or
                                    self.changes["modified"] or
                                    self.changes["deleted"])
        }

        with open(HASH_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved hashes to {HASH_FILE}")

    async def fetch_swagger(self, url: str) -> Optional[Dict]:
        """Fetch Swagger spec from URL."""
        try:
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _compute_spec_hash(self, spec: Dict) -> str:
        """
        Compute hash of Swagger spec.

        Uses a normalized representation to detect meaningful changes:
        - Ignores whitespace and formatting
        - Includes paths, methods, parameters, and schemas
        - Excludes timestamps and metadata
        """
        # Extract meaningful parts
        paths = spec.get("paths", {})
        components = spec.get("components", spec.get("definitions", {}))

        # Build normalized representation
        normalized = {
            "paths": {},
            "schemas": {}
        }

        for path, methods in paths.items():
            normalized["paths"][path] = {}
            for method, operation in methods.items():
                if method.lower() not in ["get", "post", "put", "patch", "delete"]:
                    continue

                # Include operation signature
                normalized["paths"][path][method] = {
                    "operationId": operation.get("operationId", ""),
                    "parameters": self._normalize_params(operation.get("parameters", [])),
                    "requestBody": self._normalize_request_body(operation.get("requestBody", {})),
                    "responses": list(operation.get("responses", {}).keys())
                }

        # Include schema definitions
        schemas = components.get("schemas", components)
        for schema_name, schema in schemas.items():
            if isinstance(schema, dict):
                normalized["schemas"][schema_name] = {
                    "properties": list(schema.get("properties", {}).keys()),
                    "required": schema.get("required", [])
                }

        # Compute hash of normalized JSON
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _normalize_params(self, params: List[Dict]) -> List[Dict]:
        """Normalize parameters for hashing."""
        normalized = []
        for param in params:
            normalized.append({
                "name": param.get("name", ""),
                "in": param.get("in", ""),
                "required": param.get("required", False),
                "type": param.get("schema", {}).get("type", "string")
            })
        return sorted(normalized, key=lambda x: x["name"])

    def _normalize_request_body(self, body: Dict) -> Dict:
        """Normalize request body for hashing."""
        if not body:
            return {}

        content = body.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})

        return {
            "required": body.get("required", False),
            "properties": list(schema.get("properties", {}).keys()) if "$ref" not in schema else [schema.get("$ref", "")]
        }

    def _compute_operation_hashes(self, spec: Dict) -> Dict[str, str]:
        """
        Compute hash for each operation individually.

        This enables delta/incremental updates - only changed operations
        trigger documentation regeneration.
        """
        operation_hashes = {}

        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.lower() not in ["get", "post", "put", "patch", "delete"]:
                    continue

                operation_id = operation.get("operationId", f"{method}_{path}")

                # Build operation signature
                signature = {
                    "path": path,
                    "method": method,
                    "parameters": self._normalize_params(operation.get("parameters", [])),
                    "requestBody": self._normalize_request_body(operation.get("requestBody", {})),
                    "responses": list(operation.get("responses", {}).keys()),
                    "description": operation.get("description", ""),
                    "summary": operation.get("summary", "")
                }

                json_str = json.dumps(signature, sort_keys=True, ensure_ascii=False)
                operation_hashes[operation_id] = hashlib.md5(json_str.encode()).hexdigest()

        return operation_hashes

    async def check_for_changes(self, swagger_sources: List[str]) -> bool:
        """
        Check all Swagger sources for changes.

        Returns:
            True if any changes detected
        """
        logger.info(f"Checking {len(swagger_sources)} Swagger sources for changes...")

        has_changes = False
        all_current_operations: Dict[str, str] = {}

        for url in swagger_sources:
            logger.info(f"Fetching: {url}")
            spec = await self.fetch_swagger(url)

            if not spec:
                logger.error(f"Could not fetch {url}")
                continue

            # Compute overall hash
            spec_hash = self._compute_spec_hash(spec)
            self.current_hashes[url] = spec_hash

            stored_hash = self.stored_hashes.get(url)

            if stored_hash != spec_hash:
                logger.info(f"CHANGE DETECTED in {url}")
                has_changes = True
            else:
                logger.info(f"No changes in {url}")

            # Compute per-operation hashes for delta detection
            op_hashes = self._compute_operation_hashes(spec)
            all_current_operations.update(op_hashes)

        # Detect specific changes (new, modified, deleted)
        if has_changes:
            self._detect_delta_changes(all_current_operations)

        return has_changes

    def _detect_delta_changes(self, current_operations: Dict[str, str]) -> None:
        """
        Detect which specific operations changed.

        Compares current operation hashes with stored documentation
        to identify new, modified, and deleted operations.
        """
        # Load existing documentation to get stored operation hashes
        doc_path = CONFIG_DIR / "tool_documentation.json"
        stored_ops: Dict[str, str] = {}

        if doc_path.exists():
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    docs = json.load(f)
                    # Each documented tool has a hash in the metadata
                    for op_id, doc in docs.items():
                        stored_ops[op_id] = doc.get("_hash", "")
            except Exception as e:
                logger.warning(f"Could not load existing documentation: {e}")

        # Compare
        current_ids = set(current_operations.keys())
        stored_ids = set(stored_ops.keys())

        # New operations
        self.changes["new"] = list(current_ids - stored_ids)

        # Deleted operations
        self.changes["deleted"] = list(stored_ids - current_ids)

        # Modified operations
        self.changes["modified"] = []
        for op_id in current_ids & stored_ids:
            if current_operations[op_id] != stored_ops.get(op_id, ""):
                self.changes["modified"].append(op_id)

        # Log summary
        logger.info("=" * 50)
        logger.info("CHANGE SUMMARY:")
        logger.info(f"New operations: {len(self.changes['new'])}")
        logger.info(f"Modified operations: {len(self.changes['modified'])}")
        logger.info(f"Deleted operations: {len(self.changes['deleted'])}")
        logger.info("=" * 50)

        if self.changes["new"]:
            logger.info(f"New: {self.changes['new'][:10]}{'...' if len(self.changes['new']) > 10 else ''}")
        if self.changes["modified"]:
            logger.info(f"Modified: {self.changes['modified'][:10]}{'...' if len(self.changes['modified']) > 10 else ''}")
        if self.changes["deleted"]:
            logger.info(f"Deleted: {self.changes['deleted'][:10]}{'...' if len(self.changes['deleted']) > 10 else ''}")

    async def trigger_regeneration(self, delta_only: bool = True) -> bool:
        """
        Trigger documentation regeneration.

        Args:
            delta_only: If True, only regenerate changed operations (Pillar 9)
                       If False, regenerate everything

        Returns:
            True if regeneration succeeded
        """
        from scripts.generate_documentation import DocumentationGenerator, load_tools_from_registry

        try:
            # Load tools from registry
            tools = await load_tools_from_registry()

            if delta_only and (self.changes["new"] or self.changes["modified"]):
                # Pillar 9: Incremental regeneration
                changed_ops = set(self.changes["new"]) | set(self.changes["modified"])
                tools_to_process = {
                    op_id: tool for op_id, tool in tools.items()
                    if op_id in changed_ops
                }

                logger.info(f"Delta mode: Processing {len(tools_to_process)} changed tools")
            else:
                # Full regeneration
                tools_to_process = tools
                logger.info(f"Full mode: Processing all {len(tools_to_process)} tools")

            # Generate documentation
            generator = DocumentationGenerator()
            await generator.generate_all(tools_to_process)

            # Save updated hashes
            self._save_hashes()

            logger.info("Documentation regeneration completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Regeneration failed: {e}", exc_info=True)
            return False

    def get_change_summary(self) -> Dict:
        """Get summary of detected changes."""
        return {
            "has_changes": bool(
                self.changes["new"] or
                self.changes["modified"] or
                self.changes["deleted"]
            ),
            "new_count": len(self.changes["new"]),
            "modified_count": len(self.changes["modified"]),
            "deleted_count": len(self.changes["deleted"]),
            "new_operations": self.changes["new"],
            "modified_operations": self.changes["modified"],
            "deleted_operations": self.changes["deleted"],
            "checked_at": datetime.now().isoformat()
        }


class WebhookHandler:
    """
    Handle webhook notifications from API deployment systems.

    When integrated with CI/CD, receives POST requests when
    new API versions are deployed.
    """

    def __init__(self, watcher: SwaggerWatcher):
        """Initialize handler."""
        self.watcher = watcher

    async def handle_webhook(self, payload: Dict) -> Dict:
        """
        Handle incoming webhook notification.

        Expected payload:
        {
            "event": "api_deployed",
            "swagger_urls": ["https://..."],
            "version": "1.2.3",
            "timestamp": "..."
        }

        Returns:
            Response dict with status and details
        """
        event = payload.get("event", "")

        if event != "api_deployed":
            return {"status": "ignored", "reason": f"Unknown event: {event}"}

        swagger_urls = payload.get("swagger_urls", [])

        if not swagger_urls:
            return {"status": "error", "reason": "No swagger_urls provided"}

        logger.info(f"Webhook received: API deployed with {len(swagger_urls)} swagger URLs")

        # Check for changes
        has_changes = await self.watcher.check_for_changes(swagger_urls)

        if has_changes:
            # Trigger regeneration in delta mode
            success = await self.watcher.trigger_regeneration(delta_only=True)

            return {
                "status": "regenerated" if success else "error",
                "changes": self.watcher.get_change_summary()
            }

        return {
            "status": "no_changes",
            "message": "Swagger specs unchanged"
        }


async def watch_loop(interval_minutes: int = 30):
    """
    Background watcher loop.

    Polls Swagger sources at regular intervals.
    """
    watcher = SwaggerWatcher()
    swagger_sources = settings.swagger_sources

    if not swagger_sources:
        logger.error("No swagger_sources configured in settings")
        return

    logger.info(f"Starting watch loop (interval: {interval_minutes} minutes)")
    logger.info(f"Monitoring {len(swagger_sources)} Swagger sources")

    while True:
        try:
            has_changes = await watcher.check_for_changes(swagger_sources)

            if has_changes:
                logger.info("Changes detected - triggering regeneration...")
                await watcher.trigger_regeneration(delta_only=True)
            else:
                logger.info("No changes detected")

        except Exception as e:
            logger.error(f"Watch loop error: {e}", exc_info=True)

        # Wait for next check
        logger.info(f"Next check in {interval_minutes} minutes...")
        await asyncio.sleep(interval_minutes * 60)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Swagger Watcher - API Change Detection")
    parser.add_argument("--check-only", action="store_true", help="Only check for changes, don't regenerate")
    parser.add_argument("--force", action="store_true", help="Force full regeneration")
    parser.add_argument("--watch", action="store_true", help="Start background watcher")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in minutes (default: 30)")

    args = parser.parse_args()

    # Get Swagger sources from settings
    swagger_sources = settings.swagger_sources

    if not swagger_sources:
        # Fallback to main API
        swagger_sources = [
            f"{settings.MOBILITY_API_URL.rstrip('/')}/swagger/v1/swagger.json"
        ]

    logger.info("=" * 60)
    logger.info("SWAGGER WATCHER")
    logger.info("=" * 60)
    logger.info(f"Monitoring {len(swagger_sources)} Swagger sources")

    if args.watch:
        # Background watcher mode
        await watch_loop(args.interval)
        return

    watcher = SwaggerWatcher()

    # Check for changes
    has_changes = await watcher.check_for_changes(swagger_sources)

    if args.check_only:
        # Just report changes
        summary = watcher.get_change_summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.force or has_changes:
        # Trigger regeneration
        mode = "full" if args.force else "delta"
        logger.info(f"Triggering {mode} regeneration...")

        success = await watcher.trigger_regeneration(delta_only=not args.force)

        if not success:
            sys.exit(1)
    else:
        logger.info("No changes detected - nothing to do")

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
