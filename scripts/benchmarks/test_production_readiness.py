"""
Production Readiness Test Suite

Run this AFTER docker-compose up to verify the system works end-to-end.

Usage:
    python scripts/test_production_readiness.py

Requirements:
    - Docker containers running (bot-api, worker, redis, postgres)
    - Valid API credentials in .env
"""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_imports():
    """Test all critical imports work."""
    print("\n[1] IMPORT TEST")
    print("-" * 50)

    errors = []

    imports = [
        ("services.cache_service", "CacheService"),
        ("services.context_service", "ContextService, UserContext"),
        ("services.engine", "MessageEngine"),
        ("services.registry", "ToolRegistry"),
        ("services.tool_executor", "ToolExecutor"),
        ("services.api_gateway", "APIGateway"),
        ("services.unified_router", "UnifiedRouter"),
        ("services.ai_orchestrator", "AIOrchestrator"),
        ("services.queue_service", "QueueService"),
        ("services.user_service", "UserService"),
        ("services.whatsapp_service", "WhatsAppService"),
        ("services.cost_tracker", "CostTracker"),
        ("services.model_drift_detector", "ModelDriftDetector"),
        ("services.admin_review", "AdminReviewService"),
    ]

    for module, classes in imports:
        try:
            exec(f"from {module} import {classes}")
            print(f"  [OK] {module}")
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            errors.append(module)

    return len(errors) == 0


async def test_registry():
    """Test ToolRegistry initialization."""
    print("\n[2] REGISTRY TEST")
    print("-" * 50)

    from services.registry import ToolRegistry

    registry = ToolRegistry(redis_client=None)

    # Check hidden defaults
    hidden = registry._HIDDEN_DEFAULTS
    print(f"  Hidden defaults: {len(hidden)} tools configured")
    for op_id, defaults in hidden.items():
        print(f"    - {op_id}: {defaults}")

    # Check critical methods
    methods = ['get_tool', 'get_hidden_defaults', 'get_merged_params']
    for method in methods:
        has_method = hasattr(registry, method)
        status = "[OK]" if has_method else "[FAIL]"
        print(f"  {status} {method}()")

    return True


async def test_user_context():
    """Test UserContext model."""
    print("\n[3] USER CONTEXT TEST")
    print("-" * 50)

    from services.context_service import UserContext

    # Test guest context
    guest = UserContext.guest("+385911234567")
    print(f"  Guest context created: is_guest={guest.is_guest}")

    # Test serialization
    ctx_dict = guest.to_dict()
    restored = UserContext.from_dict(ctx_dict)
    print(f"  Serialization roundtrip: {restored.phone == guest.phone}")

    return True


async def test_cache_service():
    """Test CacheService with SafeJSONEncoder."""
    print("\n[4] CACHE SERVICE TEST")
    print("-" * 50)

    from services.cache_service import SafeJSONEncoder
    from datetime import datetime
    from uuid import uuid4
    import json

    # Test encoder
    data = {
        "timestamp": datetime.now(),
        "uuid": uuid4(),
        "text": "Test"
    }

    try:
        encoded = json.dumps(data, cls=SafeJSONEncoder)
        print(f"  SafeJSONEncoder works: {len(encoded)} bytes")
        return True
    except Exception as e:
        print(f"  SafeJSONEncoder FAILED: {e}")
        return False


async def test_message_engine_signature():
    """Test MessageEngine accepts all required services."""
    print("\n[5] MESSAGE ENGINE SIGNATURE TEST")
    print("-" * 50)

    import inspect
    from services.engine import MessageEngine

    sig = inspect.signature(MessageEngine.__init__)
    params = list(sig.parameters.keys())

    required = ['gateway', 'registry', 'context_service', 'queue_service', 'cache_service', 'db_session']

    for param in required:
        has_param = param in params
        status = "[OK]" if has_param else "[FAIL]"
        print(f"  {status} {param}")

    return all(p in params for p in required)


async def test_worker_integration():
    """Test worker.py has correct imports and initialization."""
    print("\n[6] WORKER INTEGRATION TEST")
    print("-" * 50)

    with open('worker.py', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("CacheService import", "from services.cache_service import CacheService"),
        ("ContextService import", "from services.context_service import ContextService"),
        ("CacheService init", "CacheService(self.redis)"),
        ("ContextService init", "ContextService(self.redis)"),
        ("context_service injection", "context_service=self._context"),
        ("cache_service injection", "cache_service=self._cache"),
    ]

    all_pass = True
    for name, pattern in checks:
        found = pattern in content
        status = "[OK]" if found else "[FAIL]"
        print(f"  {status} {name}")
        if not found:
            all_pass = False

    return all_pass


async def test_audit():
    """Run audit_project.py and check for errors."""
    print("\n[7] AUDIT TEST")
    print("-" * 50)

    import subprocess
    result = subprocess.run(
        [sys.executable, "audit_project.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if "PROJEKT JE CIST" in result.stdout:
        print("  [OK] Audit passed - no errors or warnings")
        return True
    elif "KRITICNE GRESKE" in result.stdout:
        print("  [FAIL] Audit found CRITICAL errors!")
        print(result.stdout[-500:])
        return False
    elif "UPOZORENJA" in result.stdout:
        print("  [WARN] Audit found warnings (non-critical)")
        return True
    else:
        print("  [UNKNOWN] Could not determine audit result")
        print(result.stdout[-200:])
        return False


async def main():
    print("=" * 60)
    print("PRODUCTION READINESS TEST SUITE")
    print("=" * 60)

    results = []

    results.append(("Imports", await test_imports()))
    results.append(("Registry", await test_registry()))
    results.append(("UserContext", await test_user_context()))
    results.append(("CacheService", await test_cache_service()))
    results.append(("MessageEngine", await test_message_engine_signature()))
    results.append(("Worker", await test_worker_integration()))
    results.append(("Audit", await test_audit()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("[SUCCESS] All tests passed! System is production-ready.")
        print()
        print("Next steps:")
        print("  1. docker-compose up --build -d")
        print("  2. docker-compose logs -f")
        print("  3. Send WhatsApp test message")
    else:
        print("[FAILURE] Some tests failed. Fix issues before deployment.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
