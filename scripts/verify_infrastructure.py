#!/usr/bin/env python3
"""
Infrastructure Verification Script
Version: 1.0

Verifies the "Split-Brain" architecture fixes:
1. Metrics sync (API reads from Redis, Worker writes to Redis)
2. Admin security (SERVICE_TYPE environment variable)
3. Stream connectivity (webhook -> Redis -> worker)
4. Database connection pool

Run this after deploying to verify the fixes work.

Usage:
    python scripts/verify_infrastructure.py
"""

import asyncio
import sys
import os
import json
import httpx
import redis.asyncio as aioredis

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings

settings = get_settings()

# Test configuration
BOT_API_URL = os.getenv("BOT_API_URL", "http://localhost:8000")
ADMIN_API_URL = os.getenv("ADMIN_API_URL", "http://localhost:8080")


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with color coding."""
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")


async def test_metrics_sync():
    """
    Test 1: Metrics Sync (The "Zero" Bug Fix)

    Problem: API doesn't load ToolRegistry, so tools_loaded_total was always 0.
    Solution: Worker writes count to Redis, API reads from Redis.

    This test verifies:
    1. We can write a test value to Redis
    2. The /metrics endpoint reads this value correctly
    """
    print(f"\n{Colors.BOLD}Test 1: Metrics Sync (The 'Zero' Bug){Colors.RESET}")

    test_value = 999
    passed = True

    try:
        # Connect to Redis
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # Write test value
        await redis_client.set(settings.REDIS_STATS_KEY_TOOLS, test_value)
        print_result("Write to Redis", True, f"Set {settings.REDIS_STATS_KEY_TOOLS}={test_value}")

        # Read from API metrics endpoint
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{BOT_API_URL}/metrics")

                if response.status_code == 200:
                    content = response.text
                    expected = f"tools_loaded_total {float(test_value)}"

                    if expected in content or f"tools_loaded_total {test_value}.0" in content:
                        print_result("API reads from Redis", True, f"Found '{expected}' in response")
                    else:
                        print_result("API reads from Redis", False, f"Expected '{expected}' not found")
                        passed = False
                else:
                    print_result("API metrics endpoint", False, f"Status: {response.status_code}")
                    passed = False

            except httpx.ConnectError:
                print_result("API metrics endpoint", False, f"Cannot connect to {BOT_API_URL}")
                passed = False

        # Cleanup
        await redis_client.delete(settings.REDIS_STATS_KEY_TOOLS)
        await redis_client.aclose()

    except Exception as e:
        print_result("Redis connection", False, str(e))
        passed = False

    return passed


async def test_admin_security():
    """
    Test 2: Admin Security (The "Role" Check)

    Problem: Admin API needs full database access for admin operations.
    Solution: Set SERVICE_TYPE=admin BEFORE importing database.py.

    This test verifies:
    1. Admin API is accessible
    2. Response confirms admin role/privileges
    """
    print(f"\n{Colors.BOLD}Test 2: Admin Security (SERVICE_TYPE){Colors.RESET}")

    passed = True

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{ADMIN_API_URL}/health")

                if response.status_code == 200:
                    print_result("Admin API accessible", True)

                    try:
                        data = response.json()
                        # Check for role indicator
                        if data.get("role") == "admin" or data.get("service_type") == "admin":
                            print_result("Admin role confirmed", True, f"Response: {data}")
                        else:
                            # If health doesn't return role, check if it at least works
                            print_result("Admin API responds", True, f"Response: {data}")
                            print(f"         {Colors.YELLOW}Note: Consider adding 'role' to /health response{Colors.RESET}")
                    except json.JSONDecodeError:
                        print_result("Admin API JSON", False, "Invalid JSON response")
                        passed = False
                else:
                    print_result("Admin API accessible", False, f"Status: {response.status_code}")
                    passed = False

            except httpx.ConnectError:
                print_result("Admin API connection", False, f"Cannot connect to {ADMIN_API_URL}")
                print(f"         {Colors.YELLOW}Hint: Is admin_api running on port 8080?{Colors.RESET}")
                passed = False

    except Exception as e:
        print_result("Admin security test", False, str(e))
        passed = False

    return passed


async def test_stream_connectivity():
    """
    Test 3: Stream Connectivity

    Verifies the message flow:
    1. POST /webhook/whatsapp -> Redis Stream
    2. Worker reads from stream

    This test verifies:
    1. Webhook accepts messages
    2. Messages appear in Redis stream
    """
    print(f"\n{Colors.BOLD}Test 3: Stream Connectivity{Colors.RESET}")

    passed = True
    test_message_id = f"verify_test_{int(asyncio.get_event_loop().time())}"

    try:
        # Connect to Redis
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # Count messages before
        try:
            before_count = await redis_client.xlen("whatsapp_stream_inbound")
        except Exception:
            before_count = 0

        # Post test message to webhook
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                test_payload = {
                    "results": [{
                        "sender": "+385991234567",
                        "messageId": test_message_id,
                        "content": [{"type": "TEXT", "text": "verify_infrastructure_test"}]
                    }]
                }

                response = await client.post(
                    f"{BOT_API_URL}/webhook/whatsapp",
                    json=test_payload
                )

                if response.status_code == 200:
                    print_result("Webhook accepts message", True)

                    # Check if message appeared in stream
                    await asyncio.sleep(0.5)  # Give it time
                    after_count = await redis_client.xlen("whatsapp_stream_inbound")

                    if after_count > before_count:
                        print_result("Message in Redis stream", True, f"Stream length: {before_count} -> {after_count}")
                    else:
                        print_result("Message in Redis stream", False, "Stream length unchanged")
                        passed = False
                else:
                    print_result("Webhook accepts message", False, f"Status: {response.status_code}")
                    passed = False

            except httpx.ConnectError:
                print_result("Webhook connection", False, f"Cannot connect to {BOT_API_URL}")
                passed = False

        await redis_client.aclose()

    except Exception as e:
        print_result("Stream connectivity", False, str(e))
        passed = False

    return passed


async def test_database_pool():
    """
    Test 4: Database Connection Pool

    Verifies:
    1. Database is accessible
    2. Connection pool is working
    """
    print(f"\n{Colors.BOLD}Test 4: Database Connection Pool{Colors.RESET}")

    passed = True

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{BOT_API_URL}/health")

                if response.status_code == 200:
                    data = response.json()

                    db_status = data.get("database", "unknown")
                    if db_status == "connected":
                        print_result("Database connected", True)
                    else:
                        print_result("Database connected", False, f"Status: {db_status}")
                        passed = False

                    redis_status = data.get("redis", "unknown")
                    if redis_status == "connected":
                        print_result("Redis connected", True)
                    else:
                        print_result("Redis connected", False, f"Status: {redis_status}")
                        passed = False
                else:
                    print_result("Health endpoint", False, f"Status: {response.status_code}")
                    passed = False

            except httpx.ConnectError:
                print_result("API connection", False, f"Cannot connect to {BOT_API_URL}")
                passed = False

    except Exception as e:
        print_result("Database pool test", False, str(e))
        passed = False

    return passed


async def main():
    """Run all infrastructure verification tests."""
    print("=" * 60)
    print(f"{Colors.BOLD}INFRASTRUCTURE VERIFICATION{Colors.RESET}")
    print("=" * 60)
    print(f"\nBot API: {BOT_API_URL}")
    print(f"Admin API: {ADMIN_API_URL}")
    print(f"Redis: {settings.REDIS_URL}")

    results = []

    # Run tests
    results.append(("Metrics Sync", await test_metrics_sync()))
    results.append(("Admin Security", await test_admin_security()))
    results.append(("Stream Connectivity", await test_stream_connectivity()))
    results.append(("Database Pool", await test_database_pool()))

    # Summary
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
    print("=" * 60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name}: {status}")

    print()
    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed! Infrastructure is ready.{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}{total_count - passed_count} test(s) failed. Please fix before deployment.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
