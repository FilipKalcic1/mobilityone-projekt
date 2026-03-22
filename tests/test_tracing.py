"""
Tests for services.tracing — NoOp fallback behaviour.

These tests verify the no-op tracing path (OTEL_ENABLED=false)
without requiring opentelemetry packages to be installed.
"""

import os
import unittest
from unittest.mock import patch

# Ensure OTEL is disabled for these tests
os.environ["OTEL_ENABLED"] = "false"

# Reset module state so each test starts fresh
import importlib
import services.tracing as tracing_mod


class TestNoOpTracer(unittest.TestCase):
    """Verify _NoOpTracer / _NoOpSpan behave correctly."""

    def setUp(self):
        # Reset module-level state
        tracing_mod._initialized = False
        tracing_mod._tracer_provider = None

    def test_get_tracer_returns_noop_when_disabled(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            tracing_mod._initialized = False
            tracer = tracing_mod.get_tracer("test_module")
            self.assertIsInstance(tracer, tracing_mod._NoOpTracer)

    def test_noop_span_set_attribute(self):
        span = tracing_mod._NoOpSpan()
        # Should not raise
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        span.set_attribute("flag", True)

    def test_noop_span_set_status(self):
        span = tracing_mod._NoOpSpan()
        span.set_status("OK")

    def test_noop_span_record_exception(self):
        span = tracing_mod._NoOpSpan()
        span.record_exception(ValueError("test"))

    def test_noop_span_add_event(self):
        span = tracing_mod._NoOpSpan()
        span.add_event("my_event", {"detail": "info"})

    def test_noop_tracer_start_as_current_span(self):
        tracer = tracing_mod._NoOpTracer()
        ctx = tracer.start_as_current_span("test_span")
        # Should be usable as context manager
        with ctx as span:
            self.assertIsInstance(span, tracing_mod._NoOpSpan)
            span.set_attribute("inside", True)

    def test_noop_tracer_start_span(self):
        tracer = tracing_mod._NoOpTracer()
        span = tracer.start_span("test_span")
        self.assertIsInstance(span, tracing_mod._NoOpSpan)


class TestTraceSpan(unittest.TestCase):
    """Verify trace_span context manager with NoOp tracer."""

    def setUp(self):
        tracing_mod._initialized = False
        tracing_mod._tracer_provider = None

    def test_trace_span_with_noop_tracer(self):
        tracer = tracing_mod._NoOpTracer()
        with tracing_mod.trace_span(tracer, "test.operation") as span:
            self.assertIsInstance(span, tracing_mod._NoOpSpan)
            span.set_attribute("test", "value")

    def test_trace_span_with_attributes(self):
        tracer = tracing_mod._NoOpTracer()
        attrs = {"query": "test query", "count": 5, "flag": True}
        with tracing_mod.trace_span(tracer, "test.with_attrs", attrs) as span:
            self.assertIsInstance(span, tracing_mod._NoOpSpan)

    def test_trace_span_no_attributes(self):
        tracer = tracing_mod._NoOpTracer()
        with tracing_mod.trace_span(tracer, "test.no_attrs", None) as span:
            span.set_attribute("added_later", "yes")

    def test_trace_span_exception_inside_span(self):
        """Exceptions inside a span should propagate normally."""
        tracer = tracing_mod._NoOpTracer()
        with self.assertRaises(ValueError):
            with tracing_mod.trace_span(tracer, "test.error"):
                raise ValueError("test error")


class TestIsEnabled(unittest.TestCase):
    """Verify _is_enabled() respects environment variable."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OTEL_ENABLED", None)
            self.assertFalse(tracing_mod._is_enabled())

    def test_enabled_true(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            self.assertTrue(tracing_mod._is_enabled())

    def test_enabled_one(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "1"}):
            self.assertTrue(tracing_mod._is_enabled())

    def test_enabled_yes(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "yes"}):
            self.assertTrue(tracing_mod._is_enabled())

    def test_disabled_false(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            self.assertFalse(tracing_mod._is_enabled())

    def test_disabled_random_string(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "maybe"}):
            self.assertFalse(tracing_mod._is_enabled())


class TestInitTracing(unittest.TestCase):
    """Verify _init_tracing idempotency."""

    def setUp(self):
        tracing_mod._initialized = False
        tracing_mod._tracer_provider = None

    def test_init_is_idempotent(self):
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            tracing_mod._init_tracing()
            self.assertTrue(tracing_mod._initialized)
            self.assertIsNone(tracing_mod._tracer_provider)

            # Second call should be a no-op
            tracing_mod._init_tracing()
            self.assertTrue(tracing_mod._initialized)

    def test_init_when_otel_not_installed(self):
        """When OTEL_ENABLED=true but packages missing, should gracefully fall back."""
        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            tracing_mod._initialized = False
            tracing_mod._tracer_provider = None
            # This should not raise even if opentelemetry is not installed
            tracing_mod._init_tracing()
            self.assertTrue(tracing_mod._initialized)


class TestShutdownTracing(unittest.TestCase):
    """Verify shutdown_tracing works with None provider."""

    def test_shutdown_with_no_provider(self):
        import asyncio
        tracing_mod._tracer_provider = None
        # Should not raise
        asyncio.run(tracing_mod.shutdown_tracing())


class TestNoOpSpanContext(unittest.TestCase):
    """Verify _NoOpSpanContext context manager protocol."""

    def test_enter_returns_noop_span(self):
        ctx = tracing_mod._NoOpSpanContext()
        span = ctx.__enter__()
        self.assertIsInstance(span, tracing_mod._NoOpSpan)

    def test_exit_does_nothing(self):
        ctx = tracing_mod._NoOpSpanContext()
        ctx.__enter__()
        # Should not raise
        ctx.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
