"""Tests for SecurityHeadersMiddleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.security_headers import SecurityHeadersMiddleware


@pytest.fixture
def app():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestSecurityHeaders:
    def test_x_content_type_options(self, client):
        resp = client.get("/test")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/test")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_x_xss_protection(self, client):
        resp = client.get("/test")
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"

    def test_referrer_policy(self, client):
        resp = client.get("/test")
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_cache_control(self, client):
        resp = client.get("/test")
        assert "no-store" in resp.headers["Cache-Control"]
        assert "no-cache" in resp.headers["Cache-Control"]

    def test_pragma(self, client):
        resp = client.get("/test")
        assert resp.headers["Pragma"] == "no-cache"

    def test_server_header_removed(self, client):
        resp = client.get("/test")
        assert "server" not in resp.headers


class TestServerHeaderRemoval:
    """Test that server header is removed when present."""

    def test_server_header_stripped_when_present(self):
        """Middleware should remove server header if present in response."""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import Response

        class AddServerHeaderMiddleware(BaseHTTPMiddleware):
            """Middleware that adds a server header (to test removal)."""
            async def dispatch(self, request, call_next):
                response = await call_next(request)
                response.headers["server"] = "TestServer/1.0"
                return response

        app = FastAPI()
        # Middleware execution order is reverse of registration order
        # Register AddServerHeader first so it runs (adds header) before
        # SecurityHeaders runs (removes header)
        app.add_middleware(AddServerHeaderMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test-server")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/test-server")
        # Server header should be removed by SecurityHeadersMiddleware
        assert "server" not in resp.headers
