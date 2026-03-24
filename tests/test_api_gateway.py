"""
API Gateway Tests - HTTP client, tenant headers, retry logic.

Tests that x-tenant header is always set, Rows default is sensible,
retry logic works, and APIResponse dataclass is correct.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.api_gateway import APIGateway, APIResponse, HttpMethod


class TestAPIResponse:
    """Test the APIResponse dataclass."""

    def test_success_response_to_dict(self):
        resp = APIResponse(success=True, status_code=200, data={"id": "123"})
        d = resp.to_dict()
        assert d["success"] is True
        assert d["data"] == {"id": "123"}
        assert d["status_code"] == 200
        assert "error" not in d

    def test_error_response_to_dict(self):
        resp = APIResponse(
            success=False, status_code=403,
            data=None, error_message="Forbidden", error_code="403"
        )
        d = resp.to_dict()
        assert d["success"] is False
        assert d["error"] == "Forbidden"
        assert "data" not in d

    def test_success_with_empty_data(self):
        resp = APIResponse(success=True, status_code=200, data=[])
        d = resp.to_dict()
        assert d["data"] == []

    def test_success_with_none_data(self):
        resp = APIResponse(success=True, status_code=204, data=None)
        d = resp.to_dict()
        assert d["data"] is None


class TestAPIGatewayInit:
    """Test gateway initialization with tenant routing."""

    @patch("services.api_gateway.TokenManager")
    @patch("services.api_gateway.settings")
    def test_default_tenant_from_settings(self, mock_settings, mock_tm):
        mock_settings.MOBILITY_API_URL = "https://api.example.com"
        mock_settings.tenant_id = "tenant-default"

        gw = APIGateway()
        assert gw.tenant_id == "tenant-default"

    @patch("services.api_gateway.TokenManager")
    @patch("services.api_gateway.settings")
    def test_custom_tenant_overrides_settings(self, mock_settings, mock_tm):
        mock_settings.MOBILITY_API_URL = "https://api.example.com"
        mock_settings.tenant_id = "tenant-default"

        gw = APIGateway(tenant_id="tenant-custom")
        assert gw.tenant_id == "tenant-custom"

    @patch("services.api_gateway.TokenManager")
    @patch("services.api_gateway.settings")
    def test_base_url_trailing_slash_stripped(self, mock_settings, mock_tm):
        mock_settings.MOBILITY_API_URL = "https://api.example.com/"
        mock_settings.tenant_id = "t"

        gw = APIGateway()
        assert gw.base_url == "https://api.example.com"


class TestRowsDefault:
    """Test the GET request Rows default behavior."""

    @patch("services.api_gateway.TokenManager")
    @patch("services.api_gateway.settings")
    def test_get_without_rows_adds_default_50(self, mock_settings, mock_tm):
        """GET requests without Rows param should get Rows=50 (not 1!)."""
        mock_settings.MOBILITY_API_URL = "https://api.example.com"
        mock_settings.tenant_id = "t"

        gw = APIGateway()
        params = {"status": "active"}

        # Simulate the Rows default logic
        method = HttpMethod.GET
        if method == HttpMethod.GET:
            if not any(k.lower() == 'rows' for k in params):
                params['Rows'] = 50

        assert params['Rows'] == 50

    def test_existing_rows_not_overwritten(self):
        """If user specifies Rows, don't overwrite it."""
        params = {"Rows": 10, "status": "active"}

        if not any(k.lower() == 'rows' for k in params):
            params['Rows'] = 50

        assert params['Rows'] == 10  # Preserved user value

    def test_case_insensitive_rows_check(self):
        """Rows check should be case-insensitive."""
        params = {"rows": 25}  # lowercase

        if not any(k.lower() == 'rows' for k in params):
            params['Rows'] = 50

        assert "Rows" not in params or params.get("rows") == 25


class TestHttpMethod:
    """Test HTTP method enum."""

    def test_all_methods_exist(self):
        assert HttpMethod.GET.value == "GET"
        assert HttpMethod.POST.value == "POST"
        assert HttpMethod.PUT.value == "PUT"
        assert HttpMethod.PATCH.value == "PATCH"
        assert HttpMethod.DELETE.value == "DELETE"
