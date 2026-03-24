"""
Unit tests for UserContextManager.

Tests the centralized context management system.
"""

import pytest
from services.context import (
    UserContextManager,
    VehicleContext,
    MissingContextError,
    VehicleSelectionRequired,
    InvalidContextError,
    get_missing_param_prompt,
    get_multiple_missing_prompts,
)


class TestUserContextManager:
    """Tests for UserContextManager class."""

    def test_full_context(self):
        """Test with complete context data."""
        ctx_dict = {
            "person_id": "12345678-1234-1234-1234-123456789012",
            "phone": "+385991234567",
            "tenant_id": "87654321-4321-4321-4321-210987654321",
            "display_name": "Test User",
            "vehicle": {
                "Id": "aabbccdd-1111-2222-3333-444455556666",
                "LicencePlate": "ZG-1234-AB",
                "FullVehicleName": "Skoda Octavia",
                "Mileage": 45000,
            },
        }

        ctx = UserContextManager(ctx_dict)

        assert ctx.person_id == "12345678-1234-1234-1234-123456789012"
        assert ctx.phone == "+385991234567"
        assert ctx.display_name == "Test User"
        assert ctx.vehicle_id == "aabbccdd-1111-2222-3333-444455556666"
        assert ctx.vehicle_plate == "ZG-1234-AB"
        assert ctx.vehicle_name == "Skoda Octavia"
        assert ctx.has_vehicle() is True
        assert ctx.is_guest is False

    def test_missing_vehicle(self):
        """Test with missing vehicle context."""
        ctx_dict = {
            "person_id": "12345678-1234-1234-1234-123456789012",
            "phone": "+385991234567",
        }

        ctx = UserContextManager(ctx_dict)

        assert ctx.has_vehicle() is False
        assert ctx.vehicle_id is None
        assert ctx.vehicle_plate is None

    def test_require_vehicle_id_raises(self):
        """Test that require_vehicle_id raises MissingContextError."""
        ctx_dict = {"person_id": "12345678-1234-1234-1234-123456789012"}

        ctx = UserContextManager(ctx_dict)

        with pytest.raises(MissingContextError) as exc_info:
            ctx.require_vehicle_id()

        assert exc_info.value.param == "vehicle"
        assert "vozilo" in exc_info.value.prompt_hr.lower()

    def test_require_person_id_raises(self):
        """Test that require_person_id raises MissingContextError for guest."""
        ctx_dict = {"phone": "+385991234567"}

        ctx = UserContextManager(ctx_dict)

        with pytest.raises(MissingContextError) as exc_info:
            ctx.require_person_id()

        assert exc_info.value.param == "person_id"
        assert "prijavljeni" in exc_info.value.prompt_hr.lower()

    def test_invalid_uuid_ignored(self):
        """Test that invalid UUID is treated as None."""
        ctx_dict = {
            "person_id": "not-a-valid-uuid",
            "phone": "+385991234567",
        }

        ctx = UserContextManager(ctx_dict)

        assert ctx.person_id is None  # Invalid UUID should return None
        assert ctx.is_guest is True

    def test_validation(self):
        """Test validation method."""
        ctx_dict = {
            "person_id": "12345678-1234-1234-1234-123456789012",
            "vehicle": {
                "Id": "invalid-id",  # Invalid UUID
            },
        }

        ctx = UserContextManager(ctx_dict)
        issues = ctx.validate()

        assert len(issues) > 0
        assert any("vehicle ID" in issue for issue in issues)


class TestVehicleContext:
    """Tests for VehicleContext dataclass."""

    def test_from_dict_with_various_field_names(self):
        """Test VehicleContext handles various field name conventions."""
        # Test with Id
        v1 = VehicleContext.from_dict({"Id": "uuid-123", "LicencePlate": "ZG-1234"})
        assert v1.id == "uuid-123"
        assert v1.plate == "ZG-1234"

        # Test with VehicleId
        v2 = VehicleContext.from_dict({"VehicleId": "uuid-456"})
        assert v2.id == "uuid-456"

        # Test with RegistrationNumber
        v3 = VehicleContext.from_dict({"RegistrationNumber": "ST-5678"})
        assert v3.plate == "ST-5678"

    def test_display_string(self):
        """Test display_string formatting."""
        vehicle = VehicleContext.from_dict({
            "LicencePlate": "ZG-1234-AB",
            "FullVehicleName": "Skoda Octavia",
        })

        display = vehicle.display_string()
        assert "ZG-1234-AB" in display
        assert "Skoda Octavia" in display


class TestParamPrompts:
    """Tests for param_prompts functions."""

    def test_get_missing_param_prompt_exact(self):
        """Test exact param name match."""
        prompt = get_missing_param_prompt("VehicleId")
        assert "vozilo" in prompt.lower()
        assert "tablice" in prompt.lower()

    def test_get_missing_param_prompt_case_insensitive(self):
        """Test case-insensitive match."""
        prompt = get_missing_param_prompt("vehicleid")
        assert "vozilo" in prompt.lower()

    def test_get_missing_param_prompt_fallback(self):
        """Test fallback for unknown params."""
        prompt = get_missing_param_prompt("SomeUnknownParam")
        assert "Trebam" in prompt or "SomeUnknownParam" in prompt

    def test_get_multiple_missing_prompts(self):
        """Test multiple param prompt generation."""
        prompts = get_multiple_missing_prompts(["VehicleId", "from", "PersonId"])
        assert "informacija" in prompts.lower()
        # Should contain bullet points
        assert "•" in prompts or "*" in prompts

    def test_single_param_prompt(self):
        """Test single param returns direct prompt."""
        prompt = get_multiple_missing_prompts(["VehicleId"])
        assert "vozilo" in prompt.lower()
        assert "•" not in prompt  # No bullet for single param


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
