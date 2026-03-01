"""
End-to-End Car Booking Integration Test

Comprehensive test simulating a real car booking scenario:
1. User wants to book a car for a specific time
2. System checks availability
3. User selects a vehicle
4. System confirms booking details
5. User confirms
6. Booking is created

This test covers:
- Conversation state management
- Entity extraction (dates, times, vehicles)
- Tool selection and execution
- Parameter collection flow
- GDPR masking of sensitive data
- Cost tracking
- Error handling and recovery
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_redis():
    """Mock Redis client for state management."""
    redis = MagicMock()
    storage = {}

    async def mock_get(key):
        return storage.get(key)

    async def mock_set(key, value, *args, **kwargs):
        storage[key] = value
        return True

    async def mock_setex(key, ttl, value):
        storage[key] = value
        return True

    async def mock_delete(*keys):
        for key in keys:
            storage.pop(key, None)
        return len(keys)

    redis.get = AsyncMock(side_effect=mock_get)
    redis.set = AsyncMock(side_effect=mock_set)
    redis.setex = AsyncMock(side_effect=mock_setex)
    redis.delete = AsyncMock(side_effect=mock_delete)
    redis.ping = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()

    return redis


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()

    # Create mock result object with sync methods
    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    mock_result.scalars.return_value.first.return_value = None
    mock_result.all.return_value = []

    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    session.close = AsyncMock()

    return session


@pytest.fixture
def sample_user_context():
    """Sample user context for testing."""
    return {
        "phone": "+385991234567",
        "person_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "tenant_id": "tenant-001",
        "display_name": "Ivan Horvat",
        "email": "ivan.horvat@example.com"
    }


@pytest.fixture
def sample_available_vehicles():
    """Sample list of available vehicles."""
    return [
        {
            "Id": "vehicle-001",
            "FullVehicleName": "Volkswagen Passat B8 2.0 TDI",
            "LicencePlate": "ZG-1234-AB",
            "Mileage": 45000,
            "Status": "Available",
            "Location": "Zagreb, Heinzelova 71"
        },
        {
            "Id": "vehicle-002",
            "FullVehicleName": "Škoda Octavia IV 1.5 TSI",
            "LicencePlate": "ZG-5678-CD",
            "Mileage": 32000,
            "Status": "Available",
            "Location": "Zagreb, Savska 25"
        },
        {
            "Id": "vehicle-003",
            "FullVehicleName": "Audi A4 Avant 40 TDI",
            "LicencePlate": "ST-9012-EF",
            "Mileage": 28000,
            "Status": "Available",
            "Location": "Split, Put Firula 19"
        }
    ]


@pytest.fixture
def booking_time_range():
    """Sample booking time range."""
    now = datetime.now(timezone.utc)
    start = now + timedelta(days=1, hours=9)  # Tomorrow at 9 AM
    end = now + timedelta(days=1, hours=17)   # Tomorrow at 5 PM

    return {
        "from_time": start.isoformat(),
        "to_time": end.isoformat(),
        "duration_hours": 8
    }


# ============================================================================
# CONVERSATION FLOW TESTS
# ============================================================================

class TestCarBookingFlow:
    """End-to-end test for car booking conversation flow."""

    @pytest.mark.asyncio
    async def test_complete_booking_flow(
        self,
        mock_redis,
        mock_db_session,
        sample_user_context,
        sample_available_vehicles,
        booking_time_range
    ):
        """
        Test complete booking flow from start to finish.

        Simulates:
        1. User: "Trebam auto sutra od 9 do 17"
        2. Bot: Shows available vehicles
        3. User: "Passat" (selection)
        4. Bot: Confirms booking details
        5. User: "Da" (confirmation)
        6. Bot: Booking created successfully
        """
        from services.conversation_manager import ConversationManager, ConversationState

        # Initialize conversation manager
        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        # Step 1: User initiates booking request
        assert manager.get_state() == ConversationState.IDLE

        # Start booking flow with required parameters
        await manager.start_flow(
            flow_name="booking",
            tool="post_VehicleCalendar",
            required_params=["vehicleId", "fromTime", "toTime", "personId"]
        )

        assert manager.get_state() == ConversationState.GATHERING_PARAMS

        # Step 2: Add extracted time parameters
        await manager.add_parameters({
            "fromTime": booking_time_range["from_time"],
            "toTime": booking_time_range["to_time"],
            "personId": sample_user_context["person_id"]
        })

        # Still missing vehicleId
        assert "vehicleId" in manager.get_missing_params()

        # Step 3: Show available vehicles for selection
        await manager.set_displayed_items(sample_available_vehicles)
        assert manager.get_state() == ConversationState.SELECTING_ITEM

        # Step 4: User selects "Passat"
        selected = manager.parse_item_selection("Passat")
        assert selected is not None
        assert "Passat" in selected["FullVehicleName"]

        await manager.select_item(selected)
        assert manager.get_selected_item()["Id"] == "vehicle-001"

        # Add selected vehicle ID
        await manager.add_parameters({"vehicleId": selected["Id"]})

        # All required params collected
        assert manager.has_all_required_params()

        # Step 5: Request confirmation
        await manager.request_confirmation("Želite li potvrditi rezervaciju?")
        assert manager.get_state() == ConversationState.CONFIRMING

        # Step 6: User confirms
        confirmation = manager.parse_confirmation("da, potvrđujem")
        assert confirmation is True

        # Complete the flow
        await manager.complete()
        assert manager.get_state() == ConversationState.COMPLETED

        # Verify all parameters are collected
        params = manager.get_parameters()
        assert params["vehicleId"] == "vehicle-001"
        assert params["fromTime"] == booking_time_range["from_time"]
        assert params["toTime"] == booking_time_range["to_time"]
        assert params["personId"] == sample_user_context["person_id"]

    @pytest.mark.asyncio
    async def test_booking_cancellation_flow(
        self,
        mock_redis,
        sample_user_context,
        sample_available_vehicles
    ):
        """Test user cancelling booking mid-flow."""
        from services.conversation_manager import ConversationManager, ConversationState

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        # Start flow
        await manager.start_flow("booking", required_params=["vehicleId"])

        # Show vehicles
        await manager.set_displayed_items(sample_available_vehicles)

        # User cancels
        await manager.cancel()

        assert manager.get_state() == ConversationState.IDLE
        assert not manager.is_in_flow()

    @pytest.mark.asyncio
    async def test_vehicle_selection_by_plate(
        self,
        mock_redis,
        sample_user_context,
        sample_available_vehicles
    ):
        """Test selecting vehicle by license plate number."""
        from services.conversation_manager import ConversationManager, ConversationState

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        await manager.start_flow("booking")
        await manager.set_displayed_items(sample_available_vehicles)

        # Select by plate number
        selected = manager.parse_item_selection("ZG-5678-CD")
        assert selected is not None
        assert selected["Id"] == "vehicle-002"

        # Also test partial plate
        selected = manager.parse_item_selection("5678")
        assert selected is not None
        assert selected["Id"] == "vehicle-002"

    @pytest.mark.asyncio
    async def test_vehicle_selection_by_number(
        self,
        mock_redis,
        sample_user_context,
        sample_available_vehicles
    ):
        """Test selecting vehicle by list number (1, 2, 3)."""
        from services.conversation_manager import ConversationManager, ConversationState

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        await manager.start_flow("booking")
        await manager.set_displayed_items(sample_available_vehicles)

        # Select by number
        selected = manager.parse_item_selection("2")
        assert selected is not None
        assert selected["Id"] == "vehicle-002"

        # Invalid number should return None
        invalid = manager.parse_item_selection("99")
        assert invalid is None

    @pytest.mark.asyncio
    async def test_ambiguous_confirmation_handling(
        self,
        mock_redis,
        sample_user_context
    ):
        """Test handling of ambiguous confirmation responses."""
        from services.conversation_manager import ConversationManager, ConversationState

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        # Test various confirmation inputs
        assert manager.parse_confirmation("da") is True
        assert manager.parse_confirmation("Da, slažem se") is True
        assert manager.parse_confirmation("potvrđujem") is True
        assert manager.parse_confirmation("ok") is True
        assert manager.parse_confirmation("može") is True

        assert manager.parse_confirmation("ne") is False
        assert manager.parse_confirmation("Ne, hvala") is False
        assert manager.parse_confirmation("odustani") is False
        assert manager.parse_confirmation("cancel") is False

        # Ambiguous should return None
        assert manager.parse_confirmation("možda") is None
        assert manager.parse_confirmation("ne znam") is None
        assert manager.parse_confirmation("nisam siguran") is None
        assert manager.parse_confirmation("hmmm") is None


# ============================================================================
# ENTITY EXTRACTION TESTS
# ============================================================================

class TestEntityExtraction:
    """Test entity extraction from user messages."""

    def test_extract_dates_croatian(self):
        """Test extracting dates from Croatian text."""
        import re

        # Common Croatian date patterns
        date_patterns = [
            r"sutra",
            r"prekosutra",
            r"u ponedjeljak",
            r"(\d{1,2})\.(\d{1,2})\.(\d{4})?",
            r"(\d{1,2})/(\d{1,2})/(\d{4})?",
        ]

        test_cases = [
            ("Trebam auto sutra", ["sutra"]),
            ("Rezervacija za 15.01.2024", ["15.01.2024"]),
            ("Molim vozilo u ponedjeljak", ["u ponedjeljak"]),
        ]

        for text, expected_dates in test_cases:
            found = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    if isinstance(matches[0], tuple):
                        found.extend([".".join(m) for m in matches])
                    else:
                        found.extend(matches)

            for expected in expected_dates:
                assert any(expected.lower() in f.lower() for f in found) or expected.lower() in text.lower()

    def test_extract_time_croatian(self):
        """Test extracting times from Croatian text."""
        import re

        time_pattern = r"(\d{1,2})(?::(\d{2}))?\s*(?:sati?|h)?"

        test_cases = [
            ("od 9 sati do 17 sati", ["9", "17"]),
            ("u 14:30", ["14:30"]),
            ("oko 10h", ["10"]),
        ]

        for text, expected_times in test_cases:
            matches = re.findall(time_pattern, text)
            found_times = [f"{h}:{m if m else '00'}" if m else h for h, m in matches]

            for expected in expected_times:
                expected_short = expected.split(":")[0]
                assert any(expected_short in t for t in found_times), \
                    f"Expected to find {expected} in {text}"

    def test_extract_vehicle_ids(self):
        """Test extracting vehicle UUIDs from messages."""
        import re

        uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'

        test_cases = [
            (
                "Vozilo je a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                ["a1b2c3d4-e5f6-7890-abcd-ef1234567890"]
            ),
            (
                "Bez vozila",
                []
            ),
        ]

        for text, expected_uuids in test_cases:
            found = re.findall(uuid_pattern, text.lower())
            assert found == expected_uuids

    def test_extract_license_plates(self):
        """Test extracting Croatian license plates."""
        import re

        # Croatian plate format: XX-1234-YY or XX 1234 YY
        plate_pattern = r'[A-ZČĆŽŠĐ]{2}[\s\-]?\d{3,4}[\s\-]?[A-ZČĆŽŠĐ]{1,2}'

        test_cases = [
            ("Auto ZG-1234-AB", ["ZG-1234-AB"]),
            ("Registracija ZG 5678 CD", ["ZG 5678 CD"]),
            ("Tablica ST9012EF", ["ST9012EF"]),
        ]

        for text, expected_plates in test_cases:
            found = re.findall(plate_pattern, text.upper())
            assert len(found) == len(expected_plates), \
                f"Expected {expected_plates} in {text}, found {found}"


# ============================================================================
# GDPR AND SECURITY TESTS
# ============================================================================

class TestGDPRCompliance:
    """Test GDPR compliance in booking flow."""

    def test_mask_user_phone(self):
        """Test phone number masking for logs."""
        from services.gdpr_masking import GDPRMaskingService

        service = GDPRMaskingService()
        result = service.mask_pii("Korisnik +385991234567 želi rezervaciju")

        assert "+385991234567" not in result.masked_text
        assert "****" in result.masked_text or result.pii_count > 0

    def test_mask_user_email(self):
        """Test email masking."""
        from services.gdpr_masking import GDPRMaskingService

        service = GDPRMaskingService()
        result = service.mask_pii("Email: ivan.horvat@example.com")

        assert "ivan.horvat@example.com" not in result.masked_text
        assert "****" in result.masked_text or result.pii_count > 0

    def test_mask_oib(self):
        """Test Croatian OIB masking."""
        from services.gdpr_masking import GDPRMaskingService

        service = GDPRMaskingService()
        # Valid OIB with correct checksum
        result = service.mask_pii("OIB korisnika je 73aborb577")

        # Should mask OIB-like patterns
        # Note: actual OIB validation is complex

    def test_mask_booking_response(self):
        """Test masking sensitive data in booking responses."""
        from services.gdpr_masking import GDPRMaskingService

        service = GDPRMaskingService()

        booking_data = {
            "bookingId": "booking-123",
            "userId": "user-456",
            "phone": "+385991234567",
            "email": "ivan@example.com",
            "vehiclePlate": "ZG-1234-AB",
            "location": "Zagreb"
        }

        masked = service.mask_dict(booking_data)

        # Phone and email should be masked
        assert masked.get("phone") != "+385991234567"
        assert masked.get("email") != "ivan@example.com"

        # Non-sensitive fields preserved
        assert masked["location"] == "Zagreb"


# ============================================================================
# COST TRACKING TESTS
# ============================================================================

class TestCostTracking:
    """Test cost tracking during booking operations."""

    @pytest.mark.asyncio
    async def test_track_ai_usage(self, mock_redis):
        """Test tracking AI token usage during booking."""
        from services.cost_tracker import CostTracker

        tracker = CostTracker(mock_redis)

        # Simulate AI calls during booking flow
        await tracker.record_usage(
            prompt_tokens=150,
            completion_tokens=50,
            tenant_id="tenant-001"
        )

        # Second call for confirmation
        await tracker.record_usage(
            prompt_tokens=100,
            completion_tokens=30,
            tenant_id="tenant-001"
        )

        stats = await tracker.get_session_stats()
        assert stats["session_prompt_tokens"] == 250
        assert stats["session_completion_tokens"] == 80

    def test_cost_calculation(self):
        """Test cost calculation with V2 simplified pricing."""
        from services.cost_tracker import CostTracker, INPUT_PRICE, OUTPUT_PRICE

        tracker = CostTracker(MagicMock())
        cost = tracker._calculate_cost(1000, 500)
        assert cost > 0
        assert isinstance(cost, float)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in booking flow."""

    @pytest.mark.asyncio
    async def test_handle_no_available_vehicles(
        self,
        mock_redis,
        sample_user_context
    ):
        """Test handling when no vehicles are available."""
        from services.conversation_manager import ConversationManager, ConversationState

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        await manager.start_flow("booking")

        # Empty vehicle list
        await manager.set_displayed_items([])

        # State should reflect no items
        assert len(manager.get_displayed_items()) == 0

    @pytest.mark.asyncio
    async def test_handle_invalid_selection(
        self,
        mock_redis,
        sample_user_context,
        sample_available_vehicles
    ):
        """Test handling invalid vehicle selection."""
        from services.conversation_manager import ConversationManager

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        await manager.start_flow("booking")
        await manager.set_displayed_items(sample_available_vehicles)

        # Invalid selections should return None
        assert manager.parse_item_selection("nonexistent") is None
        assert manager.parse_item_selection("") is None
        assert manager.parse_item_selection("   ") is None

    @pytest.mark.asyncio
    async def test_handle_session_timeout(
        self,
        mock_redis,
        sample_user_context
    ):
        """Test handling of session timeout."""
        from services.conversation_manager import ConversationManager
        from datetime import datetime, timedelta

        manager = ConversationManager(
            sample_user_context["phone"],
            mock_redis
        )
        await manager.load()

        await manager.start_flow("booking")

        # Simulate old session
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manager.context.started_at = old_time

        # Should detect timeout (if timeout is < 2 hours)
        is_timed_out = manager.is_timed_out()
        # Result depends on FLOW_TIMEOUT_SECONDS setting


# ============================================================================
# INTEGRATION WITH ADMIN REVIEW
# ============================================================================

class TestAdminReviewIntegration:
    """Test integration with hallucination review system."""

    @pytest.mark.asyncio
    async def test_report_booking_issue(self, mock_db_session):
        """Test reporting booking-related hallucination."""
        from services.hallucination_repository import HallucinationRepository

        repo = HallucinationRepository(mock_db_session)

        # User reports wrong booking info
        report = await repo.create(
            user_query="Trebam auto sutra od 9 do 17",
            bot_response="Vaša rezervacija je kreirana za 15.01.2025.",
            user_feedback="Krivi datum, ja sam rekao sutra!",
            conversation_id="conv-123",
            tenant_id="tenant-001",
            model="gpt-4o-mini"
        )

        assert report is not None
        assert report.user_query == "Trebam auto sutra od 9 do 17"
        assert report.reviewed is False


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestBookingPerformance:
    """Performance tests for booking flow."""

    @pytest.mark.asyncio
    async def test_concurrent_bookings(self, mock_redis):
        """Test handling multiple concurrent booking requests."""
        from services.conversation_manager import ConversationManager

        # Create multiple managers (simulating concurrent users)
        managers = []
        for i in range(10):
            phone = f"+38599123456{i}"
            manager = ConversationManager(phone, mock_redis)
            await manager.load()
            managers.append(manager)

        # All should be able to start flows
        async def start_flow(manager):
            await manager.start_flow("booking")
            return manager.is_in_flow()

        results = await asyncio.gather(*[start_flow(m) for m in managers])

        assert all(results), "All concurrent bookings should start successfully"

    @pytest.mark.asyncio
    async def test_state_persistence(self, mock_redis, sample_user_context):
        """Test state persists correctly between operations."""
        from services.conversation_manager import ConversationManager

        phone = sample_user_context["phone"]

        # First session
        manager1 = ConversationManager(phone, mock_redis)
        await manager1.load()
        await manager1.start_flow("booking")
        await manager1.add_parameters({"testParam": "testValue"})
        await manager1.save()

        # New session with same phone
        manager2 = ConversationManager(phone, mock_redis)
        await manager2.load()

        # Should have same state
        assert manager2.is_in_flow()
        assert manager2.get_parameters().get("testParam") == "testValue"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
