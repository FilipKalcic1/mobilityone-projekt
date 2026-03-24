"""Tests for GDPR masking service - PII detection and anonymization."""

import pytest
from services.gdpr_masking import (
    GDPRMaskingService,
    PIIType,
    PIIMatch,
    MaskingResult,
    reset_masking_service,
)


@pytest.fixture
def masker():
    """Create masker with a known salt for deterministic hashing."""
    return GDPRMaskingService(
        use_hashing=True,
        hash_salt="test-salt-for-gdpr-unit-tests-deterministic-output"
    )


@pytest.fixture
def masker_no_hash():
    """Create masker with hashing disabled."""
    return GDPRMaskingService(
        use_hashing=False,
        hash_salt="test-salt-for-gdpr-unit-tests-deterministic-output"
    )


class TestInit:
    def test_init_with_salt(self, masker):
        assert masker.hash_salt == "test-salt-for-gdpr-unit-tests-deterministic-output"
        assert masker.use_hashing is True

    def test_init_no_hash(self, masker_no_hash):
        assert masker_no_hash.use_hashing is False

    def test_patterns_compiled(self, masker):
        assert PIIType.PHONE in masker._compiled_patterns
        assert PIIType.EMAIL in masker._compiled_patterns
        assert PIIType.OIB in masker._compiled_patterns
        assert len(masker._compiled_patterns[PIIType.PHONE]) > 0

    def test_short_salt_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            GDPRMaskingService(use_hashing=True, hash_salt="short")


class TestHashValue:
    def test_hash_deterministic(self, masker):
        h1 = masker._hash_value("test@email.com")
        h2 = masker._hash_value("test@email.com")
        assert h1 == h2

    def test_hash_different_inputs(self, masker):
        h1 = masker._hash_value("test1@email.com")
        h2 = masker._hash_value("test2@email.com")
        assert h1 != h2

    def test_hash_length_default(self, masker):
        h = masker._hash_value("test")
        assert len(h) == 12

    def test_hash_custom_length(self, masker):
        h = masker._hash_value("test", length=8)
        assert len(h) == 8

    def test_hash_disabled(self, masker_no_hash):
        h = masker_no_hash._hash_value("test")
        assert h == "X" * 12


class TestMaskValue:
    def test_mask_oib(self, masker):
        masked = masker._mask_value(PIIType.OIB, "12345678901")
        assert masked == "[OIB-MASKED]"

    def test_mask_address(self, masker):
        masked = masker._mask_value(PIIType.ADDRESS, "Ilica 123")
        assert masked == "[ADDRESS-MASKED]"

    def test_mask_email_has_hash(self, masker):
        masked = masker._mask_value(PIIType.EMAIL, "test@example.com")
        assert masked.startswith("[EMAIL-")
        assert masked.endswith("]")

    def test_mask_phone_has_hash(self, masker):
        masked = masker._mask_value(PIIType.PHONE, "+385912345678")
        assert masked.startswith("[PHONE-")

    def test_mask_credit_card_last4(self, masker):
        masked = masker._mask_value(PIIType.CREDIT_CARD, "4111111111111111")
        assert "1111" in masked
        assert masked.startswith("[CARD-****")


class TestValidateOIB:
    def test_valid_oib(self, masker):
        assert masker._validate_oib("94577403194") is True

    def test_invalid_oib_checksum(self, masker):
        assert masker._validate_oib("12345678900") is False

    def test_all_same_digits_rejected(self, masker):
        assert masker._validate_oib("11111111111") is False

    def test_too_short(self, masker):
        assert masker._validate_oib("1234567890") is False

    def test_non_digits(self, masker):
        assert masker._validate_oib("1234567890a") is False


class TestValidateCreditCard:
    def test_valid_visa(self, masker):
        assert masker._validate_credit_card("4111111111111111") is True

    def test_invalid_luhn(self, masker):
        assert masker._validate_credit_card("4111111111111112") is False

    def test_too_short(self, masker):
        assert masker._validate_credit_card("411111") is False

    def test_fake_number_rejected(self, masker):
        assert masker._validate_credit_card("0000000000000000") is False

    def test_with_separators(self, masker):
        assert masker._validate_credit_card("4111-1111-1111-1111") is True


class TestDetectPII:
    def test_detect_email(self, masker):
        matches = masker.detect_pii("Contact: user@example.com")
        types = [m.pii_type for m in matches]
        assert PIIType.EMAIL in types

    def test_detect_croatian_phone(self, masker):
        matches = masker.detect_pii("Nazovite 091 234 5678")
        types = [m.pii_type for m in matches]
        assert PIIType.PHONE in types

    def test_detect_international_phone(self, masker):
        matches = masker.detect_pii("Telefon: +385 91 234 5678")
        types = [m.pii_type for m in matches]
        assert PIIType.PHONE in types

    def test_detect_ipv4(self, masker):
        matches = masker.detect_pii("Server IP: 192.168.1.100")
        types = [m.pii_type for m in matches]
        assert PIIType.IP_ADDRESS in types

    def test_detect_iban(self, masker):
        matches = masker.detect_pii("IBAN: HR1234567890123456789")
        types = [m.pii_type for m in matches]
        assert PIIType.IBAN in types

    def test_empty_text_returns_empty(self, masker):
        assert masker.detect_pii("") == []

    def test_no_pii_returns_empty(self, masker):
        matches = masker.detect_pii("Hello, this is a normal message.")
        assert len(matches) == 0

    def test_match_has_position_info(self, masker):
        matches = masker.detect_pii("Email: user@example.com here")
        if matches:
            assert matches[0].start >= 0
            assert matches[0].end > matches[0].start


class TestMaskPII:
    def test_mask_email_in_text(self, masker):
        result = masker.mask_pii("Email: user@example.com")
        assert "user@example.com" not in result.masked_text
        assert "[EMAIL-" in result.masked_text
        assert result.has_pii()
        assert result.pii_count == 1

    def test_mask_phone_in_text(self, masker):
        result = masker.mask_pii("Zovi me na 091 234 5678")
        assert "091 234 5678" not in result.masked_text
        assert "[PHONE-" in result.masked_text

    def test_empty_text(self, masker):
        result = masker.mask_pii("")
        assert result.masked_text == ""
        assert result.pii_count == 0
        assert result.has_pii() is False

    def test_no_pii_unchanged(self, masker):
        text = "Dobar dan, kako ste?"
        result = masker.mask_pii(text)
        assert result.masked_text == text
        assert result.pii_count == 0

    def test_multiple_pii_masked(self, masker):
        text = "Email: a@b.com, Tel: +385 91 234 5678"
        result = masker.mask_pii(text)
        assert "a@b.com" not in result.masked_text
        assert result.pii_count >= 1

    def test_preserves_surrounding_text(self, masker):
        result = masker.mask_pii("Pozdrav user@example.com hvala")
        assert result.masked_text.startswith("Pozdrav")
        assert result.masked_text.endswith("hvala")

    def test_original_text_preserved(self, masker):
        text = "Email: user@example.com"
        result = masker.mask_pii(text)
        assert result.original_text == text


class TestMaskPIIAsync:
    async def test_async_same_as_sync(self, masker):
        sync_result = masker.mask_pii("Email: user@example.com")
        async_result = await masker.mask_pii_async("Email: user@example.com")
        assert sync_result.masked_text == async_result.masked_text


class TestMaskDict:
    def test_mask_default_fields(self, masker):
        data = {"message": "Email: user@example.com", "id": 123}
        result = masker.mask_dict(data)
        assert "user@example.com" not in result["message"]
        assert result["id"] == 123

    def test_nested_dict(self, masker):
        data = {"outer": {"message": "Call 091 234 5678"}}
        result = masker.mask_dict(data)
        assert "091 234 5678" not in result["outer"]["message"]

    def test_custom_fields(self, masker):
        data = {"custom_field": "user@test.com", "message": "safe text"}
        result = masker.mask_dict(data, fields_to_mask=["custom_field"])
        assert "user@test.com" not in result["custom_field"]
        assert result["message"] == "safe text"

    def test_empty_dict(self, masker):
        assert masker.mask_dict({}) == {}

    def test_non_dict_returns_as_is(self, masker):
        assert masker.mask_dict(None) is None

    def test_max_depth_stops_recursion(self, masker):
        deep = {"a": {"b": {"message": "user@test.com"}}}
        result = masker.mask_dict(deep, max_depth=1)
        assert result["a"]["b"]["message"] == "user@test.com"

    def test_list_in_dict(self, masker):
        data = {"items": [{"message": "user@test.com"}]}
        result = masker.mask_dict(data)
        assert "user@test.com" not in result["items"][0]["message"]

    def test_non_string_field_unchanged(self, masker):
        data = {"message": 12345, "count": 10}
        result = masker.mask_dict(data)
        assert result["message"] == 12345


class TestMaskList:
    def test_mask_strings_in_list(self, masker):
        data = {"items": ["user@test.com", "safe text"]}
        result = masker.mask_dict(data)
        assert "user@test.com" not in str(result["items"])

    def test_nested_list(self, masker):
        items = [{"message": "user@test.com"}]
        result = masker._mask_list(items, None, 10, 0)
        assert "user@test.com" not in result[0]["message"]

    def test_list_max_depth(self, masker):
        items = [{"message": "user@test.com"}]
        result = masker._mask_list(items, None, 10, 10)
        assert result[0]["message"] == "user@test.com"

    def test_nested_list_in_list(self, masker):
        items = [["user@test.com"]]
        result = masker._mask_list(items, None, 10, 0)
        assert "user@test.com" not in str(result)

    def test_non_maskable_items(self, masker):
        items = [123, True, None]
        result = masker._mask_list(items, None, 10, 0)
        assert result == [123, True, None]


class TestMaskLogMessage:
    def test_mask_email_in_log(self, masker):
        msg = "User user@example.com logged in"
        result = masker.mask_log_message(msg)
        assert "user@example.com" not in result

    def test_empty_message(self, masker):
        assert masker.mask_log_message("") == ""
        assert masker.mask_log_message(None) is None


class TestSingleton:
    def test_reset_masking_service(self):
        import services.gdpr_masking as mod
        mod._masking_service = "something"
        reset_masking_service()
        assert mod._masking_service is None
