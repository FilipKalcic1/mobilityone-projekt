"""Tests for PII filter patterns (services/pii_filter.py)."""

import pytest
from services.pii_filter import _scrub_pii


class TestPhonePatterns:
    def test_croatian_international(self):
        assert "[PHONE]" in _scrub_pii("Call +385991234567")

    def test_croatian_mobile(self):
        assert "[PHONE]" in _scrub_pii("Mob: 0991234567")

    def test_croatian_with_dashes(self):
        assert "[PHONE]" in _scrub_pii("Tel: +385-99-123-4567")

    def test_generic_international(self):
        assert "[PHONE]" in _scrub_pii("Number: +49 151 12345678")

    def test_no_false_positive_on_short_numbers(self):
        result = _scrub_pii("Order #12345")
        assert "[PHONE]" not in result


class TestEmailPattern:
    def test_standard_email(self):
        assert _scrub_pii("user@example.com") == "[EMAIL]"

    def test_email_with_dots(self):
        assert "[EMAIL]" in _scrub_pii("first.last@company.hr")

    def test_no_false_positive(self):
        assert "[EMAIL]" not in _scrub_pii("not an email")


class TestOIBPattern:
    def test_oib_with_keyword(self):
        result = _scrub_pii("oib: 12345678901")
        assert "12345678901" not in result
        assert "[OIB]" in result

    def test_oib_full_keyword(self):
        result = _scrub_pii("osobni identifikacijski 12345678901")
        assert "12345678901" not in result

    def test_no_false_positive_without_keyword(self):
        """11-digit number without OIB keyword should NOT be scrubbed."""
        result = _scrub_pii("ID: 12345678901")
        assert "12345678901" in result


class TestIPv4Pattern:
    def test_standard_ip(self):
        assert _scrub_pii("Server: 192.168.1.1") == "Server: [IP]"

    def test_edge_ip(self):
        assert "[IP]" in _scrub_pii("255.255.255.255")

    def test_no_false_positive_on_version(self):
        """Version numbers like 1.2.3 should not match (only 3 octets)."""
        result = _scrub_pii("Version 1.2.3")
        assert "[IP]" not in result


class TestIBANPattern:
    def test_croatian_iban(self):
        """IBAN is matched; phone pattern may also match digits within."""
        result = _scrub_pii("IBAN: HR1210010051863000160")
        # At minimum, original number should be scrubbed
        assert "1210010051863000160" not in result

    def test_iban_with_spaces(self):
        result = _scrub_pii("HR12 1001 0051 8630 0016 0")
        assert "[IBAN]" in result

    def test_german_iban(self):
        result = _scrub_pii("DE89 3704 0044 0532 0130 00")
        assert "[IBAN]" in result


class TestCombinedScrubbing:
    def test_multiple_pii_types(self):
        text = "User user@test.com called +385991234567 from 10.0.0.1"
        result = _scrub_pii(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[IP]" in result
        assert "user@test.com" not in result

    def test_no_pii_passes_through(self):
        text = "Normal log message with no PII"
        assert _scrub_pii(text) == text
