"""
PII-Safe Logging Filter (EU Privacy / GDPR)

Regex-scrubs ALL PII types from log output (messages + tracebacks):
- Phone numbers (Croatian + international)
- Email addresses
- Croatian OIB (11-digit personal ID numbers)
- IPv4 addresses
- IBAN numbers

Shared by main.py (API) and worker.py — single source of truth.
"""

import logging
import re

_PHONE_PATTERNS = re.compile(
    r'(?:\+?385[\s/-]?\d{2}[\s/-]?\d{3}[\s/-]?\d{3,4})'       # Croatian international
    r'|(?:09[1-9][\s/-]?\d{3}[\s/-]?\d{3,4})'                   # Croatian mobile
    r'|(?:0[1-9][\s/-]?\d{2,3}[\s/-]?\d{3}[\s/-]?\d{3,4})'     # Croatian landline (01-05x)
    r'|(?:\+\d{1,3}[\s/-]?\d{2,3}[\s/-]?\d{3,4}[\s/-]?\d{3,4})'  # Generic international
)

_EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
)

_OIB_PATTERN = re.compile(
    r'(?i)(?:oib|osobni\s+identifikacijski)[:\s]+(\d{11})\b'  # Only match OIB near keyword context
)

_IPV4_PATTERN = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
)

_IBAN_PATTERN = re.compile(
    r'\b[A-Z]{2}\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{0,4}\b'
)

# Combined pattern for efficiency — applied in one pass
_ALL_PII_PATTERNS = [
    (_PHONE_PATTERNS, '[PHONE]'),
    (_EMAIL_PATTERN, '[EMAIL]'),
    (_IBAN_PATTERN, '[IBAN]'),
    (_OIB_PATTERN, '[OIB]'),
    (_IPV4_PATTERN, '[IP]'),
]


def _scrub_pii(text: str) -> str:
    """Remove all PII patterns from a string."""
    for pattern, replacement in _ALL_PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class PIIScrubFilter(logging.Filter):
    """Scrub all PII from log messages and exception tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub the formatted message
        if record.msg and isinstance(record.msg, str):
            record.msg = _scrub_pii(record.msg)
        # Scrub args used in %-style formatting
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _scrub_pii(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _scrub_pii(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        # Scrub exception text if present
        if record.exc_text and isinstance(record.exc_text, str):
            record.exc_text = _scrub_pii(record.exc_text)
        return True
