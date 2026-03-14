"""
PII-Safe Logging Filter (EU Privacy / GDPR)

Regex-replaces phone numbers in ALL log output (messages + tracebacks).
Catches: +385 91 234 5678, 385912345678, 091-234-5678, generic international.

Shared by main.py (API) and worker.py — single source of truth.
"""

import logging
import re

_PHONE_PATTERNS = re.compile(
    r'(?:\+?385[\s/-]?\d{2}[\s/-]?\d{3}[\s/-]?\d{3,4})'    # Croatian international
    r'|(?:09[1-9][\s/-]?\d{3}[\s/-]?\d{3,4})'                # Croatian mobile
    r'|(?:0[1-5][1-9][\s/-]?\d{3}[\s/-]?\d{3,4})'            # Croatian landline
    r'|(?:\+\d{1,3}[\s/-]?\d{2,3}[\s/-]?\d{3,4}[\s/-]?\d{3,4})'  # Generic international
)


class PIIScrubFilter(logging.Filter):
    """Scrub phone numbers from log messages and exception tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Scrub the formatted message
        if record.msg and isinstance(record.msg, str):
            record.msg = _PHONE_PATTERNS.sub('***', record.msg)
        # Scrub args used in %-style formatting
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _PHONE_PATTERNS.sub('***', str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _PHONE_PATTERNS.sub('***', str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        # Scrub exception text if present
        if record.exc_text and isinstance(record.exc_text, str):
            record.exc_text = _PHONE_PATTERNS.sub('***', record.exc_text)
        return True
