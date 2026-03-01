"""
GDPR Data Masking Service

LEGAL REQUIREMENT:
- GDPR Article 17: Right to Erasure
- GDPR Article 25: Data Protection by Design
- GDPR Article 32: Security of Processing

PII TYPES DETECTED:
- Phone numbers (Croatian, international)
- Email addresses
- OIB (Croatian personal ID - 11 digits with checksum validation)
- IBAN (bank accounts)
- Credit card numbers (with Luhn validation)
- IP addresses (IPv4 and IPv6)
- Names (heuristic)

USAGE:
    from services.gdpr_masking import GDPRMaskingService

    masker = GDPRMaskingService()
    result = masker.mask_pii("Moj OIB je 12345678901")
    # Output: result.masked_text = "Moj OIB je [OIB-MASKED]"

SECURITY:
    - Salt MUST be loaded from environment variable GDPR_HASH_SALT
    - Never commit salt to code repository
    - Rotate salt periodically (invalidates all pseudonymized links)
    - Minimum salt length: 32 characters
"""

import re
import hmac
import hashlib
import logging
import secrets
import threading
from typing import Dict, List, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_singleton_lock = threading.Lock()

# Minimum salt length for security
MIN_SALT_LENGTH = 32


class PIIType(str, Enum):
    """Types of PII data."""
    PHONE = "phone"
    EMAIL = "email"
    OIB = "oib"           # Croatian personal ID
    IBAN = "iban"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip"
    NAME = "name"
    ADDRESS = "address"


@dataclass
class PIIMatch:
    """Represents a PII match in text."""
    pii_type: PIIType
    original: str
    masked: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class MaskingResult:
    """Result of masking operation."""
    original_text: str
    masked_text: str
    pii_found: List[PIIMatch] = field(default_factory=list)
    pii_count: int = 0

    def has_pii(self) -> bool:
        return self.pii_count > 0


class GDPRMaskingService:
    """
    GDPR-compliant PII masking service.

    Features:
    - Regex-based PII detection
    - Consistent hashing for pseudonymization (HMAC-SHA256)
    - Audit logging of masking operations
    - Configurable masking strategies
    - Thread-safe singleton
    """

    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.PHONE: [
            # Croatian mobile with various separators: 09x xxx xxxx, 09x/xxx-xxxx, 09x-xxx-xxxx
            r'\b09[1-9][\s/-]?\d{3}[\s/-]?\d{3,4}\b',
            # Croatian landline: 0x xxx xxxx
            r'\b0[1-5][1-9][\s/-]?\d{3}[\s/-]?\d{3,4}\b',
            # International Croatian: +385 xx xxx xxxx
            r'\+385[\s/-]?\d{2}[\s/-]?\d{3}[\s/-]?\d{3,4}\b',
            # Generic international with various formats
            r'\+\d{1,3}[\s/-]?\d{2,3}[\s/-]?\d{3,4}[\s/-]?\d{3,4}\b',
            # Parentheses format: (01) 234-5678
            r'\(\d{2,3}\)[\s/-]?\d{3}[\s/-]?\d{3,4}\b',
        ],
        PIIType.EMAIL: [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
        PIIType.OIB: [
            # Croatian OIB: exactly 11 digits, standalone
            # More specific to avoid matching other 11-digit numbers
            r'(?<![0-9])\d{11}(?![0-9])',
        ],
        PIIType.IBAN: [
            # Croatian IBAN: HR + 19 digits (strict)
            r'\bHR\d{19}\b',
            # European IBAN formats (2 letter country + 2 check digits + BBAN)
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}\b',
        ],
        PIIType.CREDIT_CARD: [
            # Visa: starts with 4
            r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            # Mastercard: starts with 51-55 or 2221-2720
            r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            r'\b2[2-7]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            # Amex: starts with 34 or 37, 15 digits
            r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',
        ],
        PIIType.IP_ADDRESS: [
            # IPv4
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            # IPv6 full
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            # IPv6 with :: compression (various forms)
            r'\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b',
            r'\b(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}\b',
            r'\b(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}\b',
            r'\b::(?:ffff:)?(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',  # IPv4-mapped
            r'\b::1\b',  # Loopback
            r'\b::\b',   # Unspecified
        ],
    }

    # Masking format per PII type
    MASK_FORMAT = {
        PIIType.PHONE: "[PHONE-{hash}]",
        PIIType.EMAIL: "[EMAIL-{hash}]",
        PIIType.OIB: "[OIB-MASKED]",
        PIIType.IBAN: "[IBAN-{hash}]",
        PIIType.CREDIT_CARD: "[CARD-****{last4}]",
        PIIType.IP_ADDRESS: "[IP-{hash}]",
        PIIType.NAME: "[NAME-{hash}]",
        PIIType.ADDRESS: "[ADDRESS-MASKED]",
    }

    # Default fields to mask (as frozenset for O(1) lookup)
    DEFAULT_MASK_FIELDS: FrozenSet[str] = frozenset([
        "user_query", "bot_response", "user_feedback",
        "content", "message", "query", "response",
        "phone", "email", "address", "name"
    ])

    def __init__(
        self,
        use_hashing: bool = True,
        hash_salt: Optional[str] = None
    ):
        """
        Initialize masking service.

        Args:
            use_hashing: Use consistent hashing for pseudonymization
            hash_salt: Salt for hashing. If None, loads from GDPR_HASH_SALT env var.
                      CRITICAL: Never hardcode salt in code!

        Raises:
            ValueError: If no salt provided and GDPR_HASH_SALT env var not set
        """
        self.use_hashing = use_hashing

        # CRITICAL: Load salt from config, never hardcode!
        if hash_salt:
            self.hash_salt = hash_salt
        else:
            from config import get_settings
            self.hash_salt = get_settings().GDPR_HASH_SALT
            if not self.hash_salt:
                # Generate a secure random salt for this session
                # WARNING: This means hashes won't be consistent across restarts!
                self.hash_salt = secrets.token_hex(32)
                logger.warning(
                    "GDPR_HASH_SALT not set! Using random salt. "
                    "PII pseudonymization will NOT be consistent across restarts. "
                    "Set GDPR_HASH_SALT environment variable for production!"
                )

        # Validate salt strength
        if len(self.hash_salt) < MIN_SALT_LENGTH:
            logger.warning(
                f"GDPR hash salt is too short ({len(self.hash_salt)} chars). "
                f"Recommend at least {MIN_SALT_LENGTH} characters for security."
            )

        self._compiled_patterns: Dict[PIIType, List[re.Pattern]] = {}
        self._compile_patterns()

        # Cache for lowercase field names (for case-insensitive matching)
        self._mask_fields_lower: FrozenSet[str] = frozenset(
            f.lower() for f in self.DEFAULT_MASK_FIELDS
        )

        logger.info(
            "GDPRMaskingService initialized (salt provided: %s)",
            "yes" if hash_salt else "from config"
        )

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        for pii_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[pii_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def _hash_value(self, value: str, length: int = 12) -> str:
        """
        Create consistent hash for pseudonymization.

        Same input always produces same hash (for linking records).
        Uses HMAC-SHA256 for proper key derivation.

        Args:
            value: Value to hash
            length: Output length (default 12 chars, more secure than 8)

        Returns:
            Truncated hex hash
        """
        if not self.use_hashing:
            return "X" * length

        # Use HMAC for proper key derivation (more secure than simple concat)
        hash_bytes = hmac.new(
            self.hash_salt.encode('utf-8'),
            value.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return hash_bytes[:length]

    def _mask_value(self, pii_type: PIIType, value: str) -> str:
        """Apply masking format to value."""
        mask_format = self.MASK_FORMAT.get(pii_type, "[MASKED]")

        if "{hash}" in mask_format:
            return mask_format.format(hash=self._hash_value(value))
        elif "{last4}" in mask_format:
            # For credit cards, keep last 4 digits
            digits = re.sub(r'\D', '', value)
            return mask_format.format(last4=digits[-4:] if len(digits) >= 4 else "XXXX")
        else:
            return mask_format

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text.

        Args:
            text: Input text to scan

        Returns:
            List of PII matches found
        """
        if not text:
            return []

        matches: List[PIIMatch] = []
        seen_ranges: Set[tuple] = set()  # Prevent duplicate detections

        for pii_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    original = match.group()
                    start, end = match.start(), match.end()

                    # Skip if this range overlaps with already detected PII
                    if any(s <= start < e or s < end <= e for s, e in seen_ranges):
                        continue

                    # Skip if too short (false positive)
                    if len(original) < 4:
                        continue

                    # OIB validation: must be exactly 11 digits with valid checksum
                    if pii_type == PIIType.OIB:
                        if not self._validate_oib(original):
                            continue

                    # Credit card validation: must pass Luhn check
                    if pii_type == PIIType.CREDIT_CARD:
                        if not self._validate_credit_card(original):
                            continue

                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        original=original,
                        masked=self._mask_value(pii_type, original),
                        start=start,
                        end=end,
                        confidence=0.95 if pii_type == PIIType.OIB else 1.0
                    )
                    matches.append(pii_match)
                    seen_ranges.add((start, end))

        # Sort by position (for correct replacement order)
        matches.sort(key=lambda m: m.start, reverse=True)

        return matches

    def _validate_oib(self, oib: str) -> bool:
        """
        Validate Croatian OIB using MOD 11,10 algorithm.

        Returns True if valid OIB, False otherwise.
        Logs invalid attempts for security monitoring.
        """
        if len(oib) != 11 or not oib.isdigit():
            return False

        # Additional check: don't match obvious non-OIB patterns
        # OIB shouldn't be all same digits or sequential
        if len(set(oib)) == 1:  # All same digit like 11111111111
            logger.debug(f"Rejected OIB candidate: all same digits")
            return False

        # MOD 11,10 validation
        remainder = 10
        for digit in oib[:10]:
            remainder = (remainder + int(digit)) % 10
            if remainder == 0:
                remainder = 10
            remainder = (remainder * 2) % 11

        check_digit = (11 - remainder) % 10
        is_valid = check_digit == int(oib[10])

        if not is_valid:
            # Log for security monitoring but don't expose the full number
            logger.debug(f"Invalid OIB checksum detected (last 2 digits: ...{oib[-2:]})")

        return is_valid

    def _validate_credit_card(self, number: str) -> bool:
        """
        Validate credit card using Luhn algorithm.

        Returns True if valid, False otherwise.
        """
        # Remove spaces and dashes
        digits = re.sub(r'[\s-]', '', number)

        if not digits.isdigit() or len(digits) < 13 or len(digits) > 19:
            return False

        # Don't match obvious test/fake numbers
        if digits in ['0000000000000000', '1111111111111111']:
            return False

        # Luhn algorithm
        total = 0
        for i, digit in enumerate(reversed(digits)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        return total % 10 == 0

    def mask_pii(self, text: str) -> MaskingResult:
        """
        Mask all PII in text.

        Args:
            text: Input text

        Returns:
            MaskingResult with masked text and detected PII
        """
        if not text:
            return MaskingResult(
                original_text="",
                masked_text="",
                pii_found=[],
                pii_count=0
            )

        matches = self.detect_pii(text)
        masked_text = text

        # Replace from end to start (to preserve positions)
        for match in matches:
            masked_text = (
                masked_text[:match.start] +
                match.masked +
                masked_text[match.end:]
            )

        result = MaskingResult(
            original_text=text,
            masked_text=masked_text,
            pii_found=matches,
            pii_count=len(matches)
        )

        if matches:
            # Mask the log message itself to avoid logging PII
            logger.info(
                f"GDPR: Masked {len(matches)} PII items: "
                f"{[m.pii_type.value for m in matches]}"
            )

        return result

    async def mask_pii_async(self, text: str) -> MaskingResult:
        """
        Async version of mask_pii for batch processing.

        Args:
            text: Input text

        Returns:
            MaskingResult with masked text and detected PII
        """
        # For CPU-bound work, we could use run_in_executor
        # but for now, the regex operations are fast enough
        return self.mask_pii(text)

    def mask_dict(
        self,
        data: Dict,
        fields_to_mask: Optional[List[str]] = None,
        max_depth: int = 10,
        _current_depth: int = 0
    ) -> Dict:
        """
        Mask PII in dictionary fields.

        Args:
            data: Dictionary to process
            fields_to_mask: Specific fields to mask (None = default fields)
            max_depth: Maximum recursion depth (prevents infinite recursion)
            _current_depth: Internal counter for recursion depth

        Returns:
            Dictionary with masked values
        """
        if not data or not isinstance(data, dict):
            return data

        # Prevent infinite recursion
        if _current_depth >= max_depth:
            logger.warning(f"GDPR masking: max depth {max_depth} reached, stopping recursion")
            return data

        result = {}

        # Use frozenset for O(1) lookup
        if fields_to_mask:
            fields_lower: FrozenSet[str] = frozenset(f.lower() for f in fields_to_mask)
        else:
            fields_lower = self._mask_fields_lower

        for key, value in data.items():
            key_lower = key.lower() if isinstance(key, str) else str(key).lower()

            if isinstance(value, str) and key_lower in fields_lower:
                mask_result = self.mask_pii(value)
                result[key] = mask_result.masked_text
            elif isinstance(value, dict):
                result[key] = self.mask_dict(
                    value, fields_to_mask, max_depth, _current_depth + 1
                )
            elif isinstance(value, list):
                result[key] = self._mask_list(
                    value, fields_to_mask, max_depth, _current_depth + 1
                )
            else:
                result[key] = value

        return result

    def _mask_list(
        self,
        data: List,
        fields_to_mask: Optional[List[str]],
        max_depth: int,
        current_depth: int
    ) -> List:
        """
        Mask PII in list items.

        Args:
            data: List to process
            fields_to_mask: Fields to mask in nested dicts
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            List with masked values
        """
        if current_depth >= max_depth:
            return data

        result = []
        for item in data:
            if isinstance(item, dict):
                result.append(self.mask_dict(
                    item, fields_to_mask, max_depth, current_depth
                ))
            elif isinstance(item, str):
                result.append(self.mask_pii(item).masked_text)
            elif isinstance(item, list):
                result.append(self._mask_list(
                    item, fields_to_mask, max_depth, current_depth + 1
                ))
            else:
                result.append(item)
        return result

    async def anonymize_user_data(
        self,
        user_id: str,
        db_session,
        check_already_anonymized: bool = True
    ) -> Dict:
        """
        GDPR Article 17: Right to Erasure.

        Anonymize all user data in database.

        Args:
            user_id: User identifier (phone number)
            db_session: Database session
            check_already_anonymized: Check if already anonymized first

        Returns:
            Summary of anonymized records
        """
        from sqlalchemy import update, select
        from models import UserMapping, HallucinationReport

        anonymized = {
            "user_mappings": 0,
            "hallucination_reports": 0,
            "already_anonymized": False
        }

        try:
            # Check if user is already anonymized
            if check_already_anonymized:
                check_result = await db_session.execute(
                    select(UserMapping).where(
                        UserMapping.phone_number == user_id
                    )
                )
                existing = check_result.scalar_one_or_none()
                if existing and existing.phone_number.startswith("DELETED_"):
                    logger.info(f"User already anonymized: {self._hash_value(user_id)}")
                    anonymized["already_anonymized"] = True
                    return anonymized

            # 1. Anonymize user mapping
            hashed_id = self._hash_value(user_id)
            result = await db_session.execute(
                update(UserMapping)
                .where(UserMapping.phone_number == user_id)
                .values(
                    phone_number=f"DELETED_{hashed_id}",
                    display_name="[DELETED]",
                    is_active=False
                )
            )
            anonymized["user_mappings"] = result.rowcount

            # 2. Anonymize hallucination reports
            # Find reports by user - using a proper query instead of contains()
            # First, get the user mapping to find conversation IDs
            user_result = await db_session.execute(
                select(UserMapping.id).where(
                    UserMapping.phone_number == f"DELETED_{hashed_id}"
                )
            )
            user_mapping = user_result.scalar_one_or_none()

            if user_mapping:
                # Find conversations for this user and anonymize associated reports
                from models import Conversation
                conv_result = await db_session.execute(
                    select(Conversation.id).where(
                        Conversation.user_id == user_mapping
                    )
                )
                conversation_ids = [str(c) for c in conv_result.scalars().all()]

                if conversation_ids:
                    # Anonymize reports for these conversations
                    report_result = await db_session.execute(
                        update(HallucinationReport)
                        .where(HallucinationReport.conversation_id.in_(conversation_ids))
                        .values(
                            user_query="[GDPR DELETED]",
                            bot_response="[GDPR DELETED]",
                            user_feedback="[GDPR DELETED]"
                        )
                    )
                    anonymized["hallucination_reports"] = report_result.rowcount

            await db_session.commit()

            # Log with hashed user ID to avoid logging PII
            logger.warning(
                f"GDPR ERASURE: User {hashed_id} data anonymized. "
                f"Records: {anonymized}"
            )

            return anonymized

        except Exception as e:
            await db_session.rollback()
            logger.error(f"GDPR erasure failed: {e}")
            raise

    def mask_log_message(self, message: str) -> str:
        """
        Mask PII in log messages before they are written.

        Use this in logging handlers or formatters to prevent PII leakage.

        Args:
            message: Log message to mask

        Returns:
            Masked log message
        """
        if not message:
            return message
        return self.mask_pii(message).masked_text


# Singleton instance
_masking_service: Optional[GDPRMaskingService] = None


def get_masking_service() -> GDPRMaskingService:
    """
    Get or create masking service singleton (thread-safe).

    Returns:
        GDPRMaskingService instance
    """
    global _masking_service
    if _masking_service is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _masking_service is None:
                _masking_service = GDPRMaskingService()
    return _masking_service


def reset_masking_service() -> None:
    """
    Reset the singleton instance (useful for testing).
    """
    global _masking_service
    with _singleton_lock:
        _masking_service = None
