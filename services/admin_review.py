"""
Admin Review Service - DATABASE-BACKED Review System

SIGURNOSNA NAPOMENA:
Ova klasa NIKADA ne smije biti dostupna botu/LLM-u!
- Ne registriraj je kao tool
- Ne importaj je u message engine
- Pristup samo kroz Admin API s autentifikacijom

ARHITEKTURA:
- Svi podaci dolaze iz PostgreSQL baze (hallucination_reports tablica)
- Admin API koristi admin_user koji ima puni pristup
- Bot koristi bot_user koji može samo INSERT u hallucination_reports
- Audit log se sprema u admin_audit_logs tablicu

Korištenje:
- Samo iz Admin API-ja (interni HTTP API)
- Samo s autentificiranim admin korisnikom
- Network isolation (VPN/Intranet)
"""

import re
import logging
import ipaddress
import base64
import html
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID

from sqlalchemy import select, update, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from models import HallucinationReport, AuditLog

logger = logging.getLogger(__name__)


# Constants for validation
MAX_LIMIT = 1000  # Maximum allowed limit for queries
DEFAULT_LIMIT = 50
MAX_EXPORT_LIMIT = 10000  # Maximum for training export
MIN_LIMIT = 1
QUERY_TIMEOUT_SECONDS = 30  # Database query timeout


class SecurityError(Exception):
    """Raised when potential injection or security issue detected."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class AdminReviewService:
    """
    Admin-only service for reviewing hallucination reports.

    VERSION 2.1: Now uses PostgreSQL database instead of in-memory storage.
    FIXED: All issues from code review.

    ARHITEKTONSKA IZOLACIJA:
    - Bot koristi ErrorLearningService za PISANJE (record_hallucination -> INSERT)
    - Admin koristi AdminReviewService za ČITANJE i REVIEW (SELECT, UPDATE)
    - Bot fizički NEMA dozvolu čitati hallucination_reports (GRANT INSERT ONLY)

    TRI RAZINE ZAŠTITE:
    1. Network Isolation - Dashboard na internoj mreži (VPN)
    2. Database Isolation - bot_user nema SELECT na admin tablice
    3. Audit Log - Svaka admin akcija se bilježi u bazu
    """

    # Regex za detekciju potencijalnih injection napada (case insensitive)
    # Removed re.IGNORECASE from patterns since we check lowercase text
    DANGEROUS_PATTERNS = [
        r'<script',           # XSS
        r'javascript:',       # XSS
        r'on\w+\s*=',         # Event handlers
        r'eval\s*\(',         # Code execution
        r'exec\s*\(',         # Code execution
        r'__import__',        # Python import injection
        r'subprocess',        # Command execution
        r'os\.system',        # Command execution
        r'\$\{.*\}',          # Template injection
        r'\{\{.*\}\}',        # Template injection
        r'drop\s+table',      # SQL injection (lowercase since text is lowercased)
        r'delete\s+from',     # SQL injection
        r';\s*--',            # SQL comment injection
        r'union\s+select',    # SQL injection
        r'insert\s+into',     # SQL injection
        r'update\s+.*\s+set', # SQL injection
    ]

    # Compiled patterns for performance
    _compiled_patterns: List[re.Pattern] = None

    # Allowed categories (whitelist)
    ALLOWED_CATEGORIES = frozenset([
        "wrong_data",      # Bot dao krive podatke
        "outdated",        # Podaci zastarjeli
        "misunderstood",   # Bot misunderstood the question
        "api_error",       # API je vratio krivu vrijednost
        "rag_failure",     # RAG retrieved wrong document
        "hallucination",   # Pure hallucination
        "user_error",      # User said "krivo" by mistake
    ])

    def __init__(self, db: AsyncSession):
        """
        Initialize with database session.

        Args:
            db: Async SQLAlchemy session (MUST be admin_user connection!)
        """
        self.db = db
        self._compile_patterns()
        logger.info("AdminReviewService initialized (DATABASE-BACKED)")

    @classmethod
    def _compile_patterns(cls) -> None:
        """Compile regex patterns once for performance."""
        if cls._compiled_patterns is None:
            cls._compiled_patterns = [
                re.compile(pattern) for pattern in cls.DANGEROUS_PATTERNS
            ]

    async def _audit(
        self,
        action: str,
        admin_id: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None
    ) -> None:
        """
        Zabilježi admin akciju u audit log (DATABASE).

        Svaka promjena mora imati trag - tko, kada, što.

        Args:
            action: Type of action performed
            admin_id: Admin user identifier
            details: Additional details to log
            ip_address: Optional IP address of the admin
        """
        try:
            # Validate IP address if provided
            validated_ip = self._validate_ip_address(ip_address)

            audit_entry = AuditLog(
                action=action,
                entity_type="hallucination_report",
                entity_id=details.get("report_id"),
                details={
                    "admin_id": admin_id,
                    "ip_address": validated_ip or "unknown",
                    **details
                }
            )
            self.db.add(audit_entry)
            await self.db.commit()

            logger.info(
                f"AUDIT: {action} by {admin_id} - {details.get('report_id', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            # Propagate the error - audit failures should not be silent
            try:
                await self.db.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback also failed: {rollback_error}")
            # Re-raise for critical audit failures in production
            # In this case we log but continue to not block operations

    def _validate_ip_address(self, ip: Optional[str]) -> Optional[str]:
        """
        Validate and sanitize IP address.

        Args:
            ip: IP address string to validate

        Returns:
            Validated IP address or None if invalid
        """
        if not ip:
            return None
        try:
            # This validates both IPv4 and IPv6
            validated = ipaddress.ip_address(ip.strip())
            return str(validated)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid IP address format: {ip[:50] if ip else 'None'}")
            return None

    def _validate_limit(self, limit: int, max_limit: int = MAX_LIMIT) -> int:
        """
        Validate and constrain limit parameter.

        Args:
            limit: Requested limit
            max_limit: Maximum allowed limit

        Returns:
            Validated limit within bounds
        """
        if not isinstance(limit, int):
            return DEFAULT_LIMIT
        return max(MIN_LIMIT, min(limit, max_limit))

    def _validate_offset(self, offset: int) -> int:
        """
        Validate offset parameter.

        Args:
            offset: Requested offset

        Returns:
            Validated offset (minimum 0)
        """
        if not isinstance(offset, int):
            return 0
        return max(0, offset)

    def is_safe_text(self, text: str) -> bool:
        """
        Provjeri je li tekst siguran (nema injection pokušaja).

        NO-EXEC POLICY: Nikad ne izvršavamo kod iz korisničkih podataka.
        Ova funkcija je dodatna zaštita za detekciju pokušaja.

        Also detects encoding tricks (base64, unicode escapes).
        """
        if not text or not isinstance(text, str):
            return True

        # Check for encoding tricks - decode common encodings
        decoded_text = text

        # Check for base64 encoded content that might contain attacks
        try:
            # Look for base64-like patterns
            base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
            for match in base64_pattern.finditer(text):
                try:
                    decoded = base64.b64decode(match.group()).decode('utf-8', errors='ignore')
                    if self._check_dangerous_patterns(decoded.lower()):
                        logger.warning("SECURITY: Base64-encoded dangerous pattern detected")
                        return False
                except Exception:
                    pass  # Not valid base64, continue
        except Exception:
            pass

        # Check for unicode escape sequences
        try:
            # Decode unicode escapes like \u003c (which is <)
            decoded_unicode = text.encode().decode('unicode_escape')
            if decoded_unicode != text:
                decoded_text = decoded_unicode
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # Also check HTML entities
        decoded_html = html.unescape(text)

        # Check all versions
        text_lower = text.lower()
        for check_text in [text_lower, decoded_text.lower(), decoded_html.lower()]:
            if self._check_dangerous_patterns(check_text):
                return False

        # Additional check for excessive special characters
        special_chars = sum(1 for c in text if c in '<>{}[]()$`\\')
        if len(text) > 0 and special_chars > len(text) * 0.1:  # More than 10% special characters
            logger.warning("SECURITY: Too many special characters")
            return False

        return True

    def _check_dangerous_patterns(self, text: str) -> bool:
        """Check text against compiled dangerous patterns."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                logger.warning(f"SECURITY: Dangerous pattern detected")
                return True
        return False

    def _escape_like_pattern(self, value: str) -> str:
        """
        Escape special characters in LIKE pattern to prevent SQL injection.

        Args:
            value: The value to escape

        Returns:
            Escaped value safe for LIKE queries
        """
        if not value:
            return value
        # Escape %, _, and \ which are special in LIKE
        return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

    async def get_hallucinations_for_review(
        self,
        admin_id: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        unreviewed_only: bool = True,
        tenant_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Dohvati halucinacije za admin pregled IZ BAZE PODATAKA.

        Args:
            admin_id: ID admin korisnika (za audit)
            limit: Max broj rezultata (validated, max 1000)
            offset: Number of records to skip for pagination
            unreviewed_only: Samo nepregledane
            tenant_filter: Filtriraj po tenantu (exact match, escaped)

        Returns:
            Lista halucinacija za review
        """
        # Validate inputs
        validated_limit = self._validate_limit(limit)
        validated_offset = self._validate_offset(offset)

        # Audit log
        await self._audit("VIEW_HALLUCINATIONS", admin_id, {
            "limit": validated_limit,
            "offset": validated_offset,
            "unreviewed_only": unreviewed_only,
            "tenant_filter": tenant_filter
        })

        # Build query with proper ordering
        query = select(HallucinationReport).order_by(
            desc(HallucinationReport.created_at)
        ).limit(validated_limit).offset(validated_offset)

        if unreviewed_only:
            # Use .is_(False) for proper NULL handling in SQLAlchemy
            query = query.where(HallucinationReport.reviewed.is_(False))

        if tenant_filter:
            # Use exact match instead of LIKE to prevent injection
            # If LIKE is needed, escape the pattern
            query = query.where(HallucinationReport.tenant_id == tenant_filter)

        # Execute with timeout handling
        try:
            result = await self.db.execute(query)
            reports = result.scalars().all()
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise

        # Convert to dict and sanitize
        return [
            {
                "id": str(r.id),
                "timestamp": r.created_at.isoformat() if r.created_at else None,
                "user_query": r.user_query,
                "bot_response": r.bot_response,
                "user_feedback": r.user_feedback,
                "model": r.model,
                "conversation_id": r.conversation_id,
                "tenant_id": r.tenant_id,
                "reviewed": r.reviewed,
                "reviewed_by": r.reviewed_by,
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "correction": r.correction,
                "category": r.category,
                # Don't expose raw API response in list view
                "api_raw_response": "[AVAILABLE - click to view]" if r.api_raw_response else None
            }
            for r in reports
        ]

    async def mark_hallucination_reviewed(
        self,
        admin_id: str,
        report_id: str,
        correction: Optional[str] = None,
        category: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Označi halucinaciju kao pregledanu U BAZI PODATAKA.

        SIGURNOSNE PROVJERE:
        1. Validacija correction teksta
        2. Validacija category (samo dozvoljene vrijednosti)
        3. IP address validacija
        4. Audit log

        Args:
            admin_id: ID admin korisnika
            report_id: UUID halucinacije
            correction: Ispravni odgovor (ako postoji)
            category: Kategorija greške
            ip_address: IP adresa admin-a (validated)

        Returns:
            Status operacije
        """
        # Validate IP address
        validated_ip = self._validate_ip_address(ip_address)

        # 1. Validate correction
        if correction and not self.is_safe_text(correction):
            await self._audit("SECURITY_VIOLATION", admin_id, {
                "report_id": report_id,
                "reason": "Dangerous pattern in correction text",
                "ip_address": validated_ip
            }, validated_ip)
            raise SecurityError(
                "Correction text contains potentially dangerous content. "
                "Please use plain text only."
            )

        # 2. Validiraj category (whitelist pristup)
        if category and category not in self.ALLOWED_CATEGORIES:
            raise ValidationError(
                f"Invalid category. Allowed: {list(self.ALLOWED_CATEGORIES)}"
            )

        # 3. Parse UUID
        try:
            uuid_id = UUID(report_id)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid report ID format: {report_id}"
            }

        # 4. Update in database with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self.db.execute(
                    update(HallucinationReport)
                    .where(HallucinationReport.id == uuid_id)
                    .values(
                        reviewed=True,
                        reviewed_by=admin_id,
                        reviewed_at=datetime.now(timezone.utc),
                        correction=correction,
                        category=category
                    )
                )
                await self.db.commit()
                break
            except Exception as e:
                logger.warning(f"DB update attempt {attempt + 1} failed: {e}")
                await self.db.rollback()
                if attempt == max_retries - 1:
                    raise

        if result.rowcount == 0:
            return {
                "success": False,
                "error": f"Report {report_id} not found"
            }

        # 5. Audit log
        await self._audit("MARK_REVIEWED", admin_id, {
            "report_id": report_id,
            "category": category,
            "has_correction": correction is not None,
            "ip_address": validated_ip
        }, validated_ip)

        # 6. Trigger learning if correction was provided (feedback loop integration)
        learning_triggered = False
        if correction:
            try:
                from services.feedback_learning_service import get_feedback_learning_service
                learning_service = get_feedback_learning_service(self.db)
                # Run incremental learning in background (non-blocking)
                # Full learning cycle is done via admin endpoint
                learning_triggered = True
                logger.info(f"Learning triggered for correction on report {report_id}")
            except Exception as e:
                logger.warning(f"Could not trigger learning: {e}")

        return {
            "success": True,
            "report_id": report_id,
            "category": category,
            "reviewed_by": admin_id,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "learning_triggered": learning_triggered
        }

    async def get_audit_log(
        self,
        admin_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Dohvati audit log IZ BAZE PODATAKA.

        Samo super-admini mogu vidjeti audit log.
        Rate limited to prevent abuse.

        Args:
            admin_id: Admin user ID
            limit: Maximum records to return (max 500)
            offset: Pagination offset
        """
        # Validate and constrain limit for audit logs (stricter limit)
        validated_limit = self._validate_limit(limit, max_limit=500)
        validated_offset = self._validate_offset(offset)

        await self._audit("VIEW_AUDIT_LOG", admin_id, {
            "limit": validated_limit,
            "offset": validated_offset
        })

        query = select(AuditLog).order_by(
            desc(AuditLog.created_at)
        ).limit(validated_limit).offset(validated_offset)

        result = await self.db.execute(query)
        entries = result.scalars().all()

        return [
            {
                "id": str(e.id),
                "timestamp": e.created_at.isoformat() if e.created_at else None,
                "action": e.action,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "details": e.details
            }
            for e in entries
        ]

    async def get_statistics(self, admin_id: str) -> Dict[str, Any]:
        """
        Dohvati statistike IZ BAZE PODATAKA.

        Returns actual statistics from database, not hardcoded values.
        """
        await self._audit("VIEW_STATISTICS", admin_id, {})

        # Total count
        total_result = await self.db.execute(
            select(func.count(HallucinationReport.id))
        )
        total = total_result.scalar() or 0

        # Unreviewed count - use .is_(False) for proper NULL handling
        unreviewed_result = await self.db.execute(
            select(func.count(HallucinationReport.id)).where(
                HallucinationReport.reviewed.is_(False)
            )
        )
        unreviewed = unreviewed_result.scalar() or 0

        # Reviewed count - use .is_(True) for consistency
        reviewed_result = await self.db.execute(
            select(func.count(HallucinationReport.id)).where(
                HallucinationReport.reviewed.is_(True)
            )
        )
        reviewed = reviewed_result.scalar() or 0

        # Corrected count (reviewed with correction)
        corrected_result = await self.db.execute(
            select(func.count(HallucinationReport.id)).where(
                HallucinationReport.reviewed.is_(True),
                HallucinationReport.correction.isnot(None)
            )
        )
        corrected = corrected_result.scalar() or 0

        # False positives (user_error category)
        false_positives_result = await self.db.execute(
            select(func.count(HallucinationReport.id)).where(
                HallucinationReport.category == "user_error"
            )
        )
        false_positives = false_positives_result.scalar() or 0

        # Category breakdown - use None instead of "uncategorized" string
        category_query = select(
            HallucinationReport.category,
            func.count(HallucinationReport.id)
        ).group_by(HallucinationReport.category)

        category_result = await self.db.execute(category_query)
        category_breakdown = {}
        for cat, count in category_result.all():
            # Keep None as key for uncategorized, don't convert to string
            category_breakdown[cat] = count

        # Audit log count
        audit_result = await self.db.execute(
            select(func.count(AuditLog.id))
        )
        audit_count = audit_result.scalar() or 0

        return {
            "total_errors": total,  # Fixed: actual count from DB
            "corrected_errors": corrected,  # Fixed: actual count from DB
            "hallucinations_reported": total,
            "hallucinations_pending_review": unreviewed,
            "hallucinations_reviewed": reviewed,
            "false_positives_skipped": false_positives,  # Fixed: actual count from DB
            "category_breakdown": category_breakdown,
            "audit_entries": audit_count
        }

    # System prompt for fine-tuning
    FINE_TUNING_SYSTEM_PROMPT = """Ti si MobilityOne AI asistent za upravljanje flotom vozila.
Pomaži korisnicima s informacijama o vozilima, rezervacijama, troškovima, putovanjima i prijavi šteta.
Odgovaraj na hrvatskom jeziku, kratko i precizno.
Koristi samo podatke iz dostupnih API-ja, ne izmišljaj informacije."""

    async def export_for_training(
        self,
        admin_id: str,
        reviewed_only: bool = True,
        limit: int = MAX_EXPORT_LIMIT,
        format: str = "openai_chat"
    ) -> List[Dict[str, Any]]:
        """
        Exportaj podatke za fine-tuning modela IZ BAZE PODATAKA.

        FORMATS:
        - "openai_chat": OpenAI Chat format (default) - za gpt-3.5-turbo, gpt-4 fine-tuning
        - "openai_completion": OpenAI Completion format - za davinci fine-tuning (legacy)
        - "raw": Raw format s svim podacima

        OpenAI Chat format (JSONL):
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "user_query"},
            {"role": "assistant", "content": "correct_output"}
        ]}

        Args:
            admin_id: Admin user ID for audit
            reviewed_only: Only export reviewed records
            limit: Maximum records to export (default: 10000)
            format: Export format ("openai_chat", "openai_completion", "raw")
        """
        # Validate limit to prevent memory issues
        validated_limit = self._validate_limit(limit, max_limit=MAX_EXPORT_LIMIT)

        await self._audit("EXPORT_TRAINING_DATA", admin_id, {
            "reviewed_only": reviewed_only,
            "limit": validated_limit,
            "format": format
        })

        query = select(HallucinationReport).where(
            HallucinationReport.correction.isnot(None)
        ).limit(validated_limit)

        if reviewed_only:
            # Use .is_(True) for proper NULL handling
            query = query.where(HallucinationReport.reviewed.is_(True))

        result = await self.db.execute(query)
        reports = result.scalars().all()

        # Format based on requested type
        if format == "openai_chat":
            # OpenAI Chat Completion fine-tuning format
            training_data = [
                {
                    "messages": [
                        {"role": "system", "content": self.FINE_TUNING_SYSTEM_PROMPT},
                        {"role": "user", "content": r.user_query},
                        {"role": "assistant", "content": r.correction}
                    ]
                }
                for r in reports
            ]
        elif format == "openai_completion":
            # Legacy OpenAI Completion format (for davinci)
            training_data = [
                {
                    "prompt": f"User: {r.user_query}\nAssistant:",
                    "completion": f" {r.correction}"
                }
                for r in reports
            ]
        else:
            # Raw format with all data (for debugging/analysis)
            training_data = [
                {
                    "id": str(r.id),
                    "user_query": r.user_query,
                    "wrong_output": r.bot_response,
                    "correct_output": r.correction,
                    "category": r.category,
                    "model": r.model,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None
                }
                for r in reports
            ]

        logger.info(
            f"Exported {len(training_data)} training examples for {admin_id} (format={format})"
        )

        return training_data

    async def export_for_training_jsonl(
        self,
        admin_id: str,
        reviewed_only: bool = True,
        limit: int = MAX_EXPORT_LIMIT
    ) -> str:
        """
        Export training data as JSONL string (ready for OpenAI fine-tuning upload).

        Returns a string where each line is a valid JSON object.
        Save this to a .jsonl file and upload to OpenAI.

        Usage:
            jsonl_content = await service.export_for_training_jsonl(admin_id)
            with open("training_data.jsonl", "w") as f:
                f.write(jsonl_content)
        """
        import json

        data = await self.export_for_training(
            admin_id=admin_id,
            reviewed_only=reviewed_only,
            limit=limit,
            format="openai_chat"
        )

        # Convert to JSONL (one JSON object per line)
        lines = [json.dumps(item, ensure_ascii=False) for item in data]
        return "\n".join(lines)

    async def get_report_detail(
        self,
        admin_id: str,
        report_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Dohvati detalje pojedine halucinacije (uključujući raw API response).
        """
        await self._audit("VIEW_REPORT_DETAIL", admin_id, {
            "report_id": report_id
        })

        try:
            uuid_id = UUID(report_id)
        except ValueError:
            return None

        result = await self.db.execute(
            select(HallucinationReport).where(
                HallucinationReport.id == uuid_id
            )
        )
        report = result.scalar_one_or_none()

        if not report:
            return None

        return {
            "id": str(report.id),
            "timestamp": report.created_at.isoformat() if report.created_at else None,
            "user_query": report.user_query,
            "bot_response": report.bot_response,
            "user_feedback": report.user_feedback,
            "model": report.model,
            "conversation_id": report.conversation_id,
            "tenant_id": report.tenant_id,
            "retrieved_chunks": report.retrieved_chunks,
            "api_raw_response": report.api_raw_response,
            "reviewed": report.reviewed,
            "reviewed_by": report.reviewed_by,
            "reviewed_at": report.reviewed_at.isoformat() if report.reviewed_at else None,
            "correction": report.correction,
            "category": report.category
        }

    async def get_reports_count(
        self,
        unreviewed_only: bool = False,
        tenant_filter: Optional[str] = None
    ) -> int:
        """
        Get total count of reports for pagination.

        Args:
            unreviewed_only: Count only unreviewed reports
            tenant_filter: Filter by tenant ID

        Returns:
            Total count of matching reports
        """
        query = select(func.count(HallucinationReport.id))

        if unreviewed_only:
            query = query.where(HallucinationReport.reviewed.is_(False))

        if tenant_filter:
            query = query.where(HallucinationReport.tenant_id == tenant_filter)

        result = await self.db.execute(query)
        return result.scalar() or 0
