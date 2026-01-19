"""
Error Learning Service - Self-Correction Engine
Version: 1.0

KRITIČNA KOMPONENTA za robustan sustav.

Pamti greške i uči iz njih poput neural networka:
1. Logira greške s kontekstom
2. Prepoznaje uzorke (iste greške se ponavljaju)
3. Automatski primjenjuje popravke za poznate probleme
4. Smanjuje broj ponovljenih grešaka

Primjer:
    Greška: "405 Method Not Allowed" za /automation/MasterData (POST umjesto GET)
    Learning: Sljedeći put kad LLM pokuša POST na MasterData, automatski ispravi na GET
"""

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from services.gdpr_masking import GDPRMaskingService, get_masking_service

logger = logging.getLogger(__name__)

# Cache file path for JSON persistence
ERROR_LEARNING_CACHE_FILE = Path.cwd() / ".cache" / "error_learning.json"


@dataclass
class ErrorPattern:
    """Represents a learned error pattern."""
    error_code: str
    operation_id: str
    error_message: str
    context: Dict[str, Any]
    correction: Optional[str] = None  # What fixed it
    occurrence_count: int = 1
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved: bool = False


@dataclass
class HallucinationReport:
    """
    Korisnik je rekao "krivo" - semantička greška bota.

    Razlika od ErrorPattern:
    - ErrorPattern = tehnička greška (405, 400, timeout)
    - HallucinationReport = bot je dao krivi odgovor koji je tehnički prošao

    Primjer: Korisnik pita "Koji je limit kartice?"
             Bot odgovara "5000 EUR" ali je zapravo 3000 EUR.
    """
    timestamp: str
    user_query: str  # Što je korisnik pitao
    bot_response: str  # Što je bot odgovorio (halucinacija)
    user_feedback: str  # "krivo" ili detaljni feedback
    retrieved_chunks: List[str]  # RAG chunk ID-evi korišteni
    model: str  # Koji model je producirao odgovor
    api_raw_response: Optional[Dict[str, Any]] = None  # Sirovi API response
    conversation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    reviewed: bool = False  # Za manualni pregled
    correction: Optional[str] = None  # Ljudska korekcija
    category: Optional[str] = None  # "wrong_data", "outdated", "misunderstood"


@dataclass
class CorrectionRule:
    """Automatic correction rule learned from errors."""
    trigger_pattern: str  # Regex or exact match
    trigger_operation: Optional[str]  # Specific operation or None for all
    correction_type: str  # "method", "param", "url", "value"
    correction_action: Dict[str, Any]  # What to change
    confidence: float  # 0.0 to 1.0
    success_count: int = 0
    failure_count: int = 0


class ErrorLearningService:
    """
    Self-learning error correction system.

    Features:
    - Pattern detection across errors
    - Automatic correction rules
    - Confidence-based application
    - Redis-backed persistence
    - Model drift detection integration
    - Closed feedback loop with training data export

    Inspiracija: Neural network backpropagation - greška se propagira
    natrag i sustav se prilagođava.
    """

    # Known error patterns and their corrections
    # This is the "training data" for the system
    KNOWN_CORRECTIONS: List[CorrectionRule] = [
        # 405 errors often mean wrong HTTP method
        CorrectionRule(
            trigger_pattern="405",
            trigger_operation=None,
            correction_type="method",
            correction_action={"from": "POST", "to": "GET", "hint": "retrieval_operation"},
            confidence=0.7,
        ),
        # 400 with "required" often means missing param
        CorrectionRule(
            trigger_pattern="required",
            trigger_operation=None,
            correction_type="param",
            correction_action={"action": "inject_from_context"},
            confidence=0.6,
        ),
        # HTML response means auth issue or wrong URL
        CorrectionRule(
            trigger_pattern="HTML_RESPONSE",
            trigger_operation=None,
            correction_type="url",
            correction_action={"action": "verify_swagger_name"},
            confidence=0.8,
        ),
    ]

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        db_session: Optional[Any] = None,
        enable_gdpr_masking: bool = True
    ):
        """
        Initialize error learning service.

        Args:
            redis_client: Optional Redis client for persistence
            db_session: Optional async DB session for hallucination reports
            enable_gdpr_masking: Enable GDPR PII masking (default: True)
        """
        self.redis = redis_client
        self.db = db_session
        self._error_patterns: Dict[str, ErrorPattern] = {}
        self._correction_rules: List[CorrectionRule] = self.KNOWN_CORRECTIONS.copy()
        self._learned_rules: List[CorrectionRule] = []

        # Hallucination reports - in-memory cache (DB is source of truth)
        self._hallucination_reports: List[HallucinationReport] = []

        # Statistics
        self._total_errors = 0
        self._corrected_errors = 0
        self._pattern_matches = 0
        self._hallucinations_reported = 0
        self._false_positives_skipped = 0

        # Model drift detector integration
        self._drift_detector = None

        # GDPR PII masking - Article 25: Data Protection by Design
        self._gdpr_masking_enabled = enable_gdpr_masking
        self._gdpr_masker: Optional[GDPRMaskingService] = None
        if enable_gdpr_masking:
            self._gdpr_masker = get_masking_service()
            logger.info("GDPR masking enabled for hallucination reports")

        logger.info("ErrorLearningService initialized")

    def set_drift_detector(self, drift_detector) -> None:
        """
        Connect drift detector for closed feedback loop.

        Args:
            drift_detector: ModelDriftDetector instance
        """
        self._drift_detector = drift_detector
        logger.info("Drift detector connected to ErrorLearningService")

    def _mask_pii(self, text: str) -> str:
        """
        Mask PII data for GDPR compliance.

        Args:
            text: Input text that may contain PII

        Returns:
            Text with PII masked (e.g., phone -> [PHONE-abc123])
        """
        if not self._gdpr_masker or not text:
            return text

        result = self._gdpr_masker.mask_pii(text)
        if result.has_pii():
            logger.debug(
                f"GDPR: Masked {result.pii_count} PII items: "
                f"{[m.pii_type.value for m in result.pii_found]}"
            )
        return result.masked_text

    async def record_error(
        self,
        error_code: str,
        operation_id: str,
        error_message: str,
        context: Dict[str, Any],
        was_corrected: bool = False,
        correction: Optional[str] = None,
        http_status: Optional[int] = None,
        response_data: Optional[Any] = None
    ) -> None:
        """
        Record an error for learning.

        Args:
            error_code: Error code (HTTP status, internal code, etc.)
            operation_id: Tool/operation that failed
            error_message: Full error message
            context: Context when error occurred (params, user, etc.)
            was_corrected: Whether error was automatically corrected
            correction: What fixed it (if corrected)
            http_status: HTTP status code (for False Positive detection)
            response_data: Raw API response (for False Positive detection)
        """
        # =====================================================================
        # FALSE POSITIVE ZAŠTITA
        # 200 + prazan odgovor = "Data Not Found", NE greška modela
        # Primjer: GET /Persons?Filter=Phone(=)099... vraća [] jer osoba ne postoji
        # =====================================================================
        if self._is_false_positive(http_status, response_data, error_code):
            self._false_positives_skipped += 1
            logger.debug(
                f"⏭️ False positive skipped: {operation_id} "
                f"(HTTP {http_status}, empty response is valid)"
            )
            return

        self._total_errors += 1

        # Create pattern key
        pattern_key = f"{error_code}:{operation_id}"

        if pattern_key in self._error_patterns:
            # Update existing pattern
            pattern = self._error_patterns[pattern_key]
            pattern.occurrence_count += 1
            pattern.last_seen = datetime.utcnow().isoformat()
            if correction:
                pattern.correction = correction
                pattern.resolved = True
        else:
            # New pattern
            pattern = ErrorPattern(
                error_code=error_code,
                operation_id=operation_id,
                error_message=error_message,
                context=self._sanitize_context(context),
                correction=correction,
                resolved=was_corrected
            )
            self._error_patterns[pattern_key] = pattern

        # Log for analysis
        logger.info(
            f"📊 Error recorded: {pattern_key} "
            f"(count={pattern.occurrence_count}, resolved={pattern.resolved})"
        )

        # Persist to Redis if available
        if self.redis:
            await self._persist_pattern(pattern_key, pattern)

        # Check for new learnable patterns
        await self._analyze_patterns()

    async def suggest_correction(
        self,
        error_code: str,
        operation_id: str,
        error_message: str,
        current_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a correction based on learned patterns.

        Args:
            error_code: Current error code
            operation_id: Failed operation
            error_message: Error message
            current_params: Current parameters

        Returns:
            Correction suggestion or None
        """
        self._pattern_matches += 1

        # Check known corrections first
        for rule in self._correction_rules + self._learned_rules:
            if self._rule_matches(rule, error_code, operation_id, error_message):
                if rule.confidence >= 0.6:  # Only apply confident corrections
                    logger.info(
                        f"💡 Suggesting correction: {rule.correction_type} "
                        f"(confidence={rule.confidence:.2f})"
                    )
                    return {
                        "type": rule.correction_type,
                        "action": rule.correction_action,
                        "confidence": rule.confidence,
                        "rule_pattern": rule.trigger_pattern
                    }

        # Check if we've seen this exact error before and it was resolved
        pattern_key = f"{error_code}:{operation_id}"
        if pattern_key in self._error_patterns:
            pattern = self._error_patterns[pattern_key]
            if pattern.resolved and pattern.correction:
                return {
                    "type": "historical",
                    "action": {"correction": pattern.correction},
                    "confidence": min(0.5 + (pattern.occurrence_count * 0.1), 0.9),
                    "occurrences": pattern.occurrence_count
                }

        return None

    async def apply_correction(
        self,
        correction: Dict[str, Any],
        tool: Any,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply a suggested correction.

        Args:
            correction: Correction suggestion from suggest_correction
            tool: Tool definition
            params: Current parameters
            context: User context

        Returns:
            Corrected parameters or None if not applicable
        """
        correction_type = correction.get("type")
        action = correction.get("action", {})

        if correction_type == "method":
            # This would need tool modification - return hint for caller
            return {
                "hint": "use_get_instead_of_post",
                "original_method": action.get("from"),
                "suggested_method": action.get("to")
            }

        elif correction_type == "param":
            if action.get("action") == "inject_from_context":
                # Try to inject missing params from context
                corrected_params = params.copy()
                for key in ["person_id", "tenant_id", "PersonId", "TenantId"]:
                    if key in context and key not in corrected_params:
                        corrected_params[key] = context[key]
                return {"params": corrected_params}

        elif correction_type == "url":
            if action.get("action") == "verify_swagger_name":
                return {"hint": "check_swagger_name", "message": "URL might be incorrect"}

        elif correction_type == "historical":
            # Return the historical correction
            return {"hint": action.get("correction")}

        return None

    async def report_correction_result(
        self,
        correction: Dict[str, Any],
        success: bool
    ) -> None:
        """
        Report whether a correction worked.

        This is the "backpropagation" - we update rule confidence
        based on success/failure.

        Args:
            correction: The correction that was applied
            success: Whether it worked
        """
        rule_pattern = correction.get("rule_pattern")

        for rule in self._correction_rules + self._learned_rules:
            if rule.trigger_pattern == rule_pattern:
                if success:
                    rule.success_count += 1
                    self._corrected_errors += 1
                    # Increase confidence (max 0.95)
                    rule.confidence = min(
                        rule.confidence + 0.05,
                        0.95
                    )
                else:
                    rule.failure_count += 1
                    # Decrease confidence (min 0.1)
                    rule.confidence = max(
                        rule.confidence - 0.1,
                        0.1
                    )

                logger.info(
                    f"📈 Rule updated: {rule_pattern} "
                    f"(success={success}, new_confidence={rule.confidence:.2f})"
                )
                break

    def _rule_matches(
        self,
        rule: CorrectionRule,
        error_code: str,
        operation_id: str,
        error_message: str
    ) -> bool:
        """Check if a rule matches the current error."""
        # Check operation filter
        if rule.trigger_operation and rule.trigger_operation != operation_id:
            return False

        # Check pattern in error code or message
        pattern = rule.trigger_pattern.lower()
        return (
            pattern in str(error_code).lower() or
            pattern in error_message.lower()
        )

    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from context before storing."""
        sensitive_keys = ["token", "password", "secret", "api_key", "auth"]
        return {
            k: "[REDACTED]" if any(s in k.lower() for s in sensitive_keys) else v
            for k, v in context.items()
        }

    async def _persist_pattern(self, key: str, pattern: ErrorPattern) -> None:
        """Persist pattern to Redis."""
        if not self.redis:
            return

        try:
            await self.redis.hset(
                "error_learning:patterns",
                key,
                json.dumps(asdict(pattern))
            )
        except Exception as e:
            logger.warning(f"Failed to persist error pattern: {e}")

    async def _analyze_patterns(self) -> None:
        """
        Analyze patterns to learn new correction rules.

        This is where the "learning" happens - we look for
        patterns in resolved errors and create new rules.
        """
        # Find patterns that were resolved multiple times with same correction
        resolved_patterns = [
            p for p in self._error_patterns.values()
            if p.resolved and p.correction and p.occurrence_count >= 3
        ]

        for pattern in resolved_patterns:
            # Check if we already have a rule for this
            existing = any(
                r.trigger_pattern == pattern.error_code and
                r.trigger_operation == pattern.operation_id
                for r in self._learned_rules
            )

            if not existing:
                # Create new learned rule
                new_rule = CorrectionRule(
                    trigger_pattern=pattern.error_code,
                    trigger_operation=pattern.operation_id,
                    correction_type="historical",
                    correction_action={"correction": pattern.correction},
                    confidence=0.6 + min(pattern.occurrence_count * 0.05, 0.3),
                    success_count=pattern.occurrence_count
                )
                self._learned_rules.append(new_rule)

                logger.info(
                    f"🎓 Learned new rule: {pattern.error_code} → "
                    f"{pattern.correction} (from {pattern.occurrence_count} occurrences)"
                )

    # =========================================================================
    # FALSE POSITIVE DETEKCIJA
    # =========================================================================

    def _is_false_positive(
        self,
        http_status: Optional[int],
        response_data: Optional[Any],
        error_code: str
    ) -> bool:
        """
        Detektiraj "lažne greške" koje NISU greška modela.

        Primjeri False Positives:
        1. HTTP 200 + prazan array [] = "Nema podataka" (OK, ne greška)
        2. HTTP 200 + {"items": []} = "Prazna lista" (OK)
        3. Korisnik traži osobu koja ne postoji u sustavu

        NE uči iz ovoga jer model je radio ispravno!
        """
        # Ako nema HTTP statusa, ne možemo odlučiti
        if http_status is None:
            return False

        # Samo 2xx statusni kodovi mogu biti false positives
        if not (200 <= http_status < 300):
            return False

        # Prazan odgovor na uspješan request = Data Not Found, ne greška
        if response_data is None:
            return True

        if isinstance(response_data, list) and len(response_data) == 0:
            return True

        if isinstance(response_data, dict):
            # {"items": [], "total": 0} tipa odgovori
            items = response_data.get("items", response_data.get("data", []))
            if isinstance(items, list) and len(items) == 0:
                return True

        return False

    # =========================================================================
    # HALLUCINATION REPORTING ("krivo" feedback)
    # =========================================================================

    async def record_hallucination(
        self,
        user_query: str,
        bot_response: str,
        user_feedback: str,
        retrieved_chunks: List[str],
        model: str,
        api_raw_response: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Zabilježi halucinaciju prijavljenu od korisnika ("krivo").

        GDPR COMPLIANCE:
        - Svi PII podaci (telefon, email, OIB) se maskiraju PRIJE spremanja
        - Maskirani podaci omogućuju analizu bez exposure PII

        Persistencija:
        1. DB (PostgreSQL) - source of truth za analitiku
        2. Redis - cache za brzi pristup
        3. JSON - backup/fallback

        Returns:
            Dict s:
            - recorded: bool (uspješno zabilježeno)
            - follow_up_question: str (pitanje za korisnika za više detalja)
            - report_id: str (ID za praćenje)
        """
        self._hallucinations_reported += 1
        report_id = None

        # GDPR Article 25: Data Protection by Design
        # Mask PII BEFORE storing anywhere
        masked_query = self._mask_pii(user_query)
        masked_response = self._mask_pii(bot_response)
        masked_feedback = self._mask_pii(user_feedback)

        # 1. Persist to DATABASE (primary storage)
        if self.db:
            try:
                from services.hallucination_repository import HallucinationRepository
                repo = HallucinationRepository(self.db)
                db_report = await repo.create(
                    user_query=masked_query,  # GDPR masked
                    bot_response=masked_response,  # GDPR masked
                    user_feedback=masked_feedback,  # GDPR masked
                    model=model,
                    conversation_id=conversation_id,
                    tenant_id=tenant_id,
                    retrieved_chunks=retrieved_chunks,
                    api_raw_response=api_raw_response
                )
                report_id = str(db_report.id)
                logger.info(f"Hallucination saved to DB: {report_id} (PII masked)")
            except Exception as e:
                logger.error(f"Failed to save hallucination to DB: {e}")

        # 2. Also keep in memory cache (masked)
        report = HallucinationReport(
            timestamp=datetime.utcnow().isoformat(),
            user_query=masked_query,
            bot_response=masked_response,
            user_feedback=masked_feedback,
            retrieved_chunks=retrieved_chunks,
            model=model,
            api_raw_response=api_raw_response,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            reviewed=False
        )
        self._hallucination_reports.append(report)

        # 3. Persist to Redis if available (cache)
        if self.redis:
            await self._persist_hallucination(report)

        # Log for monitoring
        logger.warning(
            f"Hallucination reported: query='{user_query[:50]}...' "
            f"feedback='{user_feedback}' conversation={conversation_id}"
        )

        # Send to drift detector (closes the feedback loop)
        if self._drift_detector:
            await self._drift_detector.record_interaction(
                model_version=model,
                latency_ms=0,  # Not applicable for hallucination
                success=True,  # API call succeeded, but content was wrong
                hallucination_reported=True,  # This is the key metric
                tenant_id=tenant_id
            )

        # Generiraj follow-up pitanje za više konteksta
        follow_up = self._generate_hallucination_followup(user_feedback, bot_response)

        return {
            "recorded": True,
            "follow_up_question": follow_up,
            "report_id": report_id or f"hal_{conversation_id}_{self._hallucinations_reported}"
        }

    def _generate_hallucination_followup(
        self,
        user_feedback: str,
        bot_response: str
    ) -> str:
        """
        Generiraj pitanje za korisnika da dobijemo više detalja.

        Umjesto da bot šuti, pitamo što je točno bilo pogrešno.
        To nam daje "zlato" za fine-tuning i RAG poboljšanja.
        """
        # Ako je feedback kratak ("krivo", "ne"), pitaj za detalje
        if len(user_feedback.strip()) < 20:
            return (
                "Zabilježio sam grešku i proslijedio je timu na analizu. "
                "Možete li mi reći što je točno bilo pogrešno kako bih brže naučio?"
            )

        # Ako je feedback detaljniji, zahvali i potvrdi
        return (
            "Hvala na povratnoj informaciji! Zabilježio sam detalje i "
            "proslijedio ih timu. Ispričavam se za neugodnost."
        )

    async def _persist_hallucination(self, report: HallucinationReport) -> None:
        """Persist hallucination report to Redis."""
        if not self.redis:
            return

        try:
            report_key = f"hallucination:{report.conversation_id}:{report.timestamp}"
            await self.redis.hset(
                "error_learning:hallucinations",
                report_key,
                json.dumps(asdict(report), default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to persist hallucination report: {e}")

    # =========================================================================
    # JSON FILE PERSISTENCE (za analizu trendova)
    # =========================================================================

    def save_to_file(self) -> None:
        """
        Spremi stanje u JSON datoteku.

        Struktura prema preporuci:
        - patterns: Lista ErrorPattern objekata
        - hallucinations: Lista HallucinationReport objekata
        - learned_rules: Lista naučenih pravila
        - statistics: Agregirane statistike
        """
        try:
            ERROR_LEARNING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "2.0",
                "saved_at": datetime.utcnow().isoformat(),
                "patterns": [asdict(p) for p in self._error_patterns.values()],
                "hallucinations": [asdict(h) for h in self._hallucination_reports],
                "learned_rules": [
                    {
                        "trigger_pattern": r.trigger_pattern,
                        "trigger_operation": r.trigger_operation,
                        "correction_type": r.correction_type,
                        "correction_action": r.correction_action,
                        "confidence": r.confidence,
                        "success_count": r.success_count,
                        "failure_count": r.failure_count
                    }
                    for r in self._learned_rules
                ],
                "statistics": {
                    "total_errors": self._total_errors,
                    "corrected_errors": self._corrected_errors,
                    "pattern_matches": self._pattern_matches,
                    "hallucinations_reported": self._hallucinations_reported,
                    "false_positives_skipped": self._false_positives_skipped,
                    "remaining_stats": len(self._error_patterns)
                }
            }

            with open(ERROR_LEARNING_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(
                f"💾 Error learning state saved: {len(self._error_patterns)} patterns, "
                f"{len(self._hallucination_reports)} hallucinations"
            )

        except Exception as e:
            logger.warning(f"Failed to save error learning state: {e}")
