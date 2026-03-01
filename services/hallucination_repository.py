"""
Hallucination Repository - Database persistence for feedback loop.

Handles CRUD operations for HallucinationReport in PostgreSQL.
Separates DB logic from business logic in ErrorLearningService.

NOTE: This is a low-level repository layer. For admin operations with
security, audit logging, and validation, use AdminReviewService instead.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from uuid import UUID

from sqlalchemy import select, update, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from models import HallucinationReport

logger = logging.getLogger(__name__)


# Constants
MAX_LIMIT = 1000
DEFAULT_LIMIT = 50
MIN_LIMIT = 1
MAX_EXPORT_LIMIT = 10000

# Allowed categories (shared with AdminReviewService)
ALLOWED_CATEGORIES: Set[str] = frozenset([
    "wrong_data",
    "outdated",
    "misunderstood",
    "api_error",
    "rag_failure",
    "hallucination",
    "user_error",
])


class HallucinationRepository:
    """
    Database repository for hallucination reports.

    This is a LOW-LEVEL repository for raw database operations.
    For admin operations with full security, audit logging, and input
    validation, use AdminReviewService instead.

    Usage:
        repo = HallucinationRepository(db_session)
        await repo.create(user_query="...", bot_response="...", ...)
        reports = await repo.get_unreviewed(limit=50)

    NOTE: This class intentionally does NOT duplicate AdminReviewService
    functionality. It's meant for simple CRUD operations only.
    """

    def __init__(self, db: AsyncSession, gdpr_masking_service=None):
        """
        Initialize repository.

        Args:
            db: Async SQLAlchemy session
            gdpr_masking_service: Optional GDPR masking service for PII detection
        """
        self.db = db
        self._gdpr_service = gdpr_masking_service
        logger.debug("HallucinationRepository initialized")

    def _validate_limit(self, limit: int, max_limit: int = MAX_LIMIT) -> int:
        """Validate and constrain limit parameter."""
        if not isinstance(limit, int):
            return DEFAULT_LIMIT
        return max(MIN_LIMIT, min(limit, max_limit))

    def _validate_offset(self, offset: int) -> int:
        """Validate offset parameter."""
        if not isinstance(offset, int):
            return 0
        return max(0, offset)

    def _validate_category(self, category: Optional[str]) -> Optional[str]:
        """Validate category against whitelist."""
        if category is None:
            return None
        if category not in ALLOWED_CATEGORIES:
            logger.warning(f"Invalid category rejected: {category}")
            return None
        return category

    async def _apply_gdpr_masking(
        self,
        user_query: str,
        bot_response: str,
        user_feedback: str
    ) -> tuple:
        """
        Apply GDPR masking to PII in text fields.

        Returns:
            Tuple of (masked_query, masked_response, masked_feedback)
        """
        if not self._gdpr_service:
            return user_query, bot_response, user_feedback

        try:
            masked_query = self._gdpr_service.mask_pii(user_query).masked_text
            masked_response = self._gdpr_service.mask_pii(bot_response).masked_text
            masked_feedback = self._gdpr_service.mask_pii(user_feedback).masked_text
            return masked_query, masked_response, masked_feedback
        except Exception as e:
            logger.error(f"GDPR masking failed: {e}")
            # Return original if masking fails
            return user_query, bot_response, user_feedback

    async def create(
        self,
        user_query: str,
        bot_response: str,
        user_feedback: str,
        model: str,
        conversation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        retrieved_chunks: Optional[List[str]] = None,
        api_raw_response: Optional[Dict[str, Any]] = None,
        apply_gdpr_masking: bool = True
    ) -> HallucinationReport:
        """
        Create new hallucination report.

        NOTE: bot_user has INSERT-only on this table (no SELECT).
        We generate UUID in Python and don't use refresh().

        Args:
            user_query: User's original question
            bot_response: Bot's incorrect response
            user_feedback: User's feedback about the error
            model: Model name used
            conversation_id: Optional conversation identifier
            tenant_id: Optional tenant identifier
            retrieved_chunks: RAG chunks that were retrieved
            api_raw_response: Raw API response for debugging
            apply_gdpr_masking: Whether to mask PII (default: True)

        Returns:
            Created HallucinationReport instance with pre-generated ID
        """
        # Apply GDPR masking before storing
        if apply_gdpr_masking:
            user_query, bot_response, user_feedback = await self._apply_gdpr_masking(
                user_query, bot_response, user_feedback
            )

        # Generate UUID in Python (bot_user can't use RETURNING)
        report_id = uuid.uuid4()

        report = HallucinationReport(
            id=report_id,
            user_query=user_query,
            bot_response=bot_response,
            user_feedback=user_feedback,
            model=model,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            retrieved_chunks=retrieved_chunks or [],
            api_raw_response=api_raw_response,
            reviewed=False  # Explicitly set default
        )

        try:
            self.db.add(report)
            await self.db.commit()
            # NOTE: No refresh() - bot_user doesn't have SELECT privilege
            logger.info(f"Created hallucination report: {report_id}")
            return report
        except Exception as e:
            logger.error(f"Failed to create hallucination report: {e}")
            await self.db.rollback()
            raise

    async def get_by_id(self, report_id: UUID) -> Optional[HallucinationReport]:
        """
        Get report by ID.

        Args:
            report_id: UUID of the report

        Returns:
            HallucinationReport or None if not found
        """
        try:
            result = await self.db.execute(
                select(HallucinationReport).where(HallucinationReport.id == report_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get report {report_id}: {e}")
            return None

    async def get_unreviewed(
        self,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        tenant_id: Optional[str] = None
    ) -> List[HallucinationReport]:
        """
        Get unreviewed reports for admin dashboard.

        Args:
            limit: Max number of results (validated, max 1000)
            offset: Number of records to skip for pagination
            tenant_id: Optional tenant filter

        Returns:
            List of unreviewed reports, newest first
        """
        validated_limit = self._validate_limit(limit)
        validated_offset = self._validate_offset(offset)

        # Build query with ORDER BY for deterministic results
        query = select(HallucinationReport).where(
            # Use .is_(False) for proper NULL handling
            HallucinationReport.reviewed.is_(False)
        ).order_by(
            desc(HallucinationReport.created_at)
        ).limit(validated_limit).offset(validated_offset)

        if tenant_id:
            query = query.where(HallucinationReport.tenant_id == tenant_id)

        try:
            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to get unreviewed reports: {e}")
            return []

    async def get_all(
        self,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        tenant_id: Optional[str] = None,
        category: Optional[str] = None,
        reviewed_only: bool = False
    ) -> List[HallucinationReport]:
        """
        Get reports with filters.

        Args:
            limit: Max results (validated, max 1000)
            offset: Skip first N results
            tenant_id: Filter by tenant
            category: Filter by category (validated)
            reviewed_only: Only reviewed reports
        """
        validated_limit = self._validate_limit(limit)
        validated_offset = self._validate_offset(offset)
        validated_category = self._validate_category(category)

        # Build query with ORDER BY for deterministic pagination
        query = select(HallucinationReport).order_by(
            desc(HallucinationReport.created_at)
        ).limit(validated_limit).offset(validated_offset)

        if tenant_id:
            query = query.where(HallucinationReport.tenant_id == tenant_id)
        if validated_category:
            query = query.where(HallucinationReport.category == validated_category)
        if reviewed_only:
            # Use .is_(True) for proper NULL handling
            query = query.where(HallucinationReport.reviewed.is_(True))

        try:
            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Failed to get reports: {e}")
            return []

    async def exists(self, report_id: UUID) -> bool:
        """
        Check if report exists before update.

        Args:
            report_id: UUID to check

        Returns:
            True if exists, False otherwise
        """
        result = await self.db.execute(
            select(func.count(HallucinationReport.id)).where(
                HallucinationReport.id == report_id
            )
        )
        return (result.scalar() or 0) > 0

    async def mark_reviewed(
        self,
        report_id: UUID,
        reviewed_by: str,
        correction: Optional[str] = None,
        category: Optional[str] = None
    ) -> bool:
        """
        Mark report as reviewed by admin.

        NOTE: For full security validation and audit logging, use
        AdminReviewService.mark_hallucination_reviewed() instead.

        Args:
            report_id: Report UUID
            reviewed_by: Admin username
            correction: Correct answer (optional)
            category: Error category (optional, validated)

        Returns:
            True if updated, False if not found
        """
        # Validate category
        validated_category = self._validate_category(category)

        # Check if report exists first
        if not await self.exists(report_id):
            logger.warning(f"Report {report_id} not found for review")
            return False

        try:
            result = await self.db.execute(
                update(HallucinationReport)
                .where(HallucinationReport.id == report_id)
                .values(
                    reviewed=True,
                    reviewed_by=reviewed_by,
                    reviewed_at=datetime.now(timezone.utc),
                    correction=correction,
                    category=validated_category
                )
            )
            await self.db.commit()

            updated = result.rowcount > 0
            if updated:
                logger.info(f"Marked report {report_id} as reviewed by {reviewed_by}")
            return updated
        except Exception as e:
            logger.error(f"Failed to mark report as reviewed: {e}")
            await self.db.rollback()
            return False

    async def get_statistics(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for dashboard.

        Returns dict with:
        - total_reports
        - unreviewed_count
        - reviewed_count
        - category_breakdown

        NOTE: This returns a DIFFERENT structure than AdminReviewService.get_statistics()
        to avoid confusion. Use the appropriate service based on your needs.
        """
        try:
            # Base filter as a list of conditions
            base_filters = []
            if tenant_id:
                base_filters.append(HallucinationReport.tenant_id == tenant_id)

            # Total count
            total_query = select(func.count(HallucinationReport.id))
            if base_filters:
                total_query = total_query.where(*base_filters)
            total_result = await self.db.execute(total_query)
            total = total_result.scalar() or 0

            # Unreviewed count - use .is_(False) for NULL safety
            unreviewed_query = select(func.count(HallucinationReport.id)).where(
                HallucinationReport.reviewed.is_(False)
            )
            if base_filters:
                unreviewed_query = unreviewed_query.where(*base_filters)
            unreviewed_result = await self.db.execute(unreviewed_query)
            unreviewed = unreviewed_result.scalar() or 0

            # Reviewed count - use .is_(True) for NULL safety
            reviewed_query = select(func.count(HallucinationReport.id)).where(
                HallucinationReport.reviewed.is_(True)
            )
            if base_filters:
                reviewed_query = reviewed_query.where(*base_filters)
            reviewed_result = await self.db.execute(reviewed_query)
            reviewed = reviewed_result.scalar() or 0

            # Category breakdown - keep None as key, don't convert to string
            category_query = select(
                HallucinationReport.category,
                func.count(HallucinationReport.id)
            ).group_by(HallucinationReport.category)
            if base_filters:
                category_query = category_query.where(*base_filters)
            category_result = await self.db.execute(category_query)
            categories = {}
            for cat, count in category_result.all():
                # Keep None as key for uncategorized
                categories[cat] = count

            return {
                "total_reports": total,
                "unreviewed_count": unreviewed,
                "reviewed_count": reviewed,
                "category_breakdown": categories
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_reports": 0,
                "unreviewed_count": 0,
                "reviewed_count": 0,
                "category_breakdown": {},
                "error": str(e)
            }

    async def export_for_training(
        self,
        reviewed_only: bool = True,
        with_correction_only: bool = True,
        limit: int = MAX_EXPORT_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Export data for model fine-tuning.

        NOTE: This does NOT include audit logging. For audited exports,
        use AdminReviewService.export_for_training() instead.

        Args:
            reviewed_only: Only export reviewed records
            with_correction_only: Only export records with corrections
            limit: Maximum records to export (default: 10000)

        Returns list of dicts with:
        - instruction (user_query)
        - wrong_output (bot_response)
        - correct_output (correction)
        - category
        """
        validated_limit = self._validate_limit(limit, max_limit=MAX_EXPORT_LIMIT)

        query = select(HallucinationReport).limit(validated_limit)

        if reviewed_only:
            # Use .is_(True) for NULL safety
            query = query.where(HallucinationReport.reviewed.is_(True))
        if with_correction_only:
            query = query.where(HallucinationReport.correction.isnot(None))

        try:
            result = await self.db.execute(query)
            reports = result.scalars().all()

            return [
                {
                    "instruction": r.user_query,
                    "wrong_output": r.bot_response,
                    "correct_output": r.correction,
                    "category": r.category,
                    "model": r.model
                }
                for r in reports
            ]
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return []

    async def get_count(
        self,
        reviewed_only: bool = False,
        unreviewed_only: bool = False,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Get count of reports matching filters.

        Args:
            reviewed_only: Count only reviewed reports
            unreviewed_only: Count only unreviewed reports
            tenant_id: Filter by tenant

        Returns:
            Count of matching reports
        """
        query = select(func.count(HallucinationReport.id))

        if reviewed_only:
            query = query.where(HallucinationReport.reviewed.is_(True))
        elif unreviewed_only:
            query = query.where(HallucinationReport.reviewed.is_(False))

        if tenant_id:
            query = query.where(HallucinationReport.tenant_id == tenant_id)

        try:
            result = await self.db.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0

    async def delete(self, report_id: UUID) -> bool:
        """
        Delete a report (soft delete not implemented, use with caution).

        Args:
            report_id: UUID of report to delete

        Returns:
            True if deleted, False if not found
        """
        from sqlalchemy import delete as sql_delete

        try:
            result = await self.db.execute(
                sql_delete(HallucinationReport).where(
                    HallucinationReport.id == report_id
                )
            )
            await self.db.commit()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete report {report_id}: {e}")
            await self.db.rollback()
            return False

    async def bulk_mark_reviewed(
        self,
        report_ids: List[UUID],
        reviewed_by: str,
        category: Optional[str] = None
    ) -> int:
        """
        Mark multiple reports as reviewed.

        Args:
            report_ids: List of report UUIDs
            reviewed_by: Admin username
            category: Optional category to set for all

        Returns:
            Number of reports updated
        """
        if not report_ids:
            return 0

        validated_category = self._validate_category(category)

        try:
            result = await self.db.execute(
                update(HallucinationReport)
                .where(HallucinationReport.id.in_(report_ids))
                .values(
                    reviewed=True,
                    reviewed_by=reviewed_by,
                    reviewed_at=datetime.now(timezone.utc),
                    category=validated_category
                )
            )
            await self.db.commit()
            logger.info(f"Bulk marked {result.rowcount} reports as reviewed")
            return result.rowcount
        except Exception as e:
            logger.error(f"Failed to bulk mark reports: {e}")
            await self.db.rollback()
            return 0
