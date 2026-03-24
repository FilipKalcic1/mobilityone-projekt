"""Add GDPR consent tracking fields to user_mappings

Revision ID: 002_gdpr_consent
Revises: 001_initial
Create Date: 2026-03-07

Adds columns:
- gdpr_consent_given (boolean, default false)
- gdpr_consent_at (timestamp, nullable)
- gdpr_data_retention_days (integer, default 365)
- gdpr_anonymized_at (timestamp, nullable)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision: str = '002_gdpr_consent'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user_mappings', sa.Column(
        'gdpr_consent_given', sa.Boolean(), nullable=True, server_default='false'
    ))
    op.add_column('user_mappings', sa.Column(
        'gdpr_consent_at', sa.DateTime(timezone=True), nullable=True
    ))
    op.add_column('user_mappings', sa.Column(
        'gdpr_data_retention_days', sa.Integer(), nullable=True, server_default='365'
    ))
    op.add_column('user_mappings', sa.Column(
        'gdpr_anonymized_at', sa.DateTime(timezone=True), nullable=True
    ))

    # Grant bot_user access to new columns (bot can read/write consent)
    op.execute("GRANT SELECT, UPDATE ON user_mappings TO bot_user")


def downgrade() -> None:
    op.drop_column('user_mappings', 'gdpr_anonymized_at')
    op.drop_column('user_mappings', 'gdpr_data_retention_days')
    op.drop_column('user_mappings', 'gdpr_consent_at')
    op.drop_column('user_mappings', 'gdpr_consent_given')
