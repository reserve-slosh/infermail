"""add source_folder and demoted_to_spam_at to emails

Revision ID: 64494dc970e4
Revises: 54c55c7f095f
Create Date: 2026-03-30

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '64494dc970e4'
down_revision: Union[str, Sequence[str], None] = '54c55c7f095f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('emails', sa.Column('source_folder', sa.String(length=255), nullable=True))
    op.add_column('emails', sa.Column('demoted_to_spam_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('emails', 'demoted_to_spam_at')
    op.drop_column('emails', 'source_folder')
