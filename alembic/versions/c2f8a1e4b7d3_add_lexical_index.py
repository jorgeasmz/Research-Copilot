"""Add the lexical search column

Revision ID: c2f8a1e4b7d3
Revises: b12a5457d50d
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c2f8a1e4b7d3"
down_revision: str | None = "b12a5457d50d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Generated rather than written by the application: the column cannot drift
    # from the text it indexes, and ingestion does not have to know it exists.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN search tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_search ON chunks USING gin (search)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_search")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search")
