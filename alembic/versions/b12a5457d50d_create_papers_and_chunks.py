"""Create papers and chunks

Revision ID: b12a5457d50d
Revises: 
"""

from collections.abc import Sequence

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b12a5457d50d'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The vector type has to exist before a column can declare it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('papers',
    sa.Column('arxiv_id', sa.String(length=32), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('abstract', sa.Text(), nullable=False),
    sa.Column('authors', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('published', sa.String(length=32), nullable=False),
    sa.Column('updated', sa.String(length=32), nullable=False),
    sa.Column('ingested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('arxiv_id')
    )
    op.create_table('chunks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('arxiv_id', sa.String(length=32), nullable=False),
    sa.Column('section', sa.Text(), nullable=False),
    sa.Column('paragraph', sa.Integer(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=False),
    sa.ForeignKeyConstraint(['arxiv_id'], ['papers.arxiv_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chunks_arxiv_id'), 'chunks', ['arxiv_id'], unique=False)
    op.create_index('uq_chunk_location', 'chunks', ['arxiv_id', 'paragraph', 'chunk_index'], unique=True)

    # HNSW over cosine distance: the embeddings are unit-normalised, and the
    # index is built here rather than after loading so an ingestion run does
    # not have to rebuild it.
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")
    op.drop_index('uq_chunk_location', table_name='chunks')
    op.drop_index(op.f('ix_chunks_arxiv_id'), table_name='chunks')
    op.drop_table('chunks')
    op.drop_table('papers')
