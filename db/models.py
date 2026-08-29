"""Schema for the ingested corpus."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 384


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    # The arXiv identifier carries its version, so a revised paper is a distinct
    # row rather than a silent overwrite of the text a citation points at.
    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list] = mapped_column(JSONB)
    categories: Mapped[list] = mapped_column(JSONB)
    published: Mapped[str] = mapped_column(String(32))
    updated: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("papers.arxiv_id", ondelete="CASCADE"), index=True
    )
    section: Mapped[str] = mapped_column(Text)
    paragraph: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))

    paper: Mapped[Paper] = relationship(back_populates="chunks")

    __table_args__ = (
        # A citation names a paper, a paragraph and the piece within it, so that
        # triple has to identify one row.
        Index("uq_chunk_location", "arxiv_id", "paragraph", "chunk_index", unique=True),
    )
