"""
The library implementation of BM25, kept as the reference the service is checked against.

It is not what the service runs. A dictionary of term frequencies per document
costs 558 MB over this corpus against 44 MB for the same scoring over a sparse
matrix, which is why retrieval/bm25.py exists.
"""

import logging
import re
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from db.models import Chunk
from retrieval.dense import Candidate

logger = logging.getLogger(__name__)

TOKEN = re.compile(r"[a-z0-9]+")

_index: "Index | None" = None
_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Index:
    bm25: BM25Okapi
    chunks: list[Chunk]


def build(session) -> Index:
    """
    Reads every chunk and builds the term index.

    The index lives in the process rather than in the database: Postgres full
    text search ranks with ts_rank, which is not BM25, and the comparison the
    evaluation reports needs the same scoring function the literature uses. At
    corpus scale this is the component that would move into the database first.
    """
    chunks = list(session.scalars(select(Chunk).order_by(Chunk.id)))
    logger.info("indexing %d chunks for BM25", len(chunks))
    return Index(bm25=BM25Okapi([tokenize(c.text) for c in chunks]), chunks=chunks)


def get_index(session) -> Index:
    """Builds the index at most once. Locked for the same reason the loaders are."""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = build(session)
    return _index


def reset_index() -> None:
    """Drops the cached index, so a re-ingested corpus is picked up."""
    global _index
    _index = None


def search(session, query: str, limit: int) -> list[Candidate]:
    index = get_index(session)
    scores = index.bm25.get_scores(tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
    return [
        Candidate(
            chunk_id=index.chunks[i].id,
            arxiv_id=index.chunks[i].arxiv_id,
            section=index.chunks[i].section,
            paragraph=index.chunks[i].paragraph,
            text=index.chunks[i].text,
            score=float(scores[i]),
        )
        for i in ranked
        if scores[i] > 0
    ]
