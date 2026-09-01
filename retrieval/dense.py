"""Nearest-neighbour search over the stored embeddings."""

from dataclasses import dataclass

from sqlalchemy import select

from db.models import Chunk
from retrieval.encoders import encode_query


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    arxiv_id: str
    section: str
    paragraph: int
    text: str
    score: float


def search(session, query: str, limit: int) -> list[Candidate]:
    """
    Returns the nearest chunks by cosine distance.

    The ordering expression is the one the HNSW index was built on, which is what
    lets the planner use it instead of scanning the table.
    """
    vector = encode_query(query)
    distance = Chunk.embedding.cosine_distance(vector)

    rows = session.execute(
        select(Chunk, distance.label("distance")).order_by(distance).limit(limit)
    ).all()

    return [
        Candidate(
            chunk_id=chunk.id,
            arxiv_id=chunk.arxiv_id,
            section=chunk.section,
            paragraph=chunk.paragraph,
            text=chunk.text,
            score=1.0 - float(distance),
        )
        for chunk, distance in rows
    ]
