"""
Recomputes every stored embedding with the graph the service reads.

Passages and queries have to live in the same space. Embedding the corpus with
one artifact and the queries with another leaves a distance that is close enough
to look right and wrong enough to cost recall.

Usage: python -m tools.reembed
"""

import logging
import time

from sqlalchemy import select

from db.models import Chunk
from db.session import SessionLocal
from retrieval.encoders import encode_passages

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BATCH = 256


def main() -> None:
    started = time.perf_counter()

    with SessionLocal() as session:
        ids = list(session.scalars(select(Chunk.id).order_by(Chunk.id)))
        logger.info("re-embedding %d chunks", len(ids))

        for start in range(0, len(ids), BATCH):
            window = ids[start : start + BATCH]
            chunks = list(session.scalars(select(Chunk).where(Chunk.id.in_(window))))
            vectors = encode_passages([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
            session.commit()
            logger.info("  %d/%d", min(start + BATCH, len(ids)), len(ids))

    logger.info("done in %.1f min", (time.perf_counter() - started) / 60)


if __name__ == "__main__":
    main()
