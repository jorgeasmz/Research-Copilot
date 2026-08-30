"""
Builds the corpus: arXiv metadata, LaTeX source, paragraphs, chunks and embeddings.

Usage: python -m ingest.pipeline [--limit 300]
"""

import argparse
import logging
import time

import httpx
from sqlalchemy import select

from db.models import Chunk as ChunkRow
from db.models import Paper as PaperRow
from db.session import SessionLocal
from ingest import config
from ingest.arxiv import Paper, search
from ingest.chunk import chunk_paper
from ingest.embed import encode_passages
from ingest.extract import paragraphs
from ingest.source import fetch

logging.basicConfig(level=logging.INFO, format="%(message)s")
# httpx logs one line per request at INFO, which buries the ingestion output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def already_ingested(session) -> set[str]:
    return set(session.scalars(select(PaperRow.arxiv_id)))


def ingest_paper(session, paper: Paper, latex: str) -> int:
    """Stores one paper with its chunks. Returns the number of chunks written."""
    chunks = chunk_paper(paper.arxiv_id, paragraphs(latex))
    if not chunks:
        return 0

    vectors = encode_passages([c.text for c in chunks])

    session.add(
        PaperRow(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            categories=paper.categories,
            published=paper.published,
            updated=paper.updated,
        )
    )
    session.add_all(
        ChunkRow(
            arxiv_id=chunk.arxiv_id,
            section=chunk.section,
            paragraph=chunk.paragraph,
            chunk_index=chunk.index,
            text=chunk.text,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    session.commit()
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=config.CORPUS_SIZE)
    parser.add_argument("--query", default=config.SEARCH_QUERY)
    args = parser.parse_args()

    started = time.perf_counter()
    client = httpx.Client(
        headers={"User-Agent": config.USER_AGENT}, timeout=120.0, follow_redirects=True
    )

    with SessionLocal() as session:
        seen = already_ingested(session)
        logger.info("%d papers already ingested", len(seen))

        stored = skipped = 0
        for paper in search(args.query, args.limit, client):
            if paper.arxiv_id in seen:
                continue

            latex = fetch(paper, client)
            if not latex:
                skipped += 1
                continue

            written = ingest_paper(session, paper, latex)
            if written:
                stored += 1
                logger.info("%s  %3d chunks  %s", paper.arxiv_id, written, paper.title[:56])
            else:
                # A source that renders to nothing is usually a submission whose
                # body sits in a format the renderer does not read.
                skipped += 1

    client.close()
    logger.info(
        "\n%d papers stored, %d skipped, %.1f min",
        stored,
        skipped,
        (time.perf_counter() - started) / 60,
    )


if __name__ == "__main__":
    main()
