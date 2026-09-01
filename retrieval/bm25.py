"""
BM25 over a sparse term-document matrix.

The scoring is the textbook Okapi formula. What differs from the usual library
implementation is the storage: term frequencies live in one compressed sparse
matrix rather than in a dictionary per document, which is what makes the index
fit in a service rather than dominate it.
"""

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sqlalchemy import select

from db.models import Chunk
from retrieval.dense import Candidate

logger = logging.getLogger(__name__)

TOKEN = re.compile(r"[a-z0-9]+")

# The parameters Robertson and Walker report as generally effective, and the
# ones rank_bm25 defaults to, so the two implementations score alike.
K1 = 1.5
B = 0.75

_index: "Index | None" = None
_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Index:
    """Term frequencies, the vocabulary that addresses them and the document norms."""

    frequencies: sparse.csc_matrix
    vocabulary: dict[str, int]
    idf: np.ndarray
    length_norm: np.ndarray
    chunk_ids: np.ndarray

    @property
    def nbytes(self) -> int:
        return (
            self.frequencies.data.nbytes
            + self.frequencies.indices.nbytes
            + self.frequencies.indptr.nbytes
            + self.idf.nbytes
            + self.length_norm.nbytes
        )


def build(session) -> Index:
    """Reads every chunk and builds the term index."""
    rows = session.execute(select(Chunk.id, Chunk.text).order_by(Chunk.id)).all()
    logger.info("indexing %d chunks", len(rows))

    vocabulary: dict[str, int] = {}
    indices: list[int] = []
    data: list[int] = []
    indptr = [0]
    lengths = np.empty(len(rows), dtype=np.float32)

    for position, (_, text) in enumerate(rows):
        counts: dict[int, int] = {}
        tokens = tokenize(text)
        for token in tokens:
            term = vocabulary.setdefault(token, len(vocabulary))
            counts[term] = counts.get(term, 0) + 1
        indices.extend(counts)
        data.extend(counts.values())
        indptr.append(len(indices))
        lengths[position] = len(tokens)

    frequencies = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr)),
        shape=(len(rows), len(vocabulary)),
    )

    # Document frequency is the count of non-zero entries down each column, and
    # the smoothing is the form that keeps the weight of a term in every document
    # positive rather than zero.
    document_frequency = np.diff(frequencies.tocsc().indptr).astype(np.float32)
    total = len(rows)
    idf = np.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    average = float(lengths.mean()) if total else 0.0
    length_norm = K1 * (1.0 - B + B * lengths / average) if average else np.zeros(total)

    return Index(
        frequencies=frequencies.tocsc(),
        vocabulary=vocabulary,
        idf=idf.astype(np.float32),
        length_norm=length_norm.astype(np.float32),
        chunk_ids=np.asarray([row[0] for row in rows], dtype=np.int64),
    )


def scores(index: Index, question: str) -> np.ndarray:
    """
    Scores every document against the question.

    Only the columns for terms the question uses are touched, so the cost follows
    the length of the question rather than the size of the corpus.
    """
    totals = np.zeros(index.frequencies.shape[0], dtype=np.float32)

    for token in tokenize(question):
        term = index.vocabulary.get(token)
        if term is None:
            continue
        column = index.frequencies.getcol(term).tocoo()
        rows, frequency = column.row, column.data
        totals[rows] += index.idf[term] * (
            frequency * (K1 + 1.0) / (frequency + index.length_norm[rows])
        )

    return totals


def get_index(session) -> Index:
    """Builds the index at most once. Locked for the reason the model loaders are."""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = build(session)
                logger.info("index holds %.1f MB", _index.nbytes / 1048576)
    return _index


def reset_index() -> None:
    global _index
    _index = None


def search(session, question: str, limit: int) -> list[Candidate]:
    index = get_index(session)
    ranked = scores(index, question)

    order = np.argpartition(-ranked, min(limit, len(ranked) - 1))[:limit]
    order = order[np.argsort(-ranked[order])]

    chunks = {
        chunk.id: chunk
        for chunk in session.scalars(
            select(Chunk).where(Chunk.id.in_([int(index.chunk_ids[i]) for i in order]))
        )
    }

    return [
        Candidate(
            chunk_id=chunk.id,
            arxiv_id=chunk.arxiv_id,
            section=chunk.section,
            paragraph=chunk.paragraph,
            text=chunk.text,
            score=float(ranked[i]),
        )
        for i in order
        if ranked[i] > 0 and (chunk := chunks.get(int(index.chunk_ids[i])))
    ]
