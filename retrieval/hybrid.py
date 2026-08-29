"""Combines the lexical and dense rankings, then reranks the result."""

from retrieval import config, dense, rerank, sparse
from retrieval.dense import Candidate


def fuse(rankings: list[list[Candidate]], k: int = config.RRF_K) -> list[Candidate]:
    """
    Reciprocal rank fusion over several rankings of the same corpus.

    Each list contributes 1/(k + rank) to the chunks it returns. Ranks are
    comparable across retrievers while scores are not: a BM25 score and a cosine
    similarity live on different scales and neither is calibrated.
    """
    totals: dict[int, float] = {}
    seen: dict[int, Candidate] = {}

    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            totals[candidate.chunk_id] = totals.get(candidate.chunk_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(candidate.chunk_id, candidate)

    ordered = sorted(totals, key=lambda chunk_id: totals[chunk_id], reverse=True)
    return [
        Candidate(
            chunk_id=chunk_id,
            arxiv_id=seen[chunk_id].arxiv_id,
            section=seen[chunk_id].section,
            paragraph=seen[chunk_id].paragraph,
            text=seen[chunk_id].text,
            score=totals[chunk_id],
        )
        for chunk_id in ordered
    ]


def search(
    session,
    query: str,
    top_k: int = config.TOP_K,
    candidates: int = config.CANDIDATES,
    reranked: bool = config.RERANK_BY_DEFAULT,
) -> list[Candidate]:
    """
    Runs both retrievers, fuses their rankings and returns the top passages.

    Reranking is off by default: on the judged benchmark the cross-encoder
    lowers nDCG and costs seconds per query. The stage is kept because that
    result is a property of this corpus and these models, not of reranking.
    """
    rankings = [
        dense.search(session, query, candidates),
        sparse.search(session, query, candidates),
    ]
    fused = fuse(rankings)
    if reranked:
        fused = rerank.rerank(query, fused)
    return fused[:top_k]
