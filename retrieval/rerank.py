"""Cross-encoder reranking of a fused candidate list."""

from retrieval import config, encoders
from retrieval.dense import Candidate


def rerank(query: str, candidates: list[Candidate], depth: int | None = None) -> list[Candidate]:
    """
    Rescores the head of the list by reading query and passage together.

    Only the first `depth` candidates are scored: the cross-encoder costs one
    forward pass per pair, so the depth is what sets the latency of a query.
    """
    depth = depth or config.RERANK_DEPTH
    head, tail = candidates[:depth], candidates[depth:]
    if not head:
        return candidates

    scores = encoders.pairs().score(query, [candidate.text for candidate in head])
    rescored = [
        Candidate(
            chunk_id=c.chunk_id,
            arxiv_id=c.arxiv_id,
            section=c.section,
            paragraph=c.paragraph,
            text=c.text,
            score=float(score),
        )
        for c, score in zip(head, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored + tail
