from retrieval.dense import Candidate
from retrieval.hybrid import fuse


def candidate(chunk_id: int, score: float = 0.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        arxiv_id=f"paper-{chunk_id}",
        section="Results",
        paragraph=chunk_id,
        text=f"text {chunk_id}",
        score=score,
    )


def test_a_document_ranked_by_both_retrievers_outranks_one_ranked_by_either():
    dense = [candidate(1), candidate(2)]
    lexical = [candidate(3), candidate(1)]

    fused = fuse([dense, lexical])

    assert fused[0].chunk_id == 1


def test_scores_are_ignored_in_favour_of_ranks():
    """A BM25 score and a cosine similarity live on scales that are not comparable."""
    dense = [candidate(1, score=0.9), candidate(2, score=0.8)]
    lexical = [candidate(2, score=91.0), candidate(1, score=88.0)]

    assert [c.chunk_id for c in fuse([dense, lexical])] == [1, 2]


def test_every_candidate_survives_fusion():
    fused = fuse([[candidate(1)], [candidate(2)], [candidate(3)]])

    assert {c.chunk_id for c in fused} == {1, 2, 3}


def test_provenance_survives_fusion():
    fused = fuse([[candidate(7)], [candidate(7)]])

    assert fused[0].arxiv_id == "paper-7"
    assert fused[0].paragraph == 7


def test_a_smaller_k_sharpens_the_weight_of_the_top_rank():
    dense = [candidate(1), candidate(2)]
    lexical = [candidate(2), candidate(3)]

    sharp = fuse([dense, lexical], k=1)
    flat = fuse([dense, lexical], k=1000)

    assert sharp[0].chunk_id == 2
    assert flat[0].chunk_id == 2
    assert sharp[0].score > flat[0].score
