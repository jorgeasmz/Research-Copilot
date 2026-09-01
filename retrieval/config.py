"""Retrieval settings, separate from ingestion so serving does not import the pipeline."""

# Each retriever returns this many candidates before fusion, and the reranker
# scores the fused list. Widening the candidate pool costs recall nothing and
# latency little; it is the reranker depth that dominates.
CANDIDATES = 50
TOP_K = 5

# Reciprocal rank fusion weights ranks rather than scores, which is what lets a
# lexical and a dense ranking combine without calibrating either.
RRF_K = 60

# A cross-encoder reads the query and the passage together, so it resolves cases
# a bi-encoder cannot. It costs 1.5 s per query and earns it on this corpus,
# though not on the public benchmark. See the README.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 25
RERANK_MAX_LENGTH = 512
RERANK_BY_DEFAULT = True
