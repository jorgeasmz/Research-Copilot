"""
Measures the retrieval stack on BEIR SciFact, whose judgements make the numbers comparable.

The dataset stands in for the corpus so the configurations can be compared
against published baselines. The fusion and reranking code is the same code the
service runs; only the index is held in memory rather than in Postgres.

Usage: python -m evaluation.benchmark
"""

import argparse
import logging
import statistics
import time

import numpy as np

from evaluation import beir
from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ingest import config
from retrieval import config as retrieval_config
from retrieval.bm25 import tokenize
from retrieval.dense import Candidate
from retrieval.encoders import encode_passages, encode_query
from retrieval.hybrid import fuse
from retrieval.rerank import rerank

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CUTOFF = 10


def as_candidates(order, scores, ids, texts) -> list[Candidate]:
    return [
        Candidate(
            chunk_id=int(i),
            arxiv_id=ids[i],
            section="",
            paragraph=0,
            text=texts[i],
            score=float(scores[i]),
        )
        for i in order
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=0, help="Limit for a quick run.")
    parser.add_argument("--candidates", type=int, default=retrieval_config.CANDIDATES)
    parser.add_argument("--reranker", default=retrieval_config.RERANKER_MODEL)
    parser.add_argument("--depth", type=int, default=retrieval_config.RERANK_DEPTH)
    args = parser.parse_args()

    from rank_bm25 import BM25Okapi

    benchmark = beir.load()
    ids = list(benchmark.corpus)
    texts = [benchmark.corpus[i] for i in ids]
    logger.info("%d documents, %d queries", len(ids), len(benchmark.queries))

    # Encoding the corpus dominates a run, and it does not change between
    # configurations, so it is cached beside the dataset.
    cache = beir.CACHE / f"{beir.DATASET}-{config.EMBEDDING_MODEL.split('/')[-1]}.npy"
    if cache.exists():
        matrix = np.load(cache)
        logger.info("corpus embeddings read from cache")
    else:
        started = time.perf_counter()
        matrix = np.asarray(encode_passages(texts))
        np.save(cache, matrix)
        logger.info("corpus encoded in %.1f min", (time.perf_counter() - started) / 60)
    lexical = BM25Okapi([tokenize(text) for text in texts])

    query_ids = list(benchmark.queries)[: args.queries or None]
    results = {name: {"ndcg": [], "recall": [], "mrr": [], "latency": []} for name in
               ("bm25", "dense", "hybrid", "hybrid+rerank")}

    for query_id in query_ids:
        query = benchmark.queries[query_id]
        judgements = benchmark.qrels[query_id]
        relevant = {doc for doc, grade in judgements.items() if grade > 0}

        similarity = matrix @ np.asarray(encode_query(query))
        dense_order = np.argsort(-similarity)[: args.candidates]
        dense_hits = as_candidates(dense_order, similarity, ids, texts)

        lexical_scores = lexical.get_scores(tokenize(query))
        lexical_order = np.argsort(-lexical_scores)[: args.candidates]
        lexical_hits = as_candidates(lexical_order, lexical_scores, ids, texts)

        fused = fuse([dense_hits, lexical_hits])

        started = time.perf_counter()
        reranked = rerank(query, fused, depth=args.depth, name=args.reranker)
        rerank_ms = (time.perf_counter() - started) * 1000

        for name, ranking, latency in (
            ("bm25", lexical_hits, 0.0),
            ("dense", dense_hits, 0.0),
            ("hybrid", fused, 0.0),
            ("hybrid+rerank", reranked, rerank_ms),
        ):
            ranked = [candidate.arxiv_id for candidate in ranking]
            results[name]["ndcg"].append(ndcg_at_k(ranked, judgements, CUTOFF))
            results[name]["recall"].append(recall_at_k(ranked, relevant, CUTOFF))
            results[name]["mrr"].append(reciprocal_rank(ranked[:CUTOFF], relevant))
            results[name]["latency"].append(latency)

    print(f"\n| Retriever | nDCG@{CUTOFF} | Recall@{CUTOFF} | MRR@{CUTOFF} | Rerank ms |")
    print("|---|---:|---:|---:|---:|")
    for name, values in results.items():
        latency = statistics.median(values["latency"])
        print(
            f"| `{name}` | {statistics.mean(values['ndcg']):.3f} "
            f"| {statistics.mean(values['recall']):.3f} "
            f"| {statistics.mean(values['mrr']):.3f} "
            f"| {latency:.0f} |"
        )


if __name__ == "__main__":
    main()
