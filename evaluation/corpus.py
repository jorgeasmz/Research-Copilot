"""
Measures retrieval against a hand-written question set on the ingested corpus.

The questions were written from specific passages, so a result is judged twice:
whether the answering paper is retrieved at all, and whether the passage the
question came from is among the returned chunks. The set is small and
hand-built; the comparable figures come from the BEIR benchmark instead.

Usage: python -m evaluation.corpus [--top-k 5]
"""

import argparse
import json
import statistics
import time

from db.session import SessionLocal
from evaluation.metrics import reciprocal_rank
from ingest import config
from retrieval import bm25, dense, hybrid, lexical, sparse

QUESTIONS = config.DATA / "questions.json"


def evaluate(session, questions: list[dict], top_k: int) -> dict:
    strategies = {
        "bm25 (rank_bm25)": lambda q: sparse.search(session, q, top_k),
        "bm25 (sparse matrix)": lambda q: bm25.search(session, q, top_k),
        "postgres fts": lambda q: lexical.search(session, q, top_k),
        "dense": lambda q: dense.search(session, q, top_k),
        "hybrid": lambda q: hybrid.search(session, q, top_k, reranked=False),
        "hybrid+rerank": lambda q: hybrid.search(session, q, top_k, reranked=True),
    }

    results = {}
    for name, run in strategies.items():
        papers, passages, ranks, latencies = [], [], [], []

        for item in questions:
            started = time.perf_counter()
            hits = run(item["question"])
            latencies.append((time.perf_counter() - started) * 1000)

            retrieved = [h.arxiv_id for h in hits]
            papers.append(item["arxiv_id"] in retrieved)
            ranks.append(reciprocal_rank(retrieved, {item["arxiv_id"]}))

            if item.get("paragraph") is not None:
                passages.append(
                    any(
                        h.arxiv_id == item["arxiv_id"] and h.paragraph == item["paragraph"]
                        for h in hits
                    )
                )

        results[name] = {
            "paper": statistics.mean(papers),
            "passage": statistics.mean(passages) if passages else float("nan"),
            "mrr": statistics.mean(ranks),
            "latency": statistics.median(latencies),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    anchored = sum(1 for q in questions if q.get("paragraph") is not None)

    with SessionLocal() as session:
        results = evaluate(session, questions, args.top_k)

    print(f"\n{len(questions)} questions, {anchored} anchored to a passage, top-{args.top_k}\n")
    print("| Retriever | Paper found | Passage found | MRR | Latency ms |")
    print("|---|---:|---:|---:|---:|")
    for name, values in results.items():
        print(
            f"| `{name}` | {values['paper']:.2f} | {values['passage']:.2f} "
            f"| {values['mrr']:.3f} | {values['latency']:.0f} |"
        )


if __name__ == "__main__":
    main()
