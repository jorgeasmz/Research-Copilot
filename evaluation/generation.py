"""
Measures the answering pipeline: whether citations resolve, whether claims carry
them, and whether the model declines when the corpus cannot answer.

The second set is the one that matters most. A system that cites well on
questions its corpus covers, and invents an answer on questions it does not, is
worse than one that refuses, because the failure is invisible to the reader.

Usage: python -m evaluation.generation
"""

import argparse
import json
import logging
import statistics
import time

from db.session import SessionLocal
from generation import graph, prompt, provider
from generation.citations import check
from ingest import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ANSWERABLE = config.DATA / "questions.json"
UNANSWERABLE = config.DATA / "unanswerable.json"


def measure(session, question: str, backend) -> dict:
    """Runs one question through the pipeline, timing the first fragment."""
    started = time.perf_counter()
    state = graph.gather(session, question)
    retrieved = time.perf_counter()

    first = None
    fragments = []
    for fragment in backend.stream(prompt.build(question, state["passages"])):
        if first is None:
            first = time.perf_counter()
        fragments.append(fragment)

    finished = time.perf_counter()
    checked = check("".join(fragments), state["passages"])

    return {
        "refused": checked.refused,
        "invalid": len(checked.invalid),
        "citations": len(checked.citations),
        "grounded": checked.grounded,
        "retrieval_ms": (retrieved - started) * 1000,
        "first_token_ms": ((first or finished) - retrieved) * 1000,
        "total_ms": (finished - started) * 1000,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    answerable = [q["question"] for q in json.loads(ANSWERABLE.read_text(encoding="utf-8"))]
    unanswerable = json.loads(UNANSWERABLE.read_text(encoding="utf-8"))
    if args.limit:
        answerable, unanswerable = answerable[: args.limit], unanswerable[: args.limit]

    backend = provider.build()
    with SessionLocal() as session:
        # The first query pays for the BM25 index and the encoders. Measuring
        # it would report a startup cost as a per-query one.
        graph.gather(session, answerable[0])

        covered = [measure(session, q, backend) for q in answerable]
        logger.info("answered %d covered questions", len(covered))
        outside = [measure(session, q, backend) for q in unanswerable]
        logger.info("answered %d uncovered questions", len(outside))

    latencies = [r["total_ms"] for r in covered]
    first = [r["first_token_ms"] for r in covered]

    print(f"\n## Covered questions ({len(covered)})\n")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Answers with a fabricated citation | {sum(1 for r in covered if r['invalid'])} |")
    print(f"| Mean citations per answer | {statistics.mean(r['citations'] for r in covered):.1f} |")
    print(f"| Sentences carrying a citation | {statistics.mean(r['grounded'] for r in covered):.2f} |")
    print(f"| Declined to answer | {sum(1 for r in covered if r['refused'])} |")

    print(f"\n## Uncovered questions ({len(outside)})\n")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Declined to answer | {sum(1 for r in outside if r['refused'])}/{len(outside)} |")
    print(f"| Answered anyway | {sum(1 for r in outside if not r['refused'])}/{len(outside)} |")
    print(f"| Fabricated citations | {sum(r['invalid'] for r in outside)} |")

    print("\n## Latency, covered questions\n")
    print("| Stage | p50 ms | p95 ms |")
    print("|---|---:|---:|")
    print(
        f"| Retrieval | {statistics.median(r['retrieval_ms'] for r in covered):.0f} "
        f"| {percentile([r['retrieval_ms'] for r in covered], 0.95):.0f} |"
    )
    print(f"| First token | {statistics.median(first):.0f} | {percentile(first, 0.95):.0f} |")
    print(f"| Complete answer | {statistics.median(latencies):.0f} | {percentile(latencies, 0.95):.0f} |")


if __name__ == "__main__":
    main()
