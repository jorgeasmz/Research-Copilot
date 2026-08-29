"""Loads a BEIR dataset, which ships relevance judgements the metrics can be read against."""

import csv
import io
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from ingest import config

logger = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASET = "scifact"
CACHE = config.DATA / "beir"


@dataclass(frozen=True)
class Benchmark:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download(dataset: str = DATASET) -> Path:
    """Fetches and unpacks the dataset once, into the data directory."""
    target = CACHE / dataset
    if target.exists():
        return target

    CACHE.mkdir(parents=True, exist_ok=True)
    logger.info("downloading BEIR %s", dataset)
    response = httpx.get(f"{BEIR_URL}/{dataset}.zip", timeout=300.0, follow_redirects=True)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(CACHE)
    return target


def load(dataset: str = DATASET, split: str = "test") -> Benchmark:
    """
    Returns the corpus, the queries and the judgements for one split.

    A document is the title followed by its text, which is how the BEIR
    baselines index it; indexing the text alone changes the numbers.
    """
    root = download(dataset)

    corpus = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            title = record.get("title", "").strip()
            body = record.get("text", "").strip()
            corpus[record["_id"]] = f"{title} {body}".strip()

    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    queries = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["_id"] in qrels:
                queries[record["_id"]] = record["text"]

    return Benchmark(corpus=corpus, queries=queries, qrels=qrels)
