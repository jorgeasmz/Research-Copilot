"""Corpus and pipeline settings, in one place so an ingestion run is reproducible."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The corpus is scoped to quantum cryptography rather than to quant-ph as a
# whole: a retriever answering across 185,000 unrelated papers measures topic
# separation, while one answering within a field measures passage selection.
SEARCH_QUERY = (
    "cat:quant-ph AND ("
    'abs:"quantum key distribution" OR abs:"quantum cryptography" OR '
    'abs:"BB84" OR abs:"decoy state"'
    ")"
)
CORPUS_SIZE = 300

# arXiv asks for one request every three seconds and a descriptive user agent.
# Exceeding either is what gets a client blocked rather than throttled.
API_URL = "http://export.arxiv.org/api/query"
API_PAGE_SIZE = 100
API_DELAY_SECONDS = 3.0
USER_AGENT = "research-copilot/0.1 (https://github.com/jorgeasmz/Research-Copilot)"

DATA = ROOT / "data"
SOURCE_DIR = DATA / "source"
METADATA_FILE = DATA / "papers.jsonl"

# The LaTeX source carries section commands and paragraph breaks, which the
# rendered PDF does not: in a two-column preprint both are lost to the layout.
SOURCE_URL = "https://arxiv.org/e-print"

# bge-small reads 512 word pieces into 384 dimensions and runs on CPU in
# reasonable time, which the base model at 768 dimensions does not.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH = 32
EMBEDDING_MAX_LENGTH = 512

# bge was trained with an asymmetric objective: queries carry this prefix and
# passages carry none. Embedding both the same way costs measurable recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://copilot:copilot@localhost:5433/copilot")
