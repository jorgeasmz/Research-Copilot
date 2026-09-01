"""
The encoders as exported graphs, so serving needs no deep learning framework.

Pooling and normalisation are inside the graph, and tokenisation goes through
the Rust tokenizer rather than through transformers, which leaves the runtime
holding onnxruntime and numpy.
"""

import functools
import os
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from ingest import config as ingest_config
from retrieval import config as retrieval_config

ARTIFACTS = Path(os.getenv("ENCODER_DIR", str(ingest_config.ROOT / "artifacts" / "onnx")))

# One thread per session. The service handles a handful of concurrent requests
# and the platform allots a fraction of a core, where more threads only contend.
THREADS = 1


def _session(path: Path) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS

    # The arena keeps every block it allocates and each distinct tensor shape
    # asks for a new one, so a service seeing varied questions grows without
    # bound: over fifty queries it reached 1.2 GB against 257 MB allocating per
    # call. Padding to fixed widths bounds that growth, but only at 1.4 GB and
    # at the cost of computing over the padding. EVALUATION.md carries both.
    options.enable_cpu_mem_arena = False

    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _pad(encodings, max_length: int) -> dict[str, np.ndarray]:
    """Builds the input tensors, padded to the longest item in the batch."""
    width = min(max((len(e.ids) for e in encodings), default=1), max_length)

    def stack(values: list[list[int]]) -> np.ndarray:
        padded = np.zeros((len(values), width), dtype=np.int64)
        for row, sequence in enumerate(values):
            trimmed = sequence[:width]
            padded[row, : len(trimmed)] = trimmed
        return padded

    return {
        "input_ids": stack([e.ids for e in encodings]),
        "attention_mask": stack([e.attention_mask for e in encodings]),
        "token_type_ids": stack([e.type_ids for e in encodings]),
    }


class Encoder:
    """Turns text into unit vectors, one row per input."""

    def __init__(self, folder: Path, max_length: int):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.max_length = max_length
        self.session = _session(folder / "model.onnx")
        self.inputs = {tensor.name for tensor in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        feeds = _pad(self.tokenizer.encode_batch(texts), self.max_length)
        return self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]


class PairScorer:
    """Scores a query against each passage, reading the two together."""

    def __init__(self, folder: Path, max_length: int):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.max_length = max_length
        self.session = _session(folder / "model.onnx")
        self.inputs = {tensor.name for tensor in self.session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch([(query, passage) for passage in passages])
        feeds = _pad(encoded, self.max_length)
        logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]
        return logits.reshape(-1)


@functools.lru_cache(maxsize=1)
def passages() -> Encoder:
    return Encoder(ARTIFACTS / "passages", ingest_config.EMBEDDING_MAX_LENGTH)


@functools.lru_cache(maxsize=1)
def pairs() -> PairScorer:
    return PairScorer(ARTIFACTS / "pairs", retrieval_config.RERANK_MAX_LENGTH)


def encode_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    return np.concatenate(
        [
            passages().encode(texts[start : start + batch_size])
            for start in range(0, len(texts), batch_size)
        ]
    )


def encode_query(text: str) -> np.ndarray:
    """Applies the instruction prefix the bi-encoder was trained to expect."""
    return passages().encode([f"{ingest_config.QUERY_PREFIX}{text}"])[0]
