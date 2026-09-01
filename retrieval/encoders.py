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
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class Encoder:
    """Turns text into unit vectors, one row per input."""

    def __init__(self, folder: Path, max_length: int):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.session = _session(folder / "model.onnx")
        self.inputs = {tensor.name for tensor in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
        }
        return self.session.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]


class PairScorer:
    """Scores a query against each passage, reading the two together."""

    def __init__(self, folder: Path, max_length: int):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.session = _session(folder / "model.onnx")
        self.inputs = {tensor.name for tensor in self.session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch([(query, passage) for passage in passages])
        feeds = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
        }
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
        [passages().encode(texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)]
    )


def encode_query(text: str) -> np.ndarray:
    """Applies the instruction prefix the bi-encoder was trained to expect."""
    return passages().encode([f"{ingest_config.QUERY_PREFIX}{text}"])[0]
