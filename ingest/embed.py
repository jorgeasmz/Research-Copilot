"""Encodes passages and queries with the retrieval model."""

import functools

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import config


@functools.lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    """Loads the encoder once. The first call downloads roughly 130 MB."""
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


def encode_passages(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    return model().encode(
        texts,
        batch_size=batch_size or config.EMBEDDING_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def encode_query(text: str) -> np.ndarray:
    """Applies the instruction prefix the model was trained to expect on queries."""
    return model().encode(
        f"{config.QUERY_PREFIX}{text}", normalize_embeddings=True, show_progress_bar=False
    )
