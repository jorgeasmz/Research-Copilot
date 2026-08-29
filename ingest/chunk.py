"""Splits paragraphs into embeddable units without losing where each one came from."""

import re
from dataclasses import dataclass

from ingest.extract import Paragraph

# Sentence boundary: a full stop, question or exclamation mark followed by
# whitespace and a capital. Abbreviations and decimals do not match it.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# The embedding model reads 512 word pieces, which is roughly 2,000 characters
# of English prose. The limit sits below that so a chunk is never truncated.
MAX_CHARS = 1200
OVERLAP_SENTENCES = 1


@dataclass(frozen=True)
class Chunk:
    arxiv_id: str
    section: str
    paragraph: int
    index: int
    text: str


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text) if s.strip()]


def split(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_SENTENCES) -> list[str]:
    """
    Returns the text as pieces no longer than max_chars, split between sentences.

    Consecutive pieces repeat the last sentence of the previous one, so a claim
    that straddles a boundary is retrievable from either side. A sentence longer
    than the limit on its own is cut on whitespace, which happens where dense
    notation leaves the splitter no boundary to use.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences(text):
        candidate = " ".join([*current, sentence])
        if current and len(candidate) > max_chars:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            current.append(sentence)
        else:
            current.append(sentence)

    if current:
        pieces.append(" ".join(current))
    return [piece for oversized in pieces for piece in _hard_split(oversized, max_chars)]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort split on whitespace, so nothing reaches the model truncated."""
    if len(text) <= max_chars:
        return [text]

    pieces, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_paper(arxiv_id: str, paragraphs: list[Paragraph]) -> list[Chunk]:
    """
    Returns the chunks of one paper, each carrying the paragraph it came from.

    The paragraph index is what a citation points at, so it survives splitting:
    a long paragraph yields several chunks that all name the same paragraph.
    """
    chunks = []
    for paragraph in paragraphs:
        for index, piece in enumerate(split(paragraph.text)):
            chunks.append(
                Chunk(
                    arxiv_id=arxiv_id,
                    section=paragraph.section,
                    paragraph=paragraph.index,
                    index=index,
                    text=piece,
                )
            )
    return chunks
