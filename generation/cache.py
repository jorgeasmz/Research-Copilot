"""
Answers repeated questions without spending a request.

The free tier meters requests per day per model, so a hit is not a saved
millisecond but a question the system can still answer today.

Matching is by embedding similarity, and within one field that separates the two
classes poorly. Measured on this corpus, a pair asking different things about
related subjects reached 0.823, above a genuine paraphrase at 0.733: questions in
one field share vocabulary, and the embedding is dominated by what they are about
rather than by what they ask. The classes overlap, so no threshold separates
them, and the two errors are not equal. A miss spends a request; a false hit
answers a different question and cites passages retrieved for it.

The threshold is therefore set where false hits are unlikely rather than where
recall is best, which leaves this closer to an exact-match cache that tolerates
rewording than to a semantic one.
"""

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from ingest.embed import encode_query

logger = logging.getLogger(__name__)

# Above this cosine similarity two questions are treated as the same question.
# Calibrated on the 171 pairs of the evaluation question set, which reach 0.723
# at most, against five paraphrases, which start at 0.766. At 0.75 neither set
# is misclassified. Negatives drawn from unrelated fields would put the
# threshold far lower and admit questions that differ only in what they ask.
SIMILARITY = 0.90
CAPACITY = 256


@dataclass
class Entry:
    question: str
    answer: str
    vector: np.ndarray


@dataclass
class SemanticCache:
    """In-process store. It holds one node's answers, not a shared tier."""

    threshold: float = SIMILARITY
    capacity: int = CAPACITY
    entries: list[Entry] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def lookup(self, question: str) -> str | None:
        vector = encode_query(question)
        with self._lock:
            best, score = self._nearest(vector)
            if best is not None and score >= self.threshold:
                self.hits += 1
                logger.info("cache hit at %.3f for %r", score, question)
                return best.answer
            self.misses += 1
        return None

    def store(self, question: str, answer: str) -> None:
        with self._lock:
            self.entries.append(Entry(question, answer, encode_query(question)))
            if len(self.entries) > self.capacity:
                # Oldest first: a corpus does not change during a run, so age is
                # the only thing separating two otherwise equal entries.
                self.entries.pop(0)

    def _nearest(self, vector: np.ndarray) -> tuple[Entry | None, float]:
        if not self.entries:
            return None, 0.0
        scores = np.asarray([float(entry.vector @ vector) for entry in self.entries])
        index = int(scores.argmax())
        return self.entries[index], float(scores[index])

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
