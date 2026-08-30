import numpy as np
import pytest

from generation import cache as cache_module
from generation.cache import SemanticCache


@pytest.fixture
def vectors(monkeypatch):
    """Encodes each question as a fixed unit vector, so similarity is controlled."""
    table = {
        "how is the key rate computed": np.array([1.0, 0.0, 0.0]),
        "how do you compute the key rate": np.array([0.99, 0.141, 0.0]),
        "what is a decoy state": np.array([0.0, 1.0, 0.0]),
    }

    def encode(question: str) -> np.ndarray:
        vector = table[question]
        return vector / np.linalg.norm(vector)

    monkeypatch.setattr(cache_module, "encode_query", encode)
    return table


def test_the_same_question_is_answered_without_a_request(vectors):
    cache = SemanticCache()
    cache.store("how is the key rate computed", "the answer")

    assert cache.lookup("how is the key rate computed") == "the answer"
    assert cache.hits == 1


def test_a_paraphrase_hits(vectors):
    """Two wordings of one question should not cost two requests."""
    cache = SemanticCache()
    cache.store("how is the key rate computed", "the answer")

    assert cache.lookup("how do you compute the key rate") == "the answer"


def test_a_different_question_misses(vectors):
    cache = SemanticCache()
    cache.store("how is the key rate computed", "the answer")

    assert cache.lookup("what is a decoy state") is None
    assert cache.misses == 1


def test_an_empty_cache_misses(vectors):
    assert SemanticCache().lookup("what is a decoy state") is None


def test_a_higher_threshold_rejects_the_paraphrase(vectors):
    """The threshold is what trades quota against answering a different question."""
    cache = SemanticCache(threshold=0.999)
    cache.store("how is the key rate computed", "the answer")

    assert cache.lookup("how do you compute the key rate") is None


def test_the_default_threshold_rejects_a_measured_near_neighbour(vectors):
    """
    An adjacent question reached 0.823 on this corpus, above a paraphrase at 0.733.

    The classes overlap, so the default sits above both rather than between them.
    """
    cache = SemanticCache()
    near = np.array([0.823, np.sqrt(1 - 0.823**2), 0.0])
    cache.entries.append(
        cache_module.Entry("stored", "the answer", np.array([1.0, 0.0, 0.0]))
    )
    monkey = near / np.linalg.norm(near)

    assert float(monkey @ cache.entries[0].vector) == pytest.approx(0.823, abs=1e-3)
    assert 0.823 < cache.threshold


def test_the_oldest_entry_is_dropped_at_capacity(vectors):
    cache = SemanticCache(capacity=1)
    cache.store("how is the key rate computed", "first")
    cache.store("what is a decoy state", "second")

    assert len(cache.entries) == 1
    assert cache.lookup("what is a decoy state") == "second"


def test_the_hit_rate_counts_both_outcomes(vectors):
    cache = SemanticCache()
    cache.store("how is the key rate computed", "the answer")
    cache.lookup("how is the key rate computed")
    cache.lookup("what is a decoy state")

    assert cache.hit_rate == pytest.approx(0.5)
