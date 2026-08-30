import pytest

from generation import graph
from retrieval.dense import Candidate


def passage(number: int) -> Candidate:
    return Candidate(number, f"2512.0000{number}v1", "Results", number, f"passage {number}", 0.0)


class FakeProvider:
    """Answers with a fixed text and records the prompt it was given."""

    def __init__(self, text: str = "A claim [1]."):
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        for word in self.text.split(" "):
            yield word + " "


@pytest.fixture
def retrieval(monkeypatch):
    calls: list[str] = []

    def fake_search(session, query, top_k, **kwargs):
        calls.append(query)
        return [passage(len(calls))]

    monkeypatch.setattr(graph.hybrid, "search", fake_search)
    return calls


def test_a_plain_question_retrieves_once(retrieval):
    state = graph.gather(None, "How is the key rate computed?")

    assert state["comparative"] is False
    assert retrieval == ["How is the key rate computed?"]


def test_a_comparison_retrieves_for_each_side(retrieval):
    """One retrieval returns passages about whichever side dominates the query."""
    state = graph.gather(None, "Compare decoy state protocols versus measurement device independent ones")

    assert state["comparative"] is True
    assert len(retrieval) > 1


def test_a_comparison_keeps_the_original_question_as_a_query(retrieval):
    graph.gather(None, "What is the difference between discrete and continuous variable schemes?")

    assert retrieval[0].startswith("What is the difference")


def test_duplicate_passages_are_returned_once(monkeypatch):
    monkeypatch.setattr(graph.hybrid, "search", lambda s, q, k, **kw: [passage(1)])

    state = graph.gather(None, "Compare A protocol versus B protocol")

    assert len(state["passages"]) == 1


def test_the_context_is_capped(monkeypatch):
    monkeypatch.setattr(
        graph.hybrid, "search", lambda s, q, k, **kw: [passage(i) for i in range(50)]
    )

    state = graph.gather(None, "How is the key rate computed?")

    assert len(state["passages"]) == graph.config.CONTEXT_PASSAGES


def test_the_answer_is_checked_against_what_was_retrieved(retrieval):
    provider = FakeProvider("The rate falls with distance [1].")

    checked, _ = graph.answer(None, "How does distance affect the rate?", provider)

    assert [c.number for c in checked.citations] == [1]
    assert checked.grounded == 1.0


def test_the_prompt_carries_the_passages(retrieval):
    provider = FakeProvider()

    graph.answer(None, "How is the key rate computed?", provider)

    assert "passage 1" in provider.prompts[0]
    assert "2512.00001v1" in provider.prompts[0]


def test_the_stream_sends_passages_before_any_token(retrieval):
    events = list(graph.stream(None, "How is the key rate computed?", FakeProvider()))

    assert events[0]["event"] == "passages"
    assert any(e["event"] == "token" for e in events)
    assert events[-1]["event"] == "citations"


def test_the_stream_reports_a_fabricated_citation(retrieval):
    events = list(graph.stream(None, "How is it computed?", FakeProvider("A claim [9].")))

    assert events[-1]["data"]["invalid"] == [9]


class Boom(Exception):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"status {code}")


def test_a_saturated_model_falls_through_to_the_next(monkeypatch):
    """Retrying a busy model buys nothing; asking a different one does."""
    from generation.provider import GeminiProvider

    monkeypatch.setattr("generation.config.API_KEY", "test-key")
    monkeypatch.setattr("generation.config.RETRY_ATTEMPTS", 1)
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.models = ["busy", "spare"]
    tried: list[str] = []

    def call(name: str) -> str:
        tried.append(name)
        if name == "busy":
            raise Boom(503)
        return "answer"

    assert provider._over_chain(call) == "answer"
    assert tried == ["busy", "spare"]


def test_a_rejected_request_is_not_retried_on_another_model(monkeypatch):
    """A 4xx means the request is wrong, and it will be wrong everywhere."""
    from generation.provider import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.models = ["first", "second"]
    tried: list[str] = []

    def call(name: str) -> str:
        tried.append(name)
        raise Boom(400)

    with pytest.raises(Boom):
        provider._over_chain(call)
    assert tried == ["first"]


def test_an_exhausted_quota_moves_to_the_next_model():
    """The free tier meters per model per day, so another model has its own budget."""
    from generation.provider import GeminiProvider

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.models = ["spent", "fresh"]
    tried: list[str] = []

    def call(name: str) -> str:
        tried.append(name)
        if name == "spent":
            raise Boom(429)
        return "answer"

    assert provider._over_chain(call) == "answer"
    assert tried == ["spent", "fresh"]


def test_an_exhausted_quota_is_not_waited_out():
    """Waiting on a daily quota burns the retry window for nothing."""
    from generation.provider import _retryable, _switchable

    assert _switchable(Boom(429)) is True
    assert _retryable(Boom(429)) is False
    assert _retryable(Boom(503)) is True


def test_a_cached_question_reaches_no_provider(retrieval, monkeypatch):
    """A hit is not a saved millisecond, it is a request the daily quota keeps."""
    from generation.cache import SemanticCache

    monkeypatch.setattr(
        "generation.cache.encode_query", lambda q: __import__("numpy").array([1.0, 0.0])
    )
    cache = SemanticCache()
    provider = FakeProvider("A claim [1].")

    graph.answer(None, "How is the key rate computed?", provider, cache=cache)
    graph.answer(None, "How is the key rate computed?", provider, cache=cache)

    assert len(provider.prompts) == 1
    assert cache.hits == 1


def test_the_stream_reports_whether_the_answer_came_from_cache(retrieval, monkeypatch):
    from generation.cache import SemanticCache

    monkeypatch.setattr(
        "generation.cache.encode_query", lambda q: __import__("numpy").array([1.0, 0.0])
    )
    cache = SemanticCache()
    provider = FakeProvider("A claim [1].")
    question = "How is the key rate computed?"

    list(graph.stream(None, question, provider, cache=cache))
    events = list(graph.stream(None, question, provider, cache=cache))

    assert events[-1]["data"]["cached"] is True
    assert len(provider.prompts) == 1
