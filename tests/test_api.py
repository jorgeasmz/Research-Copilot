import json

import pytest
from fastapi.testclient import TestClient

from api import main
from retrieval.dense import Candidate


def passage(number: int) -> Candidate:
    return Candidate(number, f"2512.0000{number}v1", "Results", number, f"passage {number}", 0.9)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(main.app.dependency_overrides, main.get_session, lambda: None)
    return TestClient(main.app)


def test_health_reports_readiness(client):
    body = client.get("/").json()

    assert body["status"] == "ok"
    assert body["ready"] is False


def test_search_returns_numbered_passages(client, monkeypatch):
    monkeypatch.setattr(main.hybrid, "search", lambda s, q, k: [passage(1), passage(2)])

    body = client.post("/search", json={"question": "How is the key rate computed?"}).json()

    assert [p["number"] for p in body["passages"]] == [1, 2]
    assert body["passages"][0]["arxiv_id"] == "2512.00001v1"


def test_search_carries_the_paragraph_a_citation_points_at(client, monkeypatch):
    monkeypatch.setattr(main.hybrid, "search", lambda s, q, k: [passage(7)])

    body = client.post("/search", json={"question": "anything at all"}).json()

    assert body["passages"][0]["paragraph"] == 7


def test_search_rejects_a_question_that_is_too_short(client):
    assert client.post("/search", json={"question": "hi"}).status_code == 422


def test_search_rejects_an_out_of_range_top_k(client):
    response = client.post("/search", json={"question": "a real question", "top_k": 99})

    assert response.status_code == 422


def test_answer_reports_a_missing_key_as_unavailable(client, monkeypatch):
    """Without credentials the endpoint is unavailable, not broken."""
    def explode():
        raise RuntimeError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(main.provider, "build", explode)

    response = client.get("/answer", params={"question": "How is the key rate computed?"})

    assert response.status_code == 503


def test_answer_streams_passages_then_tokens_then_citations(client, monkeypatch):
    monkeypatch.setattr(main.provider, "build", lambda: object())
    monkeypatch.setattr(
        main.graph,
        "stream",
        lambda s, q, p, k: iter(
            [
                {"event": "passages", "data": [{"number": 1}]},
                {"event": "token", "data": "text"},
                {"event": "citations", "data": {"invalid": []}},
            ]
        ),
    )

    with client.stream("GET", "/answer", params={"question": "a real question"}) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body.index("event: passages") < body.index("event: token")
    assert body.index("event: token") < body.index("event: citations")


def test_streamed_data_is_json(client, monkeypatch):
    monkeypatch.setattr(main.provider, "build", lambda: object())
    monkeypatch.setattr(
        main.graph, "stream", lambda s, q, p, k: iter([{"event": "token", "data": "a token"}])
    )

    with client.stream("GET", "/answer", params={"question": "a real question"}) as response:
        payload = [line for line in response.iter_lines() if line.startswith("data:")][0]

    assert json.loads(payload.removeprefix("data:").strip()) == "a token"
