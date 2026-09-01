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


def test_answer_without_a_key_asks_for_one(client, monkeypatch):
    """Retrieval is open; generating an answer needs a key the caller supplies."""
    def explode(api_key=""):
        raise RuntimeError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(main.provider, "build", explode)

    response = client.get("/answer", params={"question": "How is the key rate computed?"})

    assert response.status_code == 401
    assert "X-Api-Key" in response.json()["detail"]


def test_the_caller_key_reaches_the_provider(client, monkeypatch):
    seen = {}

    def record(api_key=""):
        seen["key"] = api_key
        return object()

    monkeypatch.setattr(main.provider, "build", record)
    monkeypatch.setattr(main.graph, "stream", lambda s, q, p, k, c: iter([]))

    with client.stream(
        "GET",
        "/answer",
        params={"question": "a real question"},
        headers={"X-Api-Key": "visitor-key"},
    ) as response:
        list(response.iter_lines())

    assert seen["key"] == "visitor-key"


def test_answer_streams_passages_then_tokens_then_citations(client, monkeypatch):
    monkeypatch.setattr(main.provider, "build", lambda api_key="": object())
    monkeypatch.setattr(
        main.graph,
        "stream",
        lambda s, q, p, k, c: iter(
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
    monkeypatch.setattr(main.provider, "build", lambda api_key="": object())
    monkeypatch.setattr(
        main.graph, "stream", lambda s, q, p, k, c: iter([{"event": "token", "data": "a token"}])
    )

    with client.stream("GET", "/answer", params={"question": "a real question"}) as response:
        payload = [line for line in response.iter_lines() if line.startswith("data:")][0]

    assert json.loads(payload.removeprefix("data:").strip()) == "a token"


def test_health_reports_the_cache(client):
    body = client.get("/").json()

    assert body["cache"]["entries"] == 0
    assert body["cache"]["hit_rate"] == 0.0


def test_the_client_origin_is_allowed(client):
    """The browser drops the request without these headers."""
    response = client.options(
        "/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Api-Key",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "x-api-key" in response.headers["access-control-allow-headers"].lower()


def test_an_unlisted_origin_is_not_allowed(client):
    response = client.options(
        "/search",
        headers={"Origin": "https://elsewhere.example", "Access-Control-Request-Method": "POST"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_health_answers_the_platform_probe(client):
    """Render probes with HEAD, which a GET-only route rejects with 405."""
    assert client.head("/").status_code == 200
