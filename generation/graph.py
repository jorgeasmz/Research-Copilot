"""
The answering pipeline: route the question, retrieve for it, then synthesise.

A comparison question is not one retrieval. Asking how two protocols differ and
retrieving once returns passages about whichever of the two dominates the query,
so the graph fans out and retrieves for each side before answering. That branch
is what the graph is for; the rest of the path is linear.
"""

import logging
import re
from collections.abc import Iterator
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from generation import config, prompt
from generation.cache import SemanticCache
from generation.citations import Checked, check
from generation.provider import Provider
from retrieval import hybrid
from retrieval.dense import Candidate

logger = logging.getLogger(__name__)

# Routing is a regular expression rather than a model call. The free tier meters
# requests, so a call spent on classification is a call not spent on answering.
COMPARISON = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|differ from|"
    r"better than|trade-?offs? between)\b",
    re.IGNORECASE,
)
SPLIT = re.compile(r"\b(?:versus|vs\.?|and|between)\b", re.IGNORECASE)


class State(TypedDict, total=False):
    question: str
    comparative: bool
    queries: list[str]
    passages: list[Candidate]


def route(state: State) -> State:
    return {**state, "comparative": bool(COMPARISON.search(state["question"]))}


def plan(state: State) -> State:
    """
    Turns a comparison into one query per side.

    The split is lexical and imperfect. It widens the candidate pool rather than
    deciding anything, so a bad split costs recall, not correctness.
    """
    question = state["question"]
    if not state.get("comparative"):
        return {**state, "queries": [question]}

    parts = [part.strip(" ?.,") for part in SPLIT.split(question)]
    queries = [part for part in parts if len(part.split()) >= 3] or [question]
    return {**state, "queries": [question, *queries]}


def make_retrieve(session, top_k: int):
    def retrieve(state: State) -> State:
        seen: dict[int, Candidate] = {}
        for query in state["queries"]:
            for candidate in hybrid.search(session, query, top_k):
                seen.setdefault(candidate.chunk_id, candidate)
        return {**state, "passages": list(seen.values())[: config.CONTEXT_PASSAGES]}

    return retrieve


def build_graph(session, top_k: int):
    graph = StateGraph(State)
    graph.add_node("route", route)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", make_retrieve(session, top_k))
    graph.add_edge(START, "route")
    graph.add_edge("route", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()


def gather(session, question: str, top_k: int = 5) -> State:
    """Runs everything up to the point where the model is needed."""
    return build_graph(session, top_k).invoke({"question": question})


def answer(
    session,
    question: str,
    provider: Provider,
    top_k: int = 5,
    cache: SemanticCache | None = None,
) -> tuple[Checked, State]:
    state = gather(session, question, top_k)

    cached = cache.lookup(question) if cache else None
    if cached is not None:
        return check(cached, state["passages"]), state

    text = provider.complete(prompt.build(question, state["passages"]))
    if cache:
        cache.store(question, text)
    return check(text, state["passages"]), state


def stream(
    session,
    question: str,
    provider: Provider,
    top_k: int = 5,
    cache: SemanticCache | None = None,
) -> Iterator[dict]:
    """
    Yields the retrieved passages first, then the answer as it is produced.

    The passages arrive before the first token so a reader can see what the
    answer is being drawn from while it is still being written.
    """
    state = gather(session, question, top_k)
    passages = state["passages"]

    yield {
        "event": "passages",
        "data": [
            {
                "number": number,
                "arxiv_id": p.arxiv_id,
                "section": p.section,
                "paragraph": p.paragraph,
                "text": p.text,
            }
            for number, p in enumerate(passages, start=1)
        ],
    }

    cached = cache.lookup(question) if cache else None
    if cached is not None:
        # Replayed in one piece: the answer is already written, and pretending
        # otherwise would report a latency the request did not pay.
        yield {"event": "token", "data": cached}
        text = cached
    else:
        collected = []
        for fragment in provider.stream(prompt.build(question, passages)):
            collected.append(fragment)
            yield {"event": "token", "data": fragment}
        text = "".join(collected)
        if cache:
            cache.store(question, text)

    checked = check(text, passages)
    yield {
        "event": "citations",
        "data": {
            "citations": [vars(c) for c in checked.citations],
            "invalid": checked.invalid,
            "grounded": checked.grounded,
            "refused": checked.refused,
            "cached": cached is not None,
        },
    }
