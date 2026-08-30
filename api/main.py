import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import text
from sse_starlette.sse import EventSourceResponse

from api.schemas import Passage, SearchRequest, SearchResponse
from db.session import SessionLocal
from generation import graph, provider
from retrieval import hybrid, sparse

logger = logging.getLogger(__name__)


def get_session():
    with SessionLocal() as session:
        yield session


Session = Annotated[object, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warms the BM25 index and the encoders so the first caller does not pay for them.

    A failure is recorded rather than raised: the service still answers the
    health check, which lets an orchestrator report a degraded state instead of
    restarting the container in a loop.
    """
    try:
        with SessionLocal() as session:
            session.execute(text("select 1"))
            sparse.get_index(session)
        app.state.ready = True
    except Exception:
        logger.exception("startup failed")
        app.state.ready = False

    yield


app = FastAPI(
    title="Research Copilot",
    description=(
        "Retrieval and grounded answering over the quantum cryptography literature "
        "on arXiv. Every answer cites the paragraph it came from."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.ready = False


@app.get("/")
def health(request: Request) -> dict:
    return {"status": "ok", "ready": request.app.state.ready}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, session: Session) -> SearchResponse:
    """Retrieval only. It reaches no language model, so it consumes no quota."""
    hits = hybrid.search(session, request.question, request.top_k)
    return SearchResponse(
        question=request.question,
        passages=[
            Passage(
                number=number,
                arxiv_id=hit.arxiv_id,
                section=hit.section,
                paragraph=hit.paragraph,
                text=hit.text,
                score=hit.score,
            )
            for number, hit in enumerate(hits, start=1)
        ],
    )


@app.get("/answer")
async def answer(
    request: Request,
    session: Session,
    question: Annotated[str, Query(min_length=3)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> EventSourceResponse:
    """
    Streams the passages, then the answer, then the citations it resolved.

    Retrieval and generation are synchronous and CPU bound, so the generator runs
    in a worker thread and its events are handed to the event loop one at a time.
    """
    try:
        backend = provider.build()
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from None

    async def events() -> AsyncIterator[dict]:
        iterator = graph.stream(session, question, backend, top_k)
        loop = asyncio.get_running_loop()

        while True:
            if await request.is_disconnected():
                break
            item = await loop.run_in_executor(None, lambda: next(iterator, None))
            if item is None:
                break
            yield {"event": item["event"], "data": json.dumps(item["data"])}

    return EventSourceResponse(events())


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
