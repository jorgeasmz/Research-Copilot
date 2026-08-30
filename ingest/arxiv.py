"""Reads paper metadata from the arXiv Atom API."""

import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from ingest import config

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    pdf_url: str = ""


def _text(entry, path: str) -> str:
    node = entry.find(path, NAMESPACES)
    return " ".join(node.text.split()) if node is not None and node.text else ""


def parse_entry(entry) -> Paper:
    """Maps one Atom entry to a Paper. The identifier keeps its version suffix."""
    identifier = _text(entry, "atom:id").rsplit("/", 1)[-1]
    pdf = [
        link.get("href")
        for link in entry.findall("atom:link", NAMESPACES)
        if link.get("title") == "pdf"
    ]
    return Paper(
        arxiv_id=identifier,
        title=_text(entry, "atom:title"),
        abstract=_text(entry, "atom:summary"),
        authors=[_text(a, "atom:name") for a in entry.findall("atom:author", NAMESPACES)],
        categories=[c.get("term") for c in entry.findall("atom:category", NAMESPACES)],
        published=_text(entry, "atom:published"),
        updated=_text(entry, "atom:updated"),
        pdf_url=pdf[0] if pdf else "",
    )


def search(query: str, limit: int, client: httpx.Client | None = None) -> Iterator[Paper]:
    """
    Yields the most recently submitted papers matching a query, newest first.

    The API caps a response at a few hundred entries, so results are paged and
    the delay between pages is the one arXiv asks for.
    """
    owned = client is None
    client = client or httpx.Client(
        headers={"User-Agent": config.USER_AGENT}, timeout=60.0, follow_redirects=True
    )
    try:
        yielded = 0
        while yielded < limit:
            encoded = urllib.parse.urlencode(
                {
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "start": yielded,
                    "max_results": min(config.API_PAGE_SIZE, limit - yielded),
                }
            )
            response = client.get(f"{config.API_URL}?{encoded}")
            response.raise_for_status()
            entries = ElementTree.fromstring(response.content).findall("atom:entry", NAMESPACES)

            if not entries:
                return
            for entry in entries:
                yield parse_entry(entry)
                yielded += 1

            if yielded < limit:
                time.sleep(config.API_DELAY_SECONDS)
    finally:
        if owned:
            client.close()
