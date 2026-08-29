"""Fetches the LaTeX source of a paper and flattens it into a single document."""

import io
import logging
import re
import tarfile
import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ingest import config
from ingest.arxiv import Paper

logger = logging.getLogger(__name__)

INPUT_COMMAND = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
DOCUMENT_START = re.compile(r"\\begin\s*\{document\}")


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
def _download(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def _read_members(archive: bytes) -> dict[str, str]:
    """Returns every .tex member decoded, keyed by name without its extension."""
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            members = {}
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".tex"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                members[member.name.removesuffix(".tex")] = handle.read().decode(
                    "utf-8", errors="replace"
                )
            return members
    except tarfile.TarError:
        # Single-file submissions arrive gzipped rather than tarred.
        import gzip

        try:
            return {"main": gzip.decompress(archive).decode("utf-8", errors="replace")}
        except OSError:
            return {}


def flatten(members: dict[str, str]) -> str:
    """
    Returns the main file with every \\input and \\include resolved in place.

    A submission can hold a dozen fragments and only one preamble, so the entry
    point is the member that opens the document environment.
    """
    entries = [name for name, text in members.items() if DOCUMENT_START.search(text)]
    if not entries:
        return ""
    entry = min(entries, key=lambda name: (name.count("/"), len(name)))

    def resolve(name: str, seen: frozenset[str]) -> str:
        if name in seen or name not in members:
            return ""

        def substitute(match: re.Match) -> str:
            target = match.group(1).strip().removesuffix(".tex")
            return resolve(target, seen | {name})

        return INPUT_COMMAND.sub(substitute, members[name])

    return resolve(entry, frozenset())


def fetch(paper: Paper, client: httpx.Client) -> str:
    """Downloads and flattens one paper's source, caching the archive on disk."""
    config.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    cached = config.SOURCE_DIR / f"{paper.arxiv_id}.tar.gz"

    if cached.exists() and cached.stat().st_size > 0:
        return flatten(_read_members(cached.read_bytes()))

    try:
        archive = _download(client, f"{config.SOURCE_URL}/{paper.arxiv_id}")
    except httpx.HTTPError:
        logger.warning("%s has no retrievable source", paper.arxiv_id)
        return ""

    cached.write_bytes(archive)
    time.sleep(config.API_DELAY_SECONDS)
    return flatten(_read_members(archive))
