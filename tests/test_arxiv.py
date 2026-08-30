import urllib.parse

from ingest.arxiv import parse_entry, search

ENTRY = """<entry xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/abs/2608.00001v2</id>
  <title>Decoy state
  quantum key distribution</title>
  <summary>An  abstract   with odd spacing.</summary>
  <author><name>Ada Lovelace</name></author>
  <author><name>Alan Turing</name></author>
  <category term="quant-ph"/>
  <published>2026-08-01T00:00:00Z</published>
  <updated>2026-08-02T00:00:00Z</updated>
  <link title="pdf" href="https://arxiv.org/pdf/2608.00001v2"/>
</entry>"""


def feed(count: int, offset: int = 0) -> bytes:
    entries = "".join(
        ENTRY.replace("2608.00001v2", f"2608.{offset + i:05d}v1") for i in range(count)
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'.encode()


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


class FakeClient:
    """Records the query string of every request and answers with canned pages."""

    def __init__(self, pages: list[bytes]):
        self.pages = pages
        self.requests: list[dict] = []

    def get(self, url: str) -> FakeResponse:
        self.requests.append(dict(urllib.parse.parse_qsl(url.split("?", 1)[1])))
        return FakeResponse(self.pages[len(self.requests) - 1])


def test_entry_fields_are_read():
    import xml.etree.ElementTree as ElementTree

    paper = parse_entry(ElementTree.fromstring(ENTRY))

    assert paper.arxiv_id == "2608.00001v2"
    assert paper.title == "Decoy state quantum key distribution"
    assert paper.abstract == "An abstract with odd spacing."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.pdf_url.endswith("2608.00001v2")


def test_the_identifier_keeps_its_version():
    """A revised paper is a distinct row, not an overwrite of the cited text."""
    import xml.etree.ElementTree as ElementTree

    assert parse_entry(ElementTree.fromstring(ENTRY)).arxiv_id.endswith("v2")


def test_results_are_paged_until_the_limit(monkeypatch):
    monkeypatch.setattr("ingest.arxiv.time.sleep", lambda _: None)
    client = FakeClient([feed(100), feed(50, offset=100)])

    papers = list(search("cat:quant-ph", 150, client))

    assert len(papers) == 150
    assert [r["start"] for r in client.requests] == ["0", "100"]


def test_the_search_query_is_repeated_verbatim_on_every_page(monkeypatch):
    """Re-encoding the query into itself returns an empty second page."""
    monkeypatch.setattr("ingest.arxiv.time.sleep", lambda _: None)
    client = FakeClient([feed(100), feed(50, offset=100)])
    query = 'cat:quant-ph AND abs:"quantum key distribution"'

    list(search(query, 150, client))

    assert [r["search_query"] for r in client.requests] == [query, query]


def test_an_empty_page_ends_the_search(monkeypatch):
    monkeypatch.setattr("ingest.arxiv.time.sleep", lambda _: None)
    client = FakeClient([feed(10), feed(0)])

    assert len(list(search("cat:quant-ph", 100, client))) == 10


def test_the_page_never_exceeds_what_is_left(monkeypatch):
    monkeypatch.setattr("ingest.arxiv.time.sleep", lambda _: None)
    client = FakeClient([feed(5)])

    list(search("cat:quant-ph", 5, client))

    assert client.requests[0]["max_results"] == "5"
