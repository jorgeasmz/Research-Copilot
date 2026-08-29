from ingest.extract import MIN_PARAGRAPH_CHARS, body, paragraphs

LONG = "This sentence carries enough characters to clear the minimum length. " * 5


def document(content: str) -> str:
    return r"\documentclass{revtex4-2}\begin{document}" + content + r"\end{document}"


def test_body_drops_the_preamble():
    """The renderer parses preamble macros as content and stops at the first it cannot expand."""
    latex = r"\newtheorem{lemma}{Lemma}\begin{document}Body text.\end{document}"

    assert "newtheorem" not in body(latex)
    assert "Body text." in body(latex)


def test_body_falls_back_to_the_whole_input():
    assert body("no document environment here") == "no document environment here"


def test_sections_are_recovered():
    latex = document(rf"\section{{Introduction}}{LONG}\section{{Results}}{LONG}")

    sections = [p.section for p in paragraphs(latex)]

    assert "Introduction" in sections
    assert "Results" in sections


def test_subsections_do_not_keep_their_nesting_marker():
    """latex2text nests the marker as section.subsection; the title must come out clean."""
    latex = document(rf"\section{{Method}}\subsection{{Sampling}}{LONG}")

    assert [p.section for p in paragraphs(latex)] == ["Sampling"]


def test_everything_from_the_references_on_is_dropped():
    latex = document(rf"\section{{Results}}{LONG}\section{{References}}{LONG}")

    assert [p.section for p in paragraphs(latex)] == ["Results"]


def test_short_blocks_are_dropped():
    """At that length a block is a caption, an affiliation or a stranded equation."""
    latex = document(r"\section{Results}" + "Too short.")

    assert paragraphs(latex) == []
    assert len("Too short.") < MIN_PARAGRAPH_CHARS


def test_paragraphs_are_numbered_in_reading_order():
    latex = document(rf"\section{{A}}{LONG}\par {LONG}")

    assert [p.index for p in paragraphs(latex)] == [0, 1]


def test_unparseable_source_yields_nothing_instead_of_raising():
    """The corpus is arbitrary third-party LaTeX; one bad paper must not end a run."""
    assert paragraphs(r"\begin{document}\begin{") == []
