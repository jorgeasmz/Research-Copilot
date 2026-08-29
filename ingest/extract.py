"""Turns the LaTeX source of a paper into ordered paragraphs carrying their section."""

import logging
import re
from dataclasses import dataclass

from pylatexenc.latex2text import LatexNodes2Text

logger = logging.getLogger(__name__)

# latex2text opens every sectioning command with a marker, nesting it as
# "§", "§.§", "§.§.§", which is what makes the hierarchy recoverable from text.
SECTION_MARKER = re.compile(r"^\s*(§(?:\.§)*)\s+(.+?)\s*$")

# The renderer puts a marker at the head of the following line rather than in
# a block of its own, so a heading and its first paragraph arrive joined
# wherever the source carried no blank line between them.
SECTION_LINE = re.compile(r"^[ \t]*(§(?:\.§)*[ \t]+.+)$", re.MULTILINE)

# The renderer parses the preamble as content and stops at the first macro
# definition it cannot expand, so only the document body is handed to it.
BODY = re.compile(
    r"\\begin\s*\{document\}(.*?)(?:\\end\s*\{document\}|\Z)", re.DOTALL
)

# The renderer drops \par instead of breaking the paragraph, so the break is
# written as the blank line the rest of the source uses.
PARAGRAPH_BREAK = re.compile(r"\\par\b")

STOP_SECTIONS = re.compile(r"^(references|bibliography|acknowledg)", re.IGNORECASE)
MIN_PARAGRAPH_CHARS = 200
PREAMBLE = "Abstract"


@dataclass(frozen=True)
class Paragraph:
    section: str
    index: int
    text: str


def body(latex: str) -> str:
    """Returns the document body, or the whole input when no body is delimited."""
    match = BODY.search(latex)
    return PARAGRAPH_BREAK.sub("\n\n", match.group(1) if match else latex)


def to_text(latex: str) -> str:
    """
    Renders the source, keeping math as unicode and dropping citation keys.

    The renderer raises on constructs it cannot parse, and the corpus is arbitrary
    third-party LaTeX, so a failure returns nothing and drops the paper rather
    than ending the ingestion run.
    """
    try:
        return LatexNodes2Text(math_mode="text", keep_comments=False).latex_to_text(body(latex))
    except Exception:
        logger.warning("source could not be rendered", exc_info=True)
        return ""


def paragraphs(latex: str) -> list[Paragraph]:
    """
    Returns the body paragraphs of a paper, in reading order.

    Blocks shorter than MIN_PARAGRAPH_CHARS are dropped: at that length they are
    almost always a caption, an affiliation line or a displayed equation left
    stranded by the renderer.
    """
    text = SECTION_LINE.sub(r"\n\n\1\n\n", to_text(latex))
    if not text.strip():
        return []

    collected: list[Paragraph] = []
    section = PREAMBLE

    for block in re.split(r"\n\s*\n", text):
        stripped = " ".join(block.split())
        if not stripped:
            continue

        heading = SECTION_MARKER.match(stripped)
        if heading:
            title = heading.group(2)
            if STOP_SECTIONS.match(title):
                break
            section = title.title() if title.isupper() else title
            continue

        if len(stripped) >= MIN_PARAGRAPH_CHARS:
            collected.append(Paragraph(section, len(collected), stripped))

    return collected
