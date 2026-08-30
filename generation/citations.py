"""Reads the citations out of an answer and checks each one against what was retrieved."""

import re
from dataclasses import dataclass

from retrieval.dense import Candidate

MARKER = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

REFUSAL = "The passages do not answer this question"


@dataclass(frozen=True)
class Citation:
    number: int
    arxiv_id: str
    section: str
    paragraph: int


@dataclass(frozen=True)
class Checked:
    answer: str
    citations: list[Citation]
    invalid: list[int]
    uncited_sentences: int
    sentences: int
    refused: bool

    @property
    def grounded(self) -> float:
        """Share of sentences carrying at least one citation."""
        if self.refused or not self.sentences:
            return 1.0
        return 1.0 - self.uncited_sentences / self.sentences


def sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text.strip()) if s.strip()]


def check(answer: str, passages: list[Candidate]) -> Checked:
    """
    Maps every citation to the passage it names and reports the ones that miss.

    A number outside the range of what was retrieved is the failure this exists
    to catch: the model citing a source that was never put in front of it.
    """
    refused = answer.strip().startswith(REFUSAL)

    found: dict[int, Citation] = {}
    invalid: list[int] = []
    uncited = 0
    counted = sentences(answer) if not refused else []

    for sentence in counted:
        numbers = [int(n) for n in MARKER.findall(sentence)]
        if not numbers:
            uncited += 1
        for number in numbers:
            if 1 <= number <= len(passages):
                passage = passages[number - 1]
                found[number] = Citation(
                    number=number,
                    arxiv_id=passage.arxiv_id,
                    section=passage.section,
                    paragraph=passage.paragraph,
                )
            else:
                invalid.append(number)

    return Checked(
        answer=answer,
        citations=[found[n] for n in sorted(found)],
        invalid=invalid,
        uncited_sentences=uncited,
        sentences=len(counted),
        refused=refused,
    )
