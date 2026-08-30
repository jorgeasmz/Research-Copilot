"""Builds the prompt from retrieved passages and states how they must be cited."""

from retrieval.dense import Candidate

INSTRUCTIONS = """\
You answer questions about quantum cryptography using only the numbered passages \
below.

Rules:
- Every claim must carry a citation in square brackets naming the passage it \
comes from, like [2]. A sentence may carry several, like [1][4].
- Cite only the numbers given. Do not invent a number.
- If the passages do not answer the question, say exactly: The passages do not \
answer this question. Then stop.
- Do not add background the passages do not contain, however well known it is.
- Write in prose, not bullet points, and keep it under 200 words.
"""


def format_passage(number: int, candidate: Candidate) -> str:
    """One passage with the provenance a reader needs to check it."""
    return (
        f"[{number}] {candidate.arxiv_id} · {candidate.section} · "
        f"paragraph {candidate.paragraph}\n{candidate.text}"
    )


def build(question: str, passages: list[Candidate]) -> str:
    context = "\n\n".join(
        format_passage(number, passage) for number, passage in enumerate(passages, start=1)
    )
    return f"{INSTRUCTIONS}\nPassages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
