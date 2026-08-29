from ingest.chunk import MAX_CHARS, Chunk, chunk_paper, split
from ingest.extract import Paragraph


def test_short_text_is_left_whole():
    assert split("One sentence only.") == ["One sentence only."]


def test_long_text_is_split_between_sentences():
    sentence = "The stiffness tensor is measured at low temperature. "
    pieces = split(sentence * 60)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_CHARS for piece in pieces)


def test_consecutive_pieces_overlap_by_one_sentence():
    """A claim that straddles a boundary has to be retrievable from either side."""
    sentences = [f"Sentence number {i} carries enough text to fill the buffer." for i in range(60)]
    pieces = split(" ".join(sentences))

    first_tail = pieces[0].split(". ")[-1]
    assert first_tail.rstrip(".") in pieces[1]


def test_a_single_oversized_sentence_is_cut_on_whitespace():
    """Dense notation leaves the sentence splitter no boundary to use."""
    pieces = split("word " * 1000)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_CHARS for piece in pieces)


def test_chunks_keep_the_paragraph_they_came_from():
    """The paragraph index is what a citation points at, so splitting must preserve it."""
    long_text = "This paragraph is long enough to be split into several pieces. " * 40
    paragraphs = [Paragraph(section="Results", index=7, text=long_text)]

    chunks = chunk_paper("2608.00001v1", paragraphs)

    assert len(chunks) > 1
    assert {c.paragraph for c in chunks} == {7}
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.section == "Results" for c in chunks)


def test_chunk_carries_its_paper():
    chunks = chunk_paper("2608.00002v1", [Paragraph("Abstract", 0, "x" * 300)])

    assert chunks == [Chunk("2608.00002v1", "Abstract", 0, 0, "x" * 300)]
