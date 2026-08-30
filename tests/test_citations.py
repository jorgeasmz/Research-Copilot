import pytest

from generation.citations import REFUSAL, check, sentences
from retrieval.dense import Candidate


def passage(number: int) -> Candidate:
    return Candidate(
        chunk_id=number,
        arxiv_id=f"2512.0000{number}v1",
        section="Results",
        paragraph=number * 10,
        text=f"passage {number}",
        score=0.0,
    )


@pytest.fixture
def passages():
    return [passage(1), passage(2), passage(3)]


def test_a_citation_resolves_to_the_passage_it_names(passages):
    result = check("The key rate improves [2].", passages)

    assert [c.number for c in result.citations] == [2]
    assert result.citations[0].arxiv_id == "2512.00002v1"
    assert result.citations[0].paragraph == 20


def test_a_number_outside_the_retrieved_range_is_reported(passages):
    """The model citing a source it was never shown is the failure to catch."""
    result = check("This follows from earlier work [7].", passages)

    assert result.invalid == [7]
    assert result.citations == []


def test_several_citations_in_one_sentence_all_resolve(passages):
    result = check("Both report the same bound [1][3].", passages)

    assert [c.number for c in result.citations] == [1, 3]


def test_a_repeated_citation_is_reported_once(passages):
    result = check("First claim [2]. Second claim [2].", passages)

    assert len(result.citations) == 1


def test_sentences_without_a_citation_are_counted(passages):
    result = check("Supported claim [1]. Unsupported claim.", passages)

    assert result.sentences == 2
    assert result.uncited_sentences == 1
    assert result.grounded == pytest.approx(0.5)


def test_a_fully_cited_answer_is_grounded(passages):
    result = check("One [1]. Two [2].", passages)

    assert result.grounded == 1.0


def test_a_refusal_is_recognised_and_not_penalised(passages):
    """Declining to answer is the correct behaviour, not an ungrounded answer."""
    result = check(f"{REFUSAL}.", passages)

    assert result.refused is True
    assert result.grounded == 1.0
    assert result.sentences == 0


def test_zero_is_not_a_valid_citation(passages):
    """Passages are numbered from one, so a zero is a fabricated marker."""
    assert check("Claim [0].", passages).invalid == [0]


def test_sentence_splitting_survives_a_trailing_citation():
    assert len(sentences("First [1]. Second [2].")) == 2
