import numpy as np
import pytest

from retrieval.bm25 import K1, B, Index, scores, tokenize


def build(documents: list[str]) -> Index:
    """Builds an index from plain strings, without touching a database."""
    from scipy import sparse

    vocabulary: dict[str, int] = {}
    indices, data, indptr = [], [], [0]
    lengths = np.empty(len(documents), dtype=np.float32)

    for position, document in enumerate(documents):
        counts: dict[int, int] = {}
        tokens = tokenize(document)
        for token in tokens:
            term = vocabulary.setdefault(token, len(vocabulary))
            counts[term] = counts.get(term, 0) + 1
        indices.extend(counts)
        data.extend(counts.values())
        indptr.append(len(indices))
        lengths[position] = len(tokens)

    frequencies = sparse.csr_matrix(
        (np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr)),
        shape=(len(documents), len(vocabulary)),
    )
    document_frequency = np.diff(frequencies.tocsc().indptr).astype(np.float32)
    total = len(documents)
    idf = np.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
    average = float(lengths.mean())

    return Index(
        frequencies=frequencies.tocsc(),
        vocabulary=vocabulary,
        idf=idf.astype(np.float32),
        length_norm=(K1 * (1.0 - B + B * lengths / average)).astype(np.float32),
        chunk_ids=np.arange(len(documents), dtype=np.int64),
    )


def test_tokenizing_folds_case_and_drops_punctuation():
    assert tokenize("Decoy-state QKD, 2026!") == ["decoy", "state", "qkd", "2026"]


def test_a_document_containing_the_term_outranks_one_that_does_not():
    index = build(["decoy state protocol", "satellite downlink"])

    assert scores(index, "decoy")[0] > scores(index, "decoy")[1]


def test_a_rare_term_weighs_more_than_a_common_one():
    """Without inverse document frequency every word counts alike, which is what
    separates this from the ranking Postgres offers."""
    index = build(["key rate decoy", "key rate satellite", "key rate fibre"])

    assert index.idf[index.vocabulary["decoy"]] > index.idf[index.vocabulary["key"]]


def test_every_weight_stays_positive():
    """A term in every document must not score negatively, which the unsmoothed
    formula does once a term appears in more than half the corpus."""
    index = build(["key rate", "key rate", "key rate", "other words entirely"])

    assert (index.idf >= 0).all()


def test_a_term_absent_from_the_vocabulary_is_ignored():
    index = build(["decoy state"])

    assert scores(index, "unrelated")[0] == 0.0


def test_a_longer_document_is_penalised_for_the_same_count():
    """Length normalisation is what stops a long passage winning by repetition."""
    index = build(["decoy", "decoy " + "filler " * 40])

    assert scores(index, "decoy")[0] > scores(index, "decoy")[1]


def test_repeating_a_term_raises_the_score_with_diminishing_return():
    index = build(["decoy", "decoy decoy", "decoy decoy decoy"])
    one, two, three = scores(index, "decoy")

    assert one < two < three
    assert three - two < two - one


def test_scores_accumulate_across_query_terms():
    index = build(["decoy state protocol", "decoy alone", "unrelated text here"])
    both = scores(index, "decoy state")

    assert both[0] > both[1]


def test_the_index_reports_its_own_size():
    index = build(["decoy state protocol", "satellite downlink"])

    assert index.nbytes > 0
    assert index.nbytes < 10_000


def test_an_empty_question_scores_nothing():
    index = build(["decoy state"])

    assert scores(index, "").sum() == pytest.approx(0.0)
