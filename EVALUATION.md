# Evaluation

Every figure here comes from a script in `evaluation/`, run against the corpus
and the model the service uses. Nothing is estimated.

| | |
|---|---|
| Corpus | 240 arXiv papers on quantum cryptography, 18,969 chunks |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384 dimensions |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, depth 25 |
| Generation | `gemini-3.6-flash`, temperature 0 |
| Hardware | Four CPU cores, no GPU |

## Retrieval on a judged benchmark

BEIR SciFact carries relevance judgements, which is what makes these figures
comparable to published ones. The dataset stands in for the corpus; the fusion
and reranking code is the code the service runs.

| Retriever | nDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| `bm25` | 0.652 | 0.776 | 0.618 |
| `dense` | **0.713** | 0.836 | **0.682** |
| `hybrid` | 0.709 | 0.838 | 0.675 |
| `hybrid+rerank` | 0.703 | **0.846** | 0.667 |

BM25 at 0.652 and `bge-small-en-v1.5` at 0.713 sit where the published SciFact
numbers for those methods sit. That is the check that the implementation is
correct rather than merely self-consistent.

## Retrieval on the corpus

Nineteen hand-written questions, fourteen anchored to the passage the question
was written from. A result is judged twice: whether the answering paper appears
in the top five, and whether that passage does.

| Retriever | Paper found | Passage found | MRR | Latency ms |
|---|---:|---:|---:|---:|
| `bm25` | 0.84 | 0.43 | 0.680 | 101 |
| `dense` | 0.68 | 0.57 | 0.473 | **25** |
| `hybrid` | 0.79 | 0.43 | 0.646 | 135 |
| `hybrid+rerank` | **0.89** | **0.71** | **0.683** | 1530 |

Scoping the corpus to one field is what makes this hard. On a general `quant-ph`
corpus of 249 papers the same questions reached 0.82 passage accuracy without
reranking; within quantum cryptography the same configuration reaches 0.43.
Retrieval across unrelated subjects mostly measures topic separation, and that
signal is absent here.

## The reranker

The two evaluations disagree about it.

On 300 judged SciFact queries the cross-encoder lowers nDCG@10 from 0.709 to
0.703 and MRR from 0.675 to 0.667. Reducing the depth from 25 to 10 does not
recover the loss; nDCG falls to 0.699. SciFact is claim verification against
abstracts, and the model was trained on web passages.

On the corpus it raises passage selection from 0.43 to 0.71, four of the fourteen
anchored questions, and paper recall from 0.79 to 0.89.

It is enabled on that basis, because the corpus is what the service serves. The
caveat is that fourteen anchored questions is a small sample. A larger
cross-encoder is not an alternative: measured over five queries at the same
depth, `bge-reranker-base` takes 19.1 s per query against 3.2 s for the one in
use.

## Grounding

Twenty-five questions: the nineteen the corpus covers, and six on subjects it
does not contain at all.

| Covered questions | 19 |
|---|---:|
| Answers containing a fabricated citation | **0** |
| Sentences carrying a citation | 0.98 |
| Citations per answer | 1.7 |
| Declined to answer | 6 |

| Uncovered questions | 6 |
|---|---:|
| Declined to answer | **6/6** |
| Answered anyway | 0/6 |
| Fabricated citations | **0** |

A fabricated citation means a bracketed number naming a passage that was never
retrieved. There were none, on either set.

The six refusals among covered questions are retrieval failures surfacing
honestly. They are consistent with the passage accuracy above: 6 of 19 is 32%,
against the 29% of anchored questions whose passage the retriever misses. When
retrieval fails the system says so instead of writing something plausible, which
is the behaviour the refusal instruction exists to produce.

## Latency

Retrieval is measured after a warm-up query, since the first call builds the
BM25 index over 18,969 chunks and loads both encoders.

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| Retrieval | 1,583 | 1,848 |
| First token | 7,820 | 51,418 |
| Complete answer | 9,219 | 52,736 |

Retrieval is dominated by the cross-encoder. The generation p95 is one question
that hit a saturated model and fell through the chain; the median is what a
served request looks like.

## Quota

The free tier meters requests per day per model, not tokens:

```
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20
```

Twenty requests per model per day. One run of this evaluation is twenty-five
requests, so a single run exceeds the budget of a single model. The provider
holds a chain of three, and moves to the next on a 429 rather than waiting,
because a daily quota does not clear within a retry window. That is what makes
the evaluation runnable at all, and it is measured: this run logged the primary
model exhausting its quota and the second answering.

## Reproducing

```bash
python -m evaluation.benchmark    # BEIR SciFact, no API calls
python -m evaluation.corpus       # question set, no API calls
python -m evaluation.generation   # 25 API calls
```

The first two consume no quota. Retrieval can be evaluated and tuned without a
key at all.
