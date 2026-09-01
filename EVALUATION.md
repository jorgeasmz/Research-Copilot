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
| `dense` | **0.702** | 0.825 | **0.673** |
| `hybrid` | 0.698 | 0.819 | 0.666 |
| `hybrid+rerank` | 0.697 | **0.832** | 0.664 |

These are the served int8 graphs. The fp32 originals score 0.713, 0.709 and
0.703 on the same three rows, so quantisation costs 0.011 of nDCG consistently.
BM25 does not move, which is the check that the difference comes from the
encoders rather than from the harness.

BM25 at 0.652 and `bge-small-en-v1.5` at 0.702 sit where the published SciFact
numbers for those methods sit. That is the check that the implementation is
correct rather than merely self-consistent.

## Retrieval on the corpus

Nineteen hand-written questions, fourteen anchored to the passage the question
was written from. A result is judged twice: whether the answering paper appears
in the top five, and whether that passage does.

| Retriever | Paper found | Passage found | MRR | Latency ms |
|---|---:|---:|---:|---:|
| `bm25`, sparse matrix | 0.84 | 0.43 | 0.627 | **4** |
| `bm25`, dictionary per document | 0.84 | 0.43 | 0.680 | 86 |
| `postgres fts` | 0.37 | 0.07 | 0.215 | 95 |
| `dense` | 0.58 | 0.43 | 0.342 | 11 |
| `hybrid` | 0.79 | 0.43 | 0.636 | 24 |
| `hybrid+rerank` | **0.89** | **0.71** | **0.718** | 1,888 |

The two BM25 rows retrieve the same passages. They differ in storage, and in how
they treat a term common enough to earn a negative weight under the unsmoothed
formula: the library floors it, this implementation smooths it, and the ordering
inside the top five shifts accordingly. The sparse matrix holds 8.6 MB against
558 MB resident, and answers in 4 ms against 86.

Postgres full text search was measured as the alternative that needs no index in
the process at all, and rejected. `ts_rank_cd` carries no inverse document
frequency, so a term appearing in most passages weighs as much as one appearing
in two, and the ranking collapses to 0.07 passage accuracy.

### What the memory budget costs

Fitting 512 MB is not only a matter of smaller weights. ONNX Runtime allocates
from an arena that keeps every block it takes and asks for a new one per tensor
shape, so a service seeing questions of varying length grows without bound.

| Allocation | Peak over fifty queries | Retrieval p50 |
|---|---:|---:|
| Arena on, padded to the batch | 1,264 MB | 1,888 ms |
| Arena on, padded to 512-token buckets | 2,999 MB | 5,260 ms |
| Arena on, padded to 32-token buckets | 1,453 MB | 2,562 ms |
| **Arena off** | **257 MB** | 3,762 ms |

Fixed widths do bound the growth, which is what they are for, but only at 1.4 GB,
and coarse ones cost compute: rounding a 320-token batch to 512 wastes a third
of the work and made both figures worse. Allocating per call is the only
configuration that fits, and it is 2.6 times slower.

Reranking depth buys that time back, and the corpus set says where.

| Depth | Paper found | Passage found | MRR | Retrieval p50 |
|---|---:|---:|---:|---:|
| No reranking | 0.79 | 0.43 | 0.636 | 24 ms |
| 5 | 0.79 | 0.43 | 0.605 | 384 ms |
| 10 | 0.89 | 0.57 | **0.732** | 1,248 ms |
| **25** | **0.89** | **0.71** | 0.718 | 3,762 ms |
| 50 | 0.84 | 0.71 | 0.673 | — |

Five is not worth running: it costs 360 ms and recovers nothing the fusion had
not already ranked. Ten reaches the best MRR at a third of the latency. Depth 25
is what ships, because passage accuracy is what the service exists for and 0.71
against 0.57 is two more of the fourteen anchored questions, on a request whose
generation step takes nine seconds regardless.

### What quantisation costs

Both encoders are served as int8 graphs. Measured against the same questions
with the fp32 originals, the end to end result is unchanged, at 0.89 paper and
0.71 passage either way, and MRR moves from 0.683 to 0.718.

The bi-encoder alone does degrade: on its own it falls from 0.68 to 0.58 on
paper recall and from 0.57 to 0.43 on passage. Fusion with BM25 and the
cross-encoder absorb that, so the served configuration shows no loss, but a
query mix the lexical retriever could not help would expose it. Keeping the
bi-encoder at fp32 and quantising only the cross-encoder costs 114 MB more
resident and buys nothing measurable here.

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

On the corpus it raises passage selection from 0.43 to 0.71, four of the
fourteen anchored questions, and paper recall from 0.74 to 0.84.

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
term index over 18,969 chunks and loads both encoders.

| Stage | p50 ms |
|---|---:|
| Retrieval without reranking | 24 |
| Retrieval with reranking | 1,888 |
| Complete answer | 9,219 |

Retrieval is dominated by the cross-encoder: without it the same query costs
24 ms. The generation p95 is one question
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

## The semantic cache

A hit does not save a millisecond, it saves one of the twenty daily requests.
Matching is by embedding similarity, and the measurement says that separates the
two classes poorly within one field.

| Pairs | Similarity |
|---|---|
| The 171 pairs of the question set | 0.496 to 0.723 |
| Five paraphrases of set questions | 0.766 to 0.974 |
| An adjacent pair asking different things | **0.823** |
| One of those paraphrases | **0.733** |

The last two rows are the finding. A pair asking how key rates are computed for
two different protocol families scores 0.823, above a genuine paraphrase at
0.733. Questions in one field share vocabulary, and the embedding is dominated by
what a question is about rather than by what it asks, so the classes overlap and
no threshold separates them.

The two errors are not equal. A miss spends a request; a false hit answers a
different question, and attaches to it the passages retrieved for the question
actually asked. The threshold is therefore 0.90, above both measured populations,
which admits close rewordings and rejects paraphrases that change the vocabulary.
That is closer to an exact-match cache that tolerates rewording than to a
semantic one, and calling it otherwise would overstate what it does.

Calibrating against negatives drawn from unrelated fields, which is the
convenient thing to measure, puts the threshold near 0.70 and admits the 0.823
pair.

## Reproducing

```bash
python -m evaluation.benchmark    # BEIR SciFact, no API calls
python -m evaluation.corpus       # question set, no API calls
python -m evaluation.generation   # 25 API calls
```

The first two consume no quota. Retrieval can be evaluated and tuned without a
key at all.
