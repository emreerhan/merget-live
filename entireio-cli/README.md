# entireio/cli repository-work retrieval

This package turns the bounded `entireio-cli` SWE-chat slice into a searchable,
chronology-aware record of work performed on `entireio/cli`. It supports questions
such as:

- What programming languages appear in the recorded work?
- What architecture was used for a model or subsystem?
- Why was a workflow changed?
- When was a feature introduced, and how did it evolve?

The system normalizes session data into citation-addressable evidence, generates and
judges a silver retrieval benchmark, compares lexical and dense retrieval, fine-tunes
a lightweight embedding model on CPU, produces dataset-grounded answers, and records
reproducible evaluation and calibration artifacts.

## Scope and evidence boundary

The only factual inputs are the files in this directory:

- `sessions.parquet`
- `session_logs.parquet`
- `conversations.parquet`
- `checkpoints.parquet`
- `commits.parquet`
- `transcripts/*.jsonl`

The implementation does not clone, reconstruct, or inspect a complete repository
snapshot. Answers therefore describe work observed in this dataset, not an exhaustive
or necessarily current description of the full repository. Supported answer claims
must resolve to dataset-native Parquet rows or transcript events.

Source data is immutable. All generated state is written under `derived/`, which is
ignored by Git because it includes local indexes, model weights, cached responses, and
potentially sensitive derived evidence.

## What was built

The pipeline has six main stages:

1. **Evidence preparation** streams the five Parquet tables and referenced JSONL
   transcripts, preserves missing relationships, redacts sensitive values, chunks
   bounded text, and creates stable citations.
2. **Silver benchmark construction** uses OpenRouter
   `openai/gpt-5.6-luna` with medium reasoning to generate realistic
   repository-understanding questions. Separate critic and relevance-judging calls,
   deterministic checks, shuffled candidate pools, and optional double evaluation
   reduce synthetic-label errors.
3. **Retrieval** combines `BAAI/bge-small-en-v1.5` embeddings, corpus-derived BM25,
   local Qdrant persistence, metadata filtering, evidence grouping, and
   query-dependent chronology.
4. **Fine-tuning** creates multi-positive contrastive examples, admits only grounded
   grade-zero hard negatives, freezes lower encoder layers, and trains a CPU-bounded
   BGE-small checkpoint.
5. **Citation-backed answering** sends only bounded, redacted retrieved evidence to
   Luna, validates every factual claim and citation, and retries or abstains when an
   answer cannot be grounded.
6. **Evaluation and calibration** compares retrieval variants on chronological
   partitions, freezes choices on validation, seals evaluation once, calibrates cosine
   scores, and applies quality gates before enabling filtering or automatic
   answerability decisions.

## Evidence produced by the pilot

The deterministic pilot selected 275 sessions and produced 125,008 evidence records:

| Evidence type | Records |
|---|---:|
| Transcript events | 71,300 |
| Conversation turns | 50,370 |
| Commit/patch chunks | 2,478 |
| Checkpoints | 536 |
| Session cards | 275 |
| Cross-session topics | 49 |

The chronological split contains 201 training, 41 validation, and 33 evaluation
sessions. Related sessions, checkpoints, topic groups, and paraphrase families are
kept within one partition. The compact vector index contains 8,588 high-value records;
the complete evidence store remains available for provenance and citation resolution.

Evidence records cover:

- user prompts, assistant conclusions, retained reasoning, and useful tool activity;
- commit messages, changed paths, bounded patch chunks, and checkpoint attribution;
- checkpoint strategies, branches, linked sessions, commits, and change statistics;
- deterministic session summaries of problems, decisions, actions, outcomes, files,
  and time ranges;
- citation-preserving topic summaries across related sessions.

Every run records input hashes, configuration hashes, code hashes, model settings,
artifact hashes, counts, and timestamps. The dataset-boundary audit checks that
manifests and citations use only `entireio-cli`.

## Query benchmark

The accepted benchmark contains 229 questions:

| Partition | Queries |
|---|---:|
| Training | 134 |
| Validation | 65 |
| Evaluation | 30 |

It includes 189 supported, 6 partially supported, and 34 intentionally
unsupported-by-dataset questions across repository, subsystem, and change scopes.
Question categories cover technology, architecture, behavior, rationale, history,
testing, and implementation. Temporal intent includes current state, latest change,
historical evolution, introduction, and time-neutral questions.

Candidate evidence is pooled independently from BM25, pretrained dense retrieval,
metadata neighbors, and temporally diverse retrieval. Luna assigns relevance grades
from 0 through 3 and identifies which expected claims each relevant record supports.

The labels are silver rather than human-reviewed. Generator, critic, relevance judge,
and answer validator are separate calls and prompts, but model-family circularity
remains a limitation.

## Embedding model and fine-tuning

The base encoder is pinned to:

```text
BAAI/bge-small-en-v1.5
revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
384 dimensions
256-token maximum sequence length
```

Queries use the BGE asymmetric retrieval instruction and all vectors are L2
normalized. Embeddings are batched and cached. Cache metadata includes model identity,
record content identity, revision, and dimensions; unchanged vectors are reused during
incremental rebuilds.

Fine-tuning used 204 examples, batch size 1, three epochs, four frozen lower layers,
three hard negatives per example, early stopping, and a learning rate of `2e-5`. It
ran on CPU in approximately 816 seconds and peaked near 2.8 GiB RSS.

Fine-tuning did not win validation selection. It reduced validation nDCG@10 but
improved the untouched evaluation nDCG@10. That split-dependent result is reported,
not used for post-hoc model selection. The pretrained encoder remains the default.

## Retrieval and chronology

Qdrant stores named dense and lexical vectors plus payloads for evidence type,
citations, sessions, checkpoints, commits, files, branches, timestamps, and
partitions. BM25 handles symbols and exact terms; reciprocal-rank fusion combines
lexical and dense candidates. Related evidence is grouped so repeated chunks from one
session do not dominate the result list.

Temporal behavior depends on query intent:

- `current_state` and `latest_change` can apply bounded freshness after relevance;
- `historical_evolution` preserves the strongest result and deliberately selects
  evidence from distinct dates;
- `introduced_when` favors earlier relevant evidence;
- `as_of_date` excludes later evidence before ranking;
- `time_neutral` applies no chronological preference.

Retrieval traces persist dense and lexical ranks and scores, fusion and temporal
components, final rank, grouping, filters, temporal interpretation, and citations.

## Frozen pilot results

The best validation variant was selected strictly on validation data:

```text
model: pretrained_dense
relevance floor: 0.10
temporal weight: 0.00
validation nDCG@10: 0.3625
validation MRR@10: 0.4686
```

The low original relevance-floor search range was subsequently found to be ineffective
for BGE cosine scores. It is retained in the sealed experiment for reproducibility,
while the post-pilot calibration stage evaluates a realistic threshold separately.

### Retrieval ablations

| Variant | Validation nDCG@10 | Validation MRR@10 | Evaluation nDCG@10 | Evaluation MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.3480 | 0.4103 | 0.2718 | 0.3679 |
| Pretrained dense | **0.3625** | 0.4686 | 0.2809 | 0.3981 |
| Pretrained hybrid | 0.1627 | 0.3530 | 0.1279 | 0.2511 |
| Pretrained hybrid + temporal | 0.1627 | 0.3530 | 0.1279 | 0.2511 |
| Fine-tuned dense | 0.2474 | 0.4769 | **0.3238** | **0.5403** |
| Fine-tuned hybrid | 0.2494 | **0.5001** | 0.2241 | 0.4594 |

The validation winner was frozen before evaluation. The later evaluation improvement
from fine-tuned dense retrieval is diagnostic and does not change the selected model.

### Answer quality

All 30 evaluation questions produced citation-valid answers after bounded
regeneration:

| Metric | Result |
|---|---:|
| Groundedness | 1.000 |
| Temporal correctness | 1.000 |
| Answerability agreement | 0.567 |
| Expected-claim coverage | 0.250 |
| Citation precision | 0.220 |
| Citation recall | 0.446 |
| Unsupported false-answer rate | 0.200 |

The strong deterministic groundedness result means accepted claims resolve to supplied
evidence. The weaker claim coverage, citation metrics, and answerability agreement show
that retrieval and synthetic answerability labels still need improvement. This is a
research pilot, not a production-quality repository oracle.

## Similarity-score calibration

Run:

```bash
entireio-retrieval calibrate
```

Calibration is fit only on validation judgments. Evaluation traces are used only for
holdout reporting and never for threshold selection.

The calibration stage:

- fits grouped out-of-fold isotonic calibration from cosine score to relevance
  probability;
- reports ROC-AUC, PR-AUC, Brier score, expected calibration error, precision, recall,
  balanced accuracy, and reliability bins;
- tests a validation-selected raw cosine threshold;
- models answerability using top-score shape, top-1/top-2 margin, top-1/top-10 drop,
  BM25 overlap, calibrated relevant-result count, and mean top-five probability;
- applies explicit validation quality gates before enabling hard filtering or
  automatic abstention.

For this pilot, the selected raw threshold is `0.6895`. Relevance calibration achieved
grouped validation ROC-AUC `0.7445`, PR-AUC `0.6181`, Brier score `0.1621`, and ECE
`0.0459`. Hard filtering is disabled because it reduced validation nDCG@10 from
`0.3625` to `0.3533` and Recall@20 from `0.6400` to `0.5581`, even though evaluation
nDCG improved.

Automatic answerability abstention is also disabled. Only five validation questions
are unsupported, and out-of-fold answerability ROC-AUC was `0.3533`, below the quality
gate. Interactive results still expose relevance and answerability probabilities for
diagnostics, but weak calibration is not allowed to suppress answers.

Exact lexical matches in the top five can bypass a future dense threshold to protect
symbol and identifier recall. Use `--no-calibration` to compare raw interactive
behavior.

## Installation

Python 3.11 or newer is required. The pinned dependencies include PyArrow,
sentence-transformers, PyTorch, scikit-learn, Qdrant, rank-bm25, Pydantic, and HTTPX.

```bash
cd entireio-cli
python -m pip install -e '.[dev]'
chmod 600 ../temp_openrouter_key.txt
entireio-retrieval validate
```

The embedding and Qdrant workflow runs locally on CPU. The configured embedding model
must already be present in the local model cache because loading uses
`local_files_only=True`.

## End-to-end workflow

```bash
# Validate source files and key permissions.
entireio-retrieval validate

# Select 275 sessions, normalize evidence, redact, cite, and partition.
entireio-retrieval prepare

# Build pretrained dense, BM25, and local Qdrant indexes.
entireio-retrieval index --recreate

# Generate and criticize silver queries, then grade candidate evidence.
entireio-retrieval generate --max-groups 10 --double-evaluation
entireio-retrieval judge

# Train and index the CPU-bounded fine-tuned encoder.
entireio-retrieval train
entireio-retrieval index --recreate --fine-tuned

# Tune on validation and freeze the selected variant.
entireio-retrieval experiment

# Generate citation-backed evaluation answers with the frozen retriever.
entireio-retrieval answer-benchmark --partition evaluation

# Run the guarded final evaluation exactly once.
entireio-retrieval experiment --run-frozen-evaluation

# Fit post-pilot score calibration without changing the sealed evaluation.
entireio-retrieval calibrate

# Prove all factual inputs remain inside entireio-cli.
entireio-retrieval audit
```

`prepare --full` processes all sessions instead of the deterministic pilot. Do this
only as a new experiment; do not overwrite a sealed pilot and describe it as the same
run.

## Interactive use

```bash
entireio-retrieval query \
  "How did checkpoint cleanup behavior evolve over time?"

entireio-retrieval query \
  "When was checkpoint cleanup first introduced?" \
  --temporal-mode introduced_when

entireio-retrieval answer \
  "What programming languages appear in the recorded work?"

# Diagnostic comparisons
entireio-retrieval query "How does authentication work?" --no-calibration
entireio-retrieval query "How does authentication work?" --fine-tuned
```

Calibrated pretrained results include `relevance_probability`,
`answerability_probability`, calibration version, threshold decision, and lexical
rescue fields in their trace payloads. Fine-tuned queries are not passed through the
pretrained score calibrator.

## External API and security

`generate`, `judge`, `answer`, and `answer-benchmark` send bounded, redacted evidence
packets to the configured OpenRouter model. Run them only with explicit authorization
to disclose potentially sensitive repository/session content.

The key is read from `../temp_openrouter_key.txt` only at runtime. Startup requires
restrictive file permissions. The key is never copied into configuration, prompts,
logs, caches, answers, manifests, or Git. Requests use strict structured schemas,
bounded concurrency, retry/backoff, response caching, and usage accounting.

Redaction covers common secrets, email addresses, and absolute user paths. Payload
minimization and redaction reduce exposure but cannot guarantee that arbitrary source
text contains no sensitive information.

## Generated artifacts

Everything generated is under `derived/`:

- `evidence/`: normalized records, quality report, and partition report
- `benchmark/`: generated and accepted queries, critic decisions, candidate pools,
  relevance judgments, failures, and summaries
- `embeddings/`: pretrained and fine-tuned vector caches
- `index/`: BM25 state and embedded Qdrant databases
- `models/`: fine-tuned checkpoints and training resource report
- `answers/`: structured claims, citations, rendered answers, and validation results
- `evaluation/`: frozen configuration, detailed traces, pilot report, calibration,
  audit, and the one-shot completion marker
- `manifests/`: stage-level provenance and hashes
- `openrouter-cache/`: parsed non-secret structured responses and usage metadata

Runs are resumable from caches and versioned artifacts. To rebuild, remove only the
relevant subdirectory under `entireio-cli/derived/`; never delete or mutate the source
Parquet or transcript files.

## Testing

```bash
pytest
```

The normal suite covers ingestion, malformed and missing relationships, transcript
citations, deterministic chunking, redaction, partition isolation, query validation,
candidate pooling, OpenRouter caching and bounded failures, provenance manifests,
retrieval fusion/filtering/grouping/chronology, training isolation, answer validation,
evaluation metrics, calibration, and completion guards.

The external smoke test is opt-in and sends no dataset content:

```bash
ENTIREIO_LIVE_OPENROUTER=1 pytest -m live
```

Local Qdrant warns that payload indexes do not accelerate embedded mode. This is
expected for the pilot; use Qdrant server mode if the corpus grows enough to require
indexed payload filtering.

## Known limitations and next steps

- Silver labels inherit generator/judge model bias and have no human-review layer.
- Unsupported validation coverage is too small to train a trustworthy abstention
  model.
- Absolute cosine probabilities are specific to this model, corpus, and query mix.
- The fine-tuned model shows split-dependent behavior and needs repeated chronological
  splits, more balanced grades, and more hard negatives before promotion.
- Hybrid fusion underperformed both BM25 and dense retrieval and needs component-level
  retuning rather than a larger temporal weight.
- Evaluation contains only 30 questions; confidence intervals would be wide.
- Dataset-only evidence cannot establish facts about unrecorded repository state.

The next iteration should generate more unsupported and partially supported validation
questions, rerun grouped calibration, require the filtering and abstention quality
gates to pass, and evaluate repeated chronological folds before changing the default
encoder or enabling hard thresholds.
