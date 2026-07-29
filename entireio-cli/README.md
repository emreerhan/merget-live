# entireio/cli retrieval research

This package builds a dataset-bounded retrieval and cited-answering system from the
Parquet tables and JSONL transcripts in this directory. No repository clone or other
factual source is used.

## Evidence boundary

Answers describe work observed in this dataset. They are not exhaustive statements
about the complete or current `entireio/cli` repository. Every supported claim must
resolve to a dataset-native session, conversation turn, checkpoint, commit, or
transcript citation.

## Setup

```bash
chmod 600 ../temp_openrouter_key.txt
python -m pip install -e '.[dev]'
entireio-retrieval validate
```

The OpenRouter key is read only at runtime. It is never copied into configuration,
logs, caches, or result artifacts.

## Pilot workflow

```bash
entireio-retrieval prepare
entireio-retrieval index --recreate
entireio-retrieval generate --max-groups 10 --double-evaluation
entireio-retrieval judge
entireio-retrieval query "How does checkpoint cleanup work?"
entireio-retrieval answer "What programming languages appear in the recorded work?"
entireio-retrieval train
entireio-retrieval index --recreate --fine-tuned
entireio-retrieval evaluate --partition validation --fine-tuned
entireio-retrieval experiment
# Generate evaluation answers after validation freezes ranking choices:
entireio-retrieval answer-benchmark --partition evaluation
# Then seal retrieval and answer metrics in a one-shot evaluation:
entireio-retrieval experiment --run-frozen-evaluation
```

`prepare` selects a deterministic 275-session pilot by default. Use `prepare --full`
only after the pilot workflow and frozen evaluation configuration are satisfactory.
The compact CPU index retains session cards, checkpoint metadata, commits, topics,
user prompts, assistant conclusions, and summaries. Transcript events remain
individually citable in the full evidence store; low-value tool traffic is not embedded
initially.

`generate`, `judge`, and `answer` transmit bounded, redacted evidence packets to the
configured OpenRouter model. Run them only when disclosure of the captured
`entireio-cli` session content to OpenRouter has been authorized. Calls are cached and
resumable. `ENTIREIO_LIVE_OPENROUTER=1 pytest -m live` runs the opt-in connectivity
smoke test without sending dataset content.

## Generated data

All generated files are written below `derived/`:

- `evidence/`: normalized evidence, quality, and partition reports
- `benchmark/`: generated/accepted queries, critic decisions, and relevance labels
- `embeddings/`: cached dense vectors
- `index/`: BM25 state and local Qdrant storage
- `models/`: fine-tuned encoder checkpoints and training reports
- `answers/`: structured citation-backed answers
- `evaluation/`: retrieval and answer metrics
- `manifests/`: configuration, input, model, and artifact provenance
- `openrouter-cache/`: parsed non-secret structured responses and usage

Source Parquet files and `transcripts/*.jsonl` are immutable inputs. Derived runs are
recoverable by removing the relevant directory below `derived/`; source files are not
migrated or altered.

To remove all derived artifacts without touching source inputs, inspect the target and
then remove only `entireio-cli/derived/`. Re-run `prepare` and `index` to reconstruct
the local evidence and retrieval artifacts. Individual stages can instead be resumed
from their existing caches, manifests, vector archives, or training state.

## Evaluation

The benchmark is silver-labelled: GPT-5.6 Luna generates questions and separate Luna
critic/judge calls validate them. Deterministic citation checks, shuffled candidates,
leakage-resistant partitions, and optional double-pass evaluation reduce—without
eliminating—synthetic-label circularity.

Retrieval reports nDCG@10, MRR@10, Precision@5, Recall@20/50, evidence and claim
coverage, and temporal correctness. Answer evaluation reports claim coverage, citation
precision/recall, groundedness, answerability, and false answers on unsupported
questions.

`experiment` tunes only the relevance floor and temporal weight on validation data,
writes `evaluation/frozen-config.json`, and produces BM25, pretrained-dense, hybrid,
hybrid-plus-temporal, and—when present—fine-tuned ablations. It freezes the actual
best validation variant, including whether that variant uses the pretrained or
fine-tuned encoder. The `answer-benchmark` command applies that frozen configuration
and generates the
citation-backed answers needed for answer-quality measurement. The
`--run-frozen-evaluation` option records a one-shot marker so the frozen pilot
evaluation is not silently repeated.
