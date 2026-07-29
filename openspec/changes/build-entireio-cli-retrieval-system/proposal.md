## Why

The `entireio-cli` dataset contains rich but fragmented evidence about work performed in `entireio/cli`, making it difficult to answer repository-understanding questions without manually tracing sessions, commits, and transcripts. A lightweight, locally runnable retrieval system and reproducible synthetic benchmark are needed to surface and evaluate evidence-backed answers using only this dataset.

## What Changes

- Add a data-preparation pipeline that converts the Parquet tables and JSONL transcripts in `entireio-cli` into sanitized, citation-addressable session, commit, conversation, and topic evidence.
- Generate a reproducible silver query benchmark with supported, partially supported, and unsupported repository-understanding questions through OpenRouter GPT-5.6 Luna using medium reasoning, structured outputs, an independent critic pass, grounding checks, relevance grading, and leakage-resistant splits.
- Fine-tune and evaluate `BAAI/bge-small-en-v1.5` on multi-positive question-to-evidence relationships using CPU-feasible training settings and mined hard negatives.
- Add local hybrid retrieval backed by Qdrant, combining dense and lexical retrieval with metadata filtering, result grouping, and query-dependent chronological ranking.
- Generate natural-language answers whose claims cite retrieved dataset records and that qualify or abstain when the indexed evidence is partial or absent.
- Add experiment and evaluation reporting for retrieval relevance, evidence coverage, answer faithfulness, temporal correctness, and unsupported-query handling.
- Enforce a hard evidence boundary: no repository clone, source snapshot, web content, or other material outside `entireio-cli` may influence generated queries, labels, retrieval results, or answers.

## Capabilities

### New Capabilities

- `dataset-evidence-preparation`: Ingest, sanitize, normalize, summarize, and cite the allowed `entireio-cli` dataset records.
- `silver-query-benchmark`: Generate, validate, split, and persist repository-understanding queries, expected claims, answerability labels, and graded relevance judgments.
- `embedding-training-evaluation`: Fine-tune the lightweight embedding model and compare it against reproducible retrieval baselines.
- `hybrid-temporal-retrieval`: Index evidence and retrieve it through dense, lexical, metadata, and query-dependent temporal signals.
- `citation-backed-answering`: Produce grounded natural-language answers with dataset-native citations and explicit partial-support or abstention behavior.

### Modified Capabilities

None.

## Impact

- Adds a local Python-based data, training, indexing, evaluation, and answering workflow scoped to `/home/emreerhan/projects/merget-live/entireio-cli`.
- Introduces dependencies for Parquet processing, Sentence Transformers/PyTorch, Qdrant, lexical retrieval, structured validation, and evaluation.
- Uses the OpenRouter API key at `temp_openrouter_key.txt` for query generation, criticism, relevance grading, and answer generation; the key must be read at runtime, protected from logs and artifacts, and secured with restrictive permissions.
- Produces derived evidence, benchmark, model, index, run-manifest, and evaluation artifacts without modifying the original Parquet or transcript inputs.
- Targets the available CPU-only environment and an initial subset pilot before full-dataset processing.
