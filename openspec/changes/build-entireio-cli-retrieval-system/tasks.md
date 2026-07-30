## 1. Project and Configuration Setup

- [x] 1.1 Create the Python package, CLI entry point, configuration directory, tests, and versioned `derived/` artifact layout under `entireio-cli`
- [x] 1.2 Define and pin runtime, training, Qdrant, Parquet, validation, security-scanning, and evaluation dependencies
- [x] 1.3 Add typed configuration and schemas for dataset paths, pilot selection, evidence records, model settings, temporal ranking, OpenRouter requests, and run manifests
- [x] 1.4 Add startup checks that validate required input files, prevent writes to source data, protect secrets in logs, and require restrictive permissions for `temp_openrouter_key.txt`

## 2. Dataset Ingestion and Provenance

- [x] 2.1 Implement streaming readers and schema validation for all five Parquet tables and the JSONL transcripts
- [x] 2.2 Implement joins among sessions, conversations, checkpoints, commits, and transcript references while preserving missing and inconsistent relationships
- [x] 2.3 Implement stable evidence identifiers and citation resolution back to dataset-native IDs, source files, rows, and transcript events
- [x] 2.4 Implement input hashing, configuration hashing, run manifests, resumable versioned outputs, and source-data immutability checks
- [x] 2.5 Add ingestion and citation tests covering missing transcripts, absent normalized conversations, unresolved commits, duplicate relationships, and malformed transcript events

## 3. Evidence Preparation

- [x] 3.1 Implement conversation filtering and bounded chunking for prompts, assistant conclusions, retained reasoning, tool calls, and useful tool results
- [x] 3.2 Implement commit and patch evidence extraction with changed paths, logical patch chunks, timestamps, checkpoint links, and attribution metadata
- [x] 3.3 Implement deterministic session evidence cards covering problems, decisions, actions, outcomes, files, commits, and time ranges
- [x] 3.4 Implement multi-session topic clustering and citation-preserving topic evidence derived only from allowed records
- [x] 3.5 Implement redaction and payload minimization for secrets, author emails, absolute paths, duplicated output, and irrelevant raw content
- [x] 3.6 Add evidence-quality reports and tests for stable IDs, bounded size, redaction, citation resolution, missing values, and repeated-run determinism

## 4. Pilot Partitioning

- [x] 4.1 Implement a reproducible, representative 250–300-session pilot selector spanning dates, users, agents, intents, components, session sizes, and commit outcomes
- [x] 4.2 Implement chronological train, validation, and frozen evaluation assignment with checkpoint, topic, source-session, and paraphrase-group leakage prevention
- [x] 4.3 Add partition reports and automated assertions that related evidence groups do not cross partitions

## 5. Silver Query Benchmark

- [x] 5.1 Implement the OpenRouter client for `openai/gpt-5.6-luna` with medium reasoning, strict JSON schemas, bounded concurrency, retry/backoff, response caching, usage accounting, and secret-safe logging
- [x] 5.2 Define versioned generator prompts and schemas for varied repository-, subsystem-, and change-level questions across the agreed query taxonomy
- [x] 5.3 Implement query generation with expected claims, source evidence, temporal mode, scope, and `supported`, `partially_supported`, or `unsupported_by_dataset` labels
- [x] 5.4 Define and implement a separate Luna critic pass with shuffled evidence and optional double-pass consensus for frozen evaluation queries
- [x] 5.5 Implement deterministic query validation for schema, evidence resolution, quoted support, standalone wording, length, temporal consistency, and unsupported-claim phrasing
- [x] 5.6 Implement exact and embedding-based semantic deduplication without filtering queries based on baseline retrieval success
- [x] 5.7 Implement BM25, pretrained-dense, metadata-neighbor, and temporally diverse candidate pooling followed by randomized Luna relevance grading from 0 through 3
- [x] 5.8 Persist accepted and rejected queries, relevance judgments, raw non-secret responses, prompt versions, model settings, costs, hashes, and validation reasons
- [x] 5.9 Add benchmark tests using mocked API responses plus a small opt-in live smoke test that never prints or persists the API key

## 6. Baseline Embeddings and Hybrid Index

- [x] 6.1 Pin and load `BAAI/bge-small-en-v1.5`, implement consistent asymmetric query instructions, normalized 384-dimensional encoding, batching, and embedding caching
- [x] 6.2 Implement the corpus-derived lexical/BM25 representation and establish BM25-only and pretrained-dense baseline runs
- [x] 6.3 Create the local Qdrant collection with dense and sparse vectors plus indexed payload fields for citations, parents, dates, files, branches, sessions, checkpoints, commits, and evidence types
- [x] 6.4 Implement dense and lexical candidate retrieval, rank fusion, metadata filtering, evidence grouping, and retrieval trace output
- [x] 6.5 Implement temporal-mode classification or override and ranking behavior for current, latest, evolution, introduction, as-of-date, and time-neutral queries
- [x] 6.6 Add retrieval tests for exact symbols, conceptual questions, filters, grouping, chronology modes, relevance floors, deterministic traces, and citation preservation

## 7. Embedding Fine-Tuning

- [x] 7.1 Build multi-positive training examples from accepted training queries and exclude all known related positives from in-batch negatives
- [x] 7.2 Mine high-ranking hard-negative candidates and admit only candidates with grounded silver relevance grade 0
- [x] 7.3 Implement CPU-bounded BGE-small fine-tuning with 256-token inputs, frozen lower layers, small batches, early stopping, resumable checkpoints, and resource reporting
- [x] 7.4 Select the best checkpoint and support rebuilding the Qdrant dense vectors from the fine-tuned encoder
- [x] 7.5 Add training-integrity tests that prevent validation/evaluation leakage and verify model, data, prompt, and configuration provenance

## 8. Citation-Backed Answering

- [x] 8.1 Define versioned Luna answer prompts and structured schemas for claims, citations, answerability, limitations, temporal interpretation, and rendered answer text
- [x] 8.2 Implement bounded retrieval-context assembly that includes only allowed, citation-addressable evidence
- [x] 8.3 Implement supported and partially supported answers plus `unsupported_by_dataset` abstention without filling gaps from model background knowledge
- [x] 8.4 Implement citation resolution and claim-support validation with bounded regeneration or rejection on failure
- [x] 8.5 Add answering tests for dataset-scoped language, multiple citations, partial evidence, unsupported questions, invalid citations, and temporal qualifications

## 9. Evaluation and Delivery

- [x] 9.1 Implement retrieval metrics for nDCG@10, MRR@10, Precision@5, Recall@20/50, evidence coverage, claim coverage, answerability, and temporal correctness
- [x] 9.2 Implement answer metrics for expected-claim coverage, citation precision and recall, groundedness, answerability classification, temporal correctness, and unsupported-query false-answer rate
- [x] 9.3 Run and compare BM25-only, pretrained dense, fine-tuned dense, hybrid, and hybrid-plus-temporal pilot experiments on unchanged partitions
- [x] 9.4 Tune thresholds and temporal weights only on validation data, freeze the selected configuration, and run the final pilot evaluation once
- [x] 9.5 Produce a reproducible pilot report covering dataset coverage, accepted/rejected query distributions, resource usage, API cost, retrieval ablations, answer quality, limitations, and artifact hashes
- [x] 9.6 Add end-to-end documentation for preparing evidence, generating the benchmark, training, indexing, querying, answering, evaluating, resuming runs, and removing derived artifacts
- [x] 9.7 Run the complete automated test suite and a dataset-boundary audit proving that no factual evidence outside `entireio-cli` influenced the artifacts
