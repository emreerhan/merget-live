## Context

`entireio-cli` is a roughly 1.9 GB, single-repository slice of SWE-chat containing five Parquet tables and 869 raw JSONL transcripts. It records 870 sessions, 1,506 checkpoints, 1,597 commit records, and 348,196 normalized events for `entireio/cli`. The useful evidence is fragmented across conversational text, tool activity, metadata, patches, and checkpoint relationships.

The system is intended to answer repository-understanding questions about work represented in this dataset. It is not allowed to inspect a repository clone or any source snapshot. Consequently, answers describe observed work and may be partial even for apparently simple repository-wide questions.

The development environment is CPU-only with 8 cores/16 threads, approximately 7.4 GiB RAM, and ample disk. Query generation, criticism, relevance judging, and answer generation use OpenRouter GPT-5.6 Luna with medium reasoning. No human review is included in this change, so generated query and relevance assets are explicitly silver labels.

## Goals / Non-Goals

**Goals:**

- Turn the allowed dataset into compact, sanitized, citation-addressable evidence at conversation, commit, session, and topic levels.
- Generate a leakage-resistant silver benchmark of realistic repository-understanding questions with expected claims, graded relevance, temporal intent, and three-state answerability.
- Fine-tune and evaluate a lightweight dense encoder on available hardware.
- Provide local hybrid retrieval with exact-term handling, metadata filters, evidence grouping, and intent-dependent chronology.
- Produce natural-language answers whose factual claims are validated against dataset-native citations.
- Make every derived artifact and experiment reproducible through immutable inputs, explicit schemas, configuration, and hashes.

**Non-Goals:**

- Cloning, reconstructing, or inspecting the complete `entireio/cli` repository.
- Claiming an exhaustive or current description of the repository outside the dataset's January–March 2026 coverage.
- Extending the first implementation to multiple repositories or the complete SWE-chat dataset.
- Human review of generated questions, judgments, or answers.
- Training an answer-generation model or building a hosted production service.

## Decisions

### 1. Treat the dataset boundary as a provenance invariant

All factual records originate from `sessions.parquet`, `session_logs.parquet`, `conversations.parquet`, `checkpoints.parquet`, `commits.parquet`, or `transcripts/*.jsonl`. Models and software dependencies may be downloaded, and OpenRouter may transform allowed evidence, but external material cannot become factual evidence.

This boundary is enforced by carrying source identifiers through every stage and requiring all claims and relevance judgments to resolve to normalized evidence. Repository cloning was rejected because it would change the target from work-history understanding to repository-state understanding and make the benchmark depend on an additional, mutable corpus.

### 2. Build bounded multi-level evidence instead of embedding transcripts whole

The pipeline creates:

- conversation evidence for prompts, assistant conclusions, thinking where retained, tool calls, and bounded tool results;
- commit evidence for messages, changed files, patches, agent changes, and attribution;
- session evidence summarizing problems, decisions, actions, outcomes, files, commits, and time range;
- topic evidence aggregating related sessions and commits without losing their citations.

Raw progress events, snapshots, and repeated boilerplate are excluded from embedding text unless they contribute unique evidence. Long patch and tool-result content is chunked along file or logical boundaries. Each chunk receives a stable content-derived identifier and retains parent relationships.

A single session vector was rejected because broad questions can require multiple sessions and long sessions exceed the encoder context. Whole-transcript long-context encoding was rejected because it is too noisy and computationally expensive on the target machine.

### 3. Keep original inputs immutable and derived artifacts colocated

Implementation code will live under `entireio-cli/src/entireio_retrieval/`, with configuration under `entireio-cli/config/`, tests under `entireio-cli/tests/`, and generated artifacts under `entireio-cli/derived/`. Source Parquet and JSONL files are read-only inputs. Each run uses a versioned directory or manifest so a failed run can be discarded without corrupting a prior result.

### 4. Create silver queries with separate generator and critic passes

The generator receives compact evidence packets and a controlled taxonomy spanning architecture, behavior, technology, history, rationale, testing, implementation, and unsupported questions. It returns strict JSON containing the question, scope, temporal mode, answerability, expected claims, and evidence identifiers.

A separate critic call evaluates grounding, naturalness, ambiguity, evidence sufficiency, temporal correctness, and source-language copying. Deterministic validators resolve citations, verify quoted evidence where supplied, enforce schema and length constraints, and remove exact and semantic duplicates.

The same model is used for cost and operational simplicity, but calls have different prompts, fresh context, shuffled evidence, and stored provenance. Double-pass consensus can be enabled for the frozen evaluation set. A one-pass generator was rejected because invalid evidence and self-confirming labels would be too common without human review.

### 5. Split evidence groups before generating queries

Sessions and related checkpoint/topic groups are assigned to train, validation, or frozen evaluation before generation. The initial plan uses evidence before March 1 for training, March 1–9 for validation, and March 10 onward for evaluation, with relationship grouping allowed to move boundary records to prevent leakage. Repository-wide questions carry an explicit dataset cutoff.

Random query-level splitting was rejected because paraphrases from the same session or topic would make evaluation optimistic.

### 6. Use three-state answerability and graded multi-positive relevance

Queries and answers use `supported`, `partially_supported`, and `unsupported_by_dataset`. Broad questions can map to several relevant evidence units. Candidate pools combine lexical, pretrained dense, fine-tuned dense, metadata-neighbor, and temporally diverse retrieval. Luna grades randomized candidates 0–3 and must cite the expected claims supported by every nonzero grade.

Absence is never converted into a repository-wide negative claim. “Not found in this dataset” is allowed; “the repository does not use it” is not.

### 7. Fine-tune BGE-small with a CPU-bounded multi-positive objective

`BAAI/bge-small-en-v1.5` is pinned as the initial 33M-parameter, 384-dimensional encoder. The pilot uses 256-token inputs, small batches, normalized embeddings, consistent asymmetric query instructions, and frozen lower layers. Multi-positive contrastive training excludes related positives from in-batch negatives. Hard negatives come only from high-ranking candidates given a grounded relevance grade of zero.

`Snowflake/snowflake-arctic-embed-s` may be evaluated later as a challenger, but supporting multiple encoders is not required for the first apply. Larger and long-context models were rejected due to CPU and memory cost.

### 8. Use Qdrant for local hybrid and temporal retrieval

A local Qdrant collection stores named dense and sparse vectors plus payload fields for evidence IDs, parents, dates, files, branch, session, checkpoint, commit, evidence type, and citation data. Dense and lexical candidate lists are fused before grouping.

Temporal behavior is determined from explicit or classified query intent:

- `current_state` and `latest_change`: modest freshness rerank after a relevance floor;
- `historical_evolution`: favor coverage across time;
- `introduced_when`: favor early directly relevant evidence;
- `as_of_date`: exclude later evidence, then rank within the allowed window;
- `time_neutral`: no freshness adjustment.

A universal recency boost was rejected because it damages historical and origin questions.

### 9. Generate and validate answers from retrieved evidence only

Luna receives the query, temporal mode, and bounded retrieved evidence. It returns structured claims, citation IDs, answerability, and rendered answer text. A validator resolves every citation and performs a second groundedness check before accepting the answer. Unsupported answers abstain rather than using model background knowledge.

### 10. Evaluate stages independently and end to end

Retrieval experiments compare BM25, pretrained dense, fine-tuned dense, hybrid, and hybrid-plus-temporal configurations using nDCG@10, MRR@10, Precision@5, Recall@20/50, and evidence/claim coverage. Answer evaluation measures expected-claim coverage, citation precision and recall, groundedness, answerability classification, temporal correctness, and false answers on unsupported questions.

Similarity scores are diagnostic rather than probabilities. Any support threshold is selected on validation data and frozen before final evaluation.

### 11. Secure external API use

The OpenRouter key is read from `temp_openrouter_key.txt` at runtime, never copied into configuration or artifacts, and never logged. The setup verifies restrictive permissions and prompts the operator to correct an insecure key file. Evidence is minimized and sanitized before transmission. Requests use bounded concurrency, retries with backoff, cached response hashes, strict JSON schemas, and usage accounting.

## Risks / Trade-offs

- **[Silver-label circularity]** The same model family generates and judges data, which can inflate quality → Separate prompts and calls, shuffled evidence, deterministic citation validation, double-pass consensus for evaluation, and explicit reporting that labels are silver.
- **[Incomplete repository knowledge]** Dataset-only evidence cannot prove full current-state facts → Use three-state answerability and dataset-scoped wording in both queries and answers.
- **[False negatives in contrastive training]** Related sessions may be incorrectly treated as negatives → Group positives by claim, checkpoint, topic, and silver relevance before batch construction.
- **[Synthetic query style bias]** Generated questions may be unnaturally polished → Enforce varied scopes, lengths, and personas; mix cleaned authentic prompts with generated questions; deduplicate by semantics.
- **[Chronology overwhelms relevance]** Recent but weak evidence may outrank authoritative older evidence → Apply temporal scoring only after a relevance floor and tune its bounded weight on validation data.
- **[CPU training time]** Fine-tuning may still be slow → Start with 250–300 sessions, frozen lower layers, 256-token inputs, small batches, early stopping, and resumable checkpoints.
- **[External data exposure]** Transcripts and commits can contain secrets or personal information → Sanitize minimal packets, scan before sending, protect the API key, and retain auditable request hashes.
- **[Qdrant complexity for a small corpus]** Operational overhead may exceed immediate scale needs → Use embedded/local persistence and keep normalized evidence artifacts portable so another index can be substituted later.
- **[Missing or inconsistent records]** Some sessions have missing transcripts, normalized conversations, or commits → Preserve explicit missingness, report coverage, and avoid invented joins.

## Migration Plan

1. Add configuration, schemas, and immutable input validation.
2. Run a subset evidence build and inspect automated coverage reports.
3. Generate and validate the pilot silver benchmark.
4. Establish BM25 and pretrained dense baselines.
5. Fine-tune the pilot encoder and build the local hybrid index.
6. Add temporal ranking, citation-backed answering, and end-to-end evaluation.
7. Freeze a pilot evaluation report before optionally scaling evidence preparation and generation to all usable sessions.

Rollback consists of removing or selecting an earlier version under `entireio-cli/derived/`; source dataset files are never migrated or modified.

## Open Questions

None blocking. Pilot thresholds, exact temporal weights, critic acceptance scores, and resource limits are validation-tuned configuration values rather than unresolved product decisions.
