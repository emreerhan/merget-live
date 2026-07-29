## ADDED Requirements

### Requirement: Local hybrid evidence index
The system SHALL persist normalized evidence and its dense and lexical representations in a local Qdrant index under the `entireio-cli` derived-artifact area. The lexical representation MUST be learned only from allowed dataset evidence.

#### Scenario: Evidence index is built
- **WHEN** normalized evidence and embeddings are available
- **THEN** Qdrant contains named dense and sparse representations plus citation and filter payloads for every indexed record

### Requirement: Hybrid candidate retrieval
The system SHALL retrieve semantic and exact-term candidates and fuse their rankings so conceptual queries and queries containing filenames, symbols, commands, or error strings can both succeed.

#### Scenario: Query contains an exact symbol
- **WHEN** a query contains a symbol present in evidence
- **THEN** lexical candidates participate alongside dense candidates in the fused ranking

### Requirement: Metadata filtering and session grouping
The system SHALL support filters over evidence type, session, checkpoint, commit, branch, file path, timestamp, and answerability-relevant metadata when present. It SHALL group evidence chunks into coherent session or topic results while retaining individual citations.

#### Scenario: Several chunks belong to one session
- **WHEN** multiple chunks from the same session rank highly
- **THEN** the response groups them without discarding their individual evidence identifiers

### Requirement: Query-dependent temporal ranking
The system SHALL classify or receive a temporal mode and apply chronology accordingly: freshness for current-state and latest-change queries, broad chronological coverage for evolution queries, early evidence for introduction queries, cutoff filtering for as-of-date queries, and no freshness adjustment for time-neutral queries.

#### Scenario: Current-state question is searched
- **WHEN** the temporal mode is `current_state`
- **THEN** chronology modestly reranks sufficiently relevant candidates toward later dataset evidence without allowing weak recent evidence to outrank strongly relevant evidence

#### Scenario: Evolution question is searched
- **WHEN** the temporal mode is `historical_evolution`
- **THEN** results preserve useful evidence across multiple dates rather than applying a universal newest-first order

### Requirement: Retrieval traceability
Every retrieval response SHALL expose component scores or ranks, applied filters, temporal mode, final ranking, and citation identifiers sufficient to reproduce and diagnose the result.

#### Scenario: Retrieval result is evaluated
- **WHEN** an experiment records a ranked list
- **THEN** the dense, lexical, fusion, temporal, grouping, and final-rank information is available in the run artifact

