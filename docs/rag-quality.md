# RAG Quality Evaluation

This document describes the deterministic RAG quality pipeline in `forge-agent`.

The goal is to make retrieval quality measurable, comparable, and explainable through a small engineering loop:

1. load a local Markdown knowledge base;
2. retrieve candidate chunks with keyword, vector, or hybrid retrieval;
3. optionally rerank candidate chunks;
4. evaluate retrieval quality against a JSONL dataset;
5. write Markdown and JSON reports for review.

## Why RAG quality evaluation matters

A RAG pipeline is not reliable just because it can return chunks.

For an agent system, retrieval must answer several engineering questions:

- Did the retriever find the expected source?
- Did the expected source appear in the top-k results?
- Was the expected source ranked high enough?
- Did the retrieved context contain the key terms needed to answer the question?
- Did the system return no answer when the knowledge base did not contain the answer?
- Were citations present so the final answer can be grounded?

Without these metrics, RAG changes are difficult to compare.

## Retrieval backends

`forge-agent` supports three retrieval backends.

| Retriever | Purpose | Strength | Limitation |
|---|---|---|---|
| `keyword` | Lexical retrieval over Markdown chunks | Deterministic and easy to debug | Weak for semantic paraphrases |
| `vector` | Embedding-based retrieval with an in-memory vector store | Captures fuzzy similarity | May retrieve weakly related chunks |
| `hybrid` | Combines keyword and vector retrieval | Balances lexical precision and semantic recall | Needs fusion and evaluation |

### Keyword retrieval

Keyword retrieval scores chunks by lexical overlap between query terms and chunk text.

It is useful for exact domain terms such as `workspace guard`, `permission policy`, `trace events`, and `grounded citations`.

### Vector retrieval

Vector retrieval embeds chunks and queries into fixed-size vectors and compares similarity through the in-memory vector store.

In this project the embedding provider is intentionally deterministic. It is designed for local tests and architecture demonstration, not production semantic quality.

### Hybrid retrieval

Hybrid retrieval combines keyword and vector retrieval.

The current implementation uses Reciprocal Rank Fusion, or RRF, to merge ranked lists from multiple retrievers.

The scoring idea is:

    rrf_score(document) = sum(1 / (k + rank_in_each_retriever))

RRF is useful because:

- it does not require keyword and vector scores to be on the same scale;
- it rewards chunks that appear in multiple retrieval lists;
- it is deterministic and easy to test;
- it is a strong baseline before introducing heavier ranking models.

## Reranking

The current reranker is a deterministic baseline:

    keyword-overlap

It reranks retrieved chunks by query-term overlap and keeps stable tie-breaking.

This is not intended to replace a production reranker. Its purpose is to create a testable reranking interface and a deterministic local baseline.

Production alternatives include cross-encoder rerankers, BGE reranker, Cohere Rerank, Jina Reranker, and LLM-based reranking.

## Evaluation dataset

The retrieval evaluation dataset lives at:

    examples/evals/rag_retrieval.jsonl

Each line is one eval case.

Example schema:

    {
      "case_id": "rag_security_permission_system",
      "query": "How does the permission system decide whether a tool call is allowed?",
      "expected_sources": ["security.md"],
      "expected_terms": ["permission system", "tool call", "allowed", "runtime policy"],
      "top_k": 3,
      "expect_no_answer": false
    }

The dataset intentionally includes both answerable and no-answer cases.

No-answer cases are important because real RAG systems must know when the knowledge base does not contain enough information.

## Metrics

The RAG retrieval eval runner calculates deterministic metrics.

| Metric | Meaning |
|---|---|
| `source_hit_rate` | Percentage of answerable cases where at least one expected source was retrieved |
| `recall_at_k` | Percentage of expected sources found within top-k |
| `MRR` | Mean reciprocal rank of the first expected source |
| `term_hit_rate` | Percentage of expected key terms found in retrieved context |
| `no_answer_accuracy` | Percentage of no-answer cases where retrieval returned no result |
| `citation_presence_rate` | Percentage of answerable cases where citations were present |

These metrics separate different retrieval failure modes.

For example:

- high `source_hit_rate` with low `term_hit_rate` means the retriever found the right file but not enough useful context;
- high `source_hit_rate` with low `MRR` means the expected source exists but is ranked too low;
- low `no_answer_accuracy` means the system needs an abstention policy or no-answer threshold.

## CLI usage

Run keyword retrieval evaluation:

    uv run forge rag eval examples/evals/rag_retrieval.jsonl \
      --knowledge-base examples/knowledge_base \
      --retriever keyword \
      --top-k 3

Run hybrid retrieval with keyword-overlap reranking:

    uv run forge rag eval examples/evals/rag_retrieval.jsonl \
      --knowledge-base examples/knowledge_base \
      --retriever hybrid \
      --reranker keyword-overlap \
      --top-k 5 \
      --top-n 3

Generate Markdown and JSON reports:

    uv run forge rag eval examples/evals/rag_retrieval.jsonl \
      --knowledge-base examples/knowledge_base \
      --retriever hybrid \
      --reranker keyword-overlap \
      --top-k 5 \
      --top-n 3 \
      --output examples/reports/rag-eval-report.md \
      --json-output examples/reports/rag-eval-results.json

## Report artifacts

The example reports are committed under:

    examples/reports/rag-eval-report.md
    examples/reports/rag-eval-results.json

The Markdown report is intended for human review. It includes configuration, summary metrics, per-case results, failure analysis, and next steps.

The JSON report is intended for automation, CI comparison, and future regression tracking.

## Current result interpretation

The current demo result shows strong retrieval for answerable cases:

    source_hit_rate: 100.00%
    recall_at_k: 100.00%
    MRR: 100.00%
    citation_presence_rate: 100.00%

It also exposes a deliberate weakness:

    no_answer_accuracy: 0.00%

This means the current retriever still returns documents for the no-answer case.

That is expected at this stage. The current retrieval layer does not yet implement a no-answer threshold, confidence calibration, or abstention policy.

## Known limitations

The current RAG quality implementation is intentionally small and deterministic.

Known limitations:

1. The vector retriever uses a deterministic local embedding provider, not a production embedding model.
2. The reranker is a keyword-overlap baseline, not a semantic cross-encoder.
3. No-answer detection currently depends on empty retrieval results only.
4. There is no score threshold or confidence calibration yet.
5. Evaluation focuses on retrieval quality, not final answer correctness.

## Next improvements

Recommended next steps:

1. Add a no-answer threshold based on retrieval score, fused rank, or reranker confidence.
2. Add a production-grade reranker behind the existing reranker interface.
3. Compare keyword, vector, hybrid, and reranked runs in one consolidated report.
4. Add answer-level evaluation for groundedness, faithfulness, citation correctness, and answer correctness.
5. Add CI regression checks for critical RAG cases.
