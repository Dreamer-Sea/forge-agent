# RAG Retrieval Evaluation Report

## Configuration

- Dataset: `examples/evals/rag_retrieval.jsonl`
- Knowledge base: `examples/knowledge_base`
- Retriever: `hybrid`
- Reranker: `keyword-overlap`
- Total cases: `6`
- Answerable cases: `5`
- No-answer cases: `1`

## Summary Metrics

| Metric | Value |
|---|---:|
| source_hit_rate | 100.00% |
| recall_at_k | 100.00% |
| MRR | 100.00% |
| term_hit_rate | 85.00% |
| no_answer_accuracy | 0.00% |
| citation_presence_rate | 100.00% |

## Case Results

| Case | Source Hit | Recall@K | MRR | Term Hit Rate | No-answer Correct | Retrieved Sources |
|---|---:|---:|---:|---:|---:|---|
| `rag_security_permission_system` | True | 1.00 | 1.00 | 100.00% | False | security.md, security.md, rag.md |
| `rag_workspace_guard_path_escape` | True | 1.00 | 1.00 | 50.00% | False | security.md, agent-runtime.md, agent-runtime.md |
| `rag_runtime_agent_loop` | True | 1.00 | 1.00 | 100.00% | False | agent-runtime.md, agent-runtime.md, rag.md |
| `rag_evaluation_loop` | True | 1.00 | 1.00 | 100.00% | False | evaluation.md, evaluation.md, evaluation.md |
| `rag_retrieval_citations` | True | 1.00 | 1.00 | 75.00% | False | rag.md, rag.md, evaluation.md |
| `rag_no_answer_billing_saga` | False | 0.00 | 0.00 | 0.00% | False | evaluation.md, rag.md, evaluation.md |

## Failure Analysis

- `rag_workspace_guard_path_escape` hit the expected source but missed terms: guard, path.
- `rag_retrieval_citations` hit the expected source but missed terms: source.
- `rag_no_answer_billing_saga` expected no answer, but retrieved: evaluation.md, rag.md, evaluation.md. This indicates the current retrieval pipeline does not have an abstention threshold.

## Next Steps

- Add a no-answer threshold or abstention policy so retrieval can return no result when all candidates are weak.
- Replace the deterministic keyword-overlap reranker with a production reranker such as a cross-encoder, BGE reranker, Cohere rerank, Jina reranker, or LLM reranker.
- Extend retrieval eval into answer eval with groundedness, faithfulness, citation correctness, and answer correctness.
