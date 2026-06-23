# Day 2 RAG Retrieval Assessment

## Purpose

This document records the Day 2 retrieval quality assessment for the local Markdown RAG pipeline.

The assessment checks whether the keyword retriever can return the expected source chunk in top-3 results for representative user queries.

## Assessment Table

| Query | Expected Source | Actual Top 3 Sources | Hit or Miss | Miss Reason |
|---|---|---|---|---|
| Agent Runtime 有哪些核心模块？ | agent-runtime.md#Agent Runtime > Components | agent-runtime.md#Agent Runtime > Components<br>agent-runtime.md#Agent Runtime > Concept<br>security.md#Security > Components | hit | - |
| ToolRegistry 的作用是什么？ | agent-runtime.md#Agent Runtime > ToolRegistry | agent-runtime.md#Agent Runtime > ToolRegistry | hit | - |
| RAG 为什么需要 citation？ | rag.md#Retrieval Augmented Generation > Components | rag.md#Retrieval Augmented Generation > Components<br>rag.md#Retrieval Augmented Generation > Agentic RAG<br>evaluation.md#Evaluation > Design Notes | hit | - |
| Permission system 如何工作？ | security.md#Security > Permission System | security.md#Security > Permission System<br>security.md#Security > Components<br>evaluation.md#Evaluation > How Eval Works | hit | - |
| Trace event 记录哪些内容？ | security.md#Security > Trace Events | security.md#Security > Trace Events<br>agent-runtime.md#Agent Runtime > Concept<br>security.md#Security > Components | hit | - |

## Notes

- The current retriever is intentionally demo-level and uses keyword/BM25-style scoring.
- Chinese queries are supported through lightweight query alias expansion for common Day 2 assessment phrases.
- Future improvements can replace the retriever with vector search, hybrid search, reranking, or better multilingual tokenization without changing the Agent Runtime or ToolRegistry.
