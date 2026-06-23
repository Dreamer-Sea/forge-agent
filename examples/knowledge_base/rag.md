# Retrieval Augmented Generation

## Concept

Retrieval Augmented Generation, also known as RAG, allows an agent to answer questions using external knowledge instead of relying only on model parameters.

## Components

A minimal RAG pipeline includes:

- Loader: loads documents from local files or external sources.
- Chunker: splits documents into searchable chunks.
- Retriever: finds relevant chunks for a query.
- Context Builder: converts search results into prompt-ready context.
- Citation: keeps answers traceable to source documents.

## Agentic RAG

In forge-agent, RAG is exposed as a tool named search_knowledge_base. The Agent Runtime decides when to call it, what query to send, and how to use the returned context.

## Design Notes

Demo-level RAG can start with keyword or BM25 retrieval. The important design point is to keep the retriever interface replaceable, so it can later be upgraded to vector search or hybrid retrieval.