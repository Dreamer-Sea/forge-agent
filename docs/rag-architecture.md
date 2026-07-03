# RAG Pipeline Architecture

## Overview

The `forge-agent` RAG module has been upgraded from a keyword-only demo into a composable RAG pipeline.

The goal of this architecture is not to maximize retrieval quality immediately. The goal is to define stable engineering boundaries so that retrievers, embedding providers, vector stores, rerankers, context builders, and citation logic can evolve independently.

The pipeline now supports deterministic keyword retrieval and deterministic vector retrieval while keeping the upper-level Agent Runtime and RAG tool stable.

## Baseline Pipeline

The local RAG pipeline is built around Markdown knowledge bases.

Flow:

    Markdown files
      -> MarkdownLoader
      -> Heading-aware MarkdownChunker
      -> Chunk + ChunkMetadata
      -> Retriever
      -> SearchResult
      -> ContextBuilder
      -> Citation
      -> search_knowledge_base tool
      -> Agent Runtime

The important design decision is that the Agent Runtime does not know whether retrieval is implemented by keyword search, vector search, hybrid search, or a future managed search backend.

## Document, Chunk, and Metadata

A document is the original knowledge source.

A chunk is the smallest retrievable and citable unit.

Metadata stores the information needed to trace a chunk back to its source, including:

    source path
    relative path
    source id
    title
    heading path
    chunk id
    ordinal

RAG retrieval and citation should operate on chunks with metadata, not on raw document strings.

This keeps citations grounded and reproducible. A generated answer should not invent citation paths during response generation. Citation paths should come from chunk metadata.

## Chunking Strategy

`forge-agent` uses heading-aware Markdown chunking.

This is appropriate for Markdown knowledge bases because Markdown headings already encode document structure. Instead of splitting purely by character count, the chunker preserves section-level meaning and heading paths.

Compared with plain character splitting, heading-aware chunking gives better citation paths such as:

    security.md#Security > Permission System

This makes retrieved context easier to inspect, debug, and evaluate.

## Retriever Boundary

A retriever is the search strategy boundary.

Its responsibility is:

    query -> ranked SearchResult list

The retriever does not build prompts, format citations, call models, execute tools, or manage runtime state.

Current implementations:

    KeywordRetriever
    VectorRetriever

Future implementations may include:

    HybridRetriever
    BM25Retriever
    ManagedSearchRetriever
    QdrantRetriever
    ElasticsearchRetriever

Because all retrievers return the same `SearchResult` structure, the ContextBuilder, RAG tool, and Agent Runtime remain independent from the retrieval algorithm.

## Keyword Retriever

`KeywordRetriever` is the deterministic baseline retriever.

It provides stable behavior for tests, demos, and regression checks. It uses lightweight tokenization and BM25-like scoring to rank local chunks.

The keyword retriever remains the default retriever to avoid breaking existing behavior.

## Embedding Provider Boundary

An embedding provider converts text into vectors.

Its responsibility is:

    text -> embedding vector

It does not store vectors, search vectors, rerank results, build context, or format citations.

Current implementation:

    HashingEmbeddingProvider

Future implementations may include:

    OpenAI-compatible embedding provider
    local embedding model provider
    BGE embedding provider
    E5 embedding provider
    Jina embedding provider

## Deterministic Hashing Embedding

`HashingEmbeddingProvider` is not a production semantic embedding model.

It exists for engineering reasons:

    no API key required
    no network dependency
    stable in CI
    deterministic test results
    easy to replace with a real embedding model later

The purpose of this provider is to validate the vector retrieval pipeline boundary, not to maximize semantic search quality.

## Vector Store Boundary

A vector store is responsible for storing chunk vectors and running vector similarity search.

Its responsibility is:

    vectorized chunks + query vector -> vector search results

It does not embed text, rewrite queries, rerank results, build context, or format citations.

Current implementation:

    InMemoryVectorStore

Future implementations may include:

    FAISS
    Chroma
    Qdrant
    Milvus
    pgvector
    Elasticsearch dense vector search

This boundary matters because a vector store is not the same thing as a retriever.

A vector retriever composes an embedding provider and a vector store.

## Vector Retriever

`VectorRetriever` implements the retriever interface using vector similarity.

The flow is:

    query
      -> EmbeddingProvider.embed_text(query)
      -> VectorStore.similarity_search(query_vector)
      -> SearchResult list

This means `VectorRetriever` is the retrieval strategy, while `VectorStore` is the storage and similarity search backend.

The current vector retriever uses deterministic hashing embeddings and an in-memory vector store.

## Reranker Boundary

A reranker reorders first-stage retrieval results.

Its responsibility is:

    query + SearchResult list -> reranked SearchResult list

Current implementation:

    IdentityReranker

The identity reranker does not change result order. It exists to establish the reranking extension point before introducing more complex rerank logic.

Future implementations may include:

    keyword overlap reranker
    cross-encoder reranker
    LLM-based reranker
    BGE reranker
    Cohere reranker

Keeping reranking separate from retrieval avoids mixing first-stage recall logic with second-stage precision logic.

## Context Builder and Citation

The ContextBuilder is responsible for converting retrieval results into bounded model context.

It controls:

    max chunks
    max characters
    deduplication
    context ordering
    citation rendering

Retrieved chunks should not be blindly injected into the model. The context builder must keep the prompt small, traceable, and reproducible.

Citation is based on chunk metadata. This ensures that grounded answers can be traced back to the source document and heading path.

## KnowledgeBase as Pipeline Entry Point

`KnowledgeBase` is the local RAG pipeline entry point.

It is responsible for:

    loading Markdown files
    chunking documents
    building the local index
    selecting retriever type
    searching chunks
    building grounded context

It supports selectable retrievers:

    keyword
    vector

The default remains:

    keyword

This preserves old behavior while allowing the same knowledge base to be searched through different retrieval strategies.

## CLI Search

The CLI supports direct RAG search.

Keyword retrieval:

    uv run forge rag search "workspace guard permission" \
      --knowledge-base examples/knowledge_base \
      --retriever keyword \
      --top-k 3

Vector retrieval:

    uv run forge rag search "workspace guard permission" \
      --knowledge-base examples/knowledge_base \
      --retriever vector \
      --top-k 3

This makes retrieval behavior visible and testable without going through the full Agent Runtime.

## Extension Plan

The current architecture prepares the project for the next stages.

### Hybrid Retrieval

A future `HybridRetriever` can combine keyword and vector retrieval:

    query
      -> KeywordRetriever
      -> VectorRetriever
      -> merge results
      -> normalize scores
      -> deduplicate chunks
      -> return SearchResult list

This can improve recall by combining lexical matching and vector similarity.

### Rerank

A future reranker can rerank the top candidates from keyword, vector, or hybrid retrieval:

    first-stage retrieval results
      -> Reranker
      -> reranked results
      -> ContextBuilder

This can improve precision before context is sent to the model.

### Evaluation

Because all retrievers return the same `SearchResult` shape, evaluation can compare retrieval strategies using the same cases:

    keyword retrieval
    vector retrieval
    hybrid retrieval
    hybrid + rerank

This supports retrieval regression tests and makes RAG quality measurable.

## Design Summary

The upgraded RAG pipeline separates the following responsibilities:

    MarkdownLoader       -> load source documents
    MarkdownChunker      -> create retrievable chunks
    ChunkMetadata        -> preserve source and citation data
    Retriever            -> retrieve ranked chunks
    EmbeddingProvider    -> convert text to vectors
    VectorStore          -> store and search vectors
    Reranker             -> reorder first-stage results
    ContextBuilder       -> build bounded grounded context
    Citation             -> trace answers back to sources
    KnowledgeBase        -> compose the local RAG pipeline

This keeps the RAG system extensible while preserving deterministic tests and stable demos.
