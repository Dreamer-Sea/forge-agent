from forge_agent.rag.embeddings import HashingEmbeddingProvider


def test_hashing_embedding_is_deterministic() -> None:
    provider = HashingEmbeddingProvider(dimensions=16)

    first = provider.embed_text("workspace guard permission")
    second = provider.embed_text("workspace guard permission")

    assert first == second


def test_hashing_embedding_uses_configured_dimensions() -> None:
    provider = HashingEmbeddingProvider(dimensions=32)

    vector = provider.embed_text("workspace guard permission")

    assert len(vector) == 32


def test_hashing_embedding_returns_zero_vector_for_empty_text() -> None:
    provider = HashingEmbeddingProvider(dimensions=8)

    vector = provider.embed_text("")

    assert vector == [0.0] * 8


def test_hashing_embedding_rejects_invalid_dimensions() -> None:
    try:
        HashingEmbeddingProvider(dimensions=0)
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("expected ValueError")
