from __future__ import annotations

from forge_agent.memory import (
    InMemoryStore,
    MemoryRecord,
    MemoryRetriever,
    MemoryScope,
    MemoryType,
)


def _record(
    content: str,
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    scope: MemoryScope = MemoryScope.PROJECT,
    scope_id: str | None = "forge-agent",
    importance: float = 0.5,
    confidence: float = 0.5,
) -> MemoryRecord:
    return MemoryRecord(
        type=memory_type,
        scope=scope,
        scope_id=scope_id,
        content=content,
        source="test",
        importance=importance,
        confidence=confidence,
    )


def test_memory_retriever_returns_filtered_results_by_score() -> None:
    store = InMemoryStore()
    strong = store.add(_record("项目默认使用 Python 3.13 和 uv", importance=0.9))
    store.add(_record("Python 是项目语言之一", importance=0.5))

    retriever = MemoryRetriever(store)

    results = retriever.retrieve("Python 3.13 uv", min_score=0.6)

    assert [result.record for result in results] == [strong]
    assert results[0].rank == 1


def test_memory_retriever_filters_by_scope_scope_id_and_type() -> None:
    store = InMemoryStore()
    expected = store.add(
        _record(
            "demo session 默认使用 Python 3.13",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="demo",
        )
    )
    store.add(
        _record(
            "other session 默认使用 Python 3.12",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.SESSION,
            scope_id="other",
        )
    )
    store.add(
        _record(
            "demo session 任务摘要",
            memory_type=MemoryType.SUMMARY,
            scope=MemoryScope.SESSION,
            scope_id="demo",
        )
    )

    retriever = MemoryRetriever(store)
    results = retriever.retrieve(
        "Python",
        scope_filter=MemoryScope.SESSION,
        scope_id="demo",
        type_filter=MemoryType.SEMANTIC,
    )

    assert [result.record for result in results] == [expected]


def test_memory_retriever_returns_empty_for_non_positive_top_k() -> None:
    retriever = MemoryRetriever(InMemoryStore())

    assert retriever.retrieve("Python", top_k=0) == []
