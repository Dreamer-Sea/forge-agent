from __future__ import annotations

from forge_agent.memory import MemoryScope, MemoryType, MemoryWritePolicy


def test_memory_write_policy_accepts_explicit_remember_request() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_from_user_input(
        "记住：我的项目默认使用 Python 3.13 和 uv",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    assert decision.should_write is True
    assert decision.memory_type == MemoryType.SEMANTIC
    assert decision.scope == MemoryScope.PROJECT
    assert decision.scope_id == "forge-agent"
    assert decision.content == "我的项目默认使用 Python 3.13 和 uv"
    assert decision.reason == "explicit remember request"


def test_memory_write_policy_skips_temporary_user_input() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_from_user_input("帮我看一下这个测试为什么失败")

    assert decision.should_write is False
    assert decision.reason == "user input does not contain an explicit memory request"


def test_memory_write_policy_skips_sensitive_content_even_when_explicit() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_from_user_input("记住：我的 API key 是 abc123")

    assert decision.should_write is False
    assert decision.reason == "content appears to contain sensitive information"


def test_memory_write_policy_accepts_successful_agent_summary() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_from_agent_result(
        "完成 JsonlMemoryStore，并验证 JSONL reload、scope isolation 和 duplicate handling。",
        succeeded=True,
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    assert decision.should_write is True
    assert decision.memory_type == MemoryType.SUMMARY
    assert decision.scope_id == "forge-agent"
    assert decision.source == "agent_result"


def test_memory_write_policy_skips_failed_agent_summary() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_from_agent_result(
        "本次任务未完成",
        succeeded=False,
    )

    assert decision.should_write is False
    assert decision.reason == "agent result was not successful"


def test_memory_write_policy_accepts_verification_failed_memory() -> None:
    policy = MemoryWritePolicy()

    decision = policy.decide_failure_memory(
        "JsonlMemoryStore 的 list 方法名遮蔽了内置 list 类型，导致 mypy valid-type 错误",
        failure_type="verification_failed",
        scope=MemoryScope.PROJECT,
        scope_id="forge-agent",
    )

    assert decision.should_write is True
    assert decision.memory_type == MemoryType.EPISODIC
    assert decision.scope == MemoryScope.PROJECT
    assert decision.scope_id == "forge-agent"
    assert decision.content is not None
    assert decision.content.startswith("verification_failed:")
    assert decision.importance == 0.9
