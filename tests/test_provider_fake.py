from __future__ import annotations

from forge_agent.providers.base import ModelMessage
from forge_agent.providers.fake import FakeProvider


def test_fake_provider_returns_tool_calls_without_observation() -> None:
    provider = FakeProvider()

    response = provider.complete(
        messages=[ModelMessage(role="user", content="list files and read README")],
        tools=[],
    )

    assert response.final_answer is None
    assert response.has_tool_calls
    assert [tool_call.name for tool_call in response.tool_calls] == [
        "list_files",
        "read_file",
    ]


def test_fake_provider_returns_final_answer_after_observation() -> None:
    provider = FakeProvider()

    response = provider.complete(
        messages=[
            ModelMessage(role="user", content="list files and read README"),
            ModelMessage(role="tool", name="list_files", content='["README.md"]'),
            ModelMessage(role="tool", name="read_file", content="# forge-agent"),
        ],
        tools=[],
    )

    assert response.tool_calls == []
    assert response.final_answer is not None
    assert "list_files" in response.final_answer
    assert "read_file" in response.final_answer