from __future__ import annotations

from forge_agent.tools.echo import EchoTextTool


def test_echo_text_tool_returns_text() -> None:
    tool = EchoTextTool()

    result = tool.execute(
        arguments={"text": "hello agent"},
        tool_call_id="call_echo",
    )

    assert result.success is True
    assert result.tool_name == "echo_text"
    assert result.tool_call_id == "call_echo"
    assert result.payload["text"] == "hello agent"


def test_echo_text_tool_validates_arguments() -> None:
    tool = EchoTextTool()

    result = tool.execute(arguments={})

    assert result.success is False
    assert result.error_code == "validation_error"
