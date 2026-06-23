from __future__ import annotations

from typing import Any

from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall


class FakeProvider:
    """Deterministic provider for tests and demos.

    The fake provider simulates a minimal tool-calling flow:

    1. First call: request list_files and read_file.
    2. Second call: return final answer after observing tool results.
    """

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        tool_observations = [message for message in messages if message.role == "tool"]

        if not tool_observations:
            return ProviderResponse(
                content="I need to inspect the workspace first.",
                tool_calls=[
                    ToolCall(
                        id="call_list_files",
                        name="list_files",
                        arguments={},
                    ),
                    ToolCall(
                        id="call_read_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
            )

        observed_tool_names = ", ".join(
            message.name for message in tool_observations if message.name is not None
        )

        return ProviderResponse(
            final_answer=(
                "I inspected the workspace using these tools: "
                f"{observed_tool_names}. The README was requested successfully."
            )
        )