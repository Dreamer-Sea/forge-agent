from __future__ import annotations

from typing import Any

from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall


class FakeProvider:
    """Deterministic provider for tests and demos.

    The fake provider simulates minimal tool-calling flows:

    1. For regular workspace inspection tasks:
       - First call: request list_files and read_file.
       - Second call: return final answer after observing tool results.

    2. For echo tasks:
       - First call: request echo_text.
       - Second call: return final answer after observing tool results.
    """

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        tool_observations = [message for message in messages if message.role == "tool"]

        if tool_observations:
            observed_tool_names = ", ".join(
                message.name for message in tool_observations if message.name is not None
            )

            return ProviderResponse(
                final_answer=(
                    "I inspected the workspace using these tools: "
                    f"{observed_tool_names}. The README was requested successfully."
                )
            )

        latest_user_input = self._latest_user_input(messages)

        if "echo" in latest_user_input.lower():
            return ProviderResponse(
                content="I need to echo text.",
                tool_calls=[
                    ToolCall(
                        id="call_echo_text",
                        name="echo_text",
                        arguments={"text": self._extract_echo_text(latest_user_input)},
                    )
                ],
            )

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

    def _latest_user_input(self, messages: list[ModelMessage]) -> str:
        user_messages = [message for message in messages if message.role == "user"]

        if not user_messages:
            return ""

        return user_messages[-1].content

    def _extract_echo_text(self, user_input: str) -> str:
        normalized = user_input.strip()

        if normalized.lower().startswith("echo "):
            return normalized[5:].strip() or normalized

        return normalized