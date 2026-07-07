from __future__ import annotations

import json
from typing import Any

from forge_agent.providers.base import ModelMessage, ProviderResponse, ToolCall


class FakeProvider:
    """Deterministic provider for tests and demos.

    The fake provider simulates minimal tool-calling flows:

    1. For memory recall demos:
       - If long-term memory context answers a Python-version question, return it.
       - If the user explicitly asks to remember something, acknowledge it.
    2. For knowledge base tasks:
       - First call: request search_knowledge_base.
       - Second call: return a grounded answer from the tool observation.
    3. For echo tasks:
       - First call: request echo_text.
       - Second call: return final answer after observing tool results.
    4. For regular workspace inspection tasks:
       - First call: request list_files and read_file.
       - Second call: return final answer after observing tool results.
    """

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        tool_observations = [message for message in messages if message.role == "tool"]
        if tool_observations:
            rag_observations = [
                message
                for message in tool_observations
                if message.name == "search_knowledge_base"
            ]
            if rag_observations:
                return ProviderResponse(
                    final_answer=self._answer_from_knowledge_base(rag_observations[-1])
                )

            observed_tool_names = ", ".join(
                message.name for message in tool_observations if message.name is not None
            )
            return ProviderResponse(
                final_answer=(
                    "I inspected the workspace using these tools: "
                    f"{observed_tool_names}.\n"
                    "The README was requested successfully."
                )
            )

        latest_user_input = self._latest_user_input(messages)

        memory_answer = self._answer_from_memory_context(messages, latest_user_input)
        if memory_answer is not None:
            return ProviderResponse(final_answer=memory_answer)

        if self._is_memory_write_request(latest_user_input):
            return ProviderResponse(final_answer="Memory write request accepted.")

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

        if self._should_search_knowledge_base(latest_user_input, tools):
            return ProviderResponse(
                content="I need to search the local knowledge base.",
                tool_calls=[
                    ToolCall(
                        id="call_search_knowledge_base",
                        name="search_knowledge_base",
                        arguments={
                            "query": self._extract_knowledge_query(latest_user_input),
                            "top_k": 3,
                        },
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

    def _answer_from_memory_context(
        self,
        messages: list[ModelMessage],
        user_input: str,
    ) -> str | None:
        lowered = user_input.casefold()
        asks_python_version = (
            "python" in lowered
            and (
                "版本" in lowered
                or "version" in lowered
                or "默认使用" in lowered
                or "default" in lowered
            )
        )
        if not asks_python_version:
            return None

        memory_context = "\n".join(
            message.content
            for message in messages
            if message.role == "system" and "## Long-term Memory" in message.content
        )
        if not memory_context:
            return None

        for line in memory_context.splitlines():
            if "Python 3.13" in line and "uv" in line:
                return "根据长期记忆，项目默认使用 Python 3.13 和 uv。"
            if "Python 3.13" in line:
                return "根据长期记忆，项目默认使用 Python 3.13。"

        return None

    def _is_memory_write_request(self, user_input: str) -> bool:
        normalized = user_input.strip().casefold()
        return normalized.startswith(
            (
                "记住",
                "请记住",
                "记一下",
                "保存",
                "remember",
                "please remember",
            )
        )

    def _extract_echo_text(self, user_input: str) -> str:
        normalized = user_input.strip()
        if normalized.lower().startswith("echo "):
            return normalized[5:].strip() or normalized
        return normalized

    def _should_search_knowledge_base(
        self,
        user_input: str,
        tools: list[dict[str, Any]],
    ) -> bool:
        if not self._has_tool(tools, "search_knowledge_base"):
            return False

        lowered = user_input.lower()
        return any(
            keyword in lowered
            for keyword in [
                "知识库",
                "knowledge base",
                "agent runtime",
                "core modules",
                "核心模块",
                "toolregistry",
                "permission system",
                "trace event",
            ]
        )

    def _extract_knowledge_query(self, user_input: str) -> str:
        normalized = user_input.strip()
        prefixes = [
            "根据知识库回答：",
            "根据知识库回答:",
            "knowledge base:",
            "Knowledge base:",
        ]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return normalized[len(prefix) :].strip() or normalized
        return normalized

    def _answer_from_knowledge_base(self, observation: ModelMessage) -> str:
        try:
            tool_result = json.loads(observation.content)
        except json.JSONDecodeError:
            return "Knowledge base search returned an invalid tool result."

        if not tool_result.get("success", False):
            error_message = tool_result.get("error_message") or "unknown error"
            return f"Knowledge base search failed: {error_message}"

        payload = tool_result.get("payload", {})
        context = payload.get("context", "")
        citations = payload.get("citations", [])

        if not context:
            return "I could not find relevant information in the knowledge base."

        citation_lines = [
            f"- {citation['citation']}"
            for citation in citations
            if isinstance(citation, dict) and "citation" in citation
        ]
        citation_text = "\n".join(citation_lines)

        if citation_text:
            return (
                "Based on the knowledge base, here is the grounded context:\n\n"
                f"{context}\n\n"
                f"Citations:\n{citation_text}"
            )

        return f"Based on the knowledge base, here is the grounded context:\n\n{context}"

    @staticmethod
    def _has_tool(tools: list[dict[str, Any]], tool_name: str) -> bool:
        return any(tool.get("name") == tool_name for tool in tools)
