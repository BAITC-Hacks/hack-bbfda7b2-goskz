"""OpenAI function-calling agent for deterministic AML graph analysis."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from agent_tools import GraphToolService
from prompts import AML_ANALYST_SYSTEM_PROMPT
from tool_registry import ToolRegistry

MAX_TOOL_ROUNDS = 6
MAX_TOOL_CALLS_PER_REQUEST = 12
MAX_HISTORY_MESSAGES = 12
LOGGER = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured result consumable by a chat UI and graph renderer."""

    answer: str
    tool_calls: list[dict[str, Any]]
    referenced_gids: list[int]
    visualization: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AMLAnalystAgent:
    """A bounded tool-calling loop; graph facts can only enter via registered tools."""

    def __init__(self, tool_service: GraphToolService, llm_client: Any, model: str | None = None):
        self.registry = ToolRegistry(tool_service)
        self.llm_client = llm_client
        self.model = model or os.getenv("AML_AGENT_MODEL", "gpt-4.1-mini")

    @classmethod
    def from_environment(cls, tool_service: GraphToolService) -> "AMLAnalystAgent":
        """Create the OpenAI client using OPENAI_API_KEY and AML_AGENT_MODEL."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use the analyst assistant") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before starting the analyst assistant")
        return cls(tool_service, OpenAI())

    @staticmethod
    def _value(item: Any, name: str, default: Any = None) -> Any:
        return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)

    @classmethod
    def _tool_calls(cls, response: Any) -> list[Any]:
        return [item for item in cls._value(response, "output", []) if cls._value(item, "type") == "function_call"]

    @staticmethod
    def _history_items(history: Iterable[dict[str, str]] | None) -> list[dict[str, str]]:
        items = []
        for message in list(history or [])[-MAX_HISTORY_MESSAGES:]:
            role, content = message.get("role"), message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                items.append({"role": role, "content": content})
        return items

    @staticmethod
    def _collect_gids(value: Any) -> set[int]:
        found: set[int] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"gid", "source", "target", "recipient_gid", "start_gid", "focus_node"} and isinstance(item, int):
                    found.add(item)
                elif key in {"source_gids", "highlight_nodes"} and isinstance(item, list):
                    found.update(gid for gid in item if isinstance(gid, int))
                else:
                    found.update(AMLAnalystAgent._collect_gids(item))
        elif isinstance(value, list):
            for item in value:
                found.update(AMLAnalystAgent._collect_gids(item))
        return found


    @staticmethod
    def _visualization(result: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(result, dict) or not result.get("found", True):
            return None
        if result.get("results"):
            top = result["results"][0]
            if isinstance(top, dict) and isinstance(top.get("recipient_gid"), int):
                return {"focus_node": top["recipient_gid"],
                        "highlight_nodes": [top["recipient_gid"], *top.get("source_gids", [])],
                        "highlight_edges": []}
        if result.get("connections") and isinstance(result.get("gid"), int):
            gid, direction = result["gid"], result.get("direction")
            edges = []
            for connection in result["connections"]:
                if not isinstance(connection, dict) or not isinstance(connection.get("gid"), int):
                    continue
                source, target = (connection["gid"], gid) if connection.get("direction") == "incoming" else (gid, connection["gid"])
                edges.append({"source": source, "target": target, "amount": connection.get("total_amount"),
                              "transaction_count": connection.get("transaction_count")})
            return {"focus_node": gid, "highlight_nodes": sorted({gid, *(edge["source"] for edge in edges), *(edge["target"] for edge in edges)}),
                    "highlight_edges": edges}
        nodes, edges = result.get("nodes"), result.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return None
        highlights = [node.get("gid", node.get("id")) for node in nodes if isinstance(node, dict)]
        highlights = [gid for gid in highlights if isinstance(gid, int)]
        if not highlights:
            return None
        return {"focus_node": result.get("focus_node", result.get("start_gid")),
                "highlight_nodes": highlights, "highlight_edges": edges}

    def ask(self, question: str, conversation_history: list[dict[str, str]] | None = None) -> AgentResponse:
        """Ask the LLM, execute requested graph tools, and return answer plus renderer metadata."""
        if not isinstance(question, str) or not question.strip():
            return AgentResponse("Please enter an analyst question.", [], [], None)
        input_items: list[Any] = self._history_items(conversation_history)
        input_items.append({"role": "user", "content": question.strip()})
        calls: list[dict[str, Any]] = []
        referenced: set[int] = set()
        visualization = None

        for round_number in range(MAX_TOOL_ROUNDS):
            response = self.llm_client.responses.create(
                model=self.model, instructions=AML_ANALYST_SYSTEM_PROMPT,
                input=input_items, tools=self.registry.definitions(),
            )
            tool_calls = self._tool_calls(response)
            if not tool_calls:
                if not calls:
                    return AgentResponse("Please specify a gid, cluster, or graph question so I can retrieve deterministic evidence.", [], [], None)
                answer = self._value(response, "output_text", "") or "I could not produce an analyst response."
                return AgentResponse(answer, calls, sorted(referenced), visualization)
            if len(calls) + len(tool_calls) > MAX_TOOL_CALLS_PER_REQUEST:
                return AgentResponse("The request exceeded the safe tool-call limit. Please narrow the question.", calls, sorted(referenced), visualization)

            input_items.extend(self._value(response, "output", []))
            for tool_call in tool_calls:
                name = self._value(tool_call, "name")
                call_id = self._value(tool_call, "call_id")
                raw_arguments = self._value(tool_call, "arguments", "{}")
                started = time.perf_counter()
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    result = self.registry.execute(name, arguments)
                    calls.append({"name": name, "arguments": arguments, "ok": True})
                    referenced.update(self._collect_gids(result))
                    visualization = self._visualization(result) or visualization
                    LOGGER.info("AML tool completed name=%s duration_ms=%d", name, (time.perf_counter() - started) * 1000)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    result = {"error": "tool request could not be completed", "detail": str(exc)}
                    calls.append({"name": name, "arguments": raw_arguments, "ok": False})
                    LOGGER.warning("AML tool failed name=%s error=%s", name, exc)
                input_items.append({"type": "function_call_output", "call_id": call_id,
                                    "output": json.dumps(result, ensure_ascii=False)})
        return AgentResponse("I reached the safe tool-call limit before completing the analysis.", calls, sorted(referenced), visualization)
