"""Allowlisted function schemas and dispatch for the AML analyst agent."""

from __future__ import annotations

from typing import Any, Callable

from agent_tools import GraphToolService, TOOL_DESCRIPTIONS


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


TOOL_SCHEMAS = {
    "get_node": _schema({"gid": {"type": ["integer", "string"]}}, ["gid"]),
    "get_node_connections": _schema({"gid": {"type": ["integer", "string"]}, "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}, "sort_by": {"type": "string", "enum": ["amount", "transaction_count"]}}, ["gid"]),
    "trace_upstream": _schema({"gid": {"type": ["integer", "string"]}, "hops": {"type": "integer", "minimum": 1, "maximum": 5}, "max_nodes": {"type": "integer", "minimum": 1, "maximum": 500}}, ["gid"]),
    "trace_downstream": _schema({"gid": {"type": ["integer", "string"]}, "hops": {"type": "integer", "minimum": 1, "maximum": 5}, "max_nodes": {"type": "integer", "minimum": 1, "maximum": 500}}, ["gid"]),
    "find_common_recipient": _schema({"gids": {"type": "array", "items": {"type": ["integer", "string"]}, "minItems": 1, "maxItems": 200}, "max_hops": {"type": "integer", "enum": [1]}, "min_sources": {"type": "integer", "minimum": 1, "maximum": 200}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, ["gids"]),
    "get_cluster": _schema({"cluster_id": {"type": ["integer", "string"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}, "sort_by": {"type": "string", "enum": ["priority_score", "in_kzt", "out_kzt", "in_deg", "out_deg", "pagerank"]}}, ["cluster_id"]),
    "search_nodes": _schema({"role": {"type": "string"}, "cluster_id": {"type": ["integer", "string"]}, "min_priority": {"type": "number"}, "max_priority": {"type": "number"}, "min_in_degree": {"type": "integer", "minimum": 0}, "min_out_degree": {"type": "integer", "minimum": 0}, "min_received": {"type": "number", "minimum": 0}, "min_sent": {"type": "number", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}, "sort_by": {"type": "string", "enum": ["gid", "role", "cluster_id", "priority_score", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank"]}, "descending": {"type": "boolean"}}, []),
    "compare_nodes": _schema({"gids": {"type": "array", "items": {"type": ["integer", "string"]}, "minItems": 1, "maxItems": 200}}, ["gids"]),
    "get_role_explanation": _schema({"gid": {"type": ["integer", "string"]}}, ["gid"]),
    "get_priority_explanation": _schema({"gid": {"type": ["integer", "string"]}}, ["gid"]),
    "get_subgraph_for_visualization": _schema({"gids": {"type": "array", "items": {"type": ["integer", "string"]}, "maxItems": 200}, "center_gid": {"type": ["integer", "string"]}, "hops": {"type": "integer", "minimum": 1, "maximum": 5}, "cluster_id": {"type": ["integer", "string"]}, "max_nodes": {"type": "integer", "minimum": 1, "maximum": 500}}, []),
    "find_paths": _schema({"source_gid": {"type": ["integer", "string"]}, "target_gid": {"type": ["integer", "string"]}, "max_hops": {"type": "integer", "minimum": 1, "maximum": 5}, "max_paths": {"type": "integer", "minimum": 1, "maximum": 200}}, ["source_gid", "target_gid"]),
}


class ToolRegistry:
    """A closed registry that prevents a model from invoking arbitrary code."""

    def __init__(self, service: GraphToolService):
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            name: getattr(service, name) for name in TOOL_SCHEMAS
        }

    def definitions(self) -> list[dict[str, Any]]:
        return [{"type": "function", "name": name, "description": TOOL_DESCRIPTIONS[name],
                 "parameters": schema, "strict": False}
                for name, schema in TOOL_SCHEMAS.items()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run only an allowlisted service method after lightweight validation."""
        if name not in self._handlers:
            raise ValueError(f"tool {name!r} is not registered")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        missing = set(TOOL_SCHEMAS[name]["required"]) - set(arguments)
        if missing:
            raise ValueError(f"missing required arguments for {name}: {sorted(missing)}")
        allowed = set(TOOL_SCHEMAS[name]["properties"])
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(f"unexpected arguments for {name}: {sorted(unknown)}")
        return self._handlers[name](**arguments)
