"""Deterministic, JSON-safe tools for AML money-flow graph investigation.

All graph directions follow ``sender (src) -> receiver (dst)``.  The service
does not make accusations; it exposes reproducible graph facts for an analyst
or an LLM tool caller.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

MAX_HOPS = 5
MAX_NODES = 500
MAX_RESULTS = 200

TOOL_DESCRIPTIONS = {
    "get_node": "Retrieve deterministic role, cluster, priority, and graph metrics for one gid.",
    "get_node_connections": "List direct incoming and/or outgoing counterparties aggregated by directed edge.",
    "trace_upstream": "Trace accounts that can send money to a gid, following edges backwards.",
    "trace_downstream": "Trace accounts that can receive money from a gid, following outgoing edges.",
    "find_common_recipient": "Find direct recipients receiving money from multiple supplied gids.",
    "get_cluster": "Retrieve nodes and deterministic aggregate facts for one Louvain cluster.",
    "search_nodes": "Filter nodes by role, cluster, priority, degree, and money-flow thresholds.",
    "compare_nodes": "Return comparable factual metrics for several gids without conclusions.",
    "get_role_explanation": "Show the actual deterministic rule and values used for a node role.",
    "get_priority_explanation": "Show the existing priority-score components and their raw metrics.",
    "get_subgraph_for_visualization": "Return directed nodes and edges for a bounded graph visualization.",
    "find_paths": "Find a bounded number of directed money-flow paths between two gids.",
}


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class GraphToolService:
    """Reusable tool service backed by the project's aggregated directed graph."""

    def __init__(self, graph: nx.DiGraph, transactions: pd.DataFrame, node_data: pd.DataFrame):
        required = {"gid", "role", "cluster_id", "priority_score", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank"}
        missing = required - set(node_data.columns)
        if missing:
            raise ValueError(f"node_data is missing required columns: {sorted(missing)}")
        self.graph = graph
        self.node_data = node_data.copy()
        self.nodes_by_gid = self.node_data.set_index("gid", drop=False)
        self.transactions = transactions.copy()
        self.edge_times = self._build_edge_times(self.transactions)

    @classmethod
    def from_project_data(cls, data_dir: str | Path = "parquet",
                          output_dir: str | Path | None = None) -> "GraphToolService":
        """Load the graph once and reuse valid exported analytics when available."""
        from starter import basic_features, build_graph, load, write_outputs

        data_dir = Path(data_dir).resolve()
        output_dir = Path(output_dir).resolve() if output_dir else data_dir.parent / "out"
        edges, nodes, transactions = load(data_dir)
        graph = build_graph(edges)
        roles_path = output_dir / "nodes_roles.csv"
        required_columns = {"gid", "role", "cluster_id", "priority_score", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank"}
        required_artifacts = ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "network.html")
        if roles_path.is_file() and all((output_dir / name).is_file() for name in required_artifacts):
            node_data = pd.read_csv(roles_path)
            if not required_columns.issubset(node_data.columns):
                node_data = pd.DataFrame()
        else:
            node_data = pd.DataFrame()
        if node_data.empty:
            metrics = basic_features(graph, nodes)
            node_data = write_outputs(metrics, output_dir, graph)
        return cls(graph, transactions, node_data)

    @staticmethod
    def _build_edge_times(transactions: pd.DataFrame) -> dict[tuple[Any, Any], dict[str, Any]]:
        if not {"src", "dst", "date"}.issubset(transactions.columns):
            return {}
        dates = transactions[["src", "dst", "date"]].copy()
        dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
        grouped = dates.groupby(["src", "dst"], dropna=False)["date"].agg(["min", "max"])
        return {
            key: {"first_transaction": row["min"], "last_transaction": row["max"]}
            for key, row in grouped.iterrows()
        }

    @staticmethod
    def _gid(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("gid must be an integer or numeric string")
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("gid must be an integer or numeric string") from exc
        return parsed

    @staticmethod
    def _bounded(value: int, name: str, maximum: int, minimum: int = 1) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
        return value

    def _found(self, gid: Any) -> tuple[int, bool]:
        parsed = self._gid(gid)
        return parsed, parsed in self.nodes_by_gid.index

    def _gids(self, gids: Iterable[Any]) -> list[int]:
        if isinstance(gids, (str, int, np.integer)):
            gids = [gids]
        try:
            return list(dict.fromkeys(self._gid(gid) for gid in gids))
        except TypeError as exc:
            raise ValueError("gids must be an iterable of gids") from exc

    def _node_summary(self, gid: Any) -> dict[str, Any]:
        row = self.nodes_by_gid.loc[gid]
        return _json_safe({
            "gid": row["gid"],
            "role": row["role"],
            "cluster_id": row["cluster_id"],
            "priority_score": row["priority_score"],
            "in_degree": row["in_deg"],
            "out_degree": row["out_deg"],
            "total_received": row["in_kzt"],
            "total_sent": row["out_kzt"],
        })

    def _edge(self, source: Any, target: Any) -> dict[str, Any]:
        data = self.graph[source][target]
        result = {
            "source": source,
            "target": target,
            "total_amount": data["sum_kzt"],
            "transaction_count": data["n_tx"],
        }
        result.update(self.edge_times.get((source, target), {}))
        return _json_safe(result)

    def get_node(self, gid: Any) -> dict[str, Any]:
        """Return factual role, cluster, priority, and available metrics for one gid."""
        parsed, found = self._found(gid)
        if not found:
            return {"gid": parsed, "found": False, "error": "gid not found"}
        row = self.nodes_by_gid.loc[parsed]
        metrics = {
            "in_degree": row["in_deg"], "out_degree": row["out_deg"],
            "weighted_in_degree": row["in_kzt"], "weighted_out_degree": row["out_kzt"],
            "total_received": row["in_kzt"], "total_sent": row["out_kzt"],
            "net_flow": row["in_kzt"] - row["out_kzt"],
            "distinct_senders": row["in_deg"], "distinct_receivers": row["out_deg"],
            "pagerank": row["pagerank"],
            "incoming_transaction_count": int(sum(data["n_tx"] for _, _, data in self.graph.in_edges(parsed, data=True))),
            "outgoing_transaction_count": int(sum(data["n_tx"] for _, _, data in self.graph.out_edges(parsed, data=True))),
        }
        return _json_safe({"gid": parsed, "found": True, "role": row["role"],
                           "cluster_id": row["cluster_id"], "priority_score": row["priority_score"],
                           "metrics": metrics})

    def get_node_connections(self, gid: Any, direction: str = "both", limit: int = 50,
                             sort_by: str = "amount") -> dict[str, Any]:
        """Return direct counterparties for incoming, outgoing, or both directed flows."""
        parsed, found = self._found(gid)
        if not found:
            return {"gid": parsed, "found": False, "error": "gid not found", "connections": []}
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")
        if sort_by not in {"amount", "transaction_count"}:
            raise ValueError("sort_by must be amount or transaction_count")
        limit = self._bounded(limit, "limit", MAX_RESULTS)
        connections: list[dict[str, Any]] = []
        if direction in {"incoming", "both"}:
            for source in self.graph.predecessors(parsed):
                edge = self._edge(source, parsed)
                connections.append({"gid": source, "direction": "incoming", **{k: v for k, v in edge.items() if k not in {"source", "target"}}})
        if direction in {"outgoing", "both"}:
            for target in self.graph.successors(parsed):
                edge = self._edge(parsed, target)
                connections.append({"gid": target, "direction": "outgoing", **{k: v for k, v in edge.items() if k not in {"source", "target"}}})
        key = "total_amount" if sort_by == "amount" else "transaction_count"
        connections.sort(key=lambda item: (-item[key], item["gid"], item["direction"]))
        return _json_safe({"gid": parsed, "found": True, "direction": direction, "connections": connections[:limit]})

    def _trace(self, gid: Any, hops: int, max_nodes: int, upstream: bool) -> dict[str, Any]:
        parsed, found = self._found(gid)
        direction = "upstream" if upstream else "downstream"
        if not found:
            return {"start_gid": parsed, "found": False, "error": "gid not found", "nodes": [], "edges": [], "paths": []}
        hops = self._bounded(hops, "hops", MAX_HOPS)
        max_nodes = self._bounded(max_nodes, "max_nodes", MAX_NODES)
        seen, queue = {parsed}, deque([(parsed, [parsed], 0)])
        paths, edges = [], {}
        while queue and len(seen) < max_nodes:
            current, path, depth = queue.popleft()
            if depth >= hops:
                continue
            neighbours: Iterable[Any] = self.graph.predecessors(current) if upstream else self.graph.successors(current)
            for neighbour in neighbours:
                if neighbour in seen or neighbour in path or len(seen) >= max_nodes:
                    continue
                next_path = [neighbour, *path] if upstream else [*path, neighbour]
                source, target = (neighbour, current) if upstream else (current, neighbour)
                seen.add(neighbour)
                edges[(source, target)] = self._edge(source, target)
                paths.append({"nodes": next_path, "hop_count": len(next_path) - 1})
                queue.append((neighbour, next_path, depth + 1))
        return _json_safe({"start_gid": parsed, "found": True, "direction": direction,
                           "requested_hops": hops, "nodes": [self._node_summary(node) for node in sorted(seen)],
                           "edges": list(edges.values()), "paths": paths, "truncated": bool(queue)})

    def trace_upstream(self, gid: Any, hops: int = 3, max_nodes: int = 100) -> dict[str, Any]:
        """Trace bounded directed paths that can eventually send money to gid."""
        return self._trace(gid, hops, max_nodes, upstream=True)

    def trace_downstream(self, gid: Any, hops: int = 3, max_nodes: int = 100) -> dict[str, Any]:
        """Trace bounded directed paths that can receive money from gid."""
        return self._trace(gid, hops, max_nodes, upstream=False)

    def find_common_recipient(self, gids: Iterable[Any], max_hops: int = 1,
                              min_sources: int = 2, limit: int = 20) -> dict[str, Any]:
        """Find direct recipients receiving from multiple supplied source gids."""
        if max_hops != 1:
            raise ValueError("only direct recipients are supported; max_hops must be 1")
        min_sources = self._bounded(min_sources, "min_sources", MAX_RESULTS)
        limit = self._bounded(limit, "limit", MAX_RESULTS)
        source_gids = self._gids(gids)
        if not source_gids:
            raise ValueError("gids must contain at least one gid")
        recipients: dict[Any, list[tuple[Any, dict[str, Any]]]] = {}
        for source in source_gids:
            if source not in self.graph:
                continue
            for target in self.graph.successors(source):
                recipients.setdefault(target, []).append((source, self._edge(source, target)))
        results = []
        for recipient, contributions in recipients.items():
            source_ids = sorted(source for source, _ in contributions)
            if len(source_ids) < min_sources:
                continue
            results.append({"recipient_gid": recipient, "source_gids": source_ids,
                            "source_count": len(source_ids),
                            "total_amount": sum(edge["total_amount"] for _, edge in contributions),
                            "transaction_count": sum(edge["transaction_count"] for _, edge in contributions),
                            "coverage_ratio": len(source_ids) / len(source_gids), "path_type": "direct"})
        results.sort(key=lambda item: (-item["source_count"], -item["total_amount"], -item["transaction_count"], item["recipient_gid"]))
        return _json_safe({"input_gids": source_gids, "results": results[:limit]})

    def get_cluster(self, cluster_id: Any, limit: int = 100, sort_by: str = "priority_score") -> dict[str, Any]:
        """Return a cluster's members and deterministic internal-flow summary."""
        cluster_id = self._gid(cluster_id)
        limit = self._bounded(limit, "limit", MAX_RESULTS)
        allowed = {"priority_score", "in_kzt", "out_kzt", "in_deg", "out_deg", "pagerank"}
        if sort_by not in allowed:
            raise ValueError(f"sort_by must be one of {sorted(allowed)}")
        members = self.node_data[self.node_data["cluster_id"].eq(cluster_id)]
        if members.empty:
            return {"cluster_id": cluster_id, "found": False, "error": "cluster not found", "nodes": []}
        gids = set(members["gid"])
        internal_amount = sum(data["sum_kzt"] for source, target, data in self.graph.edges(data=True) if source in gids and target in gids)
        ordered = members.sort_values([sort_by, "gid"], ascending=[False, True]).head(limit)
        return _json_safe({"cluster_id": cluster_id, "found": True, "size": len(members),
                           "nodes": [self._node_summary(gid) for gid in ordered["gid"]],
                           "summary": {"total_internal_amount": internal_amount,
                                       "role_counts": members["role"].value_counts().to_dict(),
                                       "seed_count": int(members.get("is_seed", pd.Series(False, index=members.index)).sum())}})

    def search_nodes(self, *, role: str | None = None, cluster_id: Any | None = None,
                     min_priority: float | None = None, max_priority: float | None = None,
                     min_in_degree: int | None = None, min_out_degree: int | None = None,
                     min_received: float | None = None, min_sent: float | None = None,
                     limit: int = 50, sort_by: str = "priority_score", descending: bool = True) -> dict[str, Any]:
        """Search composable, validated deterministic node filters."""
        allowed = {"gid", "role", "cluster_id", "priority_score", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank"}
        if sort_by not in allowed:
            raise ValueError(f"sort_by must be one of {sorted(allowed)}")
        if not isinstance(descending, bool):
            raise ValueError("descending must be boolean")
        limit = self._bounded(limit, "limit", MAX_RESULTS)
        frame = self.node_data
        filters: dict[str, Any] = {}
        if role is not None:
            frame, filters["role"] = frame[frame["role"].eq(role)], role
        if cluster_id is not None:
            cluster_id = self._gid(cluster_id); frame, filters["cluster_id"] = frame[frame["cluster_id"].eq(cluster_id)], cluster_id
        for argument, column, value in (("min_priority", "priority_score", min_priority), ("max_priority", "priority_score", max_priority),
                                        ("min_in_degree", "in_deg", min_in_degree), ("min_out_degree", "out_deg", min_out_degree),
                                        ("min_received", "in_kzt", min_received), ("min_sent", "out_kzt", min_sent)):
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f"{argument} must be a finite number")
                frame = frame[frame[column] <= value] if argument == "max_priority" else frame[frame[column] >= value]
                filters[argument] = value
        ordered = frame.sort_values([sort_by, "gid"], ascending=[not descending, True]).head(limit)
        return _json_safe({"filters": filters, "count": len(frame), "nodes": [self._node_summary(gid) for gid in ordered["gid"]]})

    def compare_nodes(self, gids: Iterable[Any]) -> dict[str, Any]:
        """Return comparable factual summaries for several gids."""
        requested = self._gids(gids)
        if not requested:
            raise ValueError("gids must contain at least one gid")
        return _json_safe({"gids": requested, "nodes": [self.get_node(gid) for gid in requested]})

    def get_role_explanation(self, gid: Any) -> dict[str, Any]:
        """Return evidence from the exact role conditions in starter.py."""
        parsed, found = self._found(gid)
        if not found:
            return {"gid": parsed, "found": False, "error": "gid not found"}
        row = self.nodes_by_gid.loc[parsed]
        role = row["role"]
        rules = {
            "terminal": [("out_degree", row["out_deg"], "= 0"), ("truncated_by_depth", row.get("truncated_by_depth", False), "= false")],
            "distributor": [("in_degree", row["in_deg"], "= 0"), ("out_degree", row["out_deg"], "> 0")],
            "coordinator": [("in_degree", row["in_deg"], ">= 2"), ("out_degree", row["out_deg"], ">= 2")],
            "transit": [("in_degree", row["in_deg"], "> 0"), ("out_degree", row["out_deg"], "> 0"), ("pass_through", row.get("pass_through"), ">= 0.8")],
            "consolidator": [("in_degree", row["in_deg"], "> 0"), ("out_degree", row["out_deg"], "> 0")],
            "peripheral": [("role_rule", "no earlier rule matched", "peripheral")],
        }
        return _json_safe({"gid": parsed, "found": True, "role": role,
                           "role_evidence": [{"metric": metric, "value": value, "condition": condition}
                                             for metric, value, condition in rules[role]]})

    def get_priority_explanation(self, gid: Any) -> dict[str, Any]:
        """Explain the existing 0.5 role + 0.3 flow + 0.2 PageRank priority formula."""
        parsed, found = self._found(gid)
        if not found:
            return {"gid": parsed, "found": False, "error": "gid not found"}
        row = self.nodes_by_gid.loc[parsed]
        total = self.node_data["in_kzt"] + self.node_data["out_kzt"]
        max_log = np.log1p(total).max()
        max_pr = self.node_data["pagerank"].max()
        flow_score = np.log1p(row["in_kzt"] + row["out_kzt"]) / max_log if max_log > 0 else 0.0
        centrality_score = row["pagerank"] / max_pr if max_pr > 0 else 0.0
        return _json_safe({"gid": parsed, "found": True, "priority_score": row["priority_score"],
                           "formula": "0.5 * role_score + 0.3 * log_volume_score + 0.2 * pagerank_score",
                           "components": {"role_score": row.get("role_score", None), "log_volume_score": flow_score,
                                          "pagerank_score": centrality_score},
                           "raw_metrics": {"total_received": row["in_kzt"], "total_sent": row["out_kzt"],
                                           "pagerank": row["pagerank"]}})

    def get_subgraph_for_visualization(self, gids: Iterable[Any] | None = None, center_gid: Any | None = None,
                                       hops: int = 1, cluster_id: Any | None = None, max_nodes: int = 100) -> dict[str, Any]:
        """Return bounded directed graph data for a renderer, without rendering it."""
        hops, max_nodes = self._bounded(hops, "hops", MAX_HOPS), self._bounded(max_nodes, "max_nodes", MAX_NODES)
        selected: set[int] = set()
        if gids is not None:
            selected.update(gid for gid in self._gids(gids) if gid in self.nodes_by_gid.index)
        focus = None
        if center_gid is not None:
            focus, found = self._found(center_gid)
            if not found:
                return {"focus_node": focus, "found": False, "error": "gid not found", "nodes": [], "edges": [], "highlight_nodes": []}
            selected.add(focus)
            queue = deque([(focus, 0)])
            while queue and len(selected) < max_nodes:
                current, depth = queue.popleft()
                if depth >= hops:
                    continue
                for neighbour in list(self.graph.predecessors(current)) + list(self.graph.successors(current)):
                    if neighbour not in selected and len(selected) < max_nodes:
                        selected.add(neighbour); queue.append((neighbour, depth + 1))
        if cluster_id is not None:
            cluster_id = self._gid(cluster_id)
            selected.update(self.node_data[self.node_data["cluster_id"].eq(cluster_id)].nlargest(max_nodes, "priority_score")["gid"])
        if not selected:
            raise ValueError("provide gids, center_gid, or cluster_id")
        selected = set(sorted(selected, key=lambda node: (-self.nodes_by_gid.loc[node, "priority_score"], node))[:max_nodes])
        edges = []
        for source, target in self.graph.edges():
            if source in selected and target in selected:
                edge = self._edge(source, target)
                edge["amount"] = edge.pop("total_amount")
                edges.append(edge)
        nodes = [{"id": node, **{k: v for k, v in self._node_summary(node).items() if k != "gid"}} for node in sorted(selected)]
        return _json_safe({"found": True, "focus_node": focus, "nodes": nodes, "edges": edges,
                           "highlight_nodes": sorted(selected)})

    def find_paths(self, source_gid: Any, target_gid: Any, max_hops: int = 5, max_paths: int = 20) -> dict[str, Any]:
        """Find a bounded number of simple directed money-flow paths."""
        source, source_found = self._found(source_gid)
        target, target_found = self._found(target_gid)
        if not source_found or not target_found:
            return {"source_gid": source, "target_gid": target, "found": False, "error": "gid not found", "paths": []}
        max_hops, max_paths = self._bounded(max_hops, "max_hops", MAX_HOPS), self._bounded(max_paths, "max_paths", MAX_RESULTS)
        paths = []
        try:
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=max_hops):
                paths.append({"nodes": path, "hop_count": len(path) - 1,
                              "edges": [self._edge(left, right) for left, right in zip(path, path[1:])]})
                if len(paths) >= max_paths:
                    break
        except nx.NodeNotFound:
            pass
        return _json_safe({"source_gid": source, "target_gid": target, "found": True, "paths": paths})
