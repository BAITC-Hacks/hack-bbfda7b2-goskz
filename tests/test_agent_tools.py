import json
import unittest

import networkx as nx
import pandas as pd

from agent_tools import GraphToolService


def make_service(edges):
    graph = nx.DiGraph()
    for source, target, amount, count in edges:
        graph.add_edge(source, target, sum_kzt=amount, n_tx=count)
    gids = sorted(set(graph.nodes) | {5, 10, 11})
    rows = []
    for gid in gids:
        incoming = graph.in_degree(gid, weight="sum_kzt") if gid in graph else 0
        outgoing = graph.out_degree(gid, weight="sum_kzt") if gid in graph else 0
        rows.append({
            "gid": gid, "role": "consolidator", "cluster_id": 1,
            "priority_score": gid / 100, "role_score": 1.0,
            "in_deg": graph.in_degree(gid) if gid in graph else 0,
            "out_deg": graph.out_degree(gid) if gid in graph else 0,
            "in_kzt": incoming, "out_kzt": outgoing, "pagerank": gid / 1000,
            "pass_through": outgoing / incoming if incoming else None,
            "is_seed": False, "truncated_by_depth": False,
        })
    transactions = pd.DataFrame([
        {"src": source, "dst": target, "sum_kzt": amount, "date": "2024-01-01"}
        for source, target, amount, _ in edges
    ])
    return GraphToolService(graph, transactions, pd.DataFrame(rows))


class GraphToolServiceTests(unittest.TestCase):
    def test_get_node_and_json_serialization(self):
        service = make_service([(1, 3, 300, 2), (2, 3, 100, 1), (3, 4, 75, 1)])
        result = service.get_node("3")
        self.assertTrue(result["found"])
        self.assertEqual(result["metrics"]["total_received"], 400)
        self.assertFalse(service.get_node(999)["found"])
        json.dumps(result)

    def test_connections_and_aggregation(self):
        service = make_service([(1, 3, 350, 3), (2, 3, 100, 1), (3, 4, 75, 1), (3, 5, 50, 1)])
        incoming = service.get_node_connections(3, direction="incoming")
        outgoing = service.get_node_connections(3, direction="outgoing")
        self.assertEqual({item["gid"] for item in incoming["connections"]}, {1, 2})
        self.assertEqual(incoming["connections"][0]["total_amount"], 350)
        self.assertEqual({item["gid"] for item in outgoing["connections"]}, {4, 5})

    def test_directed_traversal_and_cycles(self):
        service = make_service([(1, 2, 1, 1), (2, 3, 1, 1), (3, 4, 1, 1), (3, 1, 1, 1)])
        upstream = service.trace_upstream(3, hops=2)
        downstream = service.trace_downstream(2, hops=2)
        self.assertTrue({1, 2}.issubset({node["gid"] for node in upstream["nodes"]}))
        self.assertTrue({3, 4}.issubset({node["gid"] for node in downstream["nodes"]}))
        self.assertLessEqual(len(upstream["nodes"]), 4)

    def test_common_recipient(self):
        service = make_service([(1, 10, 10, 1), (2, 10, 20, 1), (3, 10, 30, 1), (4, 11, 40, 1)])
        result = service.find_common_recipient([1, 2, 3, 4])
        self.assertEqual(result["results"][0]["recipient_gid"], 10)
        self.assertEqual(result["results"][0]["source_count"], 3)

    def test_all_tool_outputs_are_json_safe(self):
        service = make_service([(1, 2, 10, 1), (2, 3, 20, 2), (3, 4, 30, 1)])
        results = [
            service.get_node(2), service.get_node_connections(2),
            service.trace_upstream(3), service.trace_downstream(2),
            service.find_common_recipient([1, 2, 3]),
            service.get_cluster(1), service.search_nodes(min_received=1),
            service.compare_nodes([1, 2]), service.get_role_explanation(2),
            service.get_priority_explanation(2),
            service.get_subgraph_for_visualization(center_gid=2),
            service.find_paths(1, 4),
        ]
        for result in results:
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
