# Agent graph tools

`agent_tools.GraphToolService` is a provider-independent, deterministic tool
layer for the AML transaction graph. Create it once with
`GraphToolService.from_project_data("parquet")` and reuse the instance across
agent calls or a cached Streamlit resource.

The graph direction is always `src -> dst`: an incoming connection to `X` is
`other -> X`; upstream traversal follows predecessors, and downstream
traversal follows successors. Amounts and transaction counts are aggregated
per directed pair.

## Tools

| Tool | Purpose | Key parameters |
| --- | --- | --- |
| `get_node` | Facts and metrics for one account | `gid` |
| `get_node_connections` | Direct counterparties | `gid`, `direction`, `limit`, `sort_by` |
| `trace_upstream` | Bounded predecessor traversal | `gid`, `hops`, `max_nodes` |
| `trace_downstream` | Bounded successor traversal | `gid`, `hops`, `max_nodes` |
| `find_common_recipient` | Direct common recipients of supplied gids | `gids`, `min_sources`, `limit` |
| `get_cluster` | Cluster members and internal flow | `cluster_id`, `limit`, `sort_by` |
| `search_nodes` | Composable deterministic filters | role/cluster/priority/degree/amount filters |
| `compare_nodes` | Comparable evidence for several accounts | `gids` |
| `get_role_explanation` | Actual role-rule conditions and values | `gid` |
| `get_priority_explanation` | Existing priority formula components | `gid` |
| `get_subgraph_for_visualization` | Bounded nodes/edges for a renderer | gids, center, cluster, hops |
| `find_paths` | Bounded directed simple paths | source, target, max hops/paths |

All outputs consist only of JSON-serializable dictionaries and lists. Normal
unknown-gid queries return `found: false` rather than throwing. Traversals are
limited to five hops and 500 nodes; result lists are limited to 200.

## Examples

```python
from agent_tools import GraphToolService

service = GraphToolService.from_project_data("parquet")
node = service.get_node("100000002224132100")
incoming = service.get_node_connections(node["gid"], direction="incoming")
flow = service.trace_downstream(node["gid"], hops=2)
graph_data = service.get_subgraph_for_visualization(center_gid=node["gid"], hops=2)
```

Roles, community IDs, PageRank, and priority scores come from the existing
`starter.py` calculation. The priority explanation exposes its existing
formula: `0.5 * role_score + 0.3 * log_volume_score + 0.2 * pagerank_score`.
