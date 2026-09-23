#!/usr/bin/env python3
"""
Стартовый код кейса «Граф денег» — HackAlem AI.

Что он делает:
  1. грузит три parquet-файла и проверяет их консистентность;
  2. собирает направленный взвешенный граф;
  3. считает метрики узлов и назначает объяснимые поведенческие роли;
  4. строит сообщества и приоритетный список;
  5. пишет три выгрузки в схеме из ТЗ.

Запуск:
    python starter.py --data ./parquet --out ./out
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]

# Distribution-informed cutoffs for this supplied graph. out_deg p95 = 5;
# incoming amount median = 50,000 KZT. A pass-through of .8 is above the
# 80th percentile among customers with observed incoming funds.
DISTRIBUTOR_RECIPIENTS = 5
MIN_INCOMING_KZT = 50_000
TRANSIT_PASS_THROUGH = 0.8
COORDINATOR_MIN_IN_DEG = 2
COORDINATOR_MIN_OUT_DEG = 2
MAX_EVIDENCE_CHARS = 200
ANALYSIS_VERSION = "2"


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    data_dir = Path(data_dir)
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def validate_case_frames(edges: pd.DataFrame, nodes: pd.DataFrame,
                         tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and normalize the three fixed-schema case tables."""
    required = {
        "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
        "nodes": {"gid", "depth", "is_seed"},
        "transactions": {"src", "dst", "sum_kzt", "date"},
    }
    frames = {"edges": edges.copy(), "nodes": nodes.copy(), "transactions": tx.copy()}
    for name, columns in required.items():
        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} data is missing required columns: {', '.join(sorted(missing))}")
    edges, nodes, tx = frames["edges"], frames["nodes"], frames["transactions"]
    if edges.empty or nodes.empty or tx.empty:
        raise ValueError("edges, nodes, and transactions must all contain data")
    for frame, columns, label in (
        (nodes, ("gid",), "nodes.gid"),
        (edges, ("src", "dst"), "edges identifiers"),
        (tx, ("src", "dst"), "transactions identifiers"),
    ):
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or (numeric % 1 != 0).any():
                raise ValueError(f"{label} must contain integer GIDs")
            try:
                frame[column] = numeric.astype("int64")
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{label} contains a GID outside the int64 range") from exc
    if nodes["gid"].isna().any() or nodes["gid"].duplicated().any():
        raise ValueError("nodes must contain one non-empty row per unique gid")
    if nodes[["depth", "is_seed"]].isna().any().any():
        raise ValueError("nodes contains missing values in required fields")
    if edges[["src", "dst", "sum_kzt", "n_tx", "depth"]].isna().any().any():
        raise ValueError("edges contains missing values in required fields")
    if tx[["src", "dst", "sum_kzt", "date"]].isna().any().any():
        raise ValueError("transactions contains missing values in required fields")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges must contain one aggregated row per directed src/dst pair")
    if not pd.api.types.is_numeric_dtype(edges["sum_kzt"]) or not pd.api.types.is_numeric_dtype(tx["sum_kzt"]):
        raise ValueError("sum_kzt must be numeric in edges and transactions")
    if (not pd.api.types.is_numeric_dtype(edges["n_tx"])
            or (edges["n_tx"] < 1).any() or (edges["n_tx"] % 1 != 0).any()):
        raise ValueError("edges.n_tx must be a positive transaction count")
    if (edges["sum_kzt"] < 0).any() or (tx["sum_kzt"] < 0).any():
        raise ValueError("transaction amounts cannot be negative")
    if not np.isfinite(edges[["sum_kzt", "n_tx"]].to_numpy(dtype=float)).all():
        raise ValueError("edges amounts and counts must be finite")
    if not np.isfinite(tx["sum_kzt"].to_numpy(dtype=float)).all():
        raise ValueError("transaction amounts must be finite")
    for frame, column, label in ((nodes, "depth", "nodes.depth"), (edges, "depth", "edges.depth")):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{label} must contain integer hop depths")
        frame[column] = numeric.astype("int64")
    if not pd.api.types.is_bool_dtype(nodes["is_seed"]):
        if not nodes["is_seed"].isin([0, 1]).all():
            raise ValueError("nodes.is_seed must be boolean")
        nodes["is_seed"] = nodes["is_seed"].astype(bool)
    try:
        tx["date"] = pd.to_datetime(tx["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("transactions.date contains an invalid date") from exc

    node_ids = set(nodes["gid"])
    edge_ids = set(edges["src"]) | set(edges["dst"])
    if not edge_ids.issubset(node_ids):
        raise ValueError("every edge endpoint must have a row in nodes")
    tx_ids = set(tx["src"]) | set(tx["dst"])
    if not tx_ids.issubset(node_ids):
        raise ValueError("every transaction endpoint must have a row in nodes")

    aggregate = tx.groupby(["src", "dst"], as_index=False).agg(
        tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    comparison = edges[["src", "dst", "sum_kzt", "n_tx"]].merge(
        aggregate, on=["src", "dst"], how="outer", indicator=True)
    if not comparison["_merge"].eq("both").all():
        raise ValueError("edges and transactions contain different directed src/dst pairs")
    if not np.allclose(comparison["sum_kzt"], comparison["tx_sum"], rtol=1e-9, atol=1e-6):
        raise ValueError("edges.sum_kzt does not match the transaction sums")
    if not np.array_equal(comparison["n_tx"].astype(int), comparison["tx_count"].astype(int)):
        raise ValueError("edges.n_tx does not match the number of transactions")
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    edges, nodes, tx = validate_case_frames(edges, nodes, tx)
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    print("  edges == transactions : OK")

    # узлы без единого ребра
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    print(f"\n  ВНИМАНИЕ: {len(orphans)} узлов нет ни в одном ребре "
          f"(из них seed: {len(orphans & set(nodes[nodes.is_seed].gid))})")
    print("  → они всё равно должны попасть в nodes_roles.csv")
    print("=" * 64, "\n")
    return orphans


# ---------------------------------------------------------------- граф

def build_graph(edges) -> nx.DiGraph:
    """Направленный граф. sum_kzt — вес ребра, n_tx — количество переводов."""
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Базовые метрики. Это старт, а не финиш — добавляйте свои."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    in_tx = dict(G.in_degree(weight="n_tx"))
    out_tx = dict(G.out_degree(weight="n_tx"))
    # Weighted PageRank power iteration, avoiding NetworkX's optional SciPy
    # dependency. This matches the standard alpha=.85, uniform-dangling setup.
    alpha = 0.85
    graph_nodes = list(G.nodes())
    n_graph_nodes = len(graph_nodes)
    pr = {gid: 1.0 / n_graph_nodes for gid in graph_nodes} if n_graph_nodes else {}
    out_weight = {gid: sum(data["sum_kzt"] for _, _, data in G.out_edges(gid, data=True))
                  for gid in graph_nodes}
    for _ in range(100):
        dangling = sum(pr[gid] for gid in graph_nodes if out_weight[gid] == 0)
        updated = {gid: (1.0 - alpha) / n_graph_nodes + alpha * dangling / n_graph_nodes
                   for gid in graph_nodes}
        for src, dst, data in G.edges(data=True):
            if out_weight[src] > 0:
                updated[dst] += alpha * pr[src] * data["sum_kzt"] / out_weight[src]
        error = sum(abs(updated[gid] - pr[gid]) for gid in graph_nodes)
        pr = updated
        if error < 1e-12:
            break

    df = nodes[["gid", "depth", "is_seed"]].copy()
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pr).fillna(0.0)

    # доля полученного, которая ушла дальше. Около 1.0 — деньги не задерживаются.
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)

    # ЛОВУШКА КЕЙСА: узел на 4-м колене без исходящих может быть не «стоком»,
    # а просто местом, где закончился обход. Разберитесь с этим.
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    return df


def assign_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Assign one evidence-based role and a heuristic support score per customer."""
    f = df.copy()
    both = (f.in_deg > 0) & (f.out_deg > 0)
    distributor = f.out_deg >= DISTRIBUTOR_RECIPIENTS
    transit = (
        both & ~f.is_seed.astype(bool) & (f.in_kzt >= MIN_INCOMING_KZT)
        & (f.pass_through >= TRANSIT_PASS_THROUGH)
    )
    # A depth-4 zero-out endpoint is not evidence of retained funds.
    consolidator = (
        (f.in_deg >= 2) & (f.out_deg <= 1) &
        (f.in_kzt > 0) &
        (f.pass_through.fillna(0) <= 0.5) & ~f.truncated_by_depth
    )
    terminal = (f.in_deg > 0) & (f.out_deg == 0) & (f.depth < 4)
    coordinator = (
        both & (f.in_deg >= COORDINATOR_MIN_IN_DEG)
        & (f.out_deg >= COORDINATOR_MIN_OUT_DEG)
    )
    f["role"] = np.select(
        [distributor, transit, consolidator, terminal, coordinator],
        ["distributor", "transit", "consolidator", "terminal", "coordinator"],
        default="peripheral",
    )

    # Heuristic strength (0..1), not calibrated probabilities.
    scores = np.full(len(f), 0.28, dtype=float)
    scores[distributor] = np.minimum(
        0.96, 0.62 + 0.025 * (f.loc[distributor, "out_deg"] - DISTRIBUTOR_RECIPIENTS)
        + 0.06 * (f.loc[distributor, "out_tx"] >= f.loc[distributor, "out_deg"] * 2)
    )
    scores[transit] = np.minimum(
        0.90, 0.58 + 0.12 * np.minimum(f.loc[transit, "pass_through"], 2.0) / 2.0
        + 0.08 * (f.loc[transit, "in_kzt"] >= 2 * MIN_INCOMING_KZT)
        + 0.08 * (f.loc[transit, "in_deg"] + f.loc[transit, "out_deg"] >= 3)
    )
    scores[consolidator] = np.minimum(
        0.88, 0.56 + 0.08 * np.minimum(f.loc[consolidator, "in_deg"] - 2, 4) / 4
        + 0.10 * (f.loc[consolidator, "out_deg"] == 0)
        + 0.10 * (f.loc[consolidator, "in_tx"] >= f.loc[consolidator, "in_deg"] * 2)
    )
    scores[terminal] = np.minimum(
        0.82, 0.54 + 0.08 * (f.loc[terminal, "in_deg"] >= 2)
        + 0.08 * (f.loc[terminal, "in_tx"] >= 3)
        + 0.06 * (f.loc[terminal, "depth"] < 3)
    )
    scores[coordinator] = np.minimum(
        0.78, 0.46 + 0.08 * ((f.loc[coordinator, "in_deg"] >= 2)
                              & (f.loc[coordinator, "out_deg"] >= 2))
        + 0.08 * (f.loc[coordinator, "in_deg"] + f.loc[coordinator, "out_deg"] >= 3)
        + 0.08 * (f.loc[coordinator, "in_tx"] + f.loc[coordinator, "out_tx"] >= 3)
    )
    peripheral = f.role.eq("peripheral")
    scores[peripheral & f.truncated_by_depth] = 0.18
    scores[peripheral & (f.in_deg == 0) & (f.out_deg == 0)] = 0.12
    scores[peripheral & (f.in_deg == 0) & (f.out_deg > 0)] = 0.24
    f["role_score"] = np.clip(scores, 0.0, 1.0)

    def compact_kzt(value):
        value = float(value)
        magnitude = abs(value)
        for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if magnitude >= scale:
                return f"{value / scale:.1f}{suffix}"
        return f"{value:.0f}"

    def evidence(row):
        pass_text = "undefined" if pd.isna(row.pass_through) else f"{row.pass_through:.2f}"
        text = (f"{row.role}: {int(row.in_deg)} incoming/{int(row.out_deg)} outgoing; "
                f"{compact_kzt(row.in_kzt)} in/{compact_kzt(row.out_kzt)} out KZT; "
                f"pass-through {pass_text}. ")
        if row.role == "distributor":
            text += f"Fan-out meets the {DISTRIBUTOR_RECIPIENTS}+ recipient rule."
        elif row.role == "transit":
            text += (f"Non-seed forwards at least {TRANSIT_PASS_THROUGH:.1f} of observed incoming KZT "
                     f"with {MIN_INCOMING_KZT:,}+ KZT received; transit hypothesis.")
        elif row.role == "consolidator":
            text += "Multiple sources and limited onward flow suggest collection or retention."
        elif row.role == "terminal":
            text += "No outgoing flow before the depth cutoff; observed endpoint only."
        elif row.role == "coordinator":
            text += (f"{COORDINATOR_MIN_IN_DEG}+ incoming and {COORDINATOR_MIN_OUT_DEG}+ outgoing "
                     "counterparties indicate a multi-party bridge; coordinator hypothesis.")
        elif row.truncated_by_depth:
            text += "Depth-4 endpoint may be cut off; onward activity is unknown."
        elif row.in_deg == 0 and row.out_deg == 0:
            text += "No observed edges; evidence is limited."
        else:
            text += "Observed links do not meet another role rule."
        if row.is_seed:
            text += " Seed incoming history may be incomplete."
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        return text

    f["evidence"] = f.apply(evidence, axis=1)
    return f

def build_cluster_summaries(roles: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Build community summaries with flow evidence and cautious hypotheses."""
    feature_by_gid = roles.set_index("gid")
    cluster_rows = []
    for cluster_id, group in roles.groupby("cluster_id", sort=True):
        members = set(group["gid"])
        internal = [(u, v, data) for u, v, data in G.edges(data=True)
                    if u in members and v in members]
        internal_kzt = sum(data["sum_kzt"] for _, _, data in internal)
        member_features = feature_by_gid.loc[list(members)]
        member_flow = {gid: 0.0 for gid in members}
        for src, dst, data in internal:
            member_flow[src] += data["sum_kzt"]
            member_flow[dst] += data["sum_kzt"]
        top_gids = sorted(members, key=lambda gid: (-member_flow[gid], str(gid)))[:5]
        n_nodes = len(members)
        n_seed = int(member_features["is_seed"].astype(bool).sum())
        in_and_out = int(((member_features.in_deg > 0) & (member_features.out_deg > 0)).sum())
        multi_counterparty = int(((member_features.in_deg >= 2) | (member_features.out_deg >= 2)).sum())
        truncated = int(member_features["truncated_by_depth"].astype(bool).sum())

        if n_nodes == 1 and not internal:
            hypothesis = "Singleton customer with no observed transfer links; no group-level pattern can be inferred."
        elif in_and_out >= max(2, int(np.ceil(n_nodes / 2))):
            hypothesis = (f"Observed flows connect many members: {in_and_out}/{n_nodes} have both incoming and outgoing edges; "
                          f"{multi_counterparty}/{n_nodes} have 2+ counterparties on at least one side. "
                          "This may reflect transit or exchange activity.")
        elif multi_counterparty >= max(2, int(np.ceil(n_nodes / 2))):
            hypothesis = (f"{multi_counterparty}/{n_nodes} members have 2+ counterparties on at least one side; "
                          f"the observed pattern may reflect collection or distribution activity ({internal_kzt:,.0f} KZT internal).")
        else:
            hypothesis = (f"Observed links are comparatively sparse ({len(internal)} directed pairs; {internal_kzt:,.0f} KZT internal); "
                          "available evidence does not support a more specific purpose hypothesis.")
        if truncated:
            hypothesis += f" {truncated} member(s) are depth-4 endpoints, so onward flows may be unobserved."
        if n_seed:
            hypothesis += f" Includes {n_seed} seed customer(s), whose incoming flows may be incomplete."
        role_mix = member_features["role"].value_counts()
        mix_text = ", ".join(f"{role}={count}" for role, count in role_mix.items())
        hypothesis += f" Assigned role mix: {mix_text}."
        cluster_rows.append({
            "cluster_id": int(cluster_id),
            "n_nodes": n_nodes,
            "n_seed": n_seed,
            "sum_kzt_internal": internal_kzt,
            "top_gids": ",".join(map(str, top_gids)),
            "hypothesis": hypothesis,
        })
    return pd.DataFrame(cluster_rows, columns=[
        "cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis",
    ])


def write_clusters(roles: pd.DataFrame, G: nx.DiGraph, out_dir: Path):
    build_cluster_summaries(roles, G).to_csv(out_dir / "clusters.csv", index=False)


def build_top_nodes(roles: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Build a ranked list and the evidence behind each position."""
    ranked = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()
    top = ranked.head(min(limit, len(ranked)))[
        ["gid", "role", "priority_score", "evidence", "pagerank", "cluster_id"]
    ].copy()
    top["why"] = top.apply(
        lambda row: f"{row.evidence} PageRank={row.pagerank:.5g}; cluster={row.cluster_id}.", axis=1
    )
    top = top[["gid", "role", "priority_score", "why"]]
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    return top


def write_top_nodes(roles: pd.DataFrame, out_dir: Path):
    build_top_nodes(roles).to_csv(out_dir / "top_nodes.csv", index=False)


def write_network_html(roles: pd.DataFrame, G: nx.DiGraph, out_dir: Path):
    """Create a self-contained interactive graph view for the exported network."""
    graph_nodes = roles[["gid", "role", "cluster_id", "priority_score"]].copy()
    graph_nodes["id"] = graph_nodes["gid"].astype(str)
    nodes_json = graph_nodes.rename(columns={"priority_score": "score"})[
        ["id", "role", "cluster_id", "score"]
    ].to_dict("records")
    edges_json = [
        {"source": str(src), "target": str(dst), "value": float(data["sum_kzt"])}
        for src, dst, data in G.edges(data=True)
    ]
    payload = json.dumps({"nodes": nodes_json, "edges": edges_json}, ensure_ascii=False)
    html = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Network graph</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<style>
body {{ margin: 0; font: 14px Arial, sans-serif; background: #f7f8fa; color: #1f2937; }}
#controls {{ display: flex; gap: 12px; align-items: center; padding: 12px 16px; background: white; }}
#gid-search {{ padding: 7px; min-width: 220px; }} #network {{ width: 100vw; height: calc(100vh - 58px); }}
.link {{ stroke: #94a3b8; stroke-opacity: .45; }} .node {{ stroke: white; stroke-width: 1.5px; }}
.selected {{ stroke: #111827; stroke-width: 4px; }} .dim {{ opacity: .1; }}
</style></head><body>
<div id="controls"><label>Поиск gid: <input id="gid-search" placeholder="Введите gid"></label>
<label>Подсветка: <select id="color-mode"><option value="role">роль</option><option value="cluster">кластер</option></select></label>
<span id="status"></span></div><svg id="network" aria-label="Схема направленных денежных потоков"></svg>
<script>
const graph = {payload};
const roleColors = {{consolidator:'#e15759', transit:'#4e79a7', distributor:'#59a14f', terminal:'#f28e2b', coordinator:'#af7aa1', peripheral:'#9ca3af'}};
const svg = d3.select('#network'), width = innerWidth, height = innerHeight - 58;
svg.attr('viewBox', [0, 0, width, height]);
svg.append('defs').append('marker').attr('id','arrow').attr('viewBox','0 -5 10 10').attr('refX',18).attr('refY',0).attr('markerWidth',6).attr('markerHeight',6).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#64748b');
const layer = svg.append('g'); svg.call(d3.zoom().on('zoom', e => layer.attr('transform', e.transform)));
const link = layer.append('g').selectAll('line').data(graph.edges).join('line').attr('class','link').attr('marker-end','url(#arrow)');
const node = layer.append('g').selectAll('circle').data(graph.nodes).join('circle').attr('class','node').attr('r', d => 4 + 9 * d.score).append('title').text(d => `gid=${{d.id}} | ${{d.role}} | cluster=${{d.cluster_id}} | score=${{d.score.toFixed(3)}}`);
const circles = layer.selectAll('circle');
const simulation = d3.forceSimulation(graph.nodes).force('link', d3.forceLink(graph.edges).id(d => d.id).distance(42).strength(.25)).force('charge', d3.forceManyBody().strength(-75)).force('center', d3.forceCenter(width/2, height/2));
simulation.on('tick', () => {{ link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y); circles.attr('cx',d=>d.x).attr('cy',d=>d.y); }});
function recolor() {{ const mode = document.querySelector('#color-mode').value; const scale = d3.scaleOrdinal(d3.schemeTableau10); circles.attr('fill', d => mode === 'role' ? roleColors[d.role] : scale(d.cluster_id)); }}
document.querySelector('#color-mode').addEventListener('change', recolor); recolor();
document.querySelector('#gid-search').addEventListener('input', e => {{ const term = e.target.value.trim().toLowerCase(); circles.classed('selected', d => term && d.id.toLowerCase() === term).classed('dim', d => term && !d.id.toLowerCase().includes(term)); document.querySelector('#status').textContent = term ? `${{circles.filter(d=>d.id.toLowerCase()===term).size()}} совпадений` : ''; }});
</script></body></html>'''
    (out_dir / "network.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- выгрузки

def prepare_roles(df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """Assign roles, clusters, and priority scores without writing files."""
    df = assign_roles(df)
    roles = df[["gid", "role", "role_score", "evidence"]].copy()
    # Louvain works on an undirected projection; aggregate reciprocal flows.
    # Include isolated input customers so none are assigned cluster_id=-1.
    UG = nx.Graph()
    UG.add_nodes_from(df.gid)
    for src, dst, data in G.edges(data=True):
        weight = data["sum_kzt"]
        if UG.has_edge(src, dst):
            UG[src][dst]["weight"] += weight
        else:
            UG.add_edge(src, dst, weight=weight)
    communities = nx.community.louvain_communities(UG, weight="weight", seed=42)
    cluster_by_gid = {
        gid: cluster_id
        for cluster_id, community in enumerate(communities)
        for gid in community
    }
    roles["cluster_id"] = roles["gid"].map(cluster_by_gid).astype(int)
    total_kzt = df["in_kzt"] + df["out_kzt"]
    log_volume = np.log1p(total_kzt)
    volume_score = log_volume / log_volume.max() if log_volume.max() > 0 else log_volume
    pagerank_score = df["pagerank"] / df["pagerank"].max() if df["pagerank"].max() > 0 else df["pagerank"]
    roles["priority_score"] = (0.5 * roles["role_score"] + 0.3 * volume_score + 0.2 * pagerank_score).clip(0.0, 1.0)
    roles = roles.merge(
        df[["gid", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank",
            "pass_through", "depth", "is_seed", "truncated_by_depth"]],
        on="gid", how="left")
    # The ratio is undefined without observed incoming KZT; export 0 as a
    # numeric sentinel, with in_deg/in_kzt preserving the reason it is undefined.
    roles["pass_through"] = roles["pass_through"].fillna(0.0)
    return roles


def write_outputs(df: pd.DataFrame, out_dir: Path, G: nx.DiGraph):
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = prepare_roles(df, G)
    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    write_clusters(roles, G, out_dir)
    write_top_nodes(roles, out_dir)
    write_network_html(roles, G, out_dir)
    (out_dir / "analysis_version.txt").write_text(ANALYSIS_VERSION, encoding="utf-8")
    return roles

def summarize_run(G: nx.DiGraph, df: pd.DataFrame):
    """Print role totals and the main limits of the observed graph."""
    assigned = assign_roles(df)
    print("\nСВОДКА НАЗНАЧЕННЫХ РОЛЕЙ")
    print("-" * 64)
    for role in ROLES:
        print(f"  {role:<14}: {(assigned.role == role).sum():>5}")
    print(f"  узлов с обеими сторонами связей : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  depth-4 узлов без исходящих      : {int(df.truncated_by_depth.sum())}")
    print(f"  seed-клиентов                    : {int(df.is_seed.sum())}")
    print(f"  слабосвязных компонент           : {nx.number_weakly_connected_components(G)}")
    print("  У seed-клиентов входящие потоки могут быть неполными; узлы depth 4 могут быть обрезаны сбором.")
    print("  Роли — гипотезы по наблюдаемым потокам, а не утверждения о нарушениях.")


def analyze_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="папка с parquet-файлами (по умолчанию рядом со скриптом: parquet)")
    ap.add_argument("--out", default=None, help="куда писать выгрузки (по умолчанию рядом со скриптом: out)")
    a = ap.parse_args(argv)

    project_dir = Path(__file__).resolve().parent
    data_dir = Path(a.data) if a.data else project_dir / "parquet"
    out_dir = Path(a.out) if a.out else project_dir / "out"
    edges, nodes, tx = load(data_dir)
    sanity_check(edges, nodes, tx)
    G = build_graph(edges)
    df = basic_features(G, nodes)
    write_outputs(df, out_dir, G)
    summarize_run(G, df)


PROJECT_ROOT = Path(__file__).resolve().parent


def validate_startup(root: Path = PROJECT_ROOT) -> None:
    """Check the unified UI, transaction inputs, and its runtime dependencies."""
    app_path, data_dir = root / "interface" / "network_app.py", root / "parquet"
    required_data = ("edges.parquet", "nodes.parquet", "transactions.parquet")
    missing = [str(data_dir / name) for name in required_data if not (data_dir / name).is_file()]
    if not app_path.is_file():
        raise RuntimeError(f"Streamlit app is missing: {app_path}")
    if missing:
        raise RuntimeError("Missing input data: " + ", ".join(missing))
    dependencies = ("streamlit", "pandas", "pyarrow", "networkx", "plotly")
    absent = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if absent:
        raise RuntimeError("Missing packages " + ", ".join(absent) + ". Run pip install -r requirements.txt")


def launch_streamlit(root: Path = PROJECT_ROOT) -> None:
    """Run the network workspace directly using the current Python environment."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(root / "interface" / "network_app.py")],
        cwd=str(root),
        check=True,
    )


def main():
    """Validate the project and start its unified Streamlit application."""
    parser = argparse.ArgumentParser(description="AML network analysis launcher")
    parser.add_argument("--check", action="store_true", help="check the app environment and exit")
    parser.add_argument("--analyze-only", action="store_true", help="regenerate graph exports and exit")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    print("AML Network Analysis")
    print("Checking environment...")
    try:
        validate_startup()
    except RuntimeError as exc:
        raise SystemExit(f"Startup failed: {exc}") from exc
    print("Application configuration OK.")
    if args.check:
        return
    if args.analyze_only:
        analyze_main([])
        return
    print("Launching Streamlit...")
    try:
        launch_streamlit()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()

