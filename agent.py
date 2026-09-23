#!/usr/bin/env python3
"""
Стартовый код кейса «Граф денег» — HackAlem AI.

Что он делает:
  1. грузит три parquet-файла и проверяет их консистентность;
  2. собирает направленный взвешенный граф;
  3. считает БАЗОВЫЕ метрики узлов (степени, обороты, PageRank);
  4. пишет три выгрузки в требуемой ТЗ схеме — с ПУСТЫМИ ролями.

Чего он НЕ делает — это ваша работа:
  * не присваивает роли,
  * не кластеризует,
  * не ранжирует узлы,
  * не рисует граф.

Запуск:
    python starter.py --data ../data --out ./out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    data_dir=Path("parquet")
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    # транзакции должны складываться в рёбра
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
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
    pr = nx.pagerank(G, weight="sum_kzt")

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


def write_clusters(roles: pd.DataFrame, G: nx.DiGraph, out_dir: Path):
    """Write one aggregate record for every Louvain community."""
    membership = roles.set_index("gid")["cluster_id"]
    edge_frame = pd.DataFrame(
        ((src, dst, data["sum_kzt"]) for src, dst, data in G.edges(data=True)),
        columns=["src", "dst", "sum_kzt"],
    )
    edge_frame["src_cluster"] = edge_frame["src"].map(membership)
    edge_frame["dst_cluster"] = edge_frame["dst"].map(membership)
    internal_kzt = (
        edge_frame[edge_frame["src_cluster"].eq(edge_frame["dst_cluster"])]
        .groupby("src_cluster")["sum_kzt"].sum()
    )

    labels = {
        "consolidator": "кластер накопления средств",
        "transit": "транзитный кластер",
        "distributor": "кластер распределения",
        "terminal": "кластер конечных получателей",
        "coordinator": "координационный кластер",
        "peripheral": "периферийный кластер",
    }
    rows = []
    for cluster_id, group in roles.groupby("cluster_id", sort=True):
        role_counts = group["role"].value_counts()
        main_role = role_counts.index[0]
        top_gids = group.nlargest(5, "priority_score")["gid"].astype(str).tolist()
        rows.append({
            "cluster_id": int(cluster_id),
            "n_nodes": len(group),
            "n_seed": int(group["is_seed"].sum()),
            "sum_kzt_internal": float(internal_kzt.get(cluster_id, 0.0)),
            "top_gids": ",".join(top_gids),
            "hypothesis": (
                f"{labels[main_role]}; роль {main_role}: {role_counts.iloc[0]}/{len(group)}, "
                f"seed: {int(group['is_seed'].sum())}"
            )[:200],
        })
    pd.DataFrame(rows, columns=[
        "cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis",
    ]).to_csv(out_dir / "clusters.csv", index=False)


def write_top_nodes(roles: pd.DataFrame, out_dir: Path):
    """Write the twenty highest-priority nodes with their numeric evidence."""
    top = roles.nlargest(min(20, len(roles)), "priority_score").copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["why"] = (
        "cluster=" + top["cluster_id"].astype(str)
        + "; score=" + top["priority_score"].round(3).astype(str)
        + "; " + top["evidence"]
    ).str.slice(0, 200)
    top[["rank", "gid", "role", "priority_score", "why"]].to_csv(
        out_dir / "top_nodes.csv", index=False
    )


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

def write_outputs(df: pd.DataFrame, out_dir: Path, G: nx.DiGraph):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. nodes_roles.csv — схема из ТЗ, роли не заполнены
    roles = df[["gid"]].copy()
    roles["role"] = np.select(
        [
            (df.out_deg == 0) & ~df.truncated_by_depth,
            (df.in_deg == 0) & (df.out_deg > 0),
            (df.in_deg >= 2) & (df.out_deg >= 2),
            (df.in_deg > 0) & (df.out_deg > 0) & (df.pass_through >= 0.8),
            (df.in_deg > 0) & (df.out_deg > 0),
        ],
        ["terminal", "distributor", "coordinator", "transit", "consolidator"],
        default="peripheral",
    )
    roles["role_score"] = np.select(
        [
            (df.out_deg == 0) & ~df.truncated_by_depth,
            (df.in_deg == 0) & (df.out_deg > 0),
            (df.in_deg >= 2) & (df.out_deg >= 2),
            (df.in_deg > 0) & (df.out_deg > 0) & (df.pass_through >= 0.8),
            (df.in_deg > 0) & (df.out_deg > 0),
        ],
        [1.0, 1.0, 1.0, np.clip(df.pass_through, 0.0, 1.0), 1.0],
        default=0.0,
    )
    # Louvain works on an undirected projection; aggregate reciprocal flows.
    UG = nx.Graph()
    UG.add_nodes_from(G)
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
    roles["cluster_id"] = roles["gid"].map(cluster_by_gid).fillna(-1).astype(int)
    total_kzt = df["in_kzt"] + df["out_kzt"]
    log_volume = np.log1p(total_kzt)
    volume_score = log_volume / log_volume.max() if log_volume.max() > 0 else log_volume
    pagerank_score = df["pagerank"] / df["pagerank"].max() if df["pagerank"].max() > 0 else df["pagerank"]
    roles["priority_score"] = (0.5 * roles["role_score"] + 0.3 * volume_score + 0.2 * pagerank_score).clip(0.0, 1.0)
    reason = np.select(
        [
            roles["role"].eq("terminal"),
            roles["role"].eq("distributor"),
            roles["role"].eq("coordinator"),
            roles["role"].eq("transit"),
            roles["role"].eq("consolidator"),
        ],
        ["outgoing=0", "incoming=0", "in>=2,out>=2",
         "pass_through>=0.80", "has incoming and outgoing"],
        default="insufficient links",
    )
    roles["evidence"] = (
        pd.Series(reason, index=roles.index)
        + "; in=" + df["in_deg"].astype(str)
        + ", out=" + df["out_deg"].astype(str)
        + "; KZT in=" + df["in_kzt"].round().astype("int64").astype(str)
        + ", out=" + df["out_kzt"].round().astype("int64").astype(str)
        + "; pass=" + df["pass_through"].fillna(0).round(2).astype(str)
    ).str.slice(0, 200)
    roles = roles.merge(
        df[["gid", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank",
            "pass_through", "depth", "is_seed", "truncated_by_depth"]],
        on="gid", how="left")
    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    write_clusters(roles, G, out_dir)
    write_top_nodes(roles, out_dir)
    write_network_html(roles, G, out_dir)

    # 2. clusters.csv — пустой каркас

    # 3. top_nodes.csv — пустой каркас, нужно ≥20 строк

    print(f"Выгрузки записаны в {out_dir}/  (роли пока пустые — это ваша задача)")


# ---------------------------------------------------------------- подсказки

def hints(G: nx.DiGraph, df: pd.DataFrame):
    """Куда смотреть дальше. Ответов здесь нет — только направления."""
    print("\nС ЧЕГО НАЧАТЬ")
    print("-" * 64)
    print(f"  узлов, получающих от 3+ разных плательщиков : {(df.in_deg >= 3).sum()}")
    print(f"  узлов, рассылающих на 10+ получателей       : {(df.out_deg >= 10).sum()}")
    print(f"  узлов и с входом, и с выходом               : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  узлов, обрезанных 4-м коленом               : {df.truncated_by_depth.sum()}  <- разберитесь")
    print(f"  слабосвязных компонент                      : {nx.number_weakly_connected_components(G)}")
    print("""
  Вопросы, на которые стоит ответить метриками:
    * чем «деньги пришли и остались» отличается от «пришли и ушли дальше»?
    * что важнее для роли — количество плательщиков или сумма?
    * узел собирает средства от нескольких SEED — это случайность или структура?
    * если убрать узел, сеть распадётся или переживёт?

  Полезное в networkx: pagerank, hits, betweenness_centrality,
  community.louvain_communities, simple_cycles, all_simple_paths.
  Не забудьте: граф НАПРАВЛЕННЫЙ и ВЗВЕШЕННЫЙ.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data", help="папка с parquet-файлами")
    ap.add_argument("--out", default="./out", help="куда писать выгрузки")
    a = ap.parse_args()

    edges, nodes, tx = load(Path(a.data))
    sanity_check(edges, nodes, tx)
    G = build_graph(edges)
    df = basic_features(G, nodes)
    write_outputs(df, Path(a.out), G)
    hints(G, df)


if __name__ == "__main__":
    main()

