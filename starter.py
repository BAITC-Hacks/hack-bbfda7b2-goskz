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
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]

# Distribution-informed cutoffs for this supplied graph. out_deg p95 = 5;
# incoming amount median = 50,000 KZT. A pass-through of .8 is above the
# 80th percentile among customers with observed incoming funds.
DISTRIBUTOR_RECIPIENTS = 5
MIN_INCOMING_KZT = 50_000
TRANSIT_PASS_THROUGH = 0.8


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    data_dir = Path(data_dir)
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
        (f.pass_through.fillna(0) <= 0.5) & ~f.truncated_by_depth
    )
    terminal = (f.in_deg > 0) & (f.out_deg == 0) & (f.depth < 4)
    coordinator = both
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

    def evidence(row):
        pass_text = "undefined" if pd.isna(row.pass_through) else f"{row.pass_through:.2f}"
        text = (
            f"{row.role}: in={int(row.in_deg)} counterparties/{int(row.in_tx)} tx/"
            f"{row.in_kzt:.0f} KZT; out={int(row.out_deg)} counterparties/"
            f"{int(row.out_tx)} tx/{row.out_kzt:.0f} KZT; pass-through={pass_text}; "
            f"depth={int(row.depth)}; seed={bool(row.is_seed)}. "
        )
        if row.role == "distributor":
            text += f"Outgoing recipients meet the data-based {DISTRIBUTOR_RECIPIENTS}+ threshold."
        elif row.role == "transit":
            text += (f"Non-seed node forwards at least {TRANSIT_PASS_THROUGH:.1f} of observed incoming KZT "
                     f"and has at least {MIN_INCOMING_KZT} KZT incoming; flow hypothesis only.")
        elif row.role == "consolidator":
            text += "Multiple observed sources and limited outgoing amount/activity suggest retention or limited redistribution."
        elif row.role == "terminal":
            text += "Incoming activity and no outgoing edge at depth below 4 support an observed endpoint hypothesis."
        elif row.role == "coordinator":
            text += "Both incoming and outgoing edges provide observed connecting activity."
        elif row.truncated_by_depth:
            text += "Depth-4 zero-out endpoint may be collection-truncated; terminal or retention behavior is uncertain."
        elif row.in_deg == 0 and row.out_deg == 0:
            text += "No observed edges; role evidence is minimal."
        else:
            text += "Observed links do not meet a more specific role pattern; evidence is limited."
        if row.is_seed:
            text += " Seed incoming coverage may be incomplete; pass-through is not used alone."
        return text

    f["evidence"] = f.apply(evidence, axis=1)
    return f


# ---------------------------------------------------------------- выгрузки

def write_outputs(df: pd.DataFrame, out_dir: Path, G: nx.DiGraph):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assign once; every export derives from the same customer-level rows.
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
    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    # 2. Summarize each community. Louvain used the undirected projection,
    # while internal KZT is totaled from the original directed edge list.
    feature_by_gid = df.set_index("gid")
    cluster_rows = []
    for cluster_id, community in enumerate(communities):
        members = set(community)
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
            "cluster_id": cluster_id,
            "n_nodes": n_nodes,
            "n_seed": n_seed,
            "sum_kzt_internal": internal_kzt,
            "top_gids": ",".join(map(str, top_gids)),
            "hypothesis": hypothesis,
        })
    pd.DataFrame(cluster_rows, columns=["cluster_id", "n_nodes", "n_seed",
                                       "sum_kzt_internal", "top_gids", "hypothesis"]) \
        .to_csv(out_dir / "clusters.csv", index=False)

    # 3. Rank customers by the existing role/volume/PageRank priority score.
    # Reasons expose the directed counts and amounts behind each individual row.
    ranked = roles.sort_values(["priority_score", "gid"], ascending=[False, True]).copy()
    ranked_features = df.set_index("gid")
    why = []
    for row in ranked.itertuples(index=False):
        f = ranked_features.loc[row.gid]
        why.append(f"{row.evidence} PageRank={f.pagerank:.5g}; cluster={row.cluster_id}.")
    ranked["why"] = why
    top_nodes = ranked.head(min(50, len(ranked)))[["gid", "role", "priority_score", "why"]].copy()
    top_nodes.insert(0, "rank", np.arange(1, len(top_nodes) + 1))
    top_nodes.to_csv(out_dir / "top_nodes.csv", index=False)

    print(f"Выгрузки записаны в {out_dir}/")


# ---------------------------------------------------------------- сводка

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="папка с parquet-файлами (по умолчанию рядом со скриптом: parquet)")
    ap.add_argument("--out", default=None, help="куда писать выгрузки (по умолчанию рядом со скриптом: out)")
    a = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    data_dir = Path(a.data) if a.data else project_dir / "parquet"
    out_dir = Path(a.out) if a.out else project_dir / "out"
    edges, nodes, tx = load(data_dir)
    sanity_check(edges, nodes, tx)
    G = build_graph(edges)
    df = basic_features(G, nodes)
    write_outputs(df, out_dir, G)
    summarize_run(G, df)


if __name__ == "__main__":
    main()

