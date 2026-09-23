"""Interactive directed network explorer for the money-flow hackathon dataset.

Run from the repository root with:
    streamlit run interface/network_app.py

The small loader/normalizer functions below are the only data-specific part of
the app. They accept pandas DataFrames, so CSV, JSON, NetworkX, and database
adapters can be plugged in without changing the visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v2 import component
from loading import show_loading


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "parquet"
ANNOTATIONS_PATH = ROOT / "out" / "nodes_roles.csv"

NETWORK_CHART_HTML = '<div id="network" aria-label="Interactive network graph"></div>'
NETWORK_CHART_CSS = """
#network { width: 100%; height: 690px; background: #0b1220; }
"""
NETWORK_CHART_JS = r"""
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  const root = parentElement;
  const plot = root.querySelector("#network");
  let disposed = false;

  function emit(action, gid) {
    if (gid !== undefined && gid !== null && String(gid) !== "") {
      setTriggerValue("interaction", {
        action: action,
        gid: String(gid),
        nonce: Date.now() + Math.random()
      });
    }
  }

  function loadPlotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (!window.__flowAtlasPlotlyPromise) {
      window.__flowAtlasPlotlyPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
        script.onload = () => resolve(window.Plotly);
        script.onerror = () => reject(new Error("Could not load Plotly from its CDN."));
        document.head.appendChild(script);
      });
    }
    return window.__flowAtlasPlotlyPromise;
  }

  function onClick(event) {
    const points = event && event.points;
    if (!points || points.length === 0) {
      setTriggerValue("interaction", {
        action: "clear",
        gid: "",
        nonce: Date.now() + Math.random()
      });
      return;
    }
    const point = points[points.length - 1];
    const customdata = point && point.customdata;
    if (customdata !== undefined && customdata !== null) {
      emit("select", Array.isArray(customdata) ? customdata[0] : customdata);
    }
  }
  loadPlotly().then((Plotly) => {
    if (disposed || !Plotly || !data.figure_json) return;
    const figure = JSON.parse(data.figure_json);
    figure.layout.clickanywhere = true;
    return Plotly.react(plot, figure.data, figure.layout, data.config || { displaylogo: false, scrollZoom: true })
      .then(() => plot.on("plotly_click", onClick));
  }).catch(() => {
    plot.textContent = "The network graph could not load. Check access to cdn.plot.ly.";
    plot.style.color = "#fda4af";
    plot.style.padding = "24px";
  });

  return () => {
    disposed = true;
    if (window.Plotly && plot.removeListener) plot.removeListener("plotly_click", onClick);
  };
}
"""
network_chart = component(
    "flow_atlas_network_chart",
    html=NETWORK_CHART_HTML,
    css=NETWORK_CHART_CSS,
    js=NETWORK_CHART_JS,
)

ROLE_COLORS = {
    "consolidator": "#fb7185",
    "transit": "#fbbf24",
    "distributor": "#a78bfa",
    "terminal": "#34d399",
    "coordinator": "#38bdf8",
    "peripheral": "#94a3b8",
    "Unclassified": "#64748b",
}
PALETTE = ["#38bdf8", "#a78bfa", "#34d399", "#fb7185", "#fbbf24", "#f472b6", "#2dd4bf"]


def load_default_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the repo's canonical nodes/edges; optional role export is merged later."""
    nodes_path, edges_path = DATA_DIR / "nodes.parquet", DATA_DIR / "edges.parquet"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Expected {nodes_path} and {edges_path}. Upload node and edge files in the sidebar."
        )
    return pd.read_parquet(nodes_path), pd.read_parquet(edges_path)


def normalize_data(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize common input names into gid/src/dst while preserving attributes."""
    nodes, edges = nodes.copy(), edges.copy()
    if "gid" not in nodes.columns:
        for candidate in ("id", "node", "entity"):
            if candidate in nodes.columns:
                nodes = nodes.rename(columns={candidate: "gid"})
                break
    rename_edges: dict[str, str] = {}
    for target, candidates in {
        "src": ("source", "from", "sender"),
        "dst": ("target", "to", "receiver"),
    }.items():
        if target not in edges.columns:
            found = next((name for name in candidates if name in edges.columns), None)
            if found:
                rename_edges[found] = target
    edges = edges.rename(columns=rename_edges)
    missing_nodes = {"gid"} - set(nodes.columns)
    missing_edges = {"src", "dst"} - set(edges.columns)
    if missing_nodes or missing_edges:
        raise ValueError(f"Missing required columns: nodes {sorted(missing_nodes)}, edges {sorted(missing_edges)}")
    nodes["gid"] = nodes["gid"].astype(str)
    edges["src"], edges["dst"] = edges["src"].astype(str), edges["dst"].astype(str)
    nodes = nodes.drop_duplicates("gid").reset_index(drop=True)
    known = set(nodes["gid"])
    extra = sorted((set(edges["src"]) | set(edges["dst"])) - known)
    if extra:
        nodes = pd.concat([nodes, pd.DataFrame({"gid": extra})], ignore_index=True)
    if "sum_kzt" not in edges:
        edges["sum_kzt"] = 1.0
    edges["sum_kzt"] = pd.to_numeric(edges["sum_kzt"], errors="coerce").fillna(0.0)
    if "n_tx" not in edges:
        edges["n_tx"] = 1
    return nodes, edges


def add_fallback_attributes(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Derive usable demo roles/clusters when analyst annotations are absent."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["gid"].astype(str))
    graph.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))
    in_degree, out_degree = dict(graph.in_degree()), dict(graph.out_degree())
    in_flow = edges.groupby("dst")["sum_kzt"].sum().to_dict()
    out_flow = edges.groupby("src")["sum_kzt"].sum().to_dict()
    fallback_role = []
    for gid in nodes["gid"]:
        incoming, outgoing = in_degree.get(gid, 0), out_degree.get(gid, 0)
        if incoming >= 3 and outgoing <= 1:
            role = "consolidator"
        elif outgoing >= 3 and incoming <= 1:
            role = "distributor"
        elif incoming and outgoing:
            role = "transit"
        elif incoming:
            role = "terminal"
        else:
            role = "peripheral"
        fallback_role.append(role)
    nodes["_fallback_role"] = fallback_role
    nodes["_fallback_cluster"] = -1
    nodes["_fallback_in_deg"] = nodes["gid"].map(in_degree).fillna(0).astype(int)
    nodes["_fallback_out_deg"] = nodes["gid"].map(out_degree).fillna(0).astype(int)
    nodes["_fallback_in_kzt"] = nodes["gid"].map(in_flow).fillna(0.0)
    nodes["_fallback_out_kzt"] = nodes["gid"].map(out_flow).fillna(0.0)
    for index, component in enumerate(nx.weakly_connected_components(graph)):
        nodes.loc[nodes["gid"].isin(component), "_fallback_cluster"] = index
    return nodes


def enrich_nodes(nodes: pd.DataFrame, edges: pd.DataFrame, annotations: pd.DataFrame | None) -> pd.DataFrame:
    nodes = add_fallback_attributes(nodes, edges)
    if annotations is not None and "gid" in annotations:
        annotations = annotations.copy()
        annotations["gid"] = annotations["gid"].astype(str)
        available = [column for column in ("gid", "role", "cluster_id", "priority_score", "evidence") if column in annotations]
        nodes = nodes.merge(annotations[available].drop_duplicates("gid"), on="gid", how="left", suffixes=("", "_annotation"))
    if "role" not in nodes:
        nodes["role"] = nodes["_fallback_role"]
    else:
        blank = nodes["role"].isna() | nodes["role"].astype(str).str.strip().isin(("", "nan"))
        nodes.loc[blank, "role"] = nodes.loc[blank, "_fallback_role"]
    if "cluster_id" not in nodes:
        nodes["cluster_id"] = nodes["_fallback_cluster"]
    else:
        nodes["cluster_id"] = nodes["cluster_id"].fillna(nodes["_fallback_cluster"])
        nodes.loc[pd.to_numeric(nodes["cluster_id"], errors="coerce").eq(-1), "cluster_id"] = nodes["_fallback_cluster"]
    for column in ("depth", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank", "priority_score"):
        if column not in nodes:
            fallback = {"in_deg": "_fallback_in_deg", "out_deg": "_fallback_out_deg", "in_kzt": "_fallback_in_kzt", "out_kzt": "_fallback_out_kzt"}.get(column)
            nodes[column] = nodes[fallback] if fallback else 0
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0)
    if "is_seed" not in nodes:
        nodes["is_seed"] = False
    nodes["is_seed"] = nodes["is_seed"].fillna(False).astype(bool)
    return nodes


@st.cache_data(show_spinner=False)
def layout_positions(gids: tuple[str, ...], edge_pairs: tuple[tuple[str, str], ...]) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(gids)
    graph.add_edges_from(edge_pairs)
    if not gids:
        return {}
    positions = nx.spring_layout(graph, seed=27, iterations=35, k=1.1 / max(len(gids) ** 0.5, 1))
    return {gid: (float(point[0]), float(point[1])) for gid, point in positions.items()}


def build_figure(
    layout_nodes: pd.DataFrame,
    layout_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    focus_gid: str,
    color_by: str,
    show_labels: bool,
) -> go.Figure:
    # Keep coordinates stable across filters; otherwise each small filter change
    # triggers a fresh (and expensive) force-directed layout.
    gids = tuple(layout_nodes["gid"].astype(str))
    pairs = tuple((str(src), str(dst)) for src, dst in layout_edges[["src", "dst"]].itertuples(index=False, name=None))
    positions = layout_positions(gids, pairs)
    selected = focus_gid if focus_gid in positions else None
    visible_nodes = set(gids)
    visible_edges = edges[edges["src"].isin(visible_nodes) & edges["dst"].isin(visible_nodes)]

    fig = go.Figure()
    # Draw edges in one trace for responsive pan/zoom, and arrowheads at each edge midpoint.
    focused_edges = visible_edges[(visible_edges["src"] == selected) | (visible_edges["dst"] == selected)] if selected else visible_edges.iloc[0:0]
    other_edges = visible_edges.drop(index=focused_edges.index)
    def edge_coordinates(edge_frame: pd.DataFrame) -> tuple[list[float | None], list[float | None]]:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for source, target in edge_frame[["src", "dst"]].itertuples(index=False, name=None):
            x0, y0 = positions[str(source)]
            x1, y1 = positions[str(target)]
            edge_x.extend((x0, x1, None))
            edge_y.extend((y0, y1, None))
        return edge_x, edge_y

    edge_x, edge_y = edge_coordinates(other_edges)
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 0.55, "color": "rgba(100,116,139,.09)" if selected else "rgba(148,163,184,.24)"}, hoverinfo="skip", name="Other flows" if selected else "Flow"))
    if selected and not focused_edges.empty:
        focus_x, focus_y = edge_coordinates(focused_edges)
        fig.add_trace(go.Scatter(x=focus_x, y=focus_y, mode="lines", line={"width": 2.2, "color": "rgba(56,189,248,.88)"}, hoverinfo="skip", name="Connected flows"))

    # Arrow markers placed along directed edges communicate money-flow direction.
    annotations = []
    # Keep arrow decorations focused on the selected node. When there is no
    # selection, sample the wider graph so Plotly stays responsive.
    if selected:
        arrow_edges = focused_edges
    else:
        arrow_limit = 160
        arrow_step = max(1, (len(visible_edges) + arrow_limit - 1) // arrow_limit)
        arrow_edges = visible_edges.iloc[::arrow_step].head(arrow_limit)
    for source, target in arrow_edges[["src", "dst"]].itertuples(index=False, name=None):
        x0, y0 = positions[str(source)]
        x1, y1 = positions[str(target)]
        annotations.append({"x": x0 + (x1 - x0) * 0.68, "y": y0 + (y1 - y0) * 0.68,
                            "ax": x0 + (x1 - x0) * 0.52, "ay": y0 + (y1 - y0) * 0.52,
                            "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
                            "showarrow": True, "arrowhead": 2, "arrowsize": 0.9, "arrowwidth": 1.25,
                            "arrowcolor": "#7dd3fc" if selected else "rgba(148,163,184,.42)"})

    color_column = "role" if color_by == "Role" else "cluster_id"
    if selected and selected not in set(nodes["gid"].astype(str)):
        nodes = pd.concat([nodes, layout_nodes[layout_nodes["gid"] == selected]], ignore_index=True).drop_duplicates("gid")
    groups = nodes.groupby(color_column, dropna=False, sort=True)
    cluster_colors = {value: PALETTE[i % len(PALETTE)] for i, value in enumerate(nodes["cluster_id"].drop_duplicates())}
    neighbors: set[str] = set()
    if selected:
        neighbors = set(focused_edges["src"].astype(str)) | set(focused_edges["dst"].astype(str))
    for group_value, group in groups:
        is_role = color_by == "Role"
        color = ROLE_COLORS.get(str(group_value), "#64748b") if is_role else cluster_colors.get(group_value, "#64748b")
        x_values = [positions[str(gid)][0] for gid in group["gid"]]
        y_values = [positions[str(gid)][1] for gid in group["gid"]]
        selected_flags = group["gid"].eq(selected)
        connected_flags = group["gid"].astype(str).isin(neighbors)
        sizes = [19 if is_selected else 10 + min(10, float(row.out_deg) ** 0.5) for is_selected, row in zip(selected_flags, group.itertuples())]
        custom = group[["gid", "role", "cluster_id", "depth", "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank"]].astype(str).values
        hover = [f"<b>gid {row.gid}</b><br>Role: {row.role}<br>Cluster: {row.cluster_id}<br>Depth: {row.depth}<br>In / out: {row.in_deg} / {row.out_deg}<br>Inflow: {float(row.in_kzt):,.0f} KZT<br>Outflow: {float(row.out_kzt):,.0f} KZT" for row in group.itertuples()]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="markers+text" if show_labels else "markers",
            text=group["gid"] if show_labels else None, textposition="top center", textfont={"size": 8, "color": "#cbd5e1"},
            customdata=custom, hovertext=hover, hoverinfo="text", name=f"{color_column}: {group_value}",
            marker={"size": sizes, "color": color,
                    "opacity": [1.0 if (is_selected or connected) else (0.12 if selected else 0.92) for is_selected, connected in zip(selected_flags, connected_flags)],
                    "symbol": "diamond" if is_role and str(group_value) == "consolidator" else "circle",
                    "line": {"width": [3 if flag else (1.4 if connected else 0.35) for flag, connected in zip(selected_flags, connected_flags)],
                             "color": ["#ffffff" if flag else ("#38bdf8" if connected else "rgba(15,23,42,.5)") for flag, connected in zip(selected_flags, connected_flags)]}},
        ))
    fig.update_layout(
        height=690, margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220", font={"color": "#e2e8f0"},
        legend={"orientation": "h", "y": -0.06, "x": 0, "font": {"size": 10}},
        annotations=annotations, clickmode="event+select",
        xaxis={"visible": False, "fixedrange": False}, yaxis={"visible": False, "fixedrange": False, "scaleanchor": "x", "scaleratio": 1},
        dragmode="pan", uirevision="network-layout",
    )
    if selected:
        x, y = positions[selected]
        fig.update_xaxes(range=[x - 0.55, x + 0.55])
        fig.update_yaxes(range=[y - 0.55, y + 0.55])
    return fig


def metric_value(row: pd.Series, column: str, suffix: str = "") -> str:
    value: Any = row.get(column, "—")
    if isinstance(value, (int, float)):
        return f"{value:,.0f}{suffix}"
    return f"{value}{suffix}"


def sync_dropdown_focus() -> None:
    st.session_state["_active_gid"] = st.session_state.get("selected_gid") or ""
    st.session_state["_dismissed_search_gid"] = ""


def on_network_interaction_change() -> None:
    """The component trigger is consumed in the main app execution."""


def main() -> None:
    st.set_page_config(page_title="Flow Atlas · Network Explorer", page_icon="◉", layout="wide")
    dismissed_search_gid = st.session_state.get("_dismissed_search_gid", "")
    current_search = str(st.session_state.get("gid_search", "")).strip()
    if dismissed_search_gid and current_search.casefold() == str(dismissed_search_gid).casefold():
        st.session_state["gid_search"] = ""
    st.session_state["_dismissed_search_gid"] = ""
    loading_screen = show_loading()
    app_styles = """
    <style>
      .stApp { background: #070d18; color: #e2e8f0; }
      [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1e293b; }
      .hero { padding: 1.25rem 0 .35rem 0; }
      .eyebrow { color:#38bdf8; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
      .hero h1 { font-size:2.1rem; margin:.3rem 0; color:#f8fafc; }
      .hero p { color:#94a3b8; margin:0; }
      .stat { background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:14px 16px; }
      .stat-label { color:#94a3b8; font-size:.78rem; }
      .stat-value { color:#f8fafc; font-size:1.35rem; font-weight:700; margin-top:2px; }
    </style>
    """
    hero = """
    <div class="hero"><div class="eyebrow">NETWORK INTELLIGENCE / DEMO</div>
    <h1>Flow Atlas</h1><p>Explore entities, clusters, and the direction of value moving through the network.</p></div>
    """

    with st.sidebar:
        st.markdown("### ◉ Network controls")
        st.caption("Search an entity or narrow the network by role and cluster.")
        uploaded_nodes = st.file_uploader("Nodes · CSV or JSON", type=["csv", "json"], key="nodes_upload")
        uploaded_edges = st.file_uploader("Edges · CSV or JSON", type=["csv", "json"], key="edges_upload")
        if uploaded_nodes and uploaded_edges:
            reader = lambda file: pd.read_json(file) if file.name.lower().endswith(".json") else pd.read_csv(file)
            raw_nodes, raw_edges = reader(uploaded_nodes), reader(uploaded_edges)
            source_label = "Uploaded data"
        else:
            raw_nodes, raw_edges = load_default_data()
            source_label = "Repository Parquet"
        raw_nodes, raw_edges = normalize_data(raw_nodes, raw_edges)
        annotation_data = pd.read_csv(ANNOTATIONS_PATH) if ANNOTATIONS_PATH.exists() else None
        nodes = enrich_nodes(raw_nodes, raw_edges, annotation_data)
        query = st.text_input("Find a gid", placeholder="e.g. 1000456", key="gid_search").strip()
        role_values = sorted(nodes["role"].astype(str).unique())
        selected_roles = st.multiselect("Role", role_values, default=role_values)
        cluster_values = sorted(nodes["cluster_id"].astype(str).unique())
        selected_clusters = st.multiselect("Cluster", cluster_values, default=cluster_values)
        color_by = st.radio("Color nodes by", ["Role", "Cluster"], horizontal=True)
        show_labels = st.toggle("Show gid labels", value=False)
        st.caption(f"Data source · {source_label}")

    # Compute the initial force layout while the loading screen is visible.
    layout_positions(
        tuple(nodes["gid"].astype(str)),
        tuple((str(src), str(dst)) for src, dst in raw_edges[["src", "dst"]].itertuples(index=False, name=None)),
    )
    loading_screen.empty()
    st.markdown(app_styles + hero, unsafe_allow_html=True)

    filtered_nodes = nodes[nodes["role"].astype(str).isin(selected_roles) & nodes["cluster_id"].astype(str).isin(selected_clusters)].copy()
    filtered_ids = set(filtered_nodes["gid"].astype(str))
    filtered_edges = raw_edges[raw_edges["src"].isin(filtered_ids) & raw_edges["dst"].isin(filtered_ids)].copy()
    match = nodes[nodes["gid"].str.casefold().eq(query.casefold())] if query else pd.DataFrame()
    searched_gid = str(match.iloc[0]["gid"]) if not match.empty else ""
    if searched_gid and searched_gid not in filtered_ids:
        filtered_nodes = pd.concat([filtered_nodes, nodes[nodes["gid"] == searched_gid]]).drop_duplicates("gid")
        filtered_ids.add(searched_gid)
        filtered_edges = raw_edges[raw_edges["src"].isin(filtered_ids) & raw_edges["dst"].isin(filtered_ids)].copy()

    dropdown_gid = str(st.session_state.get("selected_gid", ""))
    active_gid = str(st.session_state.get("_active_gid", ""))
    focus_gid = searched_gid or (active_gid if active_gid in filtered_ids else "")
    if not focus_gid and dropdown_gid in filtered_ids:
        focus_gid = dropdown_gid

    edge_total = len(filtered_edges)
    flow_total = float(filtered_edges["sum_kzt"].sum())
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in (
        (k1, "VISIBLE ENTITIES", f"{len(filtered_nodes):,}"),
        (k2, "DIRECTED FLOWS", f"{edge_total:,}"),
        (k3, "FLOW VOLUME", f"{flow_total:,.0f} KZT"),
        (k4, "SOURCE", "ANNOTATED" if annotation_data is not None else "STRUCTURAL ROLES"),
    ):
        col.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)
    st.write("")

    graph_col, details_col = st.columns([3.5, 1.05], gap="large")
    with graph_col:
        st.markdown("#### Network map")
        st.caption("Click a node to focus its connections · drag to move · scroll to zoom · arrows show flow direction")
        fig = build_figure(nodes, raw_edges, filtered_nodes, filtered_edges, focus_gid, color_by, show_labels)
        chart_result = network_chart(
            data={"figure_json": fig.to_json(), "config": {"displaylogo": False, "scrollZoom": True}},
            key="network_graph",
            height=690,
            width="stretch",
            on_interaction_change=on_network_interaction_change,
        )
        chart_event = getattr(chart_result, "interaction", None)
        if isinstance(chart_event, dict) and chart_event.get("nonce") != st.session_state.get("_last_network_event"):
            st.session_state["_last_network_event"] = chart_event.get("nonce")
            event_gid = str(chart_event.get("gid", ""))
            if chart_event.get("action") == "select":
                st.session_state["_active_gid"] = event_gid
                st.session_state["_dismissed_search_gid"] = ""
                if event_gid in filtered_ids:
                    st.session_state["selected_gid"] = event_gid
                st.rerun()
            elif chart_event.get("action") == "clear":
                if active_gid or searched_gid or dropdown_gid:
                    st.session_state["_active_gid"] = ""
                    st.session_state["selected_gid"] = None
                    if searched_gid:
                        st.session_state["_dismissed_search_gid"] = searched_gid
                    st.rerun()
        if searched_gid:
            st.success(f"Located gid {searched_gid}. It is highlighted and centered in the graph.")
        elif query:
            st.warning(f"No exact match for gid {query}.")

    selected_gid = focus_gid
    with details_col:
        st.markdown("#### Entity details")
        available_ids = filtered_nodes["gid"].astype(str).tolist()
        if available_ids:
            if st.session_state.get("selected_gid") not in available_ids:
                st.session_state["selected_gid"] = None
            chosen = st.selectbox("Select a gid", available_ids, index=available_ids.index(focus_gid) if focus_gid in available_ids else None, placeholder="Choose a gid…", key="selected_gid", on_change=sync_dropdown_focus)
            selected_gid = focus_gid or chosen or ""
        if selected_gid:
            row_df = nodes[nodes["gid"] == selected_gid]
            if not row_df.empty:
                row = row_df.iloc[0]
                st.markdown(f"### `{selected_gid}`")
                st.markdown(f"**{row['role']}** · Cluster {row['cluster_id']}")
                if bool(row["is_seed"]):
                    st.info("Seed entity")
                st.metric("Depth", metric_value(row, "depth"))
                st.metric("Connections · in / out", f"{metric_value(row, 'in_deg')} / {metric_value(row, 'out_deg')}")
                st.metric("Flow · in / out", f"{metric_value(row, 'in_kzt')} / {metric_value(row, 'out_kzt')} KZT")
                if float(row.get("pagerank", 0)):
                    st.metric("PageRank", f"{float(row['pagerank']):.5f}")
                if "evidence" in row and pd.notna(row["evidence"]) and str(row["evidence"]).strip():
                    st.caption("Analyst evidence")
                    st.write(str(row["evidence"]))
                adjacent = raw_edges[(raw_edges["src"] == selected_gid) | (raw_edges["dst"] == selected_gid)]
                st.caption(f"{len(adjacent):,} direct flows · {float(adjacent['sum_kzt'].sum()):,.0f} KZT total")
        else:
            st.info("Search for a gid or choose a node to inspect its profile.")
        st.markdown("---")
        st.caption("Structural roles are inferred from degree patterns until analyst labels are supplied.")


if __name__ == "__main__":
    main()
