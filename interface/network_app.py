"""Financial network investigation workspace for the supplied hackathon data."""

from __future__ import annotations

import math
import colorsys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_tools import GraphToolService
from aml_agent import AMLAnalystAgent

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "parquet"
OUT_DIR = ROOT / "out"
ROLE_COLORS = {
    "consolidator": "#d97768", "transit": "#d7a84c", "distributor": "#8a78cf",
    "terminal": "#3e9b7d", "coordinator": "#438db0", "peripheral": "#8995a5",
    "Unclassified": "#687486",
}
PALETTE = ["#4d98b5", "#8b79cb", "#409c80", "#c96d62", "#c49c47", "#bb6c9d", "#3da6a0", "#7289ca"]


@st.cache_resource(show_spinner="Preparing transaction graph...")
def get_graph_service() -> GraphToolService:
    """Share one graph/tool service between the network workspace and assistant."""
    return GraphToolService.from_project_data(DATA_DIR, OUT_DIR)


@st.cache_resource
def get_analyst_agent() -> AMLAnalystAgent:
    return AMLAnalystAgent.from_environment(get_graph_service())


@st.cache_data(show_spinner="Loading supplied network data…")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    service = get_graph_service()
    annotations = service.node_data.copy()
    nodes = annotations[["gid", "depth", "is_seed"]].copy()
    edges = pd.DataFrame(
        ((src, dst, data.get("sum_kzt", 0.0), data.get("n_tx", 0))
         for src, dst, data in service.graph.edges(data=True)),
        columns=["src", "dst", "sum_kzt", "n_tx"],
    )
    top = pd.read_csv(OUT_DIR / "top_nodes.csv")
    nodes["gid"] = nodes["gid"].astype(str)
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)
    annotations["gid"] = annotations["gid"].astype(str)
    top["gid"] = top["gid"].astype(str)
    return nodes, edges, annotations, top


@st.cache_data(show_spinner=False)
def graph_positions(gids: tuple[str, ...], edge_pairs: tuple[tuple[str, str], ...]) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(gids)
    graph.add_edges_from(edge_pairs)
    if not gids:
        return {}
    pos = nx.spring_layout(graph, seed=27, iterations=35, k=1.1 / max(len(gids) ** 0.5, 1))
    return {gid: (float(point[0]), float(point[1])) for gid, point in pos.items()}


def display_num(value, digits=0) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value) if value is not None else "—"


def cluster_color(index: int, count: int) -> str:
    red, green, blue = colorsys.hsv_to_rgb(index / max(count, 1), 0.48, 0.74)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def clear_filters() -> None:
    st.session_state["role_filter"] = "All roles"
    st.session_state["cluster_filter"] = "All clusters"
    st.session_state["gid_query"] = ""


def reset_full_network() -> None:
    clear_filters()
    st.session_state["_last_applied_gid_query"] = ""
    st.session_state["selected_gid"] = ""


def select_gid(gid: str) -> None:
    st.session_state["selected_gid"] = str(gid)
    st.session_state["agent_highlight_nodes"] = []


def sync_inspector_selection() -> None:
    st.session_state["selected_gid"] = st.session_state.get("inspector_gid", "")


def build_network(nodes: pd.DataFrame, edges: pd.DataFrame, color_by: str, show_labels: bool,
                  focus_gid: str, neighborhood_only: bool,
                  highlighted_gids: set[str] | None = None,
                  highlighted_edges: list[dict] | None = None) -> go.Figure:
    highlighted_gids = highlighted_gids or set()
    highlighted_edge_pairs = {
        (str(edge.get("source")), str(edge.get("target")))
        for edge in (highlighted_edges or [])
        if isinstance(edge, dict) and edge.get("source") is not None and edge.get("target") is not None
    }
    gids = tuple(nodes["gid"].astype(str))
    visible = set(gids)
    edge_set = edges[edges.src.isin(visible) & edges.dst.isin(visible)].copy()
    positions = graph_positions(gids, tuple(edge_set[["src", "dst"]].itertuples(index=False, name=None)))
    focus_edges = edge_set[(edge_set.src == focus_gid) | (edge_set.dst == focus_gid)] if focus_gid else edge_set.iloc[:0]
    neighbors = set(focus_edges.src.astype(str)) | set(focus_edges.dst.astype(str))
    if neighborhood_only and focus_gid:
        keep = neighbors | {focus_gid}
        nodes = nodes[nodes.gid.astype(str).isin(keep)]
        edge_set = edge_set[edge_set.src.isin(keep) & edge_set.dst.isin(keep)]
    fig = go.Figure()
    def add_lines(frame: pd.DataFrame, color: str, width: float, name: str):
        xs, ys, labels = [], [], []
        for row in frame.itertuples(index=False):
            src, dst = str(row.src), str(row.dst)
            if str(src) not in positions or str(dst) not in positions:
                continue
            x0, y0 = positions[str(src)]; x1, y1 = positions[str(dst)]
            xs.extend((x0, x1, None)); ys.extend((y0, y1, None))
            amount = display_num(getattr(row, "sum_kzt", None))
            count = display_num(getattr(row, "n_tx", None))
            label = f"{src} → {dst}<br>Observed volume: {amount} KZT<br>Transactions: {count}"
            labels.extend((label, label, None))
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": width},
                                     text=labels, hoverinfo="text", name=name, showlegend=False))
    if focus_gid and not focus_edges.empty:
        add_lines(edge_set[~edge_set.index.isin(focus_edges.index)], "rgba(110,125,143,.13)", .6, "Other flows")
        add_lines(focus_edges, "rgba(67,141,176,.82)", 1.8, "Selected customer flows")
    else:
        add_lines(edge_set, "rgba(102,119,137,.32)", .7, "Directed flows")
    if highlighted_edge_pairs:
        agent_edges = edge_set[[
            (str(src), str(dst)) in highlighted_edge_pairs
            for src, dst in edge_set[["src", "dst"]].itertuples(index=False, name=None)
        ]]
        add_lines(agent_edges, "#d69a32", 3.2, "Agent-highlighted flows")

    color_col = "role" if color_by == "Role" else "cluster_id"
    cl_vals = sorted(nodes.cluster_id.astype(str).unique(), key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
    cluster_colors = {value: cluster_color(i, len(cl_vals)) for i, value in enumerate(cl_vals)}
    # A limited sample gives a directional cue on the overview; focus edges always receive arrows.
    arrow_edges = focus_edges if focus_gid else edge_set.iloc[::max(1, math.ceil(len(edge_set) / 110))].head(110)
    arrow_x, arrow_y, arrow_angle = [], [], []
    for src, dst in arrow_edges[["src", "dst"]].itertuples(index=False, name=None):
        if str(src) not in positions or str(dst) not in positions:
            continue
        x0, y0 = positions[str(src)]; x1, y1 = positions[str(dst)]
        arrow_x.append(x0 + .64 * (x1 - x0)); arrow_y.append(y0 + .64 * (y1 - y0))
        arrow_angle.append(math.degrees(math.atan2(x1 - x0, y1 - y0)))
    if arrow_x:
        fig.add_trace(go.Scatter(x=arrow_x, y=arrow_y, mode="markers", marker={"symbol": "triangle-up", "size": 7,
            "angle": arrow_angle, "angleref": "up", "color": "#799caf" if focus_gid else "rgba(161,177,190,.78)"},
            hoverinfo="skip", showlegend=False, name="Direction"))

    for value, group in nodes.groupby(color_col, dropna=False, sort=True):
        val = str(value)
        color = ROLE_COLORS.get(val, "#687486") if color_by == "Role" else cluster_colors.get(str(value), "#687486")
        selected = group.gid.astype(str).eq(focus_gid)
        connected = group.gid.astype(str).isin(neighbors)
        custom = [[str(gid)] for gid in group.gid]
        hover = [f"<b>GID {r.gid}</b><br>Role: {r.role}<br>Cluster: {r.cluster_id}<br>Priority: {float(r.priority_score):.3f}<br>Depth: {display_num(r.depth)} · Seed: {'Yes' if bool(r.is_seed) else 'No'}<br>In / out: {display_num(r.in_deg)} / {display_num(r.out_deg)}<br>Incoming: {display_num(r.in_kzt)} KZT<br>Outgoing: {display_num(r.out_kzt)} KZT" for r in group.itertuples()]
        sizes = [22 + min(13, max(0, float(x) * 13)) for x in group.priority_score]
        sizes = [size * 1.3 if flag else size for size, flag in zip(sizes, selected)]
        highlighted = group.gid.astype(str).isin(highlighted_gids)
        opacity = [1.0 if (sel or con or marked or not focus_gid) else .14
                   for sel, con, marked in zip(selected, connected, highlighted)]
        fig.add_trace(go.Scatter(x=[positions[str(g)][0] for g in group.gid], y=[positions[str(g)][1] for g in group.gid],
            mode="markers+text" if show_labels else "markers", text=group.gid if show_labels else None,
            textposition="top center", textfont={"size": 8, "color": "#475467"}, customdata=custom,
            hovertext=hover, hoverinfo="text", name=val, marker={"size": sizes, "color": color, "opacity": opacity,
            "line": {"width": [3 if s else (2.6 if marked else (1.8 if c else .45)) for s, c, marked in zip(selected, connected, highlighted)],
                     "color": ["#f8fafc" if s else ("#d69a32" if marked else ("#438db0" if c else "rgba(255,255,255,.6)")) for s, c, marked in zip(selected, connected, highlighted)]}}))
    fig.update_layout(height=610, margin={"l": 4, "r": 4, "t": 10, "b": 10}, paper_bgcolor="white", plot_bgcolor="white",
        font={"color": "#344054", "family": "Arial"}, legend={"orientation": "h", "y": -0.03, "font": {"size": 10}},
        clickmode="event+select", xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        dragmode="pan", uirevision="stable-network")
    if focus_gid in positions:
        x, y = positions[focus_gid]
        fig.update_xaxes(range=[x - .48, x + .48]); fig.update_yaxes(range=[y - .48, y + .48])
    return fig


def render_inspector(gid: str, nodes: pd.DataFrame, edges: pd.DataFrame, annotations: pd.DataFrame, scope: str = "inspector"):
    row_df = nodes[nodes.gid.astype(str).eq(str(gid))]
    if row_df.empty:
        st.info("Select a customer from search, the network, or the ranked list to inspect their profile.")
        return
    row = row_df.iloc[0]
    st.markdown(f"### Customer `{gid}`")
    st.markdown(f"**{row.role}** &nbsp; · &nbsp; Cluster **{row.cluster_id}**" + (" &nbsp; · &nbsp; **Seed customer**" if bool(row.is_seed) else ""), unsafe_allow_html=True)
    a, b, c, d, e = st.columns(5)
    a.metric("Priority", display_num(row.priority_score, 3))
    ann = annotations[annotations.gid.astype(str).eq(str(gid))]
    score = ann.iloc[0].get("role_score") if not ann.empty else None
    b.metric("Heuristic role score", display_num(score, 2))
    c.metric("Depth", display_num(row.depth))
    d.metric("Seed status", "Seed" if bool(row.is_seed) else "Not seed")
    e.metric("In / out counterparties", f"{display_num(row.in_deg)} / {display_num(row.out_deg)}")
    st.caption("Role score is an uncalibrated heuristic supporting the assigned role; it is not a probability or a finding of wrongdoing.")
    incoming = edges[edges.dst.astype(str).eq(str(gid))].sort_values("sum_kzt", ascending=False)
    outgoing = edges[edges.src.astype(str).eq(str(gid))].sort_values("sum_kzt", ascending=False)
    total_in, total_out = float(incoming.sum_kzt.sum()), float(outgoing.sum_kzt.sum())
    x, y, z = st.columns(3)
    x.metric("Observed incoming", f"{total_in:,.0f} KZT")
    y.metric("Observed outgoing", f"{total_out:,.0f} KZT")
    pass_value = ann.iloc[0].get("pass_through") if not ann.empty else None
    z.metric("Pass-through", f"{display_num(float(pass_value), 2)}×" if total_in > 0 and pass_value is not None and pd.notna(pass_value) else "Undefined")
    if bool(row.is_seed):
        st.info("Seed customers may have incomplete incoming flows because the network was collected outward from seed customers. Interpret pass-through cautiously.")
    trunc = ann.iloc[0].get("truncated_by_depth", False) if not ann.empty else False
    if bool(trunc) or (float(row.depth) >= 4 and float(row.out_deg) == 0):
        st.warning("This customer is at the depth cutoff with no observed outgoing edge. The extract may have stopped before later flows; this does not establish a true endpoint.")
    if ann.empty or not str(ann.iloc[0].get("evidence", "")).strip():
        st.caption("No role evidence is available in the annotations export.")
    else:
        st.markdown("**Evidence from analysis export**")
        st.write(str(ann.iloc[0].evidence))
    left, right = st.columns(2)
    def counterparty_table(frame: pd.DataFrame, gid_col: str, title: str):
        st.markdown(f"**{title}**")
        if frame.empty:
            st.caption("No observed counterparties.")
            return
        view = frame[[gid_col, "sum_kzt"] + (["n_tx"] if "n_tx" in frame else [])].copy()
        view.columns = ["GID", "Amount · KZT"] + (["Transactions"] if "n_tx" in frame else [])
        event = st.dataframe(view, hide_index=True, use_container_width=True, height=min(260, 38 + len(view) * 35),
                             on_select="rerun", selection_mode="single-row", key=f"counterparty_{scope}_{title}_{gid}")
        selected_rows = getattr(event, "selection", {}).get("rows", []) if event else []
        if selected_rows:
            select_gid(str(view.iloc[selected_rows[0]]["GID"]))
    with left: counterparty_table(incoming, "src", "Incoming from")
    with right: counterparty_table(outgoing, "dst", "Outgoing to")


def main() -> None:
    st.set_page_config(page_title="Flow Atlas · Investigation Workspace", page_icon="◉", layout="wide")
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      .stApp { background:#f5f7f8; color:#172b3a; font-family:'DM Sans',sans-serif; }
      [data-testid="stHeader"] { background:rgba(245,247,248,.92); }
      [data-testid="stSidebar"] { background:#edf1f3; border-right:1px solid #dce3e7; }
      h1,h2,h3 { font-family:'Manrope',sans-serif!important; color:#182e3d!important; letter-spacing:-.035em; }
      .hero { padding:8px 0 20px; border-bottom:1px solid #dde5e8; margin-bottom:20px; }
      .eyebrow { color:#438ba4; font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
      .hero h1 { margin:4px 0!important; font-size:2.1rem!important; }
      .hero p { margin:0; color:#667785; font-size:1rem; }
      .metric-card { background:#fff; border:1px solid #e0e6e9; border-radius:12px; padding:14px 17px; min-height:88px; }
      .metric-label { font-size:.72rem; font-weight:700; color:#6a7b87; letter-spacing:.08em; text-transform:uppercase; }
      .metric-value { font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:700; color:#183447; margin-top:3px; }
      .metric-note { font-size:.75rem; color:#81909a; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e0e6e9; padding:12px 14px; border-radius:10px; }
      .block-container { padding-top:1.5rem; max-width:1550px; }
      div[data-testid="stTabs"] button { font-weight:600; }
      .stCaption { color:#72818c; }
    </style>
    <div class="hero"><div class="eyebrow">Financial network · investigation workspace</div>
    <h1>Flow Atlas</h1><p>Explore observed money flows, customer roles, and network evidence.</p></div>
    """, unsafe_allow_html=True)
    try:
        graph_service = get_graph_service()
        raw_nodes, edges, annotations, top = load_data()
    except Exception as exc:
        st.error(f"Unable to prepare the transaction network: {exc}")
        st.stop()
    nodes = raw_nodes.merge(annotations, on="gid", how="left", suffixes=("", "_analysis"))
    # Prefer the analysis export for derived values and preserve missing values explicitly.
    for col, default in (("role", "Unclassified"), ("cluster_id", "Unknown"), ("priority_score", 0),
                         ("in_deg", 0), ("out_deg", 0), ("in_kzt", 0), ("out_kzt", 0), ("depth", 0), ("is_seed", False)):
        if col not in nodes: nodes[col] = default
        nodes[col] = nodes[col].fillna(default)
    if "gid_query" not in st.session_state: st.session_state["gid_query"] = ""
    if "selected_gid" not in st.session_state: st.session_state["selected_gid"] = ""
    with st.sidebar:
        st.markdown("### Investigation controls")
        st.caption("Filters apply to the network. Whole-dataset totals remain visible in Overview.")
        query = st.text_input("Search exact GID", placeholder="Enter a customer GID", key="gid_query").strip()
        roles = sorted(nodes.role.astype(str).unique())
        clusters = sorted(nodes.cluster_id.astype(str).unique(), key=lambda x: (not x.lstrip("-").isdigit(), int(x) if x.lstrip("-").isdigit() else x))
        st.selectbox("Role", ["All roles"] + roles, key="role_filter")
        st.selectbox("Cluster", ["All clusters"] + clusters, key="cluster_filter")
        st.button("Clear filters", on_click=clear_filters, use_container_width=True)
        last_applied_query = str(st.session_state.get("_last_applied_gid_query", ""))
        if query and query.casefold() != last_applied_query.casefold():
            st.session_state["_last_applied_gid_query"] = query
            matches = nodes[nodes.gid.astype(str).str.casefold().eq(query.casefold())]
            if matches.empty:
                st.error(f"No exact GID match for ‘{query}’. Check the value and try again.")
            else:
                select_gid(str(matches.iloc[0].gid))
                st.success(f"Customer {matches.iloc[0].gid} selected")
        elif query:
            matches = nodes[nodes.gid.astype(str).str.casefold().eq(query.casefold())]
            if matches.empty:
                st.error(f"No exact GID match for ‘{query}’. Check the value and try again.")
            elif str(matches.iloc[0].gid) == str(st.session_state.get("selected_gid", "")):
                st.success(f"Customer {matches.iloc[0].gid} selected")
        st.divider()
        st.caption("Color and label controls are available in Network.")
        st.caption("Source: parquet/nodes.parquet, parquet/edges.parquet and out/*.csv")

    selected_gid = str(st.session_state.get("selected_gid", ""))
    cluster_pick = st.session_state.get("cluster_filter", "All clusters")
    cluster_mask = nodes.cluster_id.astype(str).isin(clusters if cluster_pick == "All clusters" else [cluster_pick])
    role_pick = st.session_state.get("role_filter", "All roles")
    role_mask = nodes.role.astype(str).isin(roles if role_pick == "All roles" else [role_pick])
    filtered = nodes[role_mask & cluster_mask].copy()
    filtered_ids = set(filtered.gid.astype(str))
    filtered_edges = edges[edges.src.isin(filtered_ids) & edges.dst.isin(filtered_ids)]
    total_volume = float(edges.sum_kzt.sum())
    whole_clusters = nodes.cluster_id.nunique()
    whole_seeds = int(nodes.is_seed.astype(bool).sum())
    tabs = st.tabs(["Overview", "Network", "Top Customers", "Customer Inspector"])
    with tabs[0]:
        st.markdown("### Network at a glance")
        st.caption("Whole-dataset figures describe the complete supplied extract. Filtered figures reflect the current role and cluster selections.")
        cols = st.columns(5)
        for col, label, value in zip(cols, ["Customers", "Directed flows", "Observed volume", "Clusters", "Seed customers"],
                                     [f"{len(nodes):,}", f"{len(edges):,}", f"{total_volume:,.0f} KZT", f"{whole_clusters:,}", f"{whole_seeds:,}"]):
            col.markdown(f'<div class="metric-card"><div class="metric-label">{label} · whole dataset</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.write("")
        a, b, c = st.columns(3)
        a.metric("Filtered customers", f"{len(filtered):,}")
        b.metric("Filtered directed flows", f"{len(filtered_edges):,}")
        c.metric("Filtered observed volume", f"{float(filtered_edges.sum_kzt.sum()):,.0f} KZT")
        st.markdown("#### Reading this extract")
        st.write("Edges represent observed directed customer-to-customer transfers; volumes are within the extracted graph. Role labels and rankings are analysis hypotheses based on observed structure and amounts.")
        st.info("Seed customers can have incomplete incoming history. Customers at the depth cutoff with no outgoing edge may have unobserved later activity.")
        st.markdown("#### Priority leaders")
        preview = top.head(5)[["rank", "gid", "role", "priority_score"]].copy()
        st.dataframe(preview, hide_index=True, use_container_width=True)

    with tabs[1]:
        heading, options = st.columns([2, 1])
        heading.markdown("### Directed customer network")
        with options:
            control1, control2 = st.columns(2)
            with control1: color_by = st.radio("Color by", ["Role", "Cluster"], horizontal=True, key="color_by")
            with control2: show_labels = st.toggle("Show GIDs", value=False, key="show_labels")
        st.caption("Arrows indicate source → recipient. Node size reflects priority score. Click any node to select it; pan and zoom to explore.")
        n1, n2, n3 = st.columns([2, 2, 1])
        with n1: st.markdown("**Color legend** · " + ("Role" if color_by == "Role" else "Cluster ID"))
        with n2: neighborhood_only = st.toggle("Focus on selected neighborhood", value=False, disabled=not bool(selected_gid), key="neighborhood_only")
        with n3:
            st.button("Full network", use_container_width=True, on_click=reset_full_network)
        if color_by == "Role":
            st.markdown("　".join(f'<span style="color:{ROLE_COLORS.get(role, "#687486")}">●</span> {role}' for role in roles), unsafe_allow_html=True)
        else:
            with st.expander(f"Cluster color key · {len(clusters)} clusters"):
                st.markdown("　".join(f'<span style="color:{cluster_color(i, len(clusters))}">●</span> {cluster}' for i, cluster in enumerate(clusters)), unsafe_allow_html=True)
        st.caption("Node size = analysis priority score · pale arrows are sampled to reduce clutter; selecting a customer highlights every direct flow.")
        if filtered.empty and selected_gid not in set(nodes.gid.astype(str)):
            st.warning("No customers match the current filters. Clear filters or choose another role or cluster.")
        else:
            graph_nodes, graph_edges = filtered, filtered_edges
            if selected_gid in set(nodes.gid.astype(str)):
                focus_edges = edges[(edges.src == selected_gid) | (edges.dst == selected_gid)]
                context_ids = {selected_gid} | set(focus_edges.src.astype(str)) | set(focus_edges.dst.astype(str))
                extra_ids = context_ids - filtered_ids
                if extra_ids:
                    graph_nodes = pd.concat([filtered, nodes[nodes.gid.astype(str).isin(extra_ids)]], ignore_index=True).drop_duplicates("gid")
                    graph_edges = pd.concat([filtered_edges, focus_edges], ignore_index=True).drop_duplicates(["src", "dst"])
                    st.caption("The selected customer and direct counterparties stay visible even when they fall outside the current role or cluster filter.")
            fig = build_network(
                graph_nodes, graph_edges, color_by, show_labels, selected_gid, neighborhood_only,
                set(st.session_state.get("agent_highlight_nodes", [])),
            st.session_state.get("agent_highlight_edges", []),
            )
            event = st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True},
                                    on_select="rerun", selection_mode="points", key="network_plot")
            points = getattr(event, "selection", {}).get("points", []) if event else []
            if points:
                custom = points[-1].get("customdata")
                if isinstance(custom, (list, tuple)): custom = custom[0]
                if custom and str(custom) != selected_gid:
                    select_gid(str(custom)); st.rerun()
            if selected_gid in set(nodes.gid.astype(str)):
                st.caption(f"Filters match {len(filtered):,} customers and {len(filtered_edges):,} directed flows; the graph also retains the selected customer’s direct counterparties.")
            else:
                st.caption(f"Showing {len(filtered):,} customers and {len(filtered_edges):,} directed flows after filters.")
        if selected_gid and selected_gid in set(nodes.gid.astype(str)):
            st.markdown("#### Selected customer")
            render_inspector(selected_gid, nodes, edges, annotations, "network")

    with tabs[2]:
        st.markdown("### Ranked customers")
        st.caption("Ranking and explanations come from out/top_nodes.csv. Priority is a heuristic ranking score, not a probability or finding of wrongdoing.")
        if top.empty:
            st.warning("The top customer export is empty.")
        else:
            view = top.copy()
            view["priority_score"] = pd.to_numeric(view.priority_score, errors="coerce")
            event = st.dataframe(view[[c for c in ["rank", "gid", "role", "priority_score", "why"] if c in view]],
                hide_index=True, use_container_width=True, height=540, on_select="rerun", selection_mode="single-row", key="top_customer_table")
            rows = getattr(event, "selection", {}).get("rows", []) if event else []
            if rows:
                select_gid(str(view.iloc[rows[0]].gid))
            gid_input = st.text_input("Open a customer by GID", placeholder="Exact GID", key="top_gid")
            if gid_input:
                match = nodes[nodes.gid.astype(str).str.casefold().eq(gid_input.strip().casefold())]
                if match.empty: st.error(f"GID {gid_input.strip()} is not in the supplied customer data.")
                else:
                    select_gid(str(match.iloc[0].gid))
                    selected_gid = str(match.iloc[0].gid)
            selected_gid = str(st.session_state.get("selected_gid", ""))
            if selected_gid and selected_gid in set(nodes.gid.astype(str)):
                st.markdown("#### Customer preview")
                render_inspector(selected_gid, nodes, edges, annotations, "top")

    with tabs[3]:
        st.markdown("### Customer inspector")
        available = nodes.gid.astype(str).tolist()
        if selected_gid and selected_gid not in set(available):
            st.warning(f"Selected GID {selected_gid} is not present in the current data. Search for a valid customer.")
        if available:
            choices = [""] + available
            current = selected_gid if selected_gid in set(available) else ""
            if st.session_state.get("inspector_gid") != current:
                st.session_state["inspector_gid"] = current
            chosen = st.selectbox("Select customer GID", choices, index=choices.index(current), format_func=lambda x: x or "Choose a customer…",
                                  key="inspector_gid", on_change=sync_inspector_selection)
            if chosen != selected_gid: select_gid(chosen)
        render_inspector(str(st.session_state.get("selected_gid", "")), nodes, edges, annotations, "inspector")
        st.caption("Observed values describe the supplied extract. Seed incompleteness and depth-cutoff truncation limit interpretation.")
    st.divider()
    st.markdown("## Analyst Assistant")
    st.caption("Ask questions about the observed network. Answers use deterministic graph analysis tools.")
    try:
        analyst_agent = get_analyst_agent()
        assistant_available = True
    except RuntimeError as exc:
        assistant_available = False
        st.info(f"AI analyst is unavailable: {exc}")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("tool_calls"):
                with st.expander("How this answer was generated"):
                    st.json(message["tool_calls"])

    question = st.chat_input(
        "Ask about roles, flows, clusters, or a GID...",
        disabled=not assistant_available,
    )
    if question:
        history = st.session_state["chat_messages"][-12:]
        st.session_state["chat_messages"].append({"role": "user", "content": question})
        try:
            response = analyst_agent.ask(question, history)
        except Exception as exc:
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": f"I could not complete the analysis request: {exc}",
                "tool_calls": [],
            })
        else:
            st.session_state["chat_messages"].append({
                "role": "assistant",
                "content": response.answer,
                "tool_calls": response.tool_calls,
            })
            if response.visualization:
                focus = response.visualization.get("focus_node")
                if focus is not None and str(focus) in set(nodes.gid.astype(str)):
                    st.session_state["selected_gid"] = str(focus)
                    st.session_state["agent_highlight_nodes"] = [
                        str(gid) for gid in response.visualization.get("highlight_nodes", [])
                    ]
                    st.session_state["agent_highlight_edges"] = response.visualization.get("highlight_edges", [])
        st.rerun()


if __name__ == "__main__":
    main()
