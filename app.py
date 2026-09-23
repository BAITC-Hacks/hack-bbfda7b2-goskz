"""Streamlit entry point for the AML graph analyst assistant."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from agent_tools import GraphToolService
from aml_agent import AMLAnalystAgent

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading transaction graph...")
def get_graph_service() -> GraphToolService:
    return GraphToolService.from_project_data(PROJECT_ROOT / "parquet", PROJECT_ROOT / "out")


@st.cache_resource
def get_agent() -> AMLAnalystAgent:
    service = get_graph_service()
    return AMLAnalystAgent.from_environment(service)


def render_network(highlight_nodes: list[int]) -> None:
    """Render the existing network HTML and apply the current agent selection."""
    network_path = PROJECT_ROOT / "out" / "network.html"
    if not network_path.is_file():
        return
    script = """<script>
setTimeout(() => {
  const highlighted = new Set(__HIGHLIGHTS__.map(String));
  if (highlighted.size) {
    d3.selectAll('.node').classed('selected', d => highlighted.has(d.id))
      .classed('dim', d => !highlighted.has(d.id));
  }
}, 500);
</script>""".replace("__HIGHLIGHTS__", json.dumps(highlight_nodes))
    html = network_path.read_text(encoding="utf-8").replace("</body>", script + "</body>")
    components.html(html, height=720, scrolling=True)



def main() -> None:
    st.set_page_config(page_title="AML Analyst Assistant", layout="wide")
    st.title("AML Analyst Assistant")
    st.caption("Network facts are retrieved through deterministic graph tools.")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_highlight_nodes" not in st.session_state:
        st.session_state.agent_highlight_nodes = []
    try:
        service = get_graph_service()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    with st.sidebar:
        st.subheader("Node search")
        gid = st.text_input("GID")
        if gid:
            try:
                st.json(service.get_node(gid))
            except ValueError as exc:
                st.warning(str(exc))
    st.subheader("Transaction network")
    render_network(st.session_state.agent_highlight_nodes)
    try:
        agent = get_agent()
        ai_available = True
    except RuntimeError as exc:
        ai_available = False
        st.info(f"AI analyst is unavailable because {exc}")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("tool_calls"):
                with st.expander("How this answer was generated"):
                    st.json(message["tool_calls"])
    question = st.chat_input("Ask about the transaction network...", disabled=not ai_available)
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Inspecting the graph..."):
                response = agent.ask(question, st.session_state.chat_messages[:-1])
            st.write(response.answer)
            if response.tool_calls:
                with st.expander("How this answer was generated"):
                    st.json(response.tool_calls)
        st.session_state.chat_messages.append({"role": "assistant", "content": response.answer,
                                                "tool_calls": response.tool_calls})
        if response.visualization:
            st.session_state["agent_focus_node"] = response.visualization["focus_node"]
            st.session_state["agent_highlight_nodes"] = response.visualization["highlight_nodes"]
            st.session_state["agent_highlight_edges"] = response.visualization["highlight_edges"]
            st.info(f"Graph selection prepared for {len(response.visualization['highlight_nodes'])} node(s).")


if __name__ == "__main__":
    main()
