# AML analyst agent

`AMLAnalystAgent` is the conversational layer over `GraphToolService`. It does
not calculate graph metrics: it supplies only the allowlisted graph tools to
the model through OpenAI Responses API function calling.

## Flow

```text
Streamlit app -> AMLAnalystAgent -> OpenAI function call -> ToolRegistry
                                                        -> GraphToolService
```

The agent records each tool call, sends the JSON result back to the model, and
returns an `AgentResponse` containing text, tool-call metadata, referenced
gids, and optional renderer-ready highlighting data. It limits a request to
six tool rounds and twelve calls. Recent in-session messages (up to 12) are
passed as context, so follow-ups such as “Why is it high priority?” can resolve
the previously discussed gid.

## Run

```powershell
pip install -r requirements.txt
$env:OPENAI_API_KEY = "..."
$env:AML_AGENT_MODEL = "gpt-4.1-mini"  # optional
python starter.py
```

`starter.py` validates the project paths and launches `app.py` through the same
Python interpreter. The page caches the graph service for Streamlit reruns, preserves chat history
in `st.session_state`, and stores `agent_focus_node`, `agent_highlight_nodes`,
and `agent_highlight_edges` for a graph component to consume.

The project currently has no reusable Streamlit graph component; `app.py`
stores the selection state and displays the deterministic call trace, while a
renderer can read those session-state fields without importing the LLM layer.
