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
Python interpreter. `app.py` delegates to `interface/network_app.py`, which
renders the network dashboard and embeds the analyst chat below it. The UI and
agent share one cached `GraphToolService`; chat history and visualization
highlights persist in `st.session_state`. Agent-selected nodes and directed
flows are highlighted in the Plotly graph, and the focus gid becomes the
dashboard selection. Deterministic tool-call results remain visible in the chat.
