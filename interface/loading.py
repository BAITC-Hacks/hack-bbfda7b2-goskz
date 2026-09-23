"""Reusable Streamlit loading screen for interface apps."""

import streamlit as st


def show_loading(message: str = "Preparing the network and loading its data"):
    """Render a centered loading screen and return its placeholder for clearing."""
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <style>
        .loader-container {{ min-height: 70vh; display:flex; flex-direction:column;
          justify-content:center; align-items:center; text-align:center; }}
        .loader {{ width:56px; height:56px; border:6px solid #1e293b;
          border-top-color:#38bdf8; border-radius:50%; animation:spin .9s linear infinite; }}
        @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
        .loading-text {{ margin-top:20px; font-size:1.25rem; font-weight:700; color:#f8fafc; }}
        .loading-description {{ margin-top:6px; color:#94a3b8; font-size:.95rem; }}
        </style>
        <div class="loader-container">
          <div class="loader" role="progressbar" aria-label="Loading"></div>
          <div class="loading-text">Loading Flow Atlas</div>
          <div class="loading-description">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return placeholder


if __name__ == "__main__":
    st.set_page_config(page_title="Loading", page_icon="◉", layout="wide")
    show_loading()
