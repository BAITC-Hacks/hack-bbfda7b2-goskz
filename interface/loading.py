import time

import streamlit as st


loading = st.empty()

loading.markdown(
    """
    <style>
    .loader-container {
        min-height: 70vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .loader {
        width: 56px;
        height: 56px;
        border: 6px solid #e5e7eb;
        border-top-color: #3498db;
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .loading-text {
        margin-top: 20px;
        font-size: 1.25rem;
        font-weight: 600;
    }

    .loading-description {
        margin-top: 6px;
        color: #9ca3af;
        font-size: 0.95rem;
    }

    .loading-details {
        margin-top: 14px;
        color: #9ca3af;
        font-size: 0.9rem;
        text-align: left;
        cursor: pointer;
        position: relative;
    }

    .loading-details summary {
        text-align: center;
        color: #9ca3af;
    }

    .loading-details p {
        position: absolute;
        z-index: 1;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        max-width: 420px;
        width: max-content;
        margin: 8px 0 0;
        padding: 10px 12px;
        border-radius: 6px;
        background: #fff;
        box-shadow: 0 2px 10px rgb(0 0 0 / 12%);
    }
    </style>

    <div class="loader-container">
        <div class="loader" role="progressbar" aria-label="Loading"></div>
        <div class="loading-text">Loading...</div>
        <div class="loading-description">Preparing the app and loading its resources.</div>
        <details class="loading-details">
            <summary>More details</summary>
            <p>The app is getting ready. This demo keeps the loading screen visible
            for five seconds; replace the delay with your setup or data-loading work.</p>
        </details>
    </div>
    """,
    unsafe_allow_html=True,
)

# Replace this demo delay with the work that needs to run during startup.
time.sleep(5)

loading.empty()
st.success("Loaded!")
