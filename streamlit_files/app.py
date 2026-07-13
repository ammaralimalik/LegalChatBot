import logging
import os
import sys

import streamlit as st

# Streamlit's file watcher walks every imported module (including all of
# transformers' image processors), which lazily triggers torchvision imports we
# don't have or need. The failed imports are harmless but spam WARNING-level
# tracebacks on startup, so quiet just that watcher logger. Hot-reload still works.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import database
import llm_connection
from rate_limits import (
    SESSION_TOKEN_LIMIT,
    DailyTokenLimiter,
    SessionTokenLimiter,
    estimate_tokens,
)
from rag_metrics_view import render_rag_metrics

sys.modules["torch._classes"].__path__ = []

st.set_page_config(
    page_title="Legal Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }

    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 60%, #312e81 100%);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .main-header h1 { margin: 0; font-size: 1.75rem; color: #f8fafc; }
    .main-header p { margin: 0.25rem 0 0 0; color: #cbd5e1; }

    /* ---- Chat window ---------------------------------------------------- */

    /* Center the conversation into a comfortable reading column */
    [data-testid="stChatMessage"] {
        max-width: 860px;
        margin: 0 auto;
        background: transparent;
        padding: 0.3rem 0;
        gap: 0.65rem;
    }

    /* No bubbles — every message blends straight into the page background */
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0.2rem 0 !important;
        max-width: 100%;
        line-height: 1.55;
        color: #e2e8f0;
    }

    /* Docked input — turn it into a floating, centered pill */
    [data-testid="stChatInput"] {
        max-width: 860px;
        margin: 0 auto;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 4px 20px rgba(2, 6, 23, 0.35);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #6366f1;
    }

    /* Welcome / empty state */
    .chat-welcome {
        max-width: 860px;
        margin: 1rem auto 1.25rem auto;
        text-align: center;
        padding: 1.5rem 1.25rem;
        background: linear-gradient(180deg, rgba(49, 46, 129, 0.25), rgba(15, 23, 42, 0.15));
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 16px;
    }
    .chat-welcome .icon { font-size: 2.4rem; }
    .chat-welcome h3 { margin: 0.4rem 0 0.35rem 0; color: #e0e7ff; }
    .chat-welcome p { margin: 0; color: #c7d2fe; font-size: 0.95rem; }
    .chat-suggest-label {
        max-width: 860px;
        margin: 0 auto 0.4rem auto;
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    </style>
    <div class="main-header">
        <h1>⚖️ Pakistani Legal Assistant</h1>
        <p>RAG-powered chat over Contract Act, CPC, Constitution, and Penal Code.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 RAG Metrics"])

EXAMPLE_PROMPTS = [
    "What are the essentials of a valid contract under the Contract Act?",
    "What fundamental rights does the Constitution guarantee?",
    "How is theft defined and punished under the Penal Code?",
    "What is the procedure for filing a civil suit under the CPC?",
]


def render_welcome() -> None:
    st.markdown(
        """
        <div class="chat-welcome">
            <div class="icon">⚖️</div>
            <h3>How can I help with Pakistani law today?</h3>
            <p>Ask about the Contract Act, Civil Procedure Code, Constitution, or Penal Code.
               Answers are grounded in the source documents.</p>
        </div>
        <div class="chat-suggest-label">Try asking</div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, example in enumerate(EXAMPLE_PROMPTS):
        with columns[index % 2]:
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_prompt = example
                st.rerun()


def answer_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            context = database.get_context(prompt)

        full_prompt = f"""You are a helpful legal assistant. Use the context below to answer the question.

Context:
{context}

Question: {prompt}
Answer:"""

        session_limiter = SessionTokenLimiter()
        daily_limiter = DailyTokenLimiter()
        prompt_tokens = estimate_tokens(full_prompt)

        session_check = session_limiter.check_can_use(st.session_state, prompt_tokens)
        if not session_check.allowed:
            st.error(session_check.message)
            return

        daily_check = daily_limiter.check_can_use(prompt_tokens)
        if not daily_check.allowed:
            st.error(daily_check.message)
            return

        full_response = ""
        response_container = st.empty()
        truncated = False
        session_used = session_limiter.used(st.session_state)
        daily_used = daily_limiter.used_today()

        for token in llm_connection.query_model(full_prompt):
            full_response += token
            message_tokens = prompt_tokens + estimate_tokens(full_response)
            if (
                session_used + message_tokens > SESSION_TOKEN_LIMIT
                or daily_used + message_tokens > daily_limiter.daily_limit
            ):
                truncated = True
                break
            response_container.markdown(full_response + "▌")

        tokens_used = prompt_tokens + estimate_tokens(full_response)

        if truncated:
            full_response += "\n\n*[Response truncated: token limit reached.]*"
            if session_used + tokens_used > SESSION_TOKEN_LIMIT:
                st.warning(
                    f"Stopped — session token limit is {SESSION_TOKEN_LIMIT:,} tokens per visit."
                )
            else:
                st.warning(
                    "Stopped — global daily token limit reached (shared across all users)."
                )

        response_container.markdown(full_response)
        session_limiter.record(st.session_state, tokens_used)
        daily_limiter.record(tokens_used)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


def render_chat() -> None:
    st.session_state.setdefault("messages", [])
    pending_prompt = st.session_state.pop("pending_prompt", None)

    header_col, clear_col = st.columns([6, 1])
    with clear_col:
        if st.button(
            "🗑️ Clear",
            use_container_width=True,
            disabled=not st.session_state.messages,
        ):
            st.session_state.messages = []
            st.rerun()

    # Render the whole conversation into a scrollable area declared *before* the
    # input. This keeps the input docked below the messages, and the fixed-height
    # container auto-scrolls to the newest content as tokens stream in.
    message_area = st.container(height=560, border=False)
    with message_area:
        if not st.session_state.messages and not pending_prompt:
            render_welcome()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a legal question...") or pending_prompt
    if prompt:
        with message_area:
            answer_prompt(prompt)


with chat_tab:
    render_chat()

with metrics_tab:
    render_rag_metrics()
