import logging
import os
import sys

import streamlit as st

# Streamlit's file watcher walks every imported module (including all of
# transformers' image processors), which lazily triggers torchvision imports we
# don't have or need. The failed imports are harmless but spam WARNING-level
# tracebacks on startup, so quiet just that watcher logger. Hot-reload still works.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Make imports work regardless of how the script is launched: `streamlit run`
# adds the script directory to sys.path but other entry points (tests, IDEs)
# do not, and project modules live one level up.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (_APP_DIR, os.path.dirname(_APP_DIR)):
    if _path not in sys.path:
        sys.path.append(_path)
import database
import llm_connection
from rate_limits import (
    DAILY_TOKEN_LIMIT,
    RAG_METRICS_SESSION_LIMIT,
    SESSION_TOKEN_LIMIT,
    DailyTokenLimiter,
    RagMetricsSessionLimiter,
    SessionTokenLimiter,
    estimate_tokens,
)
from rag_metrics_view import render_rag_metrics

# Same watcher problem as above: it walks torch._classes, which has no real
# __path__ and raises. Blanking it is safe, but only if torch actually got
# imported — indexing sys.modules directly would KeyError otherwise.
_torch_classes = sys.modules.get("torch._classes")
if _torch_classes is not None:
    _torch_classes.__path__ = []

st.set_page_config(
    page_title="Pakistani Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORPUS = [
    ("📜", "Constitution of Pakistan", "Fundamental rights & state structure"),
    ("🤝", "Contract Act, 1872", "Formation, consent, breach, damages"),
    ("🧑‍⚖️", "Pakistan Penal Code", "Offences and punishments"),
    ("🏛️", "Civil Procedure Code, 1908", "Suits, appeals, execution"),
]

EXAMPLE_PROMPTS = [
    ("🤝", "Contract Act", "What are the essentials of a valid contract?"),
    ("📜", "Constitution", "What safeguards apply when a person is arrested?"),
    ("🧑‍⚖️", "Penal Code", "How is theft defined and punished?"),
    ("🏛️", "CPC", "When can a court grant a temporary injunction?"),
]

st.markdown(
    """
    <style>
    /* ---- Global frame ---------------------------------------------------- */
    .block-container { padding-top: 1.2rem; padding-bottom: 0.75rem; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ---- Hero ------------------------------------------------------------ */
    .hero {
        position: relative;
        background:
            radial-gradient(ellipse 80% 120% at 85% -10%, rgba(129,140,248,0.28), transparent 55%),
            radial-gradient(ellipse 60% 120% at 10% -20%, rgba(56,189,248,0.16), transparent 50%),
            linear-gradient(135deg, #101a33 0%, #1a1f45 55%, #0b1120 100%);
        border: 1px solid rgba(129, 140, 248, 0.28);
        border-radius: 18px;
        padding: 1.35rem 1.6rem 1.2rem 1.6rem;
        margin-bottom: 0.9rem;
        overflow: hidden;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.7rem;
        letter-spacing: -0.01em;
        background: linear-gradient(90deg, #f8fafc 20%, #c7d2fe 80%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero p { margin: 0.3rem 0 0.7rem 0; color: #a5b4d4; font-size: 0.95rem; }
    .hero .chips { display: flex; gap: 0.45rem; flex-wrap: wrap; }
    .hero .chip {
        font-size: 0.78rem;
        color: #c7d2fe;
        background: rgba(99, 102, 241, 0.14);
        border: 1px solid rgba(129, 140, 248, 0.35);
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        white-space: nowrap;
    }

    /* ---- Tabs as pills --------------------------------------------------- */
    div[data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(22, 33, 58, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 0.3rem;
        width: fit-content;
    }
    button[data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 0.35rem 1.1rem !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.85), rgba(79,70,229,0.85)) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #f8fafc !important; }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none; }

    /* ---- Chat window ------------------------------------------------------ */
    [data-testid="stChatMessage"] {
        max-width: 880px;
        margin: 0 auto;
        background: transparent;
        padding: 0.3rem 0;
        gap: 0.7rem;
    }
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        padding: 0.2rem 0 !important;
        max-width: 100%;
        line-height: 1.6;
        color: #e2e8f0;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(2, 6, 23, 0.4);
    }

    [data-testid="stChatInput"] {
        max-width: 880px;
        margin: 0 auto;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        background: rgba(22, 33, 58, 0.75);
        box-shadow: 0 8px 28px rgba(2, 6, 23, 0.45);
        backdrop-filter: blur(6px);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #818cf8;
        box-shadow: 0 8px 28px rgba(79, 70, 229, 0.25);
    }

    /* ---- Welcome / empty state ------------------------------------------- */
    .chat-welcome {
        max-width: 880px;
        margin: 1.4rem auto 1.1rem auto;
        text-align: center;
        padding: 1.9rem 1.4rem 1.5rem 1.4rem;
        background:
            radial-gradient(ellipse 60% 100% at 50% -20%, rgba(129,140,248,0.18), transparent 60%),
            rgba(22, 33, 58, 0.45);
        border: 1px solid rgba(129, 140, 248, 0.22);
        border-radius: 18px;
    }
    .chat-welcome .icon { font-size: 2.6rem; filter: drop-shadow(0 4px 14px rgba(129,140,248,0.45)); }
    .chat-welcome h3 { margin: 0.5rem 0 0.3rem 0; color: #e0e7ff; }
    .chat-welcome p { margin: 0; color: #a5b4d4; font-size: 0.93rem; }
    .chat-suggest-label {
        max-width: 880px;
        margin: 0 auto 0.5rem auto;
        color: #8ea0c0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Example prompt cards (scoped by element key so other buttons keep their look) */
    div[class*="st-key-example_"] button {
        background: rgba(22, 33, 58, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 13px !important;
        padding: 0.7rem 0.9rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        color: #cbd5e1 !important;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    div[class*="st-key-example_"] button:hover {
        border-color: #818cf8 !important;
        color: #e0e7ff !important;
        transform: translateY(-1px);
    }

    /* ---- Sources expander -------------------------------------------------- */
    [data-testid="stExpander"] {
        max-width: 880px;
        margin: 0 auto;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        background: rgba(22, 33, 58, 0.4);
    }
    [data-testid="stExpander"] summary { font-size: 0.85rem; color: #a5b4d4; }
    .source-row { font-size: 0.85rem; color: #cbd5e1; padding: 0.12rem 0; }
    .source-row .src-book { color: #a5b4fc; font-weight: 600; }
    .source-row .src-quote { color: #8ea0c0; }

    /* ---- Sidebar ----------------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1730 0%, #0b1120 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    .sb-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.2rem; }
    .sb-brand .logo {
        font-size: 1.5rem;
        background: linear-gradient(135deg, rgba(99,102,241,0.35), rgba(79,70,229,0.15));
        border: 1px solid rgba(129, 140, 248, 0.4);
        border-radius: 12px;
        padding: 0.28rem 0.5rem;
    }
    .sb-brand h2 { margin: 0; font-size: 1.06rem; color: #e0e7ff; letter-spacing: -0.01em; }
    .sb-brand p { margin: 0; font-size: 0.74rem; color: #8ea0c0; }
    .sb-section {
        color: #8ea0c0;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0.9rem 0 0.35rem 0;
    }
    .corpus-item {
        display: flex; gap: 0.55rem; align-items: flex-start;
        background: rgba(22, 33, 58, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 11px;
        padding: 0.5rem 0.6rem;
        margin-bottom: 0.4rem;
    }
    .corpus-item .ci-icon { font-size: 1.05rem; margin-top: 0.05rem; }
    .corpus-item .ci-name { color: #dbe3f4; font-size: 0.84rem; font-weight: 600; margin: 0; }
    .corpus-item .ci-desc { color: #8ea0c0; font-size: 0.73rem; margin: 0; }
    .sb-disclaimer {
        margin-top: 0.9rem;
        font-size: 0.72rem;
        color: #64748b;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        padding-top: 0.7rem;
        line-height: 1.45;
    }
    [data-testid="stSidebar"] [data-testid="stProgressBar"] > div > div { height: 0.4rem; }

    /* ---- Scrollbar polish --------------------------------------------------- */
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.35); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(129, 140, 248, 0.55); }
    ::-webkit-scrollbar-track { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> None:
    session_limiter = SessionTokenLimiter()
    daily_limiter = DailyTokenLimiter()
    rag_limiter = RagMetricsSessionLimiter()

    with st.sidebar:
        st.markdown(
            """
            <div class="sb-brand">
                <div class="logo">⚖️</div>
                <div>
                    <h2>Pakistani Legal Assistant</h2>
                    <p>RAG over four core statutes</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-section">Usage</div>', unsafe_allow_html=True)
        session_used = session_limiter.used(st.session_state)
        st.progress(
            min(session_used / SESSION_TOKEN_LIMIT, 1.0),
            text=f"Session tokens · {session_used:,} / {SESSION_TOKEN_LIMIT:,}",
        )
        daily_used = daily_limiter.used_today()
        st.progress(
            min(daily_used / DAILY_TOKEN_LIMIT, 1.0),
            text=f"Daily tokens (all users) · {daily_used:,} / {DAILY_TOKEN_LIMIT:,}",
        )
        runs_used = rag_limiter.runs_used(st.session_state)
        st.progress(
            min(runs_used / RAG_METRICS_SESSION_LIMIT, 1.0),
            text=f"Eval runs · {runs_used} / {RAG_METRICS_SESSION_LIMIT}",
        )

        st.markdown('<div class="sb-section">Corpus</div>', unsafe_allow_html=True)
        for icon, name, desc in CORPUS:
            st.markdown(
                f"""
                <div class="corpus-item">
                    <div class="ci-icon">{icon}</div>
                    <div>
                        <p class="ci-name">{name}</p>
                        <p class="ci-desc">{desc}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sb-section">Conversation</div>', unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear chat",
            use_container_width=True,
            disabled=not st.session_state.get("messages"),
        ):
            st.session_state.messages = []
            st.rerun()

        st.markdown(
            """
            <div class="sb-disclaimer">
                Answers are generated from the source documents for research
                purposes and are not legal advice. Verify citations against
                the official text before relying on them.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    chips = "".join(
        f'<span class="chip">{icon}&nbsp;{name}</span>' for icon, name, _ in CORPUS
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1>Pakistani Legal Assistant</h1>
            <p>Grounded answers with section-level citations, streamed as they are written.</p>
            <div class="chips">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div class="chat-welcome">
            <div class="icon">⚖️</div>
            <h3>How can I help with Pakistani law today?</h3>
            <p>Ask about the Contract Act, Civil Procedure Code, Constitution, or Penal Code.
               Every answer cites the retrieved sections it is based on.</p>
        </div>
        <div class="chat-suggest-label">Try asking</div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, (icon, statute, example) in enumerate(EXAMPLE_PROMPTS):
        with columns[index % 2]:
            if st.button(
                f"{icon} **{statute}** — {example}",
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = example
                st.rerun()


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    """Distill retrieved chunks into unique, display-ready citations."""
    sources: list[dict] = []
    seen: set[tuple] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        book = str(metadata.get("source", "Unknown source")).removesuffix(".pdf")
        heading = str(metadata.get("heading", "")).strip()
        key = (book, heading or metadata.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        quote = " ".join(chunk.get("document", "")[:110].split())
        sources.append({"book": book, "heading": heading, "quote": quote})
    return sources


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            heading = f" §{source['heading'].rstrip('.')}" if source.get("heading") else ""
            st.markdown(
                f"""
                <div class="source-row">
                    <span class="src-book">{source['book']}{heading}</span><br>
                    <span class="src-quote">“{source['quote']}…”</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def answer_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the statutes..."):
            chunks = database.get_store().retrieve(prompt)
            context = database.context_from_chunks(chunks)
        sources = _sources_from_chunks(chunks)

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
        failed = False
        session_used = session_limiter.used(st.session_state)
        daily_used = daily_limiter.used_today()

        stream = llm_connection.query_model(full_prompt)
        try:
            for token in stream:
                full_response += token
                message_tokens = prompt_tokens + estimate_tokens(full_response)
                if (
                    session_used + message_tokens > SESSION_TOKEN_LIMIT
                    or daily_used + message_tokens > daily_limiter.daily_limit
                ):
                    truncated = True
                    break
                response_container.markdown(full_response + "▌")
        except Exception as exc:
            # Keep whatever streamed before the failure; the partial answer is
            # still useful and its tokens were genuinely consumed.
            failed = True
            st.error(f"Generation interrupted: {exc}")
        finally:
            stream.close()

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
        elif failed and full_response:
            full_response += "\n\n*[Response interrupted before completion.]*"

        if not full_response and failed:
            return

        response_container.markdown(full_response)
        render_sources(sources)
        session_limiter.record(st.session_state, tokens_used)
        daily_limiter.record(tokens_used)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )


def render_chat() -> None:
    # Render the whole conversation into a scrollable area declared *before* the
    # input. This keeps the input docked below the messages, and the fixed-height
    # container auto-scrolls to the newest content as tokens stream in.
    pending_prompt = st.session_state.pop("pending_prompt", None)
    message_area = st.container(height=560, border=False)
    with message_area:
        if not st.session_state.messages and not pending_prompt:
            render_welcome()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask a legal question...") or pending_prompt
    if prompt:
        with message_area:
            answer_prompt(prompt)


st.session_state.setdefault("messages", [])
render_sidebar()
render_hero()

chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 RAG Metrics"])

with chat_tab:
    render_chat()

with metrics_tab:
    render_rag_metrics()
