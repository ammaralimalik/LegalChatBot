"""Token and session rate limits for LLM usage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SESSION_TOKEN_LIMIT = 150_000
DAILY_TOKEN_LIMIT = 850_000
RAG_METRICS_SESSION_LIMIT = 3

STATE_PATH = Path(__file__).resolve().parent / ".rate_limit_state.json"
SESSION_TOKENS_KEY = "session_tokens_used"
RAG_METRICS_RUNS_KEY = "rag_metrics_runs"


class RateLimitError(Exception):
    """Raised when a rate limit would be exceeded."""


@dataclass(frozen=True)
class RateLimitStatus:
    allowed: bool
    message: str = ""


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token for English text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_eval_tokens(test_cases) -> int:
    """Rough token estimate for a completed RAG evaluation run."""
    pipeline_tokens = 0
    for case in test_cases:
        for field in (case.input, case.actual_output, case.expected_output):
            if field:
                pipeline_tokens += estimate_tokens(field)
        for chunk in case.retrieval_context or []:
            pipeline_tokens += estimate_tokens(chunk)

    # DeepEval judge calls re-read inputs, outputs, and context per metric.
    return int(pipeline_tokens * 3)


def estimate_eval_run_tokens(golden_count: int, metric_count: int) -> int:
    """Conservative pre-run estimate before evaluation starts."""
    per_golden = 4_000 + (metric_count * 3_000)
    return golden_count * per_golden


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"date": date.today().isoformat(), "tokens_used": 0}

    try:
        with STATE_PATH.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"date": date.today().isoformat(), "tokens_used": 0}

    if state.get("date") != date.today().isoformat():
        return {"date": date.today().isoformat(), "tokens_used": 0}

    state["tokens_used"] = int(state.get("tokens_used", 0))
    return state


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state, handle)


class DailyTokenLimiter:
    """Tracks global token usage across all users; resets at midnight local time."""

    def __init__(self, daily_limit: int = DAILY_TOKEN_LIMIT):
        self.daily_limit = daily_limit

    def used_today(self) -> int:
        return _load_state()["tokens_used"]

    def remaining(self) -> int:
        return max(0, self.daily_limit - self.used_today())

    def check_can_use(self, tokens: int) -> RateLimitStatus:
        if tokens <= 0:
            return RateLimitStatus(True)

        if tokens > self.daily_limit:
            return RateLimitStatus(
                False,
                (
                    f"This request needs about {tokens:,} tokens, which exceeds the "
                    f"global daily limit of {self.daily_limit:,} tokens."
                ),
            )

        remaining = self.remaining()
        if tokens > remaining:
            return RateLimitStatus(
                False,
                (
                    f"Global daily token limit reached (shared across all users). "
                    f"Used {self.used_today():,} of {self.daily_limit:,} tokens today. "
                    f"About {remaining:,} tokens remain until the limit resets tomorrow."
                ),
            )

        return RateLimitStatus(True)

    def record(self, tokens: int) -> None:
        if tokens <= 0:
            return

        state = _load_state()
        state["date"] = date.today().isoformat()
        state["tokens_used"] = min(
            self.daily_limit,
            state["tokens_used"] + tokens,
        )
        _save_state(state)


class SessionTokenLimiter:
    """Enforces a cumulative token cap per Streamlit session."""

    def __init__(self, session_limit: int = SESSION_TOKEN_LIMIT):
        self.session_limit = session_limit

    def used(self, session_state) -> int:
        return int(session_state.get(SESSION_TOKENS_KEY, 0))

    def remaining(self, session_state) -> int:
        return max(0, self.session_limit - self.used(session_state))

    def check_can_use(self, session_state, tokens: int) -> RateLimitStatus:
        if tokens <= 0:
            return RateLimitStatus(True)

        remaining = self.remaining(session_state)
        if tokens > remaining:
            return RateLimitStatus(
                False,
                (
                    f"Session token limit reached. Used {self.used(session_state):,} of "
                    f"{self.session_limit:,} tokens this session. "
                    f"About {remaining:,} tokens remain — refresh the page to start a new session."
                ),
            )

        return RateLimitStatus(True)

    def would_exceed(self, session_state, tokens: int) -> bool:
        return self.used(session_state) + tokens > self.session_limit

    def record(self, session_state, tokens: int) -> None:
        if tokens <= 0:
            return

        session_state[SESSION_TOKENS_KEY] = min(
            self.session_limit,
            self.used(session_state) + tokens,
        )


class RagMetricsSessionLimiter:
    """Limits RAG metric evaluation runs per Streamlit session."""

    def __init__(self, session_limit: int = RAG_METRICS_SESSION_LIMIT):
        self.session_limit = session_limit

    def runs_used(self, session_state) -> int:
        return int(session_state.get(RAG_METRICS_RUNS_KEY, 0))

    def runs_remaining(self, session_state) -> int:
        return max(0, self.session_limit - self.runs_used(session_state))

    def check_can_run(self, session_state) -> RateLimitStatus:
        if self.runs_used(session_state) >= self.session_limit:
            return RateLimitStatus(
                False,
                (
                    f"RAG metrics can only be run {self.session_limit} times per session. "
                    "Refresh the page to start a new session."
                ),
            )
        return RateLimitStatus(True)

    def record_run(self, session_state) -> None:
        session_state[RAG_METRICS_RUNS_KEY] = self.runs_used(session_state) + 1
