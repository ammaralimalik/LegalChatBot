import json
import re

import requests

API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "deepseek-r1-distill-qwen-7b"
SYSTEM_PROMPT = (
    "You are a helpful legal assistant meant to help attornies find meaningful "
    "strategies and knowledge related to their cases. Always assume you are speaking "
    "to a professional and answer each and every question they may have. Always keep "
    "legal questions in the context of Pakistan"
)
THINKING_START = "<think>"
THINKING_END = "</think>"


class _ThinkingFilter:
    """Strip model thinking blocks while streaming partial tokens."""

    def __init__(self):
        self._buffer = ""
        self._in_thinking = False

    def push(self, text: str):
        self._buffer += text

        while True:
            if self._in_thinking:
                end_idx = self._buffer.find(THINKING_END)
                if end_idx == -1:
                    return
                self._buffer = self._buffer[end_idx + len(THINKING_END) :]
                self._in_thinking = False
                continue

            start_idx = self._buffer.find(THINKING_START)
            if start_idx == -1:
                holdback = 0
                for i in range(min(len(self._buffer), len(THINKING_START) - 1), 0, -1):
                    if THINKING_START.startswith(self._buffer[-i:]):
                        holdback = i
                        break

                if holdback < len(self._buffer):
                    out = self._buffer[:-holdback] if holdback else self._buffer
                    self._buffer = self._buffer[-holdback:] if holdback else ""
                    if out:
                        yield out
                return

            if start_idx > 0:
                yield self._buffer[:start_idx]
            self._buffer = self._buffer[start_idx + len(THINKING_START) :]
            self._in_thinking = True

    def flush(self):
        if not self._in_thinking and self._buffer:
            yield self._buffer
            self._buffer = ""


def _iter_stream_tokens(response):
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        payload = line[6:].strip()
        if payload == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue

        content = choices[0].get("delta", {}).get("content")
        if content:
            yield content


def query_model(prompt):
    """Stream response tokens from the local LLM."""
    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.85,
            "stream": True,
        },
        stream=True,
        timeout=(10, 300),
    )
    response.raise_for_status()

    thinking_filter = _ThinkingFilter()
    for token in _iter_stream_tokens(response):
        yield from thinking_filter.push(token)

    yield from thinking_filter.flush()


def query_model_complete(prompt: str) -> str:
    """Collect the full streamed response as a single string."""
    content = "".join(query_model(prompt))
    cleaned = re.sub(
        rf"{re.escape(THINKING_START)}.*?{re.escape(THINKING_END)}",
        "",
        content,
        flags=re.DOTALL,
    )
    return cleaned.strip()
