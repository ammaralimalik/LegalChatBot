import json
import logging
import os
import random
import time

import requests

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

# Free OpenRouter models rate-limit hard (HTTP 429), especially during eval runs
# that fire many requests back to back. Retry with exponential backoff, honoring
# the server's Retry-After header when present.
MAX_RETRIES = int(os.environ.get("OPENROUTER_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("OPENROUTER_RETRY_BASE_DELAY", "5"))
RETRY_MAX_DELAY = float(os.environ.get("OPENROUTER_RETRY_MAX_DELAY", "60"))
SYSTEM_PROMPT = (
    "You are a helpful legal assistant meant to help attornies find meaningful "
    "strategies and knowledge related to their cases. Always assume you are speaking "
    "to a professional and answer each and every question they may have. Always keep "
    "legal questions in the context of Pakistan"
)


def _get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER in .env or OPENROUTER_API_KEY in the environment."
        )
    return api_key.strip("'\"")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _build_messages(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


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


def _retry_delay(response, attempt: int) -> float:
    """Seconds to wait before the next attempt, preferring the Retry-After header."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    backoff = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
    return backoff + random.uniform(0, 1)  # jitter to avoid lockstep retries


def _post(prompt: str, *, stream: bool):
    """POST to OpenRouter, retrying with backoff on 429. Returns the OK response."""
    payload = {
        "model": MODEL,
        "messages": _build_messages(prompt),
        "temperature": 0.85,
        "reasoning": {"enabled": True},
    }
    if stream:
        payload["stream"] = True

    for attempt in range(MAX_RETRIES + 1):
        response = requests.post(
            API_URL,
            headers=_headers(),
            json=payload,
            stream=stream,
            timeout=(10, 300),
        )
        if response.status_code == 429 and attempt < MAX_RETRIES:
            delay = _retry_delay(response, attempt)
            logger.warning(
                "OpenRouter rate limited (429); retrying in %.1fs "
                "(attempt %d/%d).",
                delay,
                attempt + 1,
                MAX_RETRIES,
            )
            response.close()
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response

    # Loop only exits via return or raise_for_status; this satisfies type checkers.
    raise RuntimeError("Exhausted OpenRouter retries without a response.")


def query_model(prompt):
    """Stream response tokens from OpenRouter."""
    response = _post(prompt, stream=True)
    yield from _iter_stream_tokens(response)


def query_model_complete(prompt: str) -> str:
    """Collect the full response as a single string."""
    response = _post(prompt, stream=False)
    message = response.json()["choices"][0]["message"]
    return (message.get("content") or "").strip()
