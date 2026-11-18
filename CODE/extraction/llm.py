"""Minimal OpenAI-compatible chat client with an on-disk cache.

Works with local Ollama and the ASU API (both speak the OpenAI API). Every
complete answer is cached by (model, messages), so reruns are free and results
are reproducible for the report. Answers that were cut off or are not valid
JSON are never cached: chat_json raises Truncated so the caller can split its
batch and ask again.
"""
import hashlib
import json
import time

import requests

from . import config

_CACHE_DIR = config.CACHE / "llm"

MAX_TOKENS = 3000
RETRIES = 6
FAILURES = []  # (system, user, error) for requests that never got an answer


class Truncated(Exception):
    """The reply was cut off at max_tokens or is not valid JSON."""


def _cache_path(payload):
    key = {k: v for k, v in payload.items() if k != "max_tokens"}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return _CACHE_DIR / digest[:2] / f"{digest}.json"


def _cached(path):
    """A cached answer, or None. Old cache files may hold a cut-off reply saved as {};
    those count as missing so they are asked again."""
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    if entry["parsed"] == {} and entry.get("raw", "").strip() not in ("", "{}"):
        return None
    return entry["parsed"]


def _wait(response, attempt):
    """Seconds to wait before the next try. Busy servers (429, 503) get long waits."""
    if response is not None and response.status_code in (429, 503):
        retry_after = response.headers.get("Retry-After", "")
        return float(retry_after) if retry_after.isdigit() else min(60, 15 * 2 ** attempt)
    return min(60, 5 * 2 ** attempt)


def chat_json(system, user, model=None):
    """Send one chat request in JSON mode and return the parsed object.

    Raises Truncated when the reply was cut off or is not valid JSON. Returns {}
    when the request keeps failing; those are recorded in FAILURES.
    """
    payload = {
        "model": model or config.LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
    }
    path = _cache_path(payload)
    cached = _cached(path)
    if cached is not None:
        return cached

    last_err = None
    for attempt in range(RETRIES):
        r = None
        try:
            r = requests.post(
                f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                json=payload, timeout=600)
            r.raise_for_status()
            choice = r.json()["choices"][0]
            content = choice["message"]["content"] or ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                raise Truncated(f"invalid JSON ({len(content)} chars, finish_reason={choice.get('finish_reason')})")
            if choice.get("finish_reason") == "length":
                raise Truncated(f"cut off at max_tokens ({len(content)} chars)")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"payload": payload, "raw": content, "parsed": parsed}))
            return parsed
        except requests.RequestException as e:
            last_err = e
            time.sleep(_wait(r, attempt))
    FAILURES.append((system[:60], user[:200], str(last_err)))
    return {}
