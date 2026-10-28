from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import __version__
from .errors import ChatError

DEFAULT_PROVIDER = "kitsune"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible chat completions endpoint."""

    name: str
    base_url: str
    api_key_env: str


PROVIDERS: dict[str, Provider] = {
    "kitsune": Provider(
        name="kitsune",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="KITSUNE_API_KEY",
    ),
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ChatError(
            "unknown provider: {} (known: {})".format(
                name, ", ".join(sorted(PROVIDERS))
            ),
            code="unknown_provider",
        )
    return PROVIDERS[name]


def _parse_response(data: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ("id", "model", "provider", "created", "usage"):
        if data.get(key) is not None:
            meta[key] = data[key]
    choices = data.get("choices") or []
    if not choices:
        return {"content": "", "meta": meta}
    choice = choices[0]
    content = (choice.get("message") or {}).get("content") or ""
    if choice.get("finish_reason") is not None:
        meta["finish_reason"] = choice["finish_reason"]
    return {"content": content, "meta": meta}


def call_model(
    provider: Provider,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    base_url: str,
    max_tokens: int,
    timeout: int,
    retry_delays: tuple,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Returns ``{"content": str, "meta": dict}`` or ``{"content": None, "error": str}``."""
    api_key = os.environ.get(provider.api_key_env)
    if not api_key:
        raise ChatError(
            "{} is not set".format(provider.api_key_env),
            code="missing_api_key",
        )

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model_id, "messages": messages, "max_tokens": max_tokens},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "User-Agent": "kitsune/" + __version__,
    }

    last_error = "request_error"
    for attempt in range(len(retry_delays) + 1):
        request = urllib.request.Request(
            url, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            return _parse_response(json.loads(raw))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            last_error = "http_{}: {}".format(exc.code, raw.strip()[:1000])
            if exc.code not in RETRYABLE_STATUS or attempt == len(retry_delays):
                return {"content": None, "error": last_error}
        except Exception as exc:  # timeouts, DNS errors, bad JSON, ...
            last_error = "{}: {}".format(type(exc).__name__, exc)
            if attempt == len(retry_delays):
                return {"content": None, "error": last_error}
        sleep(retry_delays[attempt])

    return {"content": None, "error": last_error}
