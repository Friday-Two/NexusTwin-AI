"""Ollama client using the local Ollama HTTP API.

No cloud provider or API key is used. Failures are surfaced rather than replaced
with a fabricated answer, so callers can distinguish real LLM output from rules.
"""
import json
from typing import Any

import httpx

from configs.settings import get_settings


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured local Ollama server cannot be reached."""


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns invalid or unusable model output."""


def ollama_chat_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    settings = get_settings()
    base_url = settings.llm_base_url.rstrip("/")
    base_url = base_url.removesuffix("/v1")
    endpoint = f"{base_url}/api/chat"
    payload = {
        "model": settings.llm_model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1},
    }

    try:
        response = httpx.post(
            endpoint,
            json=payload,
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
    except (httpx.HTTPError, OSError) as exc:
        raise OllamaUnavailableError(
            "Could not reach the local Ollama server. Start Ollama and pull the configured model."
        ) from exc

    try:
        body = response.json()
        content = body["message"]["content"]
        result = json.loads(content)
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaResponseError(
            "Ollama returned a response that was not valid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise OllamaResponseError("Ollama JSON response must be an object.")
    return result
