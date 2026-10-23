import json

import httpx
import pytest

from core.llm.ollama_client import (
    OllamaResponseError,
    OllamaUnavailableError,
    ollama_chat_json,
)


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad response",
                request=httpx.Request("POST", "http://localhost:11434/api/chat"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self.content


def test_ollama_client_parses_json(monkeypatch):
    import core.llm.ollama_client as client

    monkeypatch.setattr(client.httpx, "post", lambda *args, **kwargs: FakeResponse({
        "message": {"content": json.dumps({"intent": "find hotel"})}
    }))
    monkeypatch.setattr(client, "get_settings", lambda: type("Settings", (), {
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 30,
    })())

    result = ollama_chat_json("system", "user")
    assert result == {"intent": "find hotel"}


def test_ollama_client_surfaces_connection_error(monkeypatch):
    import core.llm.ollama_client as client

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(client.httpx, "post", fail)
    monkeypatch.setattr(client, "get_settings", lambda: type("Settings", (), {
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 30,
    })())

    with pytest.raises(OllamaUnavailableError):
        ollama_chat_json("system", "user")


def test_ollama_client_rejects_invalid_json(monkeypatch):
    import core.llm.ollama_client as client

    monkeypatch.setattr(client.httpx, "post", lambda *args, **kwargs: FakeResponse({
        "message": {"content": "not json"}
    }))
    monkeypatch.setattr(client, "get_settings", lambda: type("Settings", (), {
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 30,
    })())

    with pytest.raises(OllamaResponseError):
        ollama_chat_json("system", "user")
