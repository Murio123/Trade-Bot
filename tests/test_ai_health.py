"""H4 fix: the AI layer must fail loudly, not degrade silently."""
from __future__ import annotations

import asyncio

from ai import claude


def test_healthcheck_reports_missing_key(monkeypatch):
    monkeypatch.setattr(claude, "_client", None)
    monkeypatch.setattr(claude, "_HAS_ANTHROPIC", True)
    monkeypatch.setattr("config.ANTHROPIC_API_KEY", None)

    result = asyncio.run(claude.healthcheck())
    assert result["ok"] is False
    assert "ANTHROPIC_API_KEY" in result["error"]


def test_healthcheck_reports_api_error(monkeypatch):
    class _FailingMessages:
        async def create(self, **kwargs):
            raise RuntimeError("model not found")

    class _FakeClient:
        messages = _FailingMessages()

    monkeypatch.setattr(claude, "_client", _FakeClient())

    result = asyncio.run(claude.healthcheck())
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]
    assert "model not found" in result["error"]


def test_healthcheck_ok(monkeypatch):
    class _OkMessages:
        async def create(self, **kwargs):
            assert kwargs["max_tokens"] == 1  # the ping must stay cheap
            return object()

    class _FakeClient:
        messages = _OkMessages()

    monkeypatch.setattr(claude, "_client", _FakeClient())

    result = asyncio.run(claude.healthcheck())
    assert result == {"ok": True, "error": None}
