"""
Tests for backend/claude.py.

Key safety property under test: BOTH integrations (preflight + chat) must
FAIL OPEN when there's no API key configured, so the deployment doesn't
break just because someone forgot to set a secret.
"""
from __future__ import annotations

from PIL import Image

from backend import claude


def _white_image(size: int = 64) -> Image.Image:
    return Image.new("RGB", (size, size), color="white")


# ---------- is_available ----------

def test_is_available_returns_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert claude.is_available() is False


# ---------- preflight_check ----------

def test_preflight_fails_open_without_key(monkeypatch):
    """No key -> don't block the classifier, just skip the Claude check."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = claude.preflight_check(_white_image())
    assert result["ok"] is True
    assert result["used_claude"] is False
    assert result["category"] is None
    assert result["reason"] is None


def test_preflight_response_shape(monkeypatch):
    """Even on fail-open, response always has the four documented keys."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = claude.preflight_check(_white_image())
    assert set(result.keys()) == {"ok", "category", "reason", "used_claude"}


# ---------- chat_reply ----------

def test_chat_returns_error_without_key(monkeypatch):
    """No key -> ok=False + error message, never a fake reply."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = claude.chat_reply(
        [{"role": "user", "content": "what should I do?"}],
        context={"name": "Melanoma", "code": "mel", "malignancy": "cancerous", "probability": 0.8},
    )
    assert result["ok"] is False
    assert result["reply"] is None
    assert result["error"]


def test_chat_rejects_empty_messages(monkeypatch):
    """No messages -> graceful error, not a 500."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = claude.chat_reply([], context=None)
    assert result["ok"] is False
    assert result["error"]   # could be "Empty message." or "Chat is unavailable..." - either is a graceful failure


def test_chat_history_cap_does_not_crash(monkeypatch):
    """A long history should be truncated internally; no exception."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(50)]
    result = claude.chat_reply(history, None)
    # No key, so it will fail open with error - but it should NOT raise.
    assert result["ok"] is False
    assert result["error"]
