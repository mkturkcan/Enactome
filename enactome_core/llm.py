"""Optional Claude API integration.

The engine is fully usable without an API key: every experiment and data screen runs
from the /tools endpoints directly. This module adds an LLM planning layer on top: given
a natural-language request and the engine's own tool manifest, it asks Claude which
endpoints to call. The key is read from the ANTHROPIC_API_KEY environment variable and is
never stored, logged, or written to any file.
"""
from __future__ import annotations
import os
from typing import Any


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def default_model() -> str:
    return os.environ.get("ENACTOME_CLAUDE_MODEL", "claude-sonnet-4-5")


def plan(question: str, tools: list[dict], model: str | None = None,
         max_tokens: int = 1024) -> dict[str, Any]:
    """Ask Claude which engine tools to call for a request.

    Returns {"text": ...} on success, or {"error": ...} if the key or SDK is missing.
    Requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment.
    """
    if not api_key_present():
        return {"error": "ANTHROPIC_API_KEY not set; the engine runs without it, this endpoint is optional"}
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic SDK not installed; run: pip install anthropic"}
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    manifest = "\n".join(f"- {t['name']} ({t['method']} {t['path']}): {t['description']}" for t in tools)
    system = ("You plan calls to the Enactome connectome-simulation engine. Given a request and the "
              "available tools, name the tools to call in order and the parameters for each. Be concise.")
    msg = client.messages.create(
        model=model or default_model(), max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": f"Tools:\n{manifest}\n\nRequest: {question}"}])
    return {"text": "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"),
            "model": msg.model}
