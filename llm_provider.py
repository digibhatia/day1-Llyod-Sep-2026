"""
llm_provider.py — one function that calls "whichever model key is set",
Anthropic Claude or OpenAI, from the same agent loop in app.py.

WHY THIS FILE EXISTS
Anthropic and OpenAI both let a model call tools, but they describe tools
differently, return tool calls differently, and expect tool *results* fed
back differently. If app.py spoke either SDK directly, the whole solution
would be locked to one provider. Every function below takes the same
plain-Python shapes in and out; the provider-specific translation happens
only inside this file. This is the same idea Module 2 onward calls
"semantic conventions" — one shared shape, many backends.

PUBLIC SURFACE (this is all app.py needs to know):
    get_provider() -> "anthropic" | "openai"
    chat(messages, tools=None) -> AgentTurn

messages is a list of plain dicts, each one of:
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [...]}   (as returned by chat())
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

AgentTurn (see below) always has: .text, .tool_calls, .stop_reason, .raw
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory, if one exists

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Force a provider explicitly (useful for the "run it twice" exercise in the
# README), or leave unset and let get_provider() auto-detect from whichever
# key is present.
FORCED_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AgentTurn:
    """One normalized model response, regardless of provider."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    raw: object = None  # the original SDK response, kept for the Day 2/3 labs


def get_provider() -> str:
    """
    Decide which provider to use for this run. Explicit LLM_PROVIDER wins;
    otherwise we prefer Anthropic if its key is present, then OpenAI. This
    is the one place a whole class of "wrong environment" mistakes gets
    caught early, with a clear message instead of a buried SDK error.
    """
    if FORCED_PROVIDER in ("anthropic", "openai"):
        return FORCED_PROVIDER
    if FORCED_PROVIDER:
        raise ValueError(
            f"LLM_PROVIDER={FORCED_PROVIDER!r} is not recognised. Use 'anthropic' or 'openai', "
            "or leave it unset in .env to auto-detect from whichever key is present."
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No model key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file "
        "(copy .env.example to .env and fill in one or both). See README.md, Step 3."
    )


def chat(messages: list[dict], tools: list[dict] | None = None, system: str | None = None) -> AgentTurn:
    """Route to the active provider. This is the only function app.py calls."""
    provider = get_provider()
    if provider == "anthropic":
        return _chat_anthropic(messages, tools, system)
    return _chat_openai(messages, tools, system)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
def _chat_anthropic(messages, tools, system):
    import anthropic  # imported lazily so the other provider's users need not install this SDK

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment itself

    anthropic_messages = []
    for m in messages:
        if m["role"] == "tool":
            # Anthropic expects a tool result as a "user" turn containing a
            # tool_result content block keyed by the original tool_use id.
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }],
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            # Re-emit exactly what the model asked for, so it can see its own
            # prior tool_use blocks when the tool_result comes back to it.
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({
                    "type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"],
                })
            anthropic_messages.append({"role": "assistant", "content": content})
        else:
            anthropic_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = {}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=anthropic_messages,
        **kwargs,
    )

    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

    return AgentTurn(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        raw=response,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def _chat_openai(messages, tools, system):
    import openai  # imported lazily, same reason as above

    client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment itself

    openai_messages = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for m in messages:
        if m["role"] == "tool":
            openai_messages.append({
                "role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"],
            })
        elif m["role"] == "assistant" and m.get("tool_calls"):
            openai_messages.append({
                "role": "assistant",
                "content": m.get("content") or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": _json_dumps(tc["arguments"])},
                    }
                    for tc in m["tool_calls"]
                ],
            })
        else:
            openai_messages.append({"role": m["role"], "content": m["content"]})

    kwargs = {}
    if tools:
        kwargs["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools
        ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=openai_messages,
        **kwargs,
    )

    choice = response.choices[0]
    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=_json_loads(tc.function.arguments)))

    return AgentTurn(
        text=choice.message.content or "",
        tool_calls=tool_calls,
        stop_reason=choice.finish_reason,
        raw=response,
    )


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


def _json_loads(s: str) -> dict:
    import json
    return json.loads(s)
