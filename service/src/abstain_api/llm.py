"""OpenAI-compatible chat client used by the agent (Ollama /v1, vLLM, Groq, Together, OpenRouter).

Exposes the `.chat(messages, max_new_tokens)` interface the research agent expects. While the agent's
`generate` node runs (abstain.agent.STREAM_TOKENS), tokens are streamed and forwarded to LangGraph's
stream writer, so the API can push them to the browser as they arrive.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from abstain.agent import STREAM_TOKENS

from .settings import Settings

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's <think>...</think> block (an unclosed one means it ran out of tokens)."""
    text = _THINK.sub("", text or "")
    # some templates (DeepSeek-R1 in Ollama) open the block in the prompt: only "</think>" is in the output
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


def visible_text(text: str) -> str:
    """The part of a partial stream that is safe to show: nothing while a <think> block may still be open."""
    t = text.lstrip()
    if not t:
        return ""
    if "<think>".startswith(t[:7]) and "</think>" not in t:  # inside (or starting) a think block
        return ""
    return strip_thinking(text).lstrip() if "<think>" in t else text.lstrip()


class _Msg:
    def __init__(self, content: str):
        self.content = content


class OllamaNativeChat:
    """Minimal client for Ollama's /api/chat with .invoke(messages) / .stream(messages) like a LangChain chat
    model (no extra dependency). Needed because only the native API accepts options such as num_ctx."""

    _ROLES = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}

    def __init__(self, base: str, model: str, options: dict, keep_alive: str, timeout: float, think: bool = False):
        self.base, self.model, self.options, self.keep_alive, self.timeout = base, model, options, keep_alive, timeout
        self.think = think

    def _body(self, msgs, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": self._ROLES.get(getattr(m, "type", "user"), "user"), "content": str(m.content)} for m in msgs
            ],
            "stream": stream,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        if self.think:  # thinking models: the reasoning comes back in its own field, not in the answer
            body["think"] = True
        return body

    def invoke(self, msgs) -> _Msg:
        r = httpx.post(f"{self.base}/api/chat", json=self._body(msgs, False), timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return _Msg((r.json().get("message") or {}).get("content") or "")

    def stream(self, msgs):
        with httpx.stream("POST", f"{self.base}/api/chat", json=self._body(msgs, True), timeout=self.timeout) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(f"Ollama error: {d['error']}")
                text = (d.get("message") or {}).get("content") or ""
                if text:
                    yield _Msg(text)
                if d.get("done"):
                    break


class OpenAICompatChat:
    """One model on one OpenAI-compatible endpoint (default: LLM_BASE_URL / LLM_MODEL)."""

    def __init__(
        self,
        settings: Settings,
        retries: int = 3,
        wait_s: float = 5.0,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "local",
        max_tokens_field: str = "max_tokens",
        reasoning: bool = False,
    ):
        self.s, self.retries, self.wait_s = settings, retries, wait_s
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.provider, self.max_tokens_field, self.reasoning = provider, max_tokens_field, reasoning
        self._models: dict[int, object] = {}
        # local models served by Ollama (not a local vLLM / LM Studio server)
        self.is_ollama = (
            provider == "local"
            and settings.ollama_native
            and (":11434" in self.base_url or "ollama" in self.base_url.lower())
        )

    def _model(self, n: int):
        if n not in self._models:
            from langchain_openai import ChatOpenAI

            kw = {}
            if self.provider == "local" and self.s.llm_seed is not None:
                kw["seed"] = self.s.llm_seed  # deterministic research model
            if self.reasoning:  # reasoning models spend tokens on thinking and reject temperature 0
                n = n + self.s.reasoning_max_tokens  # the thinking budget comes on top of the answer's
            else:
                kw["temperature"] = 0
            headers = (
                {"HTTP-Referer": "https://github.com/cs683/abstention-drift", "X-Title": "Grounded Notebook"}
                if self.provider == "openrouter"
                else None
            )
            if self.is_ollama:
                # Ollama's native API, so the context window can be set (its OpenAI endpoint cannot)
                from .model_registry import ollama_base

                timeout = (
                    max(self.s.llm_timeout_s, self.s.reasoning_timeout_s) if self.reasoning else self.s.llm_timeout_s
                )
                base = (
                    ollama_base(self.s)
                    if self.base_url == self.s.llm_base_url
                    else re.sub(r"/v1/?$", "", self.base_url).rstrip("/")
                )
                options = {"num_ctx": self.s.ollama_num_ctx, "num_predict": n}
                if "temperature" in kw:
                    options["temperature"] = kw["temperature"]
                if "seed" in kw:
                    options["seed"] = kw["seed"]
                self._models[n] = OllamaNativeChat(
                    base, self.model, options, self.s.ollama_keep_alive, timeout, think=self.reasoning
                )
                return self._models[n]
            # "max_tokens" is understood by Ollama, vLLM, Groq, Together, OpenRouter, Anthropic and Gemini;
            # OpenAI's newer models need "max_completion_tokens"
            self._models[n] = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                # thinking first takes minutes on a small GPU
                timeout=max(self.s.llm_timeout_s, self.s.reasoning_timeout_s)
                if self.reasoning
                else self.s.llm_timeout_s,
                max_retries=0,
                extra_body={self.max_tokens_field: n},
                default_headers=headers,
                **kw,
            )
        return self._models[n]

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        cls = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        msgs = [cls[m["role"]](content=m["content"]) for m in messages]
        model = self._model(max_new_tokens or self.s.llm_max_tokens)
        for attempt in range(self.retries):
            try:
                if STREAM_TOKENS.get():
                    return self._stream(model, msgs)
                return strip_thinking(str(model.invoke(msgs).content))
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.wait_s * (attempt + 1))
        return ""

    @staticmethod
    def _stream(model, msgs) -> str:
        from langgraph.config import get_stream_writer

        try:
            write = get_stream_writer()
        except Exception:  # not running inside a graph
            write = None
        parts, shown = [], 0
        for chunk in model.stream(msgs):
            t = chunk.content if isinstance(chunk.content, str) else ""
            if not t:
                continue
            parts.append(t)
            # reasoning models (DeepSeek-R1, Qwen3, ...) think inside <think>...</think>: stream only the answer
            visible = visible_text("".join(parts))
            if write and len(visible) > shown:
                write({"type": "token", "text": visible[shown:]})
                shown = len(visible)
        return strip_thinking("".join(parts))


class RoutingChat:
    """The agent's LLM: sends each call to the model chosen for the current request (REQUEST_MODEL)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._clients: dict[str, OpenAICompatChat] = {}

    def client(self, model_id: str | None = None) -> OpenAICompatChat:
        from .model_registry import (
            PROVIDERS,
            REQUEST_MODEL,
            default_model_id,
            is_reasoning,
            provider_base_url,
            provider_key,
            split_id,
        )

        mid = model_id or REQUEST_MODEL.get() or default_model_id(self.s)
        if mid not in self._clients:
            prov, model = split_id(mid)
            p = PROVIDERS[prov]
            self._clients[mid] = OpenAICompatChat(
                self.s,
                model=model,
                base_url=provider_base_url(self.s, p),
                api_key=provider_key(self.s, p),
                provider=prov,
                max_tokens_field=p.max_tokens_field,
                reasoning=is_reasoning(model),
            )
        return self._clients[mid]

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        return self.client().chat(messages, max_new_tokens)
