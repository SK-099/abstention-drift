"""OpenAI-compatible chat client used by the agent (Ollama /v1, vLLM, Groq, Together, OpenRouter).

Exposes the `.chat(messages, max_new_tokens)` interface the research agent expects. While the agent's
`generate` node runs (abstain.agent.STREAM_TOKENS), tokens are streamed and forwarded to LangGraph's
stream writer, so the API can push them to the browser as they arrive.
"""

from __future__ import annotations

import time

from abstain.agent import STREAM_TOKENS

from .settings import Settings


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

    def _model(self, n: int):
        if n not in self._models:
            from langchain_openai import ChatOpenAI

            kw = {}
            if self.provider == "local" and self.s.llm_seed is not None:
                kw["seed"] = self.s.llm_seed  # deterministic research model
            if self.reasoning:  # reasoning models spend tokens on thinking and reject temperature 0
                n = max(n, self.s.reasoning_max_tokens)
            else:
                kw["temperature"] = 0
            headers = (
                {"HTTP-Referer": "https://github.com/cs683/abstention-drift", "X-Title": "Grounded Notebook"}
                if self.provider == "openrouter"
                else None
            )
            # "max_tokens" is understood by Ollama, vLLM, Groq, Together, OpenRouter, Anthropic and Gemini;
            # OpenAI's newer models need "max_completion_tokens"
            self._models[n] = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.s.llm_timeout_s,
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
                return str(model.invoke(msgs).content).strip()
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
        parts = []
        for chunk in model.stream(msgs):
            t = chunk.content if isinstance(chunk.content, str) else ""
            if t:
                parts.append(t)
                if write:
                    write({"type": "token", "text": t})
        return "".join(parts).strip()


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
