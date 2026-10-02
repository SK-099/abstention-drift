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
    def __init__(self, settings: Settings, retries: int = 3, wait_s: float = 5.0):
        self.s, self.retries, self.wait_s = settings, retries, wait_s
        self._models: dict[int, object] = {}

    def _model(self, n: int):
        if n not in self._models:
            from langchain_openai import ChatOpenAI

            kw = {"seed": self.s.llm_seed} if self.s.llm_seed is not None else {}
            # "max_tokens" (not "max_completion_tokens") is understood by Ollama, vLLM, Groq, Together and OpenRouter
            self._models[n] = ChatOpenAI(
                model=self.s.llm_model,
                base_url=self.s.llm_base_url,
                api_key=self.s.llm_api_key,
                temperature=0,
                timeout=self.s.llm_timeout_s,
                max_retries=0,
                extra_body={"max_tokens": n},
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
