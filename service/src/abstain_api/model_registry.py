"""Model picker: every chat model the user can choose, from every configured provider.

Like the model menu in ChatGPT or Claude.ai, the user picks a model per message. Models are listed live from
each provider's `/models` endpoint (so new models appear without code changes), filtered to chat models and
cached for MODELS_CACHE_S seconds. A provider appears only when its API key is set; local Ollama models are
discovered from the Ollama server. All providers are reached through their OpenAI-compatible endpoints.

Model ids are "<provider>:<model>", e.g. "local:qwen2.5:3b", "openrouter:anthropic/claude-sonnet-4.5",
"openai:gpt-4.1-mini", "anthropic:claude-...", "gemini:gemini-2.5-flash".
"""

from __future__ import annotations

import contextvars
import fnmatch
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field

from .settings import Settings

log = logging.getLogger(__name__)

# set per request (chat router -> agent thread): which model answers, and which documents may be searched
REQUEST_MODEL: contextvars.ContextVar[str | None] = contextvars.ContextVar("REQUEST_MODEL", default=None)
REQUEST_DOCS: contextvars.ContextVar[tuple[str, ...] | None] = contextvars.ContextVar("REQUEST_DOCS", default=None)


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    settings_key: str | None  # Settings attribute holding the API key (None: no key needed)
    list_url: str | None = None  # default: base_url + "/models"
    max_tokens_field: str = "max_tokens"


PROVIDERS: dict[str, Provider] = {
    p.key: p
    for p in [
        Provider("local", "Local (Ollama / vLLM)", "", None),
        Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openrouter_api_key"),
        Provider(
            "openai", "OpenAI", "https://api.openai.com/v1", "openai_api_key", max_tokens_field="max_completion_tokens"
        ),
        Provider("anthropic", "Anthropic", "https://api.anthropic.com/v1", "anthropic_api_key"),
        Provider(
            "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key"
        ),
        Provider("groq", "Groq", "https://api.groq.com/openai/v1", "groq_api_key"),
        Provider("mistral", "Mistral", "https://api.mistral.ai/v1", "mistral_api_key"),
        Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek_api_key"),
        Provider("together", "Together AI", "https://api.together.xyz/v1", "together_api_key"),
    ]
}

# not chat models
_SKIP = re.compile(
    r"embed|whisper|tts|transcri|audio|realtime|speech|dall-e|image|imagen|veo|moderation|guard|rerank|"
    r"search-preview|computer-use|babbage|davinci|sora|codex-mini|\blyria",
    re.I,
)
_REASONING = re.compile(r"(^|[/:])(o\d)(-|$)|gpt-5|reasoner|thinking|(^|[-/])r1\b|qwq", re.I)
# shown first in the picker (as in ChatGPT / Claude.ai), per provider
_FEATURED = re.compile(
    r"claude|gpt-4\.1|gpt-4o|gpt-5|(^|/)o[34]|gemini-2\.5|gemini-3|llama-3\.3|llama-4|qwen3|qwen-?2\.5|deepseek|mistral-large|"
    r"mistral-medium",
    re.I,
)


@dataclass
class ModelInfo:
    id: str
    provider: str
    provider_label: str
    model: str
    name: str
    context_length: int | None = None
    reasoning: bool = False
    featured: bool = False
    calibrated: bool = False  # the research thresholds (tau, min_p) were tuned for this model
    description: str = ""
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d


def provider_key(s: Settings, p: Provider) -> str | None:
    if p.key == "local":
        return s.llm_api_key or "local"
    return getattr(s, p.settings_key) if p.settings_key else None


def provider_base_url(s: Settings, p: Provider) -> str:
    return s.llm_base_url.rstrip("/") if p.key == "local" else p.base_url


def configured_providers(s: Settings) -> list[Provider]:
    return [p for p in PROVIDERS.values() if p.key == "local" or provider_key(s, p)]


def default_model_id(s: Settings) -> str:
    return s.default_model or f"local:{s.llm_model}"


def split_id(model_id: str) -> tuple[str, str]:
    prov, _, model = model_id.partition(":")
    if not model or prov not in PROVIDERS:
        raise ValueError(f"unknown model id '{model_id}' (expected '<provider>:<model>')")
    return prov, model


def is_reasoning(model: str) -> bool:
    return bool(_REASONING.search(model))


def _allowed(s: Settings, mid: str) -> bool:
    if s.model_allowlist and not any(fnmatch.fnmatch(mid, pat) for pat in s.model_allowlist):
        return False
    return not any(fnmatch.fnmatch(mid, pat) for pat in s.model_blocklist)


# ---------------------------------------------------------------- listing
def _get(url: str, headers: dict, timeout: float = 15.0):
    import httpx

    r = httpx.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _list_local(s: Settings, p: Provider) -> list[ModelInfo]:
    calibrated = ModelInfo(
        id=f"local:{s.llm_model}",
        provider="local",
        provider_label=p.label,
        model=s.llm_model,
        name=s.llm_model,
        calibrated=True,
        featured=True,
        description="Research model: abstention thresholds were calibrated for it",
    )
    out = {calibrated.id: calibrated}
    base = (s.ollama_url or re.sub(r"/v1/?$", "", s.llm_base_url)).rstrip("/")
    try:
        data = _get(f"{base}/api/tags", {}, timeout=5.0)
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if not name or _SKIP.search(name):
                continue
            mid = f"local:{name}"
            if mid in out:
                continue
            size = (m.get("details") or {}).get("parameter_size", "")
            out[mid] = ModelInfo(
                mid, "local", p.label, name, name, reasoning=is_reasoning(name), description=f"Local {size}".strip()
            )
    except Exception:
        try:  # vLLM / other OpenAI-compatible local servers
            data = _get(f"{s.llm_base_url.rstrip('/')}/models", {"Authorization": f"Bearer {s.llm_api_key}"}, 5.0)
            for m in data.get("data", []):
                mid = f"local:{m['id']}"
                out.setdefault(mid, ModelInfo(mid, "local", p.label, m["id"], m["id"], reasoning=is_reasoning(m["id"])))
        except Exception:
            log.info("local model discovery failed; listing only %s", s.llm_model)
    return list(out.values())


def _list_remote(s: Settings, p: Provider) -> list[ModelInfo]:
    key = provider_key(s, p)
    if p.key == "anthropic":
        data = _get(f"{p.base_url}/models?limit=100", {"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        data = _get(p.list_url or f"{p.base_url}/models", {"Authorization": f"Bearer {key}"})
    rows = data if isinstance(data, list) else data.get("data", [])
    out = []
    for m in rows:
        model = str(m.get("id", "")).removeprefix("models/")
        if not model or _SKIP.search(model):
            continue
        if p.key == "openrouter":
            arch = m.get("architecture") or {}
            if "text" not in (arch.get("output_modalities") or ["text"]):
                continue
        if p.key == "together" and m.get("type") not in (None, "chat", "language"):
            continue
        name = m.get("display_name") or m.get("name") or model
        out.append(
            ModelInfo(
                id=f"{p.key}:{model}",
                provider=p.key,
                provider_label=p.label,
                model=model,
                name=name,
                context_length=m.get("context_length") or m.get("context_window") or m.get("inputTokenLimit"),
                reasoning=is_reasoning(model),
                featured=bool(_FEATURED.search(model)),
            )
        )
    return out


class ModelRegistry:
    def __init__(self, s: Settings):
        self.s = s
        self._cache: dict[str, tuple[float, list[ModelInfo]]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def _provider_models(self, p: Provider) -> list[ModelInfo]:
        hit = self._cache.get(p.key)
        if hit and time.time() - hit[0] < self.s.models_cache_s:
            return hit[1]
        try:
            models = _list_local(self.s, p) if p.key == "local" else _list_remote(self.s, p)
            self._errors.pop(p.key, None)
        except Exception as e:
            log.warning("listing %s models failed: %s", p.key, e)
            self._errors[p.key] = str(e)[:200]
            models = hit[1] if hit else []
        with self._lock:
            self._cache[p.key] = (time.time(), models)
        return models

    def list(self) -> list[ModelInfo]:
        out = []
        for p in configured_providers(self.s):
            out += [m for m in self._provider_models(p) if _allowed(self.s, m.id)]
        dflt = default_model_id(self.s)
        out.sort(key=lambda m: (m.id != dflt, m.provider != "local", m.provider, not m.featured, m.name.lower()))
        return out

    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def resolve(self, model_id: str | None) -> str:
        """Validate a requested model id; None -> default. Any model of a configured provider is accepted
        (also ones not in the cached list, e.g. just released), subject to the allow/block lists."""
        mid = model_id or default_model_id(self.s)
        prov, _ = split_id(mid)
        if prov not in {p.key for p in configured_providers(self.s)}:
            raise ValueError(f"provider '{prov}' is not configured (set its API key)")
        if not _allowed(self.s, mid):
            raise ValueError(f"model '{mid}' is not allowed on this server")
        return mid


_registry: ModelRegistry | None = None


def get_registry(s: Settings) -> ModelRegistry:
    global _registry
    if _registry is None or _registry.s is not s:
        _registry = ModelRegistry(s)
    return _registry
