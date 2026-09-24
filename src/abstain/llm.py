"""Thin wrapper around a Hugging Face chat model (shared by the rewriter and the generator)."""
from __future__ import annotations


class ChatLLM:
    def __init__(self, model: str, load_in_4bit: bool = True, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(model)
        cuda = torch.cuda.is_available()
        # bf16 only on Ampere+ (RTX 30xx, A100, ...); Turing/Pascal cards (GTX 16xx, T4) use fp16
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        kw = {"device_map": "auto", "dtype": dtype}
        if load_in_4bit and cuda:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        import torch

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OllamaLLM:
    """Same .chat() interface, served by a local Ollama server (llama.cpp, GGUF 4-bit).

    On small / older GPUs (e.g. GTX 1650) this is several times faster than
    transformers + bitsandbytes 4-bit, whose per-token dequantisation is slow.
    Setup (Windows): install from https://ollama.com, then `ollama pull qwen2.5:3b`.
    """

    def __init__(self, model: str = "qwen2.5:3b", url: str = "http://localhost:11434",
                 max_new_tokens: int = 256, num_ctx: int = 8192, timeout: int = 600):
        import requests

        self.model, self.url, self.max_new_tokens, self.num_ctx, self.timeout = model, url.rstrip("/"), \
            max_new_tokens, num_ctx, timeout
        self.s = requests.Session()
        try:
            tags = self.s.get(f"{self.url}/api/tags", timeout=10).json().get("models", [])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Ollama is not reachable at {self.url}. Install it from https://ollama.com, "
                               f"start it, and run: ollama pull {model}") from e
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        if model not in names and f"{model}:latest" not in names:
            raise RuntimeError(f"Ollama model '{model}' not found. Run: ollama pull {model}")

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        r = self.s.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "messages": messages, "stream": False, "keep_alive": "60m",
            # greedy decoding for reproducibility; num_ctx must fit 5 passages (Ollama's default is small)
            "options": {"temperature": 0, "seed": 13, "num_predict": max_new_tokens or self.max_new_tokens,
                        "num_ctx": self.num_ctx},
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def make_llm(c: dict, max_new_tokens: int | None = None):
    """Build the LLM from the `llm:` config block (backend: hf | ollama)."""
    n = max_new_tokens or c.get("max_new_tokens", 256)
    if c.get("backend", "hf") == "ollama":
        return OllamaLLM(c.get("ollama_model", "qwen2.5:3b"), c.get("ollama_url", "http://localhost:11434"),
                         n, c.get("num_ctx", 8192))
    return ChatLLM(c["model"], c.get("load_in_4bit", True), n)
