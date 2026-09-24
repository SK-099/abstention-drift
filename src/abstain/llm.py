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
        kw = {"device_map": "auto", "torch_dtype": dtype}
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
