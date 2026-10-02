"""Scale-to-zero vLLM on Modal (OpenAI-compatible): you pay only while requests are being served.

    pip install modal && modal setup
    modal deploy deploy/modal/vllm_server.py
    # -> https://<workspace>--abstain-vllm-serve.modal.run/v1   (set as LLM_BASE_URL, LLM_API_KEY=$VLLM_API_KEY)

Cold start is ~1-2 min (model weights are cached in a Modal Volume after the first start).
"""

import subprocess

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SERVED_NAME = "qwen2.5:3b"  # keep LLM_MODEL unchanged in the API settings, or change both
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("abstain-vllm")


@app.function(
    image=image,
    gpu="L4",
    scaledown_window=300,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("vllm-api-key")],
)  # defines VLLM_API_KEY
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    cmd = (
        f"vllm serve {MODEL} --served-model-name {SERVED_NAME} --host 0.0.0.0 --port 8000 "
        "--max-model-len 8192 --enable-prefix-caching --api-key $VLLM_API_KEY"
    )
    subprocess.Popen(cmd, shell=True)
