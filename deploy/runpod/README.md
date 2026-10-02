# RunPod Serverless vLLM (scale to zero)

1. RunPod console -> Serverless -> New Endpoint -> **vLLM** quick deploy.
2. Model `Qwen/Qwen2.5-7B-Instruct` (or `Qwen/Qwen2.5-3B-Instruct`), GPU 24 GB (L4 / A5000), max workers 1-3,
   idle timeout 5 s (scale to zero), `OPENAI_SERVED_MODEL_NAME_OVERRIDE=qwen2.5:3b`.
3. Point the API at it:

```
LLM_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
LLM_API_KEY=<runpod api key>
LLM_MODEL=qwen2.5:3b
```
