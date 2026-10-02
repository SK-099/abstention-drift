# syntax=docker/dockerfile:1.7
# API, Celery worker and beat share this image (CPU PyTorch; the LLM runs elsewhere: Ollama / vLLM / hosted).
# Build from the repository root:  docker build -f service/Dockerfile -t abstain-api .
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv /opt/venv
# CPU-only torch first (keeps the image ~1.5 GB smaller than the default CUDA wheels)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
COPY service/pyproject.toml ./service/pyproject.toml
COPY service/src ./service/src
ARG EXTRAS=""
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "./[agent]" "./service${EXTRAS}"

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 app
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/data/hf \
    UPLOAD_DIR=/data/uploads MTRAG_DIR=/data/mt-rag-benchmark EVAL_SAMPLE_FILE=/app/service/eval/sample_ids.json \
    EVAL_BASELINE_FILE=/app/service/eval/baseline.json
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY service/alembic.ini ./service/alembic.ini
COPY service/migrations ./service/migrations
COPY service/eval ./service/eval
COPY service/scripts ./service/scripts
COPY configs ./configs
RUN mkdir -p /data/hf /data/uploads checkpoints && chown -R app /data /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "abstain_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
