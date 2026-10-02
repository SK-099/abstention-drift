# Grounded Assistant: the production service

The `service/` folder turns the S5 agent (LangGraph, retry before refusing, selective self-check) into a
multi-user web service. The research code in `src/abstain` is unchanged and is imported as a library.

```
browser / front end (later: Next.js on Vercel)
        │  JWT (Clerk or Auth.js)                     SSE: step / token / final events
        ▼
FastAPI (service/src/abstain_api)  ── Langfuse (traces of every LangGraph node and LLM call)
  │  POST /chat/stream ─▶ S5 LangGraph agent ─▶ OpenAI-compatible LLM (Ollama | vLLM | Groq | Together | OpenRouter | Modal | RunPod)
  │                        │  checkpointer: PostgreSQL (thread = conversation)
  │                        ├─ Qdrant hybrid search (dense BGE + sparse BM25, RRF, filter by collection) │ or pgvector
  │                        ├─ BGE re-ranker (in-process or TEI)
  │                        └─ fine-tuned answerability checker (gate τ) + self-check (p ≥ min_p)
  ├─ PostgreSQL (SQLAlchemy 2 + Alembic): users, conversations, messages, feedback, collections, documents, eval runs
  ├─ Celery + Redis: document indexing (Docling/basic parsing → injection screen → embed → Qdrant), nightly evaluation
  └─ OpenTelemetry metrics → Prometheus → Grafana; errors → Sentry
```

## 1. Run it on the laptop (everything in Docker except Ollama)

Prerequisites: Docker Desktop, Ollama for Windows (`ollama pull qwen2.5:3b`), and the research outputs that
already exist in this folder: `checkpoints/checker/best` (the S4 checker), `checkpoints/self_check.json`,
`indexes/<collection>/` (passages + FAISS vectors) and `data/mt-rag-benchmark`.

```powershell
copy .env.example .env                       # defaults: Ollama on the host, AUTH_MODE=dev
docker compose up -d --build                 # API, worker, beat, Postgres, Redis, Qdrant, Langfuse (+ SeaweedFS S3), Prometheus, Grafana
docker compose run --rm api python service/scripts/migrate_to_qdrant.py --index-dir /indexes --snapshot
```

`migrate_to_qdrant.py` reuses the BGE vectors stored in the FAISS files (no re-encoding of the 366K passages)
and adds the sparse BM25 vectors; it takes a few minutes. Then:

| What | Where |
| --- | --- |
| API + OpenAPI docs | http://localhost:8000/docs |
| Langfuse (login `admin@example.com` / `change-me-please`) | http://localhost:3001 |
| Grafana (admin / admin), dashboard "Abstention-aware RAG" | http://localhost:3002 |
| Prometheus | http://localhost:9090 |
| Qdrant dashboard | http://localhost:6333/dashboard |

Try it:

```bash
curl -N -X POST localhost:8000/chat/stream -H "content-type: application/json" \
     -d '{"message": "How do I create a Cloud Object Storage bucket?", "collection": "cloud"}'
```

Without Docker (development): `uv sync --all-packages`, start Postgres/Redis/Qdrant (or use
`DATABASE_URL=sqlite+aiosqlite:///dev.db CHECKPOINTER=memory CELERY_EAGER=true QDRANT_URL=:memory:`), then
`uv run alembic -c service/alembic.ini upgrade head` and `uv run abstain-api`.

## 2. API

| Endpoint | Purpose |
| --- | --- |
| `POST /chat/stream` | `{message, conversation_id?, collection}` → SSE events `meta`, `step` (rewrite, retrieve, grade, generate, self_check, retry, clarify_or_abstain), `token`, `final` (answer, decision answer/abstain/clarify, p_unanswerable, citations, trace, timings, message_id) |
| `GET /conversations`, `GET /conversations/{id}?include_state=true` | history; optionally the agent's checkpointed LangGraph state |
| `POST /feedback` | thumbs up/down on an assistant message, reason `wrong_refusal`, `wrong_answer`, `bad_citation`, … |
| `POST /collections`, `GET /collections` | private collections (public ones: the four MTRAG domains, admins only) |
| `POST /collections/{id}/documents`, `GET …/documents` | upload PDF/HTML/DOCX/MD/TXT (202, indexed in the background) |
| `GET /metrics/abstention` (admin) | refusal rate overall / per day / per turn bucket / per collection, retry rate, user-reported wrong refusals and answers, latency p50/p95, last evaluation run |
| `GET /metrics`, `/healthz`, `/readyz` | Prometheus scrape, liveness, readiness (DB, Qdrant, Redis, LLM) |

Guardrails: Pydantic validation (message length, slugs, feedback values), slowapi rate limits per user
(`RATE_LIMIT_CHAT`, `RATE_LIMIT_UPLOAD`), upload type/size checks, and a prompt-injection screen on every
uploaded passage (pattern rules, optionally a classifier via `INJECTION_MODEL`): flagged passages are
dropped, and a document where more than half is flagged is rejected.

Auth: `AUTH_MODE=jwks` verifies Clerk session tokens (RS256 via the JWKS URL, issuer and `azp`);
`AUTH_MODE=hs256` verifies a token signed by the Auth.js app with the shared `AUTH_SECRET`; `dev` (no login)
is refused when `ENV=production`. Admins: `role: admin` claim or `ADMIN_USER_IDS`.

## 3. LLMOps

* **Tracing** – with `LANGFUSE_*` keys every LangGraph node, prompt, response and duration is traced;
  session = conversation, user = user, tags `collection:<id>` and `prompt:<version>`.
* **Prompt and threshold versions** – every assistant message stores `prompt_version`, a hash of all
  prompts, the model, τ and min_p (`meta.versions`); `service/scripts/push_prompts_to_langfuse.py` registers
  them in Langfuse prompt management.
* **Release calibration** – `python service/scripts/recalibrate.py --config configs/laptop.yaml` re-tunes τ
  and the self-check cut-off on the dev split (never on test) for the serving LLM and records the versions.
* **Quality gate** – `service/eval/sample_ids.json` is a fixed stratified MTRAG-UN sample (150 turns, 34
  unanswerable). `quality_gate.py` fails (exit 1) if abstain-F1 drops > 0.03 below `baseline.json`
  (S5-selective: F1 0.540, false abstention 0.172 on the sample), if false abstention exceeds 0.25, or
  if it rises > 0.05. CI scores the committed predictions on every PR; with `LIVE_EVAL=true` and the LLM /
  Qdrant secrets it re-runs the agent live (`run_sample.py`) and gates that, plus Ragas faithfulness
  (`faithfulness.py`). Accept new numbers deliberately with `make_sample.py --keep-sample --predictions …`.
* **Nightly evaluation** – Celery beat runs the sample every night (stored in `eval_runs`, shown by
  `/metrics/abstention`); `.github/workflows/nightly-eval.yml` does the same from GitHub.
* **Metrics** – Prometheus alerts (`infra/prometheus/alerts.yml`) fire when the refusal rate jumps above
  50 % or collapses below 2 %, or p95 latency exceeds 30 s.

## 4. CI/CD (`.github/workflows/ci.yml`)

lint (Ruff) → tests (research + service, against SQLite/in-memory and against real Postgres, Qdrant and
pgvector service containers) → quality gate → Docker image to GHCR (main only) → deploy to whichever target
has secrets: Fly.io (`FLY_API_TOKEN`), Render (`RENDER_DEPLOY_HOOK`) or Kubernetes (`KUBECONFIG_B64`).

## 5. Hosting options

| Option | Front end | API + worker | LLM | Data |
| --- | --- | --- | --- | --- |
| Laptop | `localhost:3000` (later) | `docker compose up` | Ollama native on Windows (GPU) | Postgres, Redis, Qdrant in Docker |
| Free / cheap demo | Vercel | Fly.io (`deploy/fly`) or Render (`deploy/render.yaml`) or Railway (same image) | Groq / Together / OpenRouter, or scale-to-zero vLLM on Modal (`deploy/modal`) / RunPod (`deploy/runpod`) | Neon / Render Postgres, Upstash Redis, Qdrant Cloud free tier |
| Production-style | Vercel | Kubernetes on EKS / GKE / AKS (`deploy/k8s`, HPA, ingress with SSE settings) | vLLM on GPU nodes (`deploy/k8s/vllm.yaml`) | managed Postgres, managed Redis, Qdrant Cloud (restore the snapshot) |

Switching LLM provider only changes `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`; re-run the
calibration and the live quality gate before release, because τ and min_p were tuned for Qwen2.5-3B.

## 6. Tests

```bash
uv run pytest tests -q                       # research package (23)
cd service && uv run pytest -q               # service: SQLite + in-memory Qdrant (14)
TEST_DATABASE_URL=postgresql+asyncpg://app:app@localhost:5432/app TEST_CHECKPOINTER=postgres \
  TEST_QDRANT_URL=http://localhost:6333 uv run pytest -q          # real Postgres + Qdrant
TEST_DATABASE_URL=... TEST_VECTOR_STORE=pgvector uv run pytest -q  # pgvector backend
```

The service tests use scripted fakes for the LLM, embedder, re-ranker and checker, and cover the full
upload → index → streamed chat → conversation history → feedback → metrics flow, auth, validation, the
injection screen, BM25 vectors, chunking, the OpenAI-compatible client (with token streaming) and the gate.

## 7. Not included yet

The web front end (Next.js + Clerk/Auth.js + front-end Sentry) is deferred; the compose file has a
commented `web` service and CORS is already set up for `http://localhost:3000`.
