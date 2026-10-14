# Updates since the previous report and presentation (2 October 2026)

The previous `CS683_Project_Report` and `CS683_Project_Presentation` covered the research systems S0 to S5. Since then, the S5 agent has been deployed as a production web service, tested end to end on the laptop, and fixed where the tests found problems. **None of the reported research numbers change.**

## What was added to the project

1. **Production service** (`service/`, about 3,200 lines, 23 tests). It is built from:
   - FastAPI, with the agent's steps streamed as SSE
   - the LangGraph Postgres checkpointer, which keeps each conversation's agent state
   - Qdrant hybrid search over the 366,479 MTRAG passages, or pgvector
   - PostgreSQL, SQLAlchemy and Alembic for users, conversations and feedback
   - Celery and Redis for upload indexing and the nightly evaluation
   - JWT login through Clerk or Auth.js
2. **Endpoints:**
   - chat
   - conversations
   - feedback
   - private collections with document uploads
   - `/metrics/abstention`
   - health checks
3. **Guardrails with Guardrails AI.** Prompt-injection and toxic questions are blocked, and personal data and secrets are redacted. No real MTRAG question (0 of 1,349) is blocked or redacted, and no corpus passage (0 of 24,004) is flagged.
4. **LLMOps:**
   - one readable Langfuse trace per turn
   - Prometheus and Grafana dashboards and alerts
   - prompt and threshold versions stored with each answer
   - a CI quality gate on a fixed 150-turn sample
   - a nightly evaluation
5. **DevOps:**
   - the whole stack runs with Docker Compose
   - uv and Ruff
   - GitHub Actions
   - deployment files for Fly.io, Render, Kubernetes, Modal and RunPod
6. **Agent options for the service.**
   - `CLARIFY_MIN_TURN` stops clarifying questions on early turns. It is off by default.
   - A retry that would repeat an already-tried query is skipped.
   - Service answers can be up to 320 tokens.
7. **Fixes found by end-to-end testing:**
   - **Refusal detector.** "Don’t", written with a curly apostrophe, was not recognised as a refusal. This is now fixed. Re-scoring all saved research predictions with the fixed detector changes 0 decisions.
   - **Upload chunks.** Chunks of about 1,400 characters were longer than the 1,000 characters the LLM reads. Chunks are now 150 words, capped at 1,000 characters.
   - **Contextual chunk headers.** Chunks stay within one section and start with the section path, for example "Appendix A. Prompts".
   - **Metrics.** Prometheus counters now start at 0, so Grafana no longer loses the first event of each counter.

## Changes to the report (`docs/CS683_Project_Report.tex` / `.pdf`, now 50 pages)

- **New Chapter 10, "Production Service: Deploying the S5 Agent".** It covers the motivation, an architecture figure, the components table, endpoints, section-aware chunking, guardrails (with a table and the measured false-alarm rates), LLMOps and DevOps, a table of problems found by end-to-end testing, and observed latency and behaviour.
- **Later chapters renumbered.** Error Analysis is now Chapter 11, and Conclusion is now Chapter 15.
- **Updated sections:**
  - the abstract
  - the summary of outcomes
  - the report organisation
  - the hardware/software table (service stack, 26 + 23 tests)
  - the code organisation table and line counts
- **Error analysis.** New point 5, "Errors seen in the deployed service": a wrong answer that passed the selective self-check, and an answer that is itself the refusal sentence.
- **Challenges table.** Five new rows: the Docker stack, the editable install, the Grafana counters, the curly apostrophe, and upload chunking.
- **Limitations and future work.** Added the scope of the service evaluation and latency, and replaced "FastAPI back end" with a web front end and GPU serving.
- **Conclusion.** New bullet on the production service.
- **Appendices.** Service rows in Hyper-parameters and Run Times, and service commands in Appendix D.
- **Date** changed to October 2026.

## Changes to the presentation (`docs/CS683_Project_Presentation.pptx`, now 31 slides)

- **Five new slides after "The agent in action":**
  1. From research code to a production service (architecture)
  2. Service features: API, LLMOps and DevOps
  3. Guardrails with Guardrails AI
  4. What end-to-end testing found
  5. Uploads: one question, three chunking versions
- **Updated slides:**
  - the agenda (item 4)
  - the tech stack (service software, line and test counts)
  - the conclusion footer (production service; new future work)

All other slides are unchanged.

---

# Update of 15 October 2026: the notebook application

## What was added to the project

- **Model menu grouped by ability.** Standard models (answer directly) and thinking models (reason first), with
  size, measured speed per answer and a shield for calibrated models. `phi4-mini` and `phi4-mini:latest` count
  as one model.
- **Per-model refusal thresholds.** `docker compose run --rm calibrate` chooses τ and min_p for every model on
  the 224 MTRAG dev examples; ten local models are calibrated (Qwen3-4B was removed: about seven minutes per
  answer on the 4 GB GPU). The service applies each model's own thresholds.
- **Phase 2, sources:** web pages, YouTube transcripts (timestamped sections), pasted text; a source guide
  (summary, topics, suggested questions) for every source; "summarise my sources"-type questions answered from
  the guides with one citation per source.
- **Phases 3–4, Studio:** briefing doc, study guide, FAQ, timeline, interactive mind map, flashcards, quiz, and an
  audio overview (two hosts, spoken by Kokoro; `docker compose --profile tts up -d kokoro`).
- **Phase 5:** sharing (viewer / editor / anyone with the link) and "discover sources" web search.
- Service tests: 32 → 49. Migrations 0003–0005.

## Changes to the report (now 57 pages)

- **New Chapter 11, "Notebook Application: A NotebookLM-Style Front End"**: motivation, features table and
  screenshot, architecture figure, sources beyond files (with the SSRF protection), source guides and
  notebook-wide questions, Studio outputs (table and mind-map screenshot), the audio overview, choosing the model
  (model-menu screenshot), **per-model refusal thresholds (Table 11.3, ten models)**, testing and a summary.
- Later chapters renumbered (Error Analysis is now Chapter 12, Conclusion Chapter 16).
- Updated: abstract, summary of outcomes, report organisation, Chapter 10 (front end now described), limitations,
  future work and conclusion.

## Changes to the presentation (now 36 slides)

- **Five new slides after "Uploads: one question, three chunking versions":** A NotebookLM-style notebook; The
  Studio; Choose any model; Refusal thresholds for every model; Sharing, safety and testing.
- Updated: agenda item 4, the service slide footer and the conclusion footer.
