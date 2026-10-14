# Notebook UI (NotebookLM-style) — phase 1

The `web/` folder is a Next.js 16 front end for the service, modelled on Google NotebookLM:

| Panel | What it does |
| --- | --- |
| **Sources** | Upload PDF, DOCX, HTML, Markdown and text files (button or drag and drop). Tick which sources answers may use; untick to leave a source out. Click a source to read it, by section and page. |
| **Chat** | Streamed answers, grounded in the ticked sources. Citation chips show file, page and section; clicking one opens the source viewer at the cited passage, highlighted. Each answer shows an **evidence meter** (1 − P(unanswerable) from the answerability gate), "How I got this" (the agent's steps, searches and retries), the model used, save-to-notes, copy, 👍/👎 with a reason. Refusals are labelled "Not in your sources". Past chats are kept per notebook. |
| **Studio** | Notes (write your own or save answers with their citations; pin, edit in Markdown). Audio overview, guides, mind map, flashcards and quiz are added in the next phases. |
| **Model picker** | Like ChatGPT / Claude.ai: a searchable menu of every available model, grouped by provider with a "Recommended" section. The choice is remembered in the browser and sent with every message. |

## Choosing models

The picker lists:

* **Local models**: the configured `LLM_MODEL` (`qwen2.5:3b`, the research model) and every other model
  installed in Ollama (`ollama pull llama3.2:3b`,
  …), discovered automatically.
* **Free open-weight models via Ollama** (`service/scripts/pull_free_models.ps1` pulls a set):

| Model (Ollama tag) | Size | Notes |
| --- | --- | --- |
| `qwen2.5:3b` | 1.9 GB | research model, calibrated; Apache 2.0 |
| `qwen2.5:1.5b`, `qwen2.5:7b` | 1 / 4.7 GB | same family, smaller / larger |
| `qwen3:1.7b`, `qwen3:8b` | 1.4 / 5.2 GB | thinks before answering (hidden), slower |
| `llama3.2:1b`, `llama3.2:3b`, `llama3.1:8b` | 1.3 / 2 / 4.9 GB | Meta, Llama community licence |
| `gemma3:1b`, `gemma3:4b` | 0.8 / 3.3 GB | Google, Gemma terms |
| `phi4-mini` (3.8B) | 2.5 GB | Microsoft, MIT |
| `granite4:3b`, `granite4.2:8b` | | IBM, Apache 2.0, built for RAG |
| `deepseek-r1:1.5b`, `deepseek-r1:7b` | 1.1 / 4.7 GB | reasoning (distilled into Qwen), MIT; slow |
| `mistral:7b` | 4.1 GB | Mistral AI, Apache 2.0 |
| `olmo-3:7b` | | AI2, fully open (data + code) |
| `smollm2:1.7b` | | Hugging Face, Apache 2.0, very fast |

  On a 4 GB GPU, models up to ~4B parameters run fully on the GPU; 7–8B models spill to the CPU and are
  several times slower. Reasoning models' `<think>` text is hidden automatically. BLOOM/BLOOMZ (2022) exist
  only as community uploads (`gouranshitera/bloomz-1b7`); they follow instructions poorly and are not
  recommended except as a historical baseline.
* **Cloud models**: add any of these keys to `.env` and that provider's models appear (listed live from the
  provider, so new models show up without code changes):

| Key | Provider (OpenAI-compatible endpoint) |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter: one key for Claude, GPT, Gemini, Llama, Qwen, DeepSeek, Mistral and many more |
| `OPENAI_API_KEY` | OpenAI (GPT-4.1, o-series, …) |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GEMINI_API_KEY` | Google Gemini |
| `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `TOGETHER_API_KEY` | Groq, Mistral, DeepSeek, Together |

`DEFAULT_MODEL` sets the default (e.g. `openrouter:anthropic/claude-sonnet-4.5`); `MODEL_ALLOWLIST` /
`MODEL_BLOCKLIST` (JSON lists of glob patterns) limit what users can pick. Reasoning models (o-series,
GPT-5, R1, "thinking" models) get a larger output budget and no temperature setting automatically.

## Refusal thresholds for every model

The agent abstains when the gate score p ≥ τ, when the model refuses on its own, or when p ≥ min_p and the
self-check judges the answer unsupported. The gate (fine-tuned re-ranker) does not depend on the LLM, but the
model's own refusals and its self-check verdicts do, so τ and min_p are calibrated **per model**:

```powershell
docker compose run --rm calibrate                                   # every model installed in Ollama
docker compose run --rm calibrate --models llama3.2:3b gemma3:4b    # only these
docker compose run --rm calibrate --models openrouter:openai/gpt-4.1-mini   # cloud model (costs API credit)
```

`service/scripts/calibrate_models.py` uses the same 224 MTRAG **dev** examples as the research calibration
(150 answerable, 74 unanswerable; never the test set) and calls each model through the same client the app
uses. For each model it records the generator's answer and the self-check verdict per example, then picks the
(τ, min_p) pair with the best abstention F1 whose false-abstention rate is at most the model's own refusal
rate + 10 points (the research procedure allowed 5 points for the gate and 5 for the self-check). Qwen2.5-3B
keeps the research values (τ 0.611, min_p 0.5) so the report stays valid; its cached research answers are
reused, not re-generated.

Results go to `checkpoints/calibration/<model>.json` (+ `summary.md`, `summary.csv`). The API reads them
without a restart: each answer then uses its model's own τ and min_p (stored in the message's `versions`
with `thresholds: calibrated | research | default`), and the model gets the shield in the menu (hover it for
τ, min_p, recall and false-abstention rate). Models not calibrated yet use the research values.

Runs are resumable (answers cached in `checkpoints/calibration/cache/`): stop with Ctrl+C and run the same
command again. On a GTX 1650 a 1–4B model takes about 30–60 min; reasoning models (qwen3, deepseek-r1) take
several hours, so they run last. `--limit 20` does a quick smoke test (saved under `partial/`, not used).

## API additions

`GET /models`, `POST /chat/stream` with `model` and `document_ids`, `GET /conversations?collection=…`,
`DELETE /conversations/{id}`, `PATCH`/`DELETE /collections/{id}`,
`GET /collections/{id}/documents/{doc}/passages` (source viewer), `…/file` (original file),
`DELETE …/documents/{doc}`, and `/collections/{id}/notes` (CRUD). Citations now carry `source_document`,
`page` and `section`. Migration `0002` adds the `notes` table.

## Running

```powershell
docker compose up -d --build            # adds the web service; open http://localhost:3000
```

Development without Docker, models or keys (fake LLM, in-memory stores):

```bash
python service/scripts/demo_api.py      # API on :8000
cd web && npm install && npm run dev    # UI on :3000
```

## Next phases

2. Web pages, YouTube transcripts and pasted text as sources; a source guide (summary, topics, suggested
   questions) per upload; whole-notebook questions ("summarise my sources") routed to document summaries.
3. Studio: briefing doc, study guide, FAQ, timeline, flashcards, quiz, mind map.
4. Audio overview: two-host podcast script, Kokoro text-to-speech.
5. Sharing notebooks; "discover sources" web search.
