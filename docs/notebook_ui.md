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

* **Local models**: the configured `LLM_MODEL` (`qwen2.5:3b`, marked with a shield: the abstention thresholds
  τ and min_p were calibrated for it) and every other model installed in Ollama (`ollama pull llama3.2:3b`,
  …), discovered automatically.
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

**Note on abstention.** The answerability gate (the fine-tuned re-ranker) does not depend on the LLM, so it
works with every model. The generator's refusals and the self-check do depend on the model; τ and min_p
were calibrated for Qwen2.5-3B. Larger models usually answer more and hallucinate less; run
`service/scripts/recalibrate.py` with the new model before relying on its refusal rate.

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
