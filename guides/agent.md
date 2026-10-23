# S5: LangGraph self-correcting abstention agent

The fixed pipeline (S2/S4) rewrites once, retrieves once, then answers or refuses. Our error analysis
showed two failure modes: wrong refusals (18-28% of answerable turns, often after a bad rewrite) and
unsupported answers to unanswerable questions. S5 adds two agentic steps, built with LangGraph:

1. **Retry before refusing.** If the evidence grader (the S4 checker) says the passages are weak, or the
   generator refuses, the agent rewrites the question differently (expand abbreviations, keep the
   user's words, drop stale topics), retrieves again and re-grades. Still failing: it asks a clarifying
   question if the question is ambiguous, otherwise it abstains.
2. **Self-check after answering.** Qwen2.5-3B judges whether the answer is supported by the passages;
   if not, the agent abstains.

```
rewrite -> retrieve -> grade --ok--> generate --answered--> self_check --supported--> ANSWER
                         |               \--refused--\           \--unsupported--> ABSTAIN
                         \--weak---------------------> retry? --yes--> rewrite (new strategy)
                                                          \--no--> clarify_or_abstain -> CLARIFY | ABSTAIN
```

LangChain components: `ChatOllama` (langchain-ollama) for Qwen2.5-3B, a `BaseRetriever` around the
hybrid BM25 + BGE + re-ranker retriever, and a LangGraph `StateGraph` with fixed, rule-based routing
(a 3B model follows explicit steps reliably but chooses tools poorly). Code: `src/abstain/agent.py`.

## Run (Windows, conda env `abstain`, Ollama running)

```powershell
pip install "langgraph>=1.0,<2" "langchain-core>=1.0,<2" "langchain-ollama>=1.0,<2"
$C = "--config", "configs\laptop.yaml"
python scripts\run_agent.py @C --underspecified            # ~2-3 h; resumable
python scripts\evaluate_abstention.py "outputs\S2_mtragun.jsonl" "outputs\S4_mtragun.jsonl" "outputs\S5-agent_mtragun.jsonl" @C
python scripts\compare_agent.py @C                          # what changed vs S2, and why
```

Fast mode reuses the finished S2 run (the agent's first attempt is identical to S2) and the S4 gate
scores, so only the retries (~120 turns) and self-checks (~310 turns) call the LLM. `--live` runs every
step from scratch. Clarifications count as abstentions in the binary metrics; `--underspecified` also
runs the 78 underspecified MTRAG-UN turns, where asking a clarifying question is the desired behaviour.

Ablations (use `--name` so outputs don't overwrite each other):

```powershell
python scripts\run_agent.py @C --no-self-check --name S5-noselfcheck
python scripts\run_agent.py @C --max-retries 0 --name S5-noretry
python scripts\run_agent.py @C --gate reranker --name S5-rr
```

## Selective self-check (calibrated on dev)

Checking every answer made the 3B judge reject 66% of correct answers (FA .708). The agent therefore
self-checks only answers whose gate score is already doubtful (p >= min_p). min_p is chosen on the
MTRAG dev examples, never on MTRAG-UN:

```powershell
python scripts\calibrate_self_check.py @C                                  # ~40 min, writes checkpoints\self_check.json
python scripts\run_agent.py @C --self-check-min-p auto --name S5-selective  # ~1 h
```
