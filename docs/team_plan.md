# 4-week plan

Roles are for 4 members. With 3 members, merge A and B. With 5, member E runs the 50-example human check of synthetic labels and 50 abstention decisions, and owns the slides.

| Week | A: Retrieval | B: Generation | C: Checker | D: Evaluation and demo |
|---|---|---|---|---|
| 1 | `setup_data.sh`, `build_indexes.py` (all domains); retrieval recall@5 with the MTRAG scripts | `ChatLLM` on Colab; S0 on 50 dev tasks | **Novelty check**: skim SemEval-2026 Task 8 system papers for per-turn analyses | `dataset_stats.py`; `evaluate_abstention.py` on S0 |
| 2 | LLM rewriting (S1); hybrid retrieval and re-rank (S2) | Prompt tuning for abstention; IDK regex audit | `make_checker_data.py` (+`--verify-llm`); train S3 and S4 | Drift and CIs for S0–S2; hand-check 50 synthetic examples |
| 3 | Oracle-rewrite upper bound; retrieval ablations | Run S0–S4 on MTRAG-UN; official MTRAG generation eval | Threshold sweep (false-abstention budget 2/5/10%); synthetic probe | Gradio demo, Hugging Face Space |
| 4 | Clean code, README results | Error analysis (10 examples) | Report: method and results | Report figures, slides, demo GIF |

**Checkpoints:** end of week 1, the S0 numbers exist · end of week 2, first drift result (go/no-go on the narrative) · end of week 3, all tables frozen.

**Fallback:** if week 1 finds a paper that already analyses abstention by turn, reframe around the "acceptance bias" of LLM answerability judges noted by AILS-NTUA (2026). The code stays the same.

**Compute:** one free Colab or Kaggle T4 per member is enough. Qwen2.5-7B in 4-bit uses about 6 GB, and the DeBERTa-v3-base checker trains in under an hour. Use `--set llm.model=Qwen/Qwen2.5-1.5B-Instruct` for quick iterations.
