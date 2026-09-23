# 2-page report outline (ACL short-paper style)

Use the ACL LaTeX template (Overleaf: "ACL 2023 Proceedings Template" or later). Target: 2 pages + references.

1. **Introduction** (~0.4 page). The problem (confident answers without evidence) · the 21.8% statistic · RQ1 and RQ2 · three contributions: (i) the abstention-drift metric and analysis, (ii) evidence-removal synthesis, (iii) a checker that reduces drift with false abstention capped at 5%.
2. **Related work** (~0.25 page). Abstention (Wen et al. 2025 TACL survey; Feng et al. 2024; AbstentionBench 2025) · conversational RAG (MTRAG, MTRAG-UN, SemEval-2026 Task 8 systems) · query rewriting (CANARD; multi-query rewriting 2024).
3. **Method** (~0.4 page). Pipeline figure (from README) · evidence removal with its two filters · checker and threshold rule · drift definition: slope of P(abstain | unanswerable) against turn, with a cluster bootstrap CI.
4. **Experimental setup** (~0.2 page). Data table (from `results/dataset_stats.md`) · conversation-level split · models · systems S0–S4.
5. **Results** (~0.5 page). Table: F1, recall by bucket, drift slope with CI, false abstention, RB_llm · Figure: `results/drift.png` · controlled logit · synthetic probe.
6. **Analysis and limitations** (~0.25 page). Error examples (late-turn follow-ups, near-duplicate evidence) · synthetic label noise rate from the 50-example hand check.
7. **Conclusion** (2–3 sentences).

Slides (prepare, don't submit): problem → gap (21.8%) → RQs → pipeline → evidence removal → drift plot → results table → demo → limitations.
