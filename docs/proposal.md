# Project proposal (for faculty approval)

**Title:** Knowing When Not to Answer: Abstention Drift in Multi-Turn Retrieval-Augmented Conversations

**Team:** <names, roll numbers>

**Proposal.** Retrieval-augmented conversational assistants often answer confidently even when retrieved evidence cannot support an answer; the strongest SemEval-2026 multi-turn RAG systems detect only about one in five unanswerable turns. We hypothesise that this failure worsens as conversations progress ("abstention drift"), because later questions depend more on context. We propose a multi-turn RAG pipeline with query rewriting, hybrid retrieval with re-ranking, and a lightweight answerability checker trained on unanswerable examples created automatically by removing the answer-bearing documents from retrieval. We evaluate on MTRAG (SemEval-2026 Task 8, TACL 2025) and MTRAG-UN, reporting per-turn abstention quality alongside standard answer-faithfulness metrics.

**Dataset:** MTRAG and MTRAG-UN (SemEval-2026 Task 8, Apache-2.0).
