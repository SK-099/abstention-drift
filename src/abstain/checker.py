"""Answerability checker: a cross-encoder (DeBERTa-v3-base) that predicts P(unanswerable) for a
(question, retrieved passages) pair.

Two modes (config `checker.mode`):
  passage (default)  scores every (question, passage) pair separately for "this passage does NOT answer";
                     P(unanswerable) = min over the k passages. Each passage is read in full (up to
                     max_length tokens), so evidence deep inside a passage or in the 4th/5th passage is seen.
  joint              one pass over the question and all k passages concatenated (each cut to 600 chars).
                     Cheaper, but at 384-512 tokens it only sees the first 1-2 passages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pair_text(ex: dict, max_passage_chars: int = 600) -> tuple[str, str]:
    q = ex.get("query") or ex["question"]
    ctx = " || ".join(p["text"].strip().replace("\n", " ")[:max_passage_chars] for p in ex["passages"])
    return q, ctx


def passage_examples(examples: list[dict], gold: dict[str, set]) -> list[dict]:
    """Expand example-level data to (question, passage) pairs. label 1 = passage does not answer.

    unanswerable examples (real or synthetic): every passage -> 1
    answerable examples: gold passages -> 0; other passages are unlabeled and skipped
    """
    out = []
    for e in examples:
        g = set(e.get("gold_ids") or gold.get(e["task_id"], ()))
        for p in e["passages"]:
            if e["label"] == 1:
                out.append({"query": e.get("query") or e["question"], "passages": [p], "label": 1})
            elif p["document_id"] in g:
                out.append({"query": e.get("query") or e["question"], "passages": [p], "label": 0})
    return out


def choose_threshold(probs, labels, max_false_abstention: float = 0.05) -> dict:
    """Pick the threshold with the best abstention F1 among those that refuse at most
    `max_false_abstention` of answerable turns (RQ2: fix abstention *without hurting* answers)."""
    p, y = np.asarray(probs, float), np.asarray(labels, int)
    # fallback if no threshold meets the budget: never abstain
    best = {"threshold": float(p.max()) + 1e-6 if p.size else 1.0, "f1": -1.0, "recall": 0.0,
            "false_abstention_rate": 0.0}
    for t in np.unique(np.concatenate([p, [0.5]])):
        pred = p >= t
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        rec = tp / n_pos if n_pos else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        far = fp / n_neg if n_neg else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if far <= max_false_abstention and f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "recall": rec, "false_abstention_rate": far}
    return best


def train_checker(train: list[dict], dev: list[dict], cfg: dict, gold: dict[str, set] | None = None,
                  threshold_dev: list[dict] | None = None) -> dict:
    """Train on `train`, early-stop on `dev`, pick the threshold on `threshold_dev` (default: dev).
    `gold` maps task_id -> gold passage ids (needed for mode=passage)."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    c = cfg["checker"]
    mode = c.get("mode", "passage")
    ex_dev = threshold_dev if threshold_dev is not None else dev
    if mode == "passage":
        train, dev = passage_examples(train, gold or {}), passage_examples(dev, gold or {})
        print(f"passage-level pairs: train {len(train)} (no-answer {sum(e['label'] for e in train)}) | "
              f"dev {len(dev)}")
    max_chars = 4000 if mode == "passage" else 600
    tok = AutoTokenizer.from_pretrained(c["model"])
    # transformers >= 5 loads weights in their stored dtype (dtype="auto"); some checkpoints are fp16.
    # Mixed-precision training needs fp32 master weights, otherwise: "Attempting to unscale FP16 gradients".
    from transformers import AutoConfig

    # A cross-encoder re-ranker (1 relevance logit, e.g. BAAI/bge-reranker-base) is fine-tuned as is:
    # P(passage does not answer) = 1 - sigmoid(logit), so training starts from the zero-shot re-ranker.
    # Other encoders (e.g. DeBERTa) get a fresh 2-way head.
    one_logit = AutoConfig.from_pretrained(c["model"]).num_labels == 1 and mode == "passage"
    model = AutoModelForSequenceClassification.from_pretrained(c["model"], num_labels=1 if one_logit else 2).float()
    print("checker model:", c["model"], "| head:", "re-ranker logit" if one_logit else "2-way",
          "| device:", "cuda" if torch.cuda.is_available() else "CPU (slow!)")

    def enc(exs):
        a, b = zip(*[pair_text(e, max_chars) for e in exs])
        d = tok(list(a), list(b), truncation="only_second", max_length=c["max_length"])
        d["labels"] = [e["label"] for e in exs]
        return Dataset.from_dict(dict(d))

    # class weights: unanswerable is the minority class even after synthesis
    y = np.array([e["label"] for e in train])
    w = torch.tensor([1.0, max(1.0, (y == 0).sum() / max((y == 1).sum(), 1))], dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            if one_logit:   # relevance target = 1 - label; weight the minority class like the 2-way loss
                t = 1.0 - labels.float()
                wt = torch.where(labels == 1, w[1], w[0]).to(out.logits.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits[:, 0].float(), t, weight=wt)
            else:
                loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    import math
    steps = math.ceil(len(train) / (c["batch_size"] * c.get("grad_accum", 1))) * c["epochs"]
    args = TrainingArguments(
        output_dir=c["output_dir"], num_train_epochs=c["epochs"], learning_rate=c["lr"],
        per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=2 * c["batch_size"],
        gradient_accumulation_steps=c.get("grad_accum", 1), warmup_steps=max(1, int(0.1 * steps)), weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", fp16=torch.cuda.is_available(), report_to="none", seed=13,
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=enc(train), eval_dataset=enc(dev),
                              data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    out = Path(c["output_dir"]) / "best"
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    json.dump({"mode": mode}, open(out / "checker_mode.json", "w", encoding="utf-8"))

    checker = AnswerabilityChecker(str(out), max_length=c["max_length"])
    probs = checker.predict_proba(ex_dev)
    th = choose_threshold(probs, [e["label"] for e in ex_dev], c["max_false_abstention"])
    json.dump(th, open(out / "threshold.json", "w", encoding="utf-8"), indent=2)
    return th


class AnswerabilityChecker:
    def __init__(self, path: str, threshold: float | None = None, device: str | None = None,
                 max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        if not Path(path).exists():
            raise FileNotFoundError(f"No trained checker at '{path}'. Run scripts/train_checker.py first "
                                    "(and --real-only for S3).")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).float().to(self.device).eval()
        if str(self.device).startswith("cuda"):
            self.model.half()                 # fp16 on GPU: half the VRAM next to the LLM
        th_file = Path(path) / "threshold.json"
        self.threshold = threshold if threshold is not None else (
            json.load(open(th_file, encoding="utf-8"))["threshold"] if th_file.exists() else 0.5)
        mode_file = Path(path) / "checker_mode.json"
        self.mode = json.load(open(mode_file, encoding="utf-8"))["mode"] if mode_file.exists() else "joint"

    def predict_proba(self, examples: list[dict], batch_size: int = 16, max_length: int | None = None) -> np.ndarray:
        if self.mode == "passage":         # P(unanswerable) = min over passages of P(passage does not answer)
            flat = [{"query": e.get("query") or e["question"], "passages": [p]} for e in examples for p in e["passages"]]
            p = self._proba(flat, batch_size, max_length or self.max_length, 4000)
            out, i = [], 0
            for e in examples:
                n = len(e["passages"])
                out.append(float(p[i:i + n].min()) if n else 1.0)
                i += n
            return np.array(out)
        return self._proba(examples, batch_size, max_length or self.max_length, 600)

    def _proba(self, examples, batch_size, max_length, max_chars):
        import torch

        probs = []
        for i in range(0, len(examples), batch_size):
            a, b = zip(*[pair_text(e, max_chars) for e in examples[i:i + batch_size]])
            enc = self.tok(list(a), list(b), truncation="only_second", max_length=max_length,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
                p = 1 - torch.sigmoid(logits[:, 0]) if logits.shape[-1] == 1 else torch.softmax(logits, -1)[:, 1]
                probs.append(p.cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def __call__(self, query: str, passages: list[dict]) -> float:
        return float(self.predict_proba([{"query": query, "passages": passages}])[0])


def _unit(scores) -> np.ndarray:
    """Re-ranker scores as probabilities (sentence-transformers already applies a sigmoid for 1-label models)."""
    s = np.asarray(scores, float)
    return s if s.size == 0 or (s.min() >= 0 and s.max() <= 1) else 1 / (1 + np.exp(-s))


class RerankerScoreChecker:
    """Training-free baseline: P(unanswerable) = 1 - max re-ranker relevance of the retrieved passages.
    Uses the 'score' the cross-encoder re-ranker already attached to each passage, so it costs nothing."""

    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"No re-ranker threshold at '{path}'. Run scripts/fit_reranker_checker.py first.")
        self.threshold = json.load(open(path, encoding="utf-8"))["threshold"]

    @staticmethod
    def p_unanswerable(scores) -> float:
        s = _unit(scores)
        return float(1 - s.max()) if s.size else 1.0

    def __call__(self, query: str, passages: list[dict]) -> float:
        return self.p_unanswerable([p.get("score", 0.0) for p in passages])
