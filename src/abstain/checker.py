"""Answerability checker: a cross-encoder that reads (question, retrieved passages) and predicts
P(unanswerable). Small enough (DeBERTa-v3-base, 184M params) to train on a free T4 in < 1 hour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pair_text(ex: dict, max_passage_chars: int = 600) -> tuple[str, str]:
    q = ex.get("query") or ex["question"]
    ctx = " || ".join(p["text"].strip().replace("\n", " ")[:max_passage_chars] for p in ex["passages"])
    return q, ctx


def choose_threshold(probs, labels, max_false_abstention: float = 0.05) -> dict:
    """Pick the threshold with the best abstention F1 among those that refuse at most
    `max_false_abstention` of answerable turns (RQ2: fix abstention *without hurting* answers)."""
    p, y = np.asarray(probs, float), np.asarray(labels, int)
    best = {"threshold": 0.5, "f1": -1.0, "recall": 0.0, "false_abstention_rate": 1.0}
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


def train_checker(train: list[dict], dev: list[dict], cfg: dict) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    c = cfg["checker"]
    tok = AutoTokenizer.from_pretrained(c["model"])
    # transformers >= 5 loads weights in their stored dtype (dtype="auto"); some checkpoints are fp16.
    # Mixed-precision training needs fp32 master weights, otherwise: "Attempting to unscale FP16 gradients".
    model = AutoModelForSequenceClassification.from_pretrained(c["model"], num_labels=2).float()

    def enc(exs):
        a, b = zip(*[pair_text(e) for e in exs])
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

    checker = AnswerabilityChecker(str(out))
    probs = checker.predict_proba(dev)
    th = choose_threshold(probs, [e["label"] for e in dev], c["max_false_abstention"])
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
        th_file = Path(path) / "threshold.json"
        self.threshold = threshold if threshold is not None else (
            json.load(open(th_file, encoding="utf-8"))["threshold"] if th_file.exists() else 0.5)

    def predict_proba(self, examples: list[dict], batch_size: int = 16, max_length: int | None = None) -> np.ndarray:
        import torch

        max_length = max_length or self.max_length
        probs = []
        for i in range(0, len(examples), batch_size):
            a, b = zip(*[pair_text(e) for e in examples[i:i + batch_size]])
            enc = self.tok(list(a), list(b), truncation="only_second", max_length=max_length,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs.append(torch.softmax(self.model(**enc).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def __call__(self, query: str, passages: list[dict]) -> float:
        return float(self.predict_proba([{"query": query, "passages": passages}])[0])
