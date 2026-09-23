"""Abstention metrics and the abstention-drift statistic.

A *record* is a dict with at least:
    conversation_id, turn, label (1 = should abstain, 0 = should answer), pred (1 = system abstained)
and optionally collection, multi_turn, source.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import turn_bucket


def abstention_report(labels, preds) -> dict:
    y, p = np.asarray(labels, int), np.asarray(preds, int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_pos if n_pos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(rec) else 0.0
    return {
        "n": len(y), "n_unanswerable": n_pos, "n_answerable": n_neg,
        "abstain_precision": prec,
        "abstain_recall": rec,                                   # RQ1: does this fall with turn?
        "abstain_f1": f1,
        "false_abstention_rate": fp / n_neg if n_neg else float("nan"),   # RQ2: must stay low
        "accuracy": float((y == p).mean()) if len(y) else float("nan"),
    }


def report_by_bucket(records: list[dict], buckets, names) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in records:
        groups[turn_bucket(r["turn"], buckets, names)].append(r)
    out = {}
    for n in names:
        g = groups.get(n, [])
        out[n] = abstention_report([r["label"] for r in g], [r["pred"] for r in g])
    return out


def _slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def drift(records: list[dict], which: str = "recall", n_boot: int = 1000, seed: int = 13,
          max_turn: int | None = None) -> dict:
    """Abstention drift = change in abstention behaviour per extra conversation turn.

    which='recall'        -> slope of P(abstain | unanswerable) vs turn   (negative = drift, bad)
    which='false_abstain' -> slope of P(abstain | answerable) vs turn     (positive = over-refusal creeping in)

    Linear-probability slope, with a 95% CI from a cluster bootstrap that resamples
    whole conversations (turns within a conversation are not independent).
    """
    want = 1 if which == "recall" else 0
    rs = [r for r in records if r["label"] == want and (max_turn is None or r["turn"] <= max_turn)]
    if not rs:
        return {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "n_conversations": 0}
    point = _slope([r["turn"] for r in rs], [r["pred"] for r in rs])

    by_conv = defaultdict(list)
    for r in rs:
        by_conv[r["conversation_id"]].append(r)
    convs = list(by_conv)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(len(convs), size=len(convs), replace=True)
        xs, ys = [], []
        for i in sample:
            for r in by_conv[convs[i]]:
                xs.append(r["turn"]); ys.append(r["pred"])
        s = _slope(xs, ys)
        if not np.isnan(s):
            boots.append(s)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"slope": point, "ci_low": float(lo), "ci_high": float(hi), "n": len(rs), "n_conversations": len(convs)}


def controlled_logit(records: list[dict], label_value: int = 1) -> dict:
    """Logistic regression of abstaining on turn, controlling for domain and follow-up type.

    Answers: 'is the turn effect still there once we account for domain / question type?'
    Returns the turn coefficient (log-odds per turn) and its odds ratio.
    """
    from sklearn.linear_model import LogisticRegression

    rs = [r for r in records if r["label"] == label_value]
    if len({r["pred"] for r in rs}) < 2:
        return {"turn_coef": float("nan"), "odds_ratio_per_turn": float("nan"), "n": len(rs)}
    cols = sorted({r.get("collection", "na") for r in rs})
    mts = sorted({r.get("multi_turn", "N/A") for r in rs})
    X = np.array([[r["turn"]] + [r.get("collection", "na") == c for c in cols[1:]]
                  + [r.get("multi_turn", "N/A") == m for m in mts[1:]] for r in rs], float)
    y = np.array([r["pred"] for r in rs])
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)   # ~unregularised
    coef = float(m.coef_[0][0])
    return {"turn_coef": coef, "odds_ratio_per_turn": float(np.exp(coef)), "n": len(rs),
            "controls": {"collection": cols[1:], "multi_turn": mts[1:]}}
