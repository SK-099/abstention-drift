import math

from abstain.metrics import abstention_report, controlled_logit, drift, report_by_bucket


def rec(conv, turn, label, pred, **kw):
    return {"conversation_id": conv, "turn": turn, "label": label, "pred": pred, **kw}


def test_report_basic():
    r = abstention_report([1, 1, 0, 0], [1, 0, 1, 0])
    assert r["abstain_recall"] == 0.5 and r["false_abstention_rate"] == 0.5 and r["abstain_precision"] == 0.5


def test_report_no_positives():
    r = abstention_report([0, 0], [0, 1])
    assert math.isnan(r["abstain_recall"]) and r["false_abstention_rate"] == 0.5


def test_drift_detects_decline():
    # abstains on early unanswerable turns, fails on late ones -> negative slope, CI below 0
    rs = []
    for c in range(40):
        rs += [rec(f"c{c}", 1, 1, 1), rec(f"c{c}", 2, 1, 1), rec(f"c{c}", 7, 1, 0), rec(f"c{c}", 8, 1, 0)]
    d = drift(rs, "recall", n_boot=200)
    assert d["slope"] < 0 and d["ci_high"] < 0 and d["n_conversations"] == 40


def test_drift_flat():
    rs = [rec(f"c{c}", t, 1, 1) for c in range(10) for t in (1, 4, 8)]
    assert abs(drift(rs, "recall", n_boot=50)["slope"]) < 1e-9


def test_buckets():
    rs = [rec("a", 1, 1, 1), rec("a", 4, 1, 0), rec("a", 7, 0, 1)]
    b = report_by_bucket(rs, [[1, 2], [3, 5], [6, 99]], ["early", "mid", "late"])
    assert b["early"]["abstain_recall"] == 1 and b["mid"]["abstain_recall"] == 0
    assert b["late"]["false_abstention_rate"] == 1


def test_controlled_logit_sign():
    rs = []
    for c in range(30):
        for t in range(1, 9):
            rs.append(rec(f"c{c}", t, 1, int(t <= 4 or (c % 5 == 0)), collection="fiqa" if c % 2 else "govt"))
    assert controlled_logit(rs)["odds_ratio_per_turn"] < 1
