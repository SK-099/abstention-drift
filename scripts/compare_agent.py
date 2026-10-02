"""Where does the agent (S5) change decisions compared with S2, and why?

    python scripts/compare_agent.py --config configs/laptop.yaml
    python scripts/compare_agent.py --config configs/laptop.yaml --agent S5-agent --base S2

Prints a transition table (S2 decision -> agent decision, split by gold label), the effect of each
agent step (retry, self-check, clarification), and clarification statistics on underspecified turns.
Writes results/agent_comparison.md.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from abstain.config import load_config
from abstain.data import task_from_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="mtragun")
ap.add_argument("--agent", default="S5-agent")
ap.add_argument("--base", default="S2")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out = Path(cfg["data"]["output_dir"])


def load(name):
    rows = {}
    for r in map(json.loads, open(out / f"{name}_{a.split}.jsonl", encoding="utf-8")):
        t = task_from_json(r | {"targets": r.get("targets", [])}, "e")
        if t.abstain_label is not None:
            rows[r["task_id"]] = (t.abstain_label, r)
    return rows


base, agent = load(a.base), load(a.agent)
ids = sorted(set(base) & set(agent))
lines = [f"# {a.agent} vs {a.base} on {a.split} ({len(ids)} turns)", ""]

trans = Counter()
for i in ids:
    y, rb = base[i]
    ra = agent[i][1]
    trans[("unanswerable" if y else "answerable", "abstain" if rb["abstain"]["abstained"] else "answer",
           ra["agent"]["decision"])] += 1
lines += ["| Gold label | " + a.base + " | " + a.agent + " | Turns |", "| --- | --- | --- | --- |"]
lines += [f"| {k[0]} | {k[1]} | {k[2]} | {v} |" for k, v in sorted(trans.items())]


def count(cond):
    return sum(1 for i in ids if cond(base[i][0], base[i][1], agent[i][1]))


ab = lambda r: r["abstain"]["abstained"]
rec = count(lambda y, b, g: y == 0 and ab(b) and not ab(g))
lost = count(lambda y, b, g: y == 1 and ab(b) and not ab(g))
caught = count(lambda y, b, g: y == 1 and not ab(b) and ab(g))
newfa = count(lambda y, b, g: y == 0 and not ab(b) and ab(g))
sc = Counter((("unans" if y else "ans")) for i in ids for y, g in [(base[i][0], agent[i][1])]
             if g["abstain"]["abstained_by"] == "self-check")
cl = Counter((("unans" if y else "ans")) for i in ids for y, g in [(base[i][0], agent[i][1])]
             if g["agent"]["decision"] == "clarify")
retried = sum(agent[i][1]["agent"]["attempts"] > 0 for i in ids)
lines += ["", "## Effect of the agent steps", "",
          f"- Answerable turns {a.base} refused that the agent now answers (recovered): **{rec}**",
          f"- Unanswerable turns {a.base} refused that the agent now answers (lost abstentions): **{lost}**",
          f"- Unanswerable turns {a.base} answered that the agent now refuses (new catches): **{caught}**",
          f"- Answerable turns {a.base} answered that the agent now refuses (new wrong refusals): **{newfa}**",
          f"- Turns retried: {retried}",
          f"- Self-check abstentions: {dict(sc)} (unans = correct, ans = wrong)",
          f"- Clarifying questions: {dict(cl)}"]

uf = out / f"{a.agent}_underspecified.jsonl"
if uf.exists():
    ur = [json.loads(l) for l in open(uf, encoding="utf-8")]
    d = Counter(r["agent"]["decision"] for r in ur)
    lines += ["", f"## Underspecified turns ({len(ur)}): decisions {dict(d)}",
              f"Clarification rate: {d.get('clarify', 0) / max(len(ur), 1):.3f}", ""]
    lines += [f"- {r['input'][-1]['text'][:90]!r} -> {r['predictions'][0]['text'][:90]!r}"
              for r in ur if r["agent"]["decision"] == "clarify"][:8]

text = "\n".join(lines)
print(text)
Path("results").mkdir(exist_ok=True)
Path("results/agent_comparison.md").write_text(text + "\n", encoding="utf-8")
print("\nwrote results/agent_comparison.md")
