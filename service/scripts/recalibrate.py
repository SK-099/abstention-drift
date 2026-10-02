"""Release step: re-calibrate the gate threshold tau and the self-check cut-off min_p on the MTRAG *dev*
split (never on the test set), then record the new values with the prompt version.

    python service/scripts/recalibrate.py --config configs/laptop.yaml [--set llm.ollama_model=qwen2.5:7b]

1. scripts/retune_threshold.py --on all   -> checkpoints/checker/best/threshold.json   (tau)
2. scripts/calibrate_self_check.py        -> checkpoints/self_check.json                 (min_p, for the serving LLM)
3. writes service/eval/release_versions.json and, if Langfuse keys are set, pushes prompts + thresholds.
Run it whenever the serving LLM, the prompts or the checker change, then re-run the quality gate live
(service/eval/run_sample.py + quality_gate.py) before tagging the release.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service/src"), str(ROOT / "src")]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/laptop.yaml")
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--skip-tau", action="store_true")
ap.add_argument("--skip-self-check", action="store_true")
a = ap.parse_args()
extra = ["--set", *a.set] if a.set else []


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


if not a.skip_tau:
    run("scripts/retune_threshold.py", "--config", a.config, "--on", "all", *extra)
if not a.skip_self_check:
    run("scripts/calibrate_self_check.py", "--config", a.config, *extra)

from abstain_api.prompts import versions
from abstain_api.settings import get_settings

s = get_settings()
v = versions(s) | {"calibrated_at": datetime.now(timezone.utc).isoformat(), "config": a.config, "overrides": a.set}
(ROOT / "service/eval/release_versions.json").write_text(json.dumps(v, indent=1))
print("release versions:", v)
if s.langfuse_public_key and s.langfuse_secret_key:
    run("service/scripts/push_prompts_to_langfuse.py")
