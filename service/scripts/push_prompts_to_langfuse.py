"""Register the agent's prompts and thresholds in Langfuse prompt management (one version per change).

    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=http://localhost:3001 \
        python service/scripts/push_prompts_to_langfuse.py [--label production]

Each prompt is stored as "abstain/<name>" with the service's prompt version and hash in its config, and
the gate threshold tau and self-check min_p as "abstain/thresholds", so every trace (tagged with
prompt:<version>) can be matched to the exact prompts and cut-offs that produced it.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parents[1] / "src")]

from abstain_api.prompts import PROMPTS, prompt_hash, versions
from abstain_api.settings import get_settings

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="production")
a = ap.parse_args()
s = get_settings()
from langfuse import Langfuse

lf = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)
cfg = {"prompt_version": s.prompt_version, "prompt_hash": prompt_hash()}
for name, text in PROMPTS.items():
    lf.create_prompt(name=f"abstain/{name}", prompt=text, labels=[a.label], config=cfg, type="text")
    print("pushed", name)
lf.create_prompt(
    name="abstain/thresholds",
    prompt="gate tau and self-check min_p (see config)",
    labels=[a.label],
    config=versions(s),
    type="text",
)
lf.flush()
print(f"versions: {versions(s)}")
