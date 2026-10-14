"""Per-model refusal thresholds.

The agent abstains when the gate score p >= tau, when the generator refuses, or when p >= min_p and the
self-check judges the answer unsupported. The gate does not depend on the LLM, but how often the LLM refuses
on its own and how strict its self-check is do, so tau and min_p are calibrated for every model on the MTRAG
dev split (service/scripts/calibrate_models.py) and stored as one JSON file per model in CALIBRATION_DIR:

    checkpoints/calibration/local_qwen3_1.7b.json   {"model_id": "local:qwen3:1.7b", "tau": .., "min_p": .., ...}

The research model (LLM_MODEL) keeps the thresholds of the research runs (checker threshold.json and
self_check.json) unless a file for it exists. A model without a calibration file uses the research values
and is shown without the shield in the model menu.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    tau: float
    min_p: float
    calibrated: bool
    source: str  # "research" | "calibrated" | "default"
    stats: dict | None = None


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model_id).strip("_")


class CalibrationStore:
    """Reads CALIBRATION_DIR/*.json, re-reading when a file changes (no restart after a calibration run)."""

    def __init__(self, s: Settings):
        self.s = s
        self._sig: tuple = ()
        self._by_model: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict]:
        d = Path(self.s.calibration_dir)
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        sig = tuple((f.name, f.stat().st_mtime_ns) for f in files)
        if sig == self._sig:
            return self._by_model
        out = {}
        for f in files:
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
                if c.get("model_id") and c.get("complete", True):
                    out[c["model_id"]] = c
            except Exception as e:  # a half-written file while a run is in progress
                log.warning("skipping calibration file %s: %s", f, e)
        with self._lock:
            self._sig, self._by_model = sig, out
        return out

    def all(self) -> dict[str, dict]:
        return dict(self._load())

    def get(self, model_id: str) -> Thresholds:
        from .model_registry import default_model_id

        c = self._load().get(model_id)
        if c:
            return Thresholds(float(c["tau"]), float(c["min_p"]), True, "calibrated", c)
        research = model_id in {default_model_id(self.s), f"local:{self.s.llm_model}"}
        return Thresholds(
            self.s.resolved_gate_threshold(),
            self.s.resolved_min_p(),
            research,
            "research" if research else "default",
        )


_store: CalibrationStore | None = None


def get_calibrations(s: Settings) -> CalibrationStore:
    global _store
    if _store is None or _store.s is not s:
        _store = CalibrationStore(s)
    return _store
