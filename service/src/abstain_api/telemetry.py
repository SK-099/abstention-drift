"""Metrics with OpenTelemetry, exported in Prometheus format at GET /metrics; errors to Sentry."""

from __future__ import annotations

import logging

from .settings import Settings

log = logging.getLogger(__name__)


class Metrics:
    def __init__(self):
        from opentelemetry import metrics

        m = metrics.get_meter("abstain_api")
        self.turns = m.create_counter("assistant_turns", description="assistant turns by decision")
        self.latency = m.create_histogram("assistant_turn_seconds", unit="s", description="end-to-end turn latency")
        self.stage = m.create_histogram("agent_stage_seconds", unit="s", description="time per agent stage")
        self.retries = m.create_counter("agent_retries", description="turns where the agent searched again")
        self.feedback = m.create_counter("feedback_events", description="user feedback by rating and decision")
        self.documents = m.create_counter("documents_indexed", description="uploaded documents by final status")
        self.flagged = m.create_counter(
            "injection_chunks_dropped", description="uploaded chunks dropped as prompt injection"
        )
        self.guardrail = m.create_counter(
            "guardrail_events", description="chat guardrail triggers by stage/rule/action"
        )

    DECISIONS = ("answer", "abstain", "clarify", "blocked")
    GUARD_EVENTS = (
        ("input", "prompt_injection", "blocked"),
        ("input", "toxicity", "blocked"),
        ("input", "secrets_pii", "redacted"),
        ("output", "toxicity", "blocked"),
        ("output", "secrets_pii", "redacted"),
    )
    FEEDBACK_REASONS = ("", "wrong_refusal", "wrong_answer", "bad_citation", "unclear", "other")
    DOC_STATUSES = ("indexed", "rejected", "failed")

    def prime(self, collections=(), api: bool = True, worker: bool = False) -> None:
        """Create every counter series at 0 before it is first incremented.

        Prometheus' increase()/rate() need a sample *before* the first event: a series that first appears
        with value 1 counts as 0 (so the first abstain/clarify/thumbs-down after each restart would be lost).
        """
        if api:
            for c in collections:
                self.prime_collection(c)
            for r in ("1", "-1"):
                for d in self.DECISIONS:
                    for reason in self.FEEDBACK_REASONS:
                        self.feedback.add(0, {"rating": r, "decision": d, "reason": reason})
            for stage, rule, action in self.GUARD_EVENTS:
                self.guardrail.add(0, {"stage": stage, "rule": rule, "action": action})
        if worker:
            for st in self.DOC_STATUSES:
                self.documents.add(0, {"status": st})
            self.flagged.add(0)

    def prime_collection(self, collection: str) -> None:
        for d in self.DECISIONS:
            self.turns.add(0, {"decision": d, "collection": collection})
        self.retries.add(0, {"collection": collection})

    def guardrail_events(self, events) -> None:
        for e in events:
            self.guardrail.add(1, {"stage": e["stage"], "rule": e["rule"], "action": e["action"]})

    def turn(self, decision: str, collection: str, seconds: float, timings: dict, retried: bool):
        attrs = {"decision": decision, "collection": collection}
        self.turns.add(1, attrs)
        self.latency.record(seconds, attrs)
        for k, v in (timings or {}).items():
            self.stage.record(float(v), {"stage": k})
        if retried:
            self.retries.add(1, {"collection": collection})


_metrics: Metrics | None = None


def setup_metrics() -> Metrics:
    """MeterProvider with a Prometheus reader (prometheus_client default registry)."""
    global _metrics
    if _metrics is None:
        from opentelemetry import metrics
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource

        provider = MeterProvider(
            resource=Resource.create({"service.name": "abstain-api"}), metric_readers=[PrometheusMetricReader()]
        )
        try:
            metrics.set_meter_provider(provider)
        except Exception:
            pass
        _metrics = Metrics()
    return _metrics


def get_metrics() -> Metrics:
    return setup_metrics()


def setup_sentry(s: Settings, component: str) -> None:
    if not s.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.env,
        traces_sample_rate=s.sentry_traces_sample_rate,
        send_default_pii=False,
        release=f"abstain-service@{__import__('abstain_api').__version__}",
    )
    sentry_sdk.set_tag("component", component)
    log.info("Sentry enabled for %s", component)
