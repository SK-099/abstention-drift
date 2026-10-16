"""Celery worker (Redis broker): document indexing and the nightly evaluation run.

celery -A abstain_api.worker worker --loglevel=INFO
celery -A abstain_api.worker beat   --loglevel=INFO      # schedules the nightly evaluation
"""

from __future__ import annotations

import logging
import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_init

from .settings import get_settings
from .telemetry import setup_sentry

log = logging.getLogger(__name__)
s = get_settings()
celery_app = Celery("abstain", broker=s.redis_url, backend=s.redis_url)
celery_app.conf.update(
    task_always_eager=s.celery_eager,
    task_eager_propagates=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    timezone="UTC",
    beat_schedule={
        "nightly-eval": {"task": "abstain.nightly_eval", "schedule": crontab(hour=s.nightly_eval_hour_utc, minute=0)}
    },
)
setup_sentry(s, "worker")


@worker_init.connect
def _start_metrics_server(**_):
    """Prometheus endpoint of the worker (documents indexed, injection drops); scraped at worker:9101."""
    port = int(os.environ.get("WORKER_METRICS_PORT", "9101"))
    if port:
        from prometheus_client import start_http_server

        from .telemetry import setup_metrics

        setup_metrics().prime(api=False, worker=True)
        start_http_server(port)


@celery_app.task(name="abstain.index_document", bind=True, max_retries=2, default_retry_delay=30)
def index_document_task(self, doc_id: str, model: str | None = None) -> dict:
    from .db import sync_sessionmaker
    from .indexing import index_document
    from .telemetry import get_metrics

    with sync_sessionmaker()() as session:
        doc = index_document(session, doc_id, get_settings())
        get_metrics().documents.add(1, {"status": doc.status})
        if doc.n_flagged:
            get_metrics().flagged.add(doc.n_flagged)
        out = {"id": doc.id, "status": doc.status, "n_passages": doc.n_passages, "n_flagged": doc.n_flagged}
    if out["status"] == "indexed" and get_settings().source_guides:
        if get_settings().celery_eager:
            source_guide_task.apply(args=[doc_id, model])
        else:
            source_guide_task.delay(doc_id, model)
    return out


@celery_app.task(name="abstain.source_guide", max_retries=1, default_retry_delay=60)
def source_guide_task(doc_id: str, model: str | None = None) -> dict:
    """Summary, key topics and suggested questions for one source (runs after indexing; a new LLM call)."""
    from .db import sync_sessionmaker
    from .indexing import make_guide

    with sync_sessionmaker()() as session:
        doc = make_guide(session, doc_id, get_settings(), model=model)
        return {"id": doc_id, "guide_status": doc.guide_status if doc else None}


@celery_app.task(name="abstain.studio", max_retries=0)
def studio_task(artifact_id: str) -> dict:
    """One Studio output (briefing doc, study guide, FAQ, timeline, mind map, flashcards, quiz, audio)."""
    from .db import sync_sessionmaker
    from .studio import generate_artifact

    with sync_sessionmaker()() as session:
        a = generate_artifact(session, artifact_id, get_settings())
        return {"id": artifact_id, "status": a.status}


@celery_app.task(name="abstain.speak", max_retries=0)
def speak_task(artifact_id: str) -> dict:
    """Speak an audio overview's script again (e.g. after starting the voice service)."""
    from .db import sync_sessionmaker
    from .studio import respeak

    with sync_sessionmaker()() as session:
        a = respeak(session, artifact_id, get_settings())
        return {"id": artifact_id, "audio": bool(a and a.data.get("audio"))}


@celery_app.task(name="abstain.nightly_eval")
def nightly_eval_task(limit: int | None = None) -> dict:
    from .db import sync_sessionmaker
    from .evaluation import run_sample_eval
    from .models import EvalRun

    result = run_sample_eval(get_settings(), limit=limit)
    with sync_sessionmaker()() as session:
        session.add(
            EvalRun(
                kind="nightly", metrics=result["metrics"], passed=result["gate"]["passed"], versions=result["versions"]
            )
        )
        session.commit()
    log.info("nightly eval: %s", result["gate"])
    return {"metrics": result["metrics"], "gate": result["gate"]}
