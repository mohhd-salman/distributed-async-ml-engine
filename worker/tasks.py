import os
import time
from typing import Any, Dict

from celery import Celery

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "dae_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.task_default_queue = "inference.jobs"
celery_app.conf.task_default_exchange = "inference"
celery_app.conf.task_default_routing_key = "inference.jobs"

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Model loading (load once per worker process)
_PIPELINE = None


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline
        _PIPELINE = pipeline("sentiment-analysis")
    return _PIPELINE


@celery_app.task(
    name="worker.tasks.run_inference",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def run_inference(self, text: str) -> Dict[str, Any]:
    """
    Runs sentiment analysis and returns a JSON-serializable dict with timing.
    """
    time.sleep(0.05)

    queued_at = time.time()
    started_at = time.time()

    t0 = time.perf_counter()

    pipe = get_pipeline()
    out = pipe(text)[0]

    t1 = time.perf_counter()
    finished_at = time.time()

    compute_s = t1 - t0
    worker_total_s = finished_at - started_at

    return {
        "label": out.get("label"),
        "score": float(out.get("score")),
        "model": "hf-sentiment-analysis",
        "timing": {
            "worker_queued_at": queued_at,
            "worker_started_at": started_at,
            "worker_finished_at": finished_at,
            "compute_s": compute_s,
            "worker_total_s": worker_total_s,
        },
    }
