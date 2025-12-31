import os
from typing import Any, Dict, Optional

from celery import Celery
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "dae_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

app = FastAPI(title="Distributed Async ML Engine", version="0.1.0")


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class InferenceQueued(BaseModel):
    task_id: str
    status: str = "QUEUED"


class ResultResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/inference", response_model=InferenceQueued)
def inference(req: InferenceRequest) -> InferenceQueued:
    """
    Accept a text payload and enqueue a Celery task.
    Returns immediately with a task_id.
    """
    # The worker defines the task name: "worker.tasks.run_inference"
    async_result = celery_app.send_task("worker.tasks.run_inference", args=[req.text])
    return InferenceQueued(task_id=async_result.id)


@app.get("/result/{task_id}", response_model=ResultResponse)
def get_result(task_id: str) -> ResultResponse:
    """
    Fetch status/result for a given task_id from Celery backend (Redis).
    """
    res = AsyncResult(task_id, app=celery_app)

    status = res.status  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
    payload: ResultResponse = ResultResponse(task_id=task_id, status=status)

    if status == "SUCCESS":
        payload.result = res.result if isinstance(res.result, dict) else {"value": res.result}
        return payload

    if status == "FAILURE":
        payload.error = str(res.result)
        return payload
    return payload
