"""/triage, /batch, /health."""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.schemas import BatchIn, BatchOut, HealthOut, TicketIn, TriageOut
from core.model import TriageModel
from core.router import route

logger = logging.getLogger("triage.api")
router = APIRouter()


def _model(request: Request) -> TriageModel:
    model: TriageModel | None = getattr(request.app.state, "model", None)
    if model is None or not model.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


async def _infer(model: TriageModel, texts: list[str]) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    try:
        decisions = await run_in_threadpool(model.predict_batch, texts)
    except Exception as e:  # model failure must not leak internals or crash the worker
        logger.exception("inference_failed n=%d", len(texts))
        raise HTTPException(status_code=503, detail="inference failed") from e
    return decisions, (time.perf_counter() - t0) * 1000


def _out(ticket: TicketIn, decisions: dict, latency_ms: float) -> TriageOut:
    return TriageOut(
        ticket_id=ticket.ticket_id or f"TKT-{uuid.uuid4().hex[:8]}",
        input=ticket.text,
        decisions=decisions,
        routing=route(decisions),
        latency_ms=round(latency_ms, 1),
    )


@router.post("/triage", response_model=TriageOut)
async def triage(ticket: TicketIn, request: Request) -> TriageOut:
    model = _model(request)
    decisions, ms = await _infer(model, [ticket.text])
    return _out(ticket, decisions[0], ms)


@router.post("/batch", response_model=BatchOut)
async def batch(body: BatchIn, request: Request) -> BatchOut:
    model = _model(request)
    decisions, ms = await _infer(model, [t.text for t in body.tickets])
    per_item = ms / len(body.tickets)
    results = [_out(t, d, per_item) for t, d in zip(body.tickets, decisions, strict=True)]
    return BatchOut(count=len(results), total_latency_ms=round(ms, 1), results=results)


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    model: TriageModel | None = getattr(request.app.state, "model", None)
    loaded = bool(model and model.loaded)
    return HealthOut(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        model_name=model.model_name if model else "",
        warmup_latency_ms=getattr(request.app.state, "warmup_latency_ms", None),
    )
