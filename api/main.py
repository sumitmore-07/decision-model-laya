"""FastAPI app. Run: uvicorn api.main:app --port 8000 (single worker: model is ~1GB RAM)."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

load_dotenv()  # before core.model reads MODEL_NAME / HF_TOKEN

from api.endpoints import router  # noqa: E402
from core.model import TriageModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("triage.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model = TriageModel()
    await run_in_threadpool(model.load)
    t0 = time.perf_counter()
    await run_in_threadpool(model.predict, "warmup ticket: please help with my invoice")
    app.state.warmup_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    app.state.model = model
    logger.info("startup_complete warmup_ms=%s", app.state.warmup_latency_ms)
    yield


app = FastAPI(title="Laya Support Triage", version="0.1.0", lifespan=lifespan)
app.include_router(router)
