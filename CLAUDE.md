# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Customer-support ticket triage built on Laya (`laya==0.4.1`, model `convaiinnovations/laya-typed-decisions`), a non-autoregressive "System 1" model that answers typed questions (Choice / Score / Noul) in one forward pass. A deterministic router turns the answers into queue, team and SLA.

## Commands

Use the Python 3.12 venv at `.venv` (3.14 is too new for torch wheels). PowerShell on Windows.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt          # runtime + pytest/httpx
pytest                                       # 18 tests, stub model, ~2s, no weights
pytest tests/test_router.py::test_noul_thresholds   # single test
uvicorn api.main:app --port 8000             # keep ONE worker (each loads its own model copy)
python examples/run_demo.py [--all]          # real model; first run downloads ~1 GB
python benchmarks/laya_vs_llm.py --skip-llm  # Laya-only accuracy/latency on examples/tickets.json
```

No linter or formatter is configured. Swagger UI is at `/docs` when the server is up.

## Architecture

Request flow: `api/endpoints.py` -> `core.model.TriageModel.predict_batch` (in a threadpool) -> `core.model.normalize` -> `core.router.route` -> `api/schemas.py` response.

- **`core/questions.py`** holds the five `TRIAGE_QUESTIONS` in Laya's real schema: `type` + `instructions` + `criteria` (dict of label -> description for `choice`, ordered list of anchors for `score`). Not `question`/`options`.
- **`core/model.py`** has two jobs. `TriageModel` is the model wrapper: `load()` runs once in the FastAPI lifespan, `laya` is imported lazily inside it (keeps tests and stubs light), and inference is serialized by a lock. `normalize()` converts Laya's raw output into the shape the rest of the code uses.
- **`core/router.py`** is pure Python with no ML, so rules are unit-testable with hand-built decision dicts (see `decisions()` in `tests/test_router.py`).
- **`api/main.py`** loads `.env` *before* importing `core.model` (env is read at call time there). Lifespan loads the model, runs a warmup predict, and stores `app.state.model` / `warmup_latency_ms`.
- **`benchmarks/laya_vs_llm.py`** runs every system through the same `route()` rules via `decisions_from()`, so queue accuracy is comparable between Laya and the Groq LLM.

## Non-obvious behavior

- **Raw Laya output is not API-shaped.** Score is an expected ordinal level 0..2 (normalized to 0..1 by dividing by `URGENCY_MAX_LEVEL`). Noul is `P(true)`, mapped to `True` / `False` / `None` (unsure) via `NOUL_TRUE_AT_OR_ABOVE` (0.6) and `NOUL_FALSE_BELOW` (0.4). Thresholds are env-configurable and uncalibrated.
- **Use `answer_confidence` and `probabilities`, not Laya's `confidence` field.** Laya warns the checkpoint's temperatures are invalid for some choice entries, so its `confidence` is uncalibrated.
- **`assign_to` is advisory.** Routing never uses it; it is returned as `model_suggested_team` with a `team_mismatch` flag.
- **`needs_review`** is set when either noul answer is `None` or `issue_type` confidence < 0.5. Escalation needs urgency strictly `> 0.8` and both noul answers `True`.
- **`/batch` makes one `predict_batch` call.** Per-item `latency_ms` is batch wall time divided by batch size. On CPU, batching gives no speedup.
- **`os.environ.setdefault("USE_TF", "0")`** at the top of `core/model.py` stops `laya.load()` hanging when TensorFlow is probed. Keep it above the laya import.
- **Latency is hardware-bound.** The published ~33 ms is for a T4 GPU; on the dev laptop (CPU only) it is ~4.7 s per ticket. Do not put speed or cost-ratio claims in docs without measuring.
- **Groq publishes no price for `llama-3.3-70b-versatile`**, so the benchmark takes `GROQ_INPUT_USD_PER_M` / `GROQ_OUTPUT_USD_PER_M` from env and otherwise reports token counts.
- `examples/tickets.json` (20 hand-labelled tickets with an `expected` block) is both the demo input and the benchmark ground truth. `benchmarks/results.json` is generated and gitignored.
- Tests build their own `FastAPI()` app with a `StubModel` and a `TestClient` used without a context manager, which skips the lifespan.
