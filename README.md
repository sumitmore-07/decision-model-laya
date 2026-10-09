# Laya Support Triage

Customer-support ticket triage with [Laya](https://github.com/NandhaKishorM/laya), a non-autoregressive "System 1" decision model. One forward pass answers five typed questions about a ticket (no text generation), and a small deterministic router turns the answers into a queue, team and SLA.

## What is Laya / what is a System 1 model?

"System 1" (fast, intuitive) vs "System 2" (slow, deliberate) is Kahneman's split. An LLM that writes out its answer token by token is System 2-style. Laya is System 1: an encoder (ModernBERT-large based, ~421M params) that reads the text and returns **typed answers directly**, in one pass:

| Primitive | Meaning | Raw output |
|---|---|---|
| **Choice** | pick one label from a set | label + probabilities |
| **Score** | ordinal level (here 0-2) | expected level, normalized to 0-1 in this repo |
| **Noul** | yes / no / unsure | P(true); this repo maps it to `true` / `false` / `null` (unsure) |

Laya `laya==0.4.1`, Apache-2.0. Model: `convaiinnovations/laya-typed-decisions` (Apache-2.0, English-only, a specialist trained on four synthetic workflows including customer service).

## Architecture

```
POST /triage, /batch, GET /health   (api/)
        |
core/model.py     singleton Laya model, loaded once at startup, inference serialized
core/questions.py the 5 typed questions (Choice x2, Score, Noul x2)
core/router.py    deterministic rules: decisions -> queue / team / SLA
```

Questions: `issue_type` (Choice), `urgency` (Score), `mentions_refund` (Noul), `customer_angry` (Noul), `assign_to` (Choice).

Routing (first match wins):

| Condition | Team | Queue | SLA |
|---|---|---|---|
| urgency > 0.8 AND angry AND refund | escalation_team | priority | 2h |
| issue_type = billing | billing_team | standard | 8h |
| issue_type = technical | engineering_team | standard | 24h |
| otherwise | tier1_support | normal | 48h |

Design notes:
- `assign_to` is **advisory**: the rules decide the team, and the model's suggestion is returned as `model_suggested_team` with a `team_mismatch` flag.
- Noul answers are probabilities. `P >= 0.6` is `true`, `P < 0.4` is `false`, otherwise `null` (unsure) and the ticket gets `needs_review: true`. Thresholds are env-configurable and **not calibrated**: tune them on your own tickets.
- Laya's shipped checkpoint warns that some confidence values are uncalibrated (the model card reports ECE 0.213). Treat confidences as rankings, not true probabilities.

## Run locally

Python 3.12 (3.14 is too new for torch wheels).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
python examples/run_demo.py            # 5 sample tickets (--all for 20)
uvicorn api.main:app --port 8000       # single worker: one model copy in RAM
pytest                                 # tests use a stub model, no weights needed
```

First start downloads ~1 GB of weights and takes ~80s; later starts take ~20s.

```bash
curl -s localhost:8000/health

curl -s localhost:8000/triage -H "Content-Type: application/json" \
  -d '{"ticket_id":"TKT-001","text":"Hi, we were billed twice for March. Please refund today or we cancel."}'

curl -s localhost:8000/batch -H "Content-Type: application/json" \
  -d '{"tickets":[{"text":"Dark mode please!"},{"text":"API returns 500 on every call"}]}'
```

Limits: ticket text 1-4000 chars (Laya context is 1024 tokens), up to 50 tickets per `/batch`. Model failure returns `503`.

## Measured results

20 hand-labelled tickets in [examples/tickets.json](examples/tickets.json) (small and written by us, so treat as a smoke test, not a benchmark of record). Hardware: Intel i7-1250U laptop, CPU only, no GPU.

| System | issue_type | urgent | refund | angry | queue | mean latency | p95 |
|---|---|---|---|---|---|---|---|
| Laya (`laya-typed-decisions`) | 85% | 100% | 90% | 80% | 90% | **4659 ms** | 5121 ms |

Reproduce: `python benchmarks/laya_vs_llm.py --skip-llm`. Add `GROQ_API_KEY` to `.env` and drop `--skip-llm` to compare against Groq's `llama-3.3-70b-versatile` on the same tickets with the same routing rules. See the script header for how cost is computed.

### Honest caveats

- **Latency:** Laya's published ~33 ms is for a T4 GPU. On this CPU it is ~4.7 s per ticket, and `predict_batch` gave no speedup (3 tickets took 11.6 s). A hosted LLM will be faster than Laya on CPU-only hardware. Laya's advantage here is no per-call API cost and no data leaving your infrastructure, not speed. Run on a GPU to get the fast path.
- **Cost:** Laya has no API cost; you pay for the compute it runs on. Groq publishes no public price for the 70B model (checked 2026-10), so no LLM dollar figure is claimed here.
- **Urgency** separates poorly: routine tickets still score ~0.6. The 0.8 cutoff works on this sample, but verify on your data.
- The sample is 20 tickets; one ticket is 5 percentage points.
