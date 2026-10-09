"""Laya vs a Groq-hosted LLM on the labelled sample tickets: accuracy, latency, cost.

Run: python benchmarks/laya_vs_llm.py [--model NAME] [--skip-llm] [--skip-laya]

Env (see .env.example): GROQ_API_KEY, GROQ_MODEL.
Cost: Groq publishes no public price for llama-3.3-70b-versatile ("contact sales"
as of 2026-10), so set GROQ_INPUT_USD_PER_M / GROQ_OUTPUT_USD_PER_M yourself to get
a $ figure; otherwise token counts are reported. Laya has no per-call API cost;
pass --hourly-usd (your machine/instance price per hour) to estimate compute cost.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from core.model import noul_answer  # noqa: E402
from core.router import route  # noqa: E402

LLM_PROMPT = """You triage customer support tickets. Reply with ONLY a JSON object:
{"issue_type": one of ["billing","technical","account","feature_request","complaint","general_inquiry"],
 "urgency": number 0 (routine) to 1 (critical, customer threatening to cancel or escalate),
 "mentions_refund": true/false (customer explicitly wants a refund or chargeback),
 "customer_angry": true/false (frustration, anger or strong dissatisfaction)}"""


def decisions_from(issue: str, urgency: float, refund: bool | None, angry: bool | None) -> dict[str, Any]:
    """Build the router's input shape so every system is routed by the same rules."""
    return {
        "issue_type": {"answer": issue, "confidence": 1.0},
        "urgency": {"score": urgency},
        "mentions_refund": {"answer": refund},
        "customer_angry": {"answer": angry},
        "assign_to": {"answer": "tier1_support"},
    }


def expected_queue(exp: dict[str, Any]) -> str:
    d = decisions_from(exp["issue_type"], 1.0 if exp["urgent"] else 0.0, exp["mentions_refund"], exp["customer_angry"])
    return route(d)["queue"]


def run_laya(tickets: list[dict], model_name: str | None) -> tuple[list[dict], list[float], dict]:
    from core.model import TriageModel

    m = TriageModel(model_name)
    print(f"[laya] loading {m.model_name}")
    m.load()
    m.predict("warmup")
    preds, lat = [], []
    for t in tickets:
        t0 = time.perf_counter()
        d = m.predict(t["text"])
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(
            {
                "issue_type": d["issue_type"]["answer"],
                "urgent": d["urgency"]["score"] > 0.8,
                "mentions_refund": d["mentions_refund"]["answer"],
                "customer_angry": d["customer_angry"]["answer"],
                "queue": route(d)["queue"],
            }
        )
    return preds, lat, {"model": m.model_name}


def run_llm(tickets: list[dict]) -> tuple[list[dict], list[float], dict]:
    import groq

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0)
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    preds, lat = [], []
    tok_in = tok_out = failures = 0
    for t in tickets:
        resp = None
        t0 = time.perf_counter()
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": t["text"]}],
                )
                break
            except groq.RateLimitError:
                time.sleep(2**attempt)  # backoff; free tier is rate limited
            except groq.APIError as e:
                print(f"[llm] {t['ticket_id']} API error: {e}")
                break
        ms = (time.perf_counter() - t0) * 1000
        try:
            if resp is None:
                raise ValueError("no response")
            out = json.loads(resp.choices[0].message.content)
            tok_in += resp.usage.prompt_tokens
            tok_out += resp.usage.completion_tokens
            urgency = float(out["urgency"])
            p = {
                "issue_type": out["issue_type"],
                "urgent": urgency > 0.8,
                "mentions_refund": bool(out["mentions_refund"]),
                "customer_angry": bool(out["customer_angry"]),
            }
            p["queue"] = route(decisions_from(p["issue_type"], urgency, p["mentions_refund"], p["customer_angry"]))["queue"]
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            failures += 1
            p = {"issue_type": None, "urgent": None, "mentions_refund": None, "customer_angry": None, "queue": None}
        preds.append(p)
        lat.append(ms)  # includes retries: that is what a caller experiences
    return preds, lat, {"model": model, "tokens_in": tok_in, "tokens_out": tok_out, "parse_failures": failures}


def accuracy(tickets: list[dict], preds: list[dict]) -> dict[str, float]:
    n = len(tickets)

    def acc(field: str, get: Callable[[dict], Any]) -> float:
        return sum(p[field] == get(t) for t, p in zip(tickets, preds, strict=True)) / n

    return {
        "issue_type": acc("issue_type", lambda t: t["expected"]["issue_type"]),
        "urgent": acc("urgent", lambda t: t["expected"]["urgent"]),
        "mentions_refund": acc("mentions_refund", lambda t: t["expected"]["mentions_refund"]),
        "customer_angry": acc("customer_angry", lambda t: t["expected"]["customer_angry"]),
        "queue": acc("queue", lambda t: expected_queue(t["expected"])),
    }


def summarize(name: str, tickets: list[dict], preds: list[dict], lat: list[float], meta: dict) -> dict[str, Any]:
    s = sorted(lat)
    return {
        "name": name,
        **meta,
        "n": len(tickets),
        "accuracy": accuracy(tickets, preds),
        "latency_ms": {"mean": statistics.mean(lat), "p50": statistics.median(lat), "p95": s[int(0.95 * (len(s) - 1))]},
    }


def cost_per_1000(r: dict[str, Any], hourly_usd: float | None) -> str:
    if "tokens_in" in r:
        pin, pout = os.environ.get("GROQ_INPUT_USD_PER_M"), os.environ.get("GROQ_OUTPUT_USD_PER_M")
        if not (pin and pout):
            per = (r["tokens_in"] + r["tokens_out"]) / r["n"] * 1000
            return f"n/a ({per:,.0f} tokens/1000 tickets; set GROQ_*_USD_PER_M)"
        usd = (r["tokens_in"] * float(pin) + r["tokens_out"] * float(pout)) / 1e6 / r["n"] * 1000
        return f"${usd:.4f} (at your list price)"
    if hourly_usd is not None:
        return f"${hourly_usd * r['latency_ms']['mean'] / 3.6e6 * 1000:.4f} (compute only)"
    return "$0 API (compute not priced)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Laya model id")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-laya", action="store_true")
    ap.add_argument("--hourly-usd", type=float, default=None, help="price/hour of the machine running Laya")
    args = ap.parse_args()

    tickets = json.loads((ROOT / "examples" / "tickets.json").read_text(encoding="utf-8"))
    results = []
    if not args.skip_laya:
        results.append(summarize("laya", tickets, *run_laya(tickets, args.model)))
    if not args.skip_llm:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY not set (put it in .env) or pass --skip-llm")
        results.append(summarize("groq-llm", tickets, *run_llm(tickets)))

    print(f"\n{len(tickets)} labelled tickets (same routing rules applied to every system)\n")
    print(f"{'system':<10}{'model':<42}{'type':>6}{'urgent':>8}{'refund':>8}{'angry':>7}{'queue':>7}{'mean ms':>9}{'p95 ms':>9}")
    for r in results:
        a, L = r["accuracy"], r["latency_ms"]
        print(
            f"{r['name']:<10}{r['model']:<42}{a['issue_type']:>6.0%}{a['urgent']:>8.0%}{a['mentions_refund']:>8.0%}"
            f"{a['customer_angry']:>7.0%}{a['queue']:>7.0%}{L['mean']:>9.0f}{L['p95']:>9.0f}"
        )
    print("\nCost per 1000 tickets:")
    for r in results:
        print(f"  {r['name']:<10}{cost_per_1000(r, args.hourly_usd)}")
    if any(r.get("parse_failures") for r in results):
        print("  (LLM parse failures count as wrong answers)")
    (ROOT / "benchmarks" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved benchmarks/results.json")


if __name__ == "__main__":
    main()
