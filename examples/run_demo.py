"""CLI demo: triage sample tickets with Laya.

Run: python examples/run_demo.py [--all] [--model NAME]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from core.model import TriageModel  # noqa: E402
from core.router import route  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="triage all tickets, not just 5")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    tickets = json.loads((ROOT / "examples" / "tickets.json").read_text(encoding="utf-8"))
    # first of each of 5 different types makes a more interesting default demo
    if not args.all:
        seen: set[str] = set()
        picked = []
        for t in tickets:
            k = t["expected"]["issue_type"]
            if k not in seen:
                seen.add(k)
                picked.append(t)
        tickets = picked[:5]

    model = TriageModel(args.model)
    print(f"Loading {model.model_name} ...")
    model.load()
    model.predict("warmup")  # first call is slower; keep it out of the numbers

    for t in tickets:
        t0 = time.perf_counter()
        d = model.predict(t["text"])
        ms = (time.perf_counter() - t0) * 1000
        r = route(d)
        print(f"\n{t['ticket_id']}  {t['text'][:90]}")
        print(
            f"  type={d['issue_type']['answer']} ({d['issue_type']['confidence']:.2f})  "
            f"urgency={d['urgency']['score']:.2f}  "
            f"refund={d['mentions_refund']['answer']} ({d['mentions_refund']['probability']:.2f})  "
            f"angry={d['customer_angry']['answer']} ({d['customer_angry']['probability']:.2f})"
        )
        print(
            f"  -> {r['queue']} queue | {r['team']} | SLA {r['sla_hours']}h | {r['reason']}"
            f"{' | NEEDS REVIEW' if r['needs_review'] else ''}  [{ms:.0f} ms]"
        )


if __name__ == "__main__":
    main()
