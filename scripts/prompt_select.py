"""Prompt-variant selection on the TRAIN split (held-out discipline).

Phase 2 (dev-set discipline): all prompt engineering happens here, on train
records only. The winning variant is frozen (name + sha) and test is run once.

Phase 3 (leave-one-workflow-out): the per-(variant x workflow) accuracies from
one pass over train subsets support fold selection without re-running: for
each held-out workflow W, pick argmax-variant on the other three workflows'
train accuracies, then score W's TEST slice with the frozen engine.

Usage:
  python scripts/prompt_select.py --model Qwen/Qwen3.5-4B --per-workflow 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltajev.data import to_record  # noqa: E402
from deltajev.engine import VARIANTS, load_engine  # noqa: E402
from deltajev.schema import Record, Question  # noqa: E402

WORKFLOWS = [
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
]


def load_train(workflow: str, n: int) -> list[dict]:
    df = pd.read_parquet(f"data/{workflow}_train.parquet")
    if n:
        df = df.head(n)
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "id": row["id"],
                "workflow": workflow,
                "state": row["state"],
                "questions": json.loads(row["questions"]),
                "gold": json.loads(row["gold"]),
            }
        )
    return out


def score(engine, items, per_type=None):
    c = t = 0
    for item in items:
        rec, golds = to_record(item)
        result = engine.score_record(rec)
        for qi, dec in enumerate(result.decisions):
            gl, _ = golds[qi]
            if gl is None:
                continue
            pred = str(dec.value).lower()
            ok = pred == gl.lower() or pred.lstrip("0") == gl.lstrip("0")
            c += ok
            t += 1
            if per_type is not None:
                key = (item["workflow"], dec.qtype)
                cc = per_type.setdefault(key, [0, 0])
                cc[0] += ok
                cc[1] += 1
    return c, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--per-workflow", type=int, default=30)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default="results/raw/prompt_selection_train.json")
    args = ap.parse_args()

    engine = load_engine(args.model, args.device, variant="base")
    data = {
        wf: load_train(wf, args.per_workflow) for wf in WORKFLOWS
    }
    total = sum(len(v) for v in data.values())
    print(f"model={args.model} variants={args.variants} train records={total}", flush=True)

    per_type: dict = {}
    rows = []
    t0 = time.perf_counter()
    for variant in args.variants:
        engine.variant = variant  # swap prompt only; weights stay loaded
        c = n = 0
        by_wf = {}
        for wf, items in data.items():
            cw, nw = score(engine, items)
            by_wf[wf] = {"acc": round(cw / nw, 4), "n": nw} if nw else None
            c += cw
            n += nw
            print(f"  {variant} {wf}: {cw}/{nw}={cw / nw:.3f}", flush=True)
        rows.append(
            {
                "variant": variant,
                "acc": round(c / n, 4),
                "n": n,
                "by_workflow": by_wf,
            }
        )
        print(f"[{variant}] overall {c}/{n}={c / n:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # freeze manifest: variant name + sha of the variant spec
    frozen = max(rows, key=lambda r: r["acc"])
    manifest = {
        "model": args.model,
        "per_workflow_records": args.per_workflow,
        "frozen_variant": frozen["variant"],
        "frozen_sha256": hashlib.sha256(
            json.dumps(VARIANTS[frozen["variant"]], sort_keys=True).encode()
        ).hexdigest()[:16],
        "rows": rows,
        "per_type": {f"{k[0]}|{k[1]}": v for k, v in sorted(per_type.items())},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("frozen:", frozen["variant"], "->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
