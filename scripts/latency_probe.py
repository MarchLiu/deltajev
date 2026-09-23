"""Latency probe: forward wall-time vs state length, fresh vs shared.

Frozen protocol (configs/eval.yaml):
  state_lengths_tokens: [254, 1363, 5165, 7697]
  repeats: 20 (per cell; first repeat discarded as warmup)
  record shape: 5 questions (matches the benchmark), max 4 options each

Reports median/p90 seconds per record and per decision for each
(length x mode) cell. No accuracy content — states are synthetic filler.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltajev.data import load_typed_decisions, to_record  # noqa: E402
from deltajev.engine import load_engine  # noqa: E402


def pad_state_to_tokens(engine, base_state: str, target_tokens: int) -> str:
    """Grow the state with filler fields until its token count >= target."""
    filler_unit = ', "log_entry_%d": "routine step completed without anomalies, retry threshold unchanged"'
    state = base_state
    i = 0
    while len(engine.tok.encode(state, add_special_tokens=False)) < target_tokens:
        state = state.rstrip("}") + filler_unit % i + "}"
        i += 1
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--lengths", type=int, nargs="+", default=[254, 1363, 5165, 7697])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results/raw/latency_probe.json")
    args = ap.parse_args()

    engine = load_engine(args.model, args.device)

    # fixed 5-question record shape from the benchmark's first case
    item = load_typed_decisions()[0]
    rec0, _ = to_record(item)
    n_q = len(rec0.questions)

    results = {"model": args.model, "device": args.device, "cells": []}
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(results, indent=2))

    for length in args.lengths:
        state = pad_state_to_tokens(engine, rec0.state, length)
        actual = len(engine.tok.encode(state, add_special_tokens=False))
        for mode in ("fresh", "shared"):
            times = []
            for rep in range(args.repeats + 1):  # +warmup
                rec = type(rec0)(state, rec0.questions, f"probe_{length}_{mode}")
                t0 = time.perf_counter()
                if mode == "shared":
                    engine.score_record_shared(rec)
                else:
                    engine.score_record(rec)
                times.append(time.perf_counter() - t0)
                if rep == 0:  # warmup done — save a preview cell early
                    results["cells"].append(
                        {
                            "state_tokens": actual,
                            "mode": mode,
                            "repeats": 0,
                            "warmup_s": round(times[0], 3),
                        }
                    )
                    save()
            times = times[1:]  # drop warmup
            cell = {
                "state_tokens": actual,
                "mode": mode,
                "repeats": len(times),
                "median_s": round(statistics.median(times), 3),
                "p90_s": round(sorted(times)[int(len(times) * 0.9)], 3),
                "per_decision_median_s": round(statistics.median(times) / n_q, 3),
            }
            # replace the warmup preview with the full cell
            results["cells"] = [
                c
                for c in results["cells"]
                if not (c["state_tokens"] == actual and c["mode"] == mode)
            ]
            results["cells"].append(cell)
            save()
            print(json.dumps(cell), flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
