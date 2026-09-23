"""Run deltajev on LocalLLaMA/typed-decisions.

Usage:
  python scripts/run_bench.py --model Qwen/Qwen3.5-4B --n 50 --mode shared
  python scripts/run_bench.py --model Qwen/Qwen3.5-4B --n 0   # full 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltajev.data import load_typed_decisions, to_record  # noqa: E402
from deltajev.engine import load_engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=50, help="records to run; 0 = all 400")
    ap.add_argument("--mode", choices=["fresh", "shared"], default="shared")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--workflow", default=None, help="restrict to one workflow")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", default="results/raw")
    args = ap.parse_args()

    if args.split == "train":
        import pandas as pd
        from deltajev.data import load_train_items
        items = load_train_items(workflow=args.workflow)
    else:
        items = load_typed_decisions(workflow=args.workflow)
    if args.n:
        items = items[: args.n]

    engine = load_engine(args.model, args.device, variant=args.variant)
    print(f"model={args.model} device={args.device} records={len(items)} mode={args.mode}", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_{args.variant}_{args.split}_{args.mode}_{args.n or 'full'}" + (f"_{args.workflow}" if args.workflow else "")
    pred_path = outdir / f"preds_{tag}.jsonl"

    correct = total = 0
    tvd_sum = 0.0
    tvd_n = 0
    by_type: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            rec, golds = to_record(item)
            result = (
                engine.score_record_shared(rec)
                if args.mode == "shared"
                else engine.score_record(rec)
            )
            for qi, dec in enumerate(result.decisions):
                gold_label, gold_dist = golds[qi]
                if gold_label is not None:
                    total += 1
                    pred = str(dec.value).lower()  # bools -> "true"/"false" to match gold
                    ok = pred == gold_label.lower() or pred.lstrip("0") == gold_label.lstrip("0")
                    correct += ok
                    c = by_type.setdefault(dec.qtype, [0, 0])
                    c[0] += ok
                    c[1] += 1
                if gold_dist:
                    tvd = 0.5 * sum(
                        abs(dec.probabilities.get(k, 0.0) - v) for k, v in gold_dist.items()
                    )
                    tvd_sum += tvd
                    tvd_n += 1
            f.write(
                json.dumps(
                    {
                        "record_id": result.record_id,
                        "workflow": item["workflow"],
                        "timings": result.timings,
                        "decisions": [d.to_json() for d in result.decisions],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                print(
                    f"[{i + 1}/{len(items)}] {el:.0f}s  "
                    f"acc={correct}/{total}={correct / total:.3f}  "
                    f"tvd={tvd_sum / max(tvd_n, 1):.3f}",
                    flush=True,
                )

    wall = time.perf_counter() - t0
    summary = {
        "model": args.model,
        "variant": args.variant,
        "split": args.split,
        "mode": args.mode,
        "records": len(items),
        "decisions_scored": sum(1 for _ in open(pred_path)) * 5,
        "accuracy": round(correct / total, 4) if total else None,
        "n_scored": total,
        "mean_tvd": round(tvd_sum / tvd_n, 4) if tvd_n else None,
        "accuracy_by_type": {
            t: {"acc": round(c / n, 4), "n": n} for t, (c, n) in sorted(by_type.items())
        },
        "wall_seconds": round(wall, 1),
        "decisions_per_second": round(total / wall, 3) if wall else None,
        "predictions": str(pred_path),
    }
    (outdir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
