"""CLI: score a JSONL of records, fresh vs shared mode, print timings.

Record JSONL format:
  {"state": "...", "questions": [
      {"qtype": "noul", "prompt": "urgent?", "options": {"true": "...", "false": "..."}},
      {"qtype": "choice", "prompt": "...", "options": {"billing": "...", ...}},
      {"qtype": "score", "prompt": "...", "options": {"0": "low", "1": "mid", "2": "high"}}
  ]}
"""

from __future__ import annotations

import argparse
import json
import sys

from .engine import load_engine
from .schema import Record, Question


def record_from_json(obj: dict) -> Record:
    qs = [
        Question(q["qtype"], q["prompt"], {str(k): v for k, v in q["options"].items()})
        for q in obj["questions"]
    ]
    return Record(obj["state"], qs, obj.get("record_id", ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deltajev")
    ap.add_argument("input", help="JSONL of decision records ('-' for stdin)")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--mode", choices=["fresh", "shared"], default="shared")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="-", help="output JSONL ('-' for stdout)")
    args = ap.parse_args(argv)

    engine = load_engine(args.model, args.device)

    stream = sys.stdin if args.input == "-" else open(args.input)
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    for line in stream:
        line = line.strip()
        if not line:
            continue
        rec = record_from_json(json.loads(line))
        result = (
            engine.score_record_shared(rec)
            if args.mode == "shared"
            else engine.score_record(rec)
        )
        out.write(
            json.dumps(
                {
                    "record_id": result.record_id,
                    "timings": result.timings,
                    "decisions": [d.to_json() for d in result.decisions],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if out is not sys.stdout:
        out.close()
    return 0
