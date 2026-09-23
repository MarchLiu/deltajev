"""Evaluation harness: agreement with gold labels/distributions + timings.

Metrics (mirroring SemIf's frozen-matrix discipline):
  - balanced accuracy on argmax vs gold (when gold labels exist)
  - total variation distance between predicted and target distributions
  - decisions/second, fresh vs shared mode

Protocols are frozen before the run: prompts, ids and metrics live in
configs/eval.yaml; every predictions line carries a sha256 of its exact
prompt for byte-level auditability.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .engine import DecisionEngine
from .schema import Record


@dataclass
class EvalRow:
    record_id: str
    qtype: str
    prompt: str
    gold_label: str | None
    gold_dist: dict[str, float] | None
    pred_label: str
    pred_prob: float
    tvd: float | None
    correct: bool | None
    prompt_sha256: str = field(default="")

    def to_json(self) -> dict:
        return {
            "record_id": self.record_id,
            "qtype": self.qtype,
            "prompt": self.prompt,
            "gold_label": self.gold_label,
            "pred_label": self.pred_label,
            "pred_prob": round(self.pred_prob, 4),
            "tvd": None if self.tvd is None else round(self.tvd, 4),
            "correct": self.correct,
            "prompt_sha256": self.prompt_sha256,
        }


def _balanced_accuracy(pairs: list[tuple[str | None, str]]) -> float | None:
    by_class: dict[str, Counter] = {}
    for gold, pred in pairs:
        if gold is None:
            continue
        by_class.setdefault(gold, Counter())[pred == gold] += 1
    if not by_class:
        return None
    recalls = []
    for gold, c in by_class.items():
        total = c[True] + c[False]
        if total:
            recalls.append(c[True] / total)
    return sum(recalls) / len(recalls) if recalls else None


def evaluate(
    engine: DecisionEngine,
    records: Iterable[tuple[Record, list[tuple[str | None, dict | None]]]],
    mode: str = "shared",
) -> dict:
    """records: iterable of (record, golds) where golds align with
    record.questions: (gold_label | None, gold_dist | None)."""
    rows: list[EvalRow] = []
    wall0 = time.perf_counter()
    for rec, golds in records:
        t0 = time.perf_counter()
        result = engine.score_record_shared(rec) if mode == "shared" else engine.score_record(rec)
        elapsed = time.perf_counter() - t0
        for i, dec in enumerate(result.decisions):
            gold_label, gold_dist = golds[i] if i < len(golds) else (None, None)
            gold_label = (gold_label or "").strip() or None
            tvd = None
            if gold_dist:
                tvd = 0.5 * sum(
                    abs(dec.probabilities.get(k, 0.0) - v) for k, v in gold_dist.items()
                )
            correct = None if gold_label is None else str(dec.value) == gold_label
            prompt = engine.encode_prompt(rec, i)
            rows.append(
                EvalRow(
                    rec.record_id,
                    dec.qtype,
                    dec.prompt,
                    gold_label,
                    gold_dist,
                    str(dec.value),
                    dec.confidence,
                    tvd,
                    correct,
                    hashlib.sha256(prompt.encode()).hexdigest(),
                )
            )
        del elapsed  # per-record wall time is reported via aggregates below
    wall = time.perf_counter() - wall0

    n = len(rows)
    correct_rows = [r.correct for r in rows if r.correct is not None]
    tvds = [r.tvd for r in rows if r.tvd is not None]
    bal_acc = _balanced_accuracy([(r.gold_label, r.pred_label) for r in rows])
    return {
        "mode": mode,
        "rows": n,
        "accuracy": round(sum(correct_rows) / len(correct_rows), 4) if correct_rows else None,
        "balanced_accuracy": None if bal_acc is None else round(bal_acc, 4),
        "mean_tvd": round(sum(tvds) / len(tvds), 4) if tvds else None,
        "decisions_per_second": round(n / wall, 3) if wall else None,
        "wall_seconds": round(wall, 3),
        "raw": rows,
    }


def write_predictions(rows: list[EvalRow], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
