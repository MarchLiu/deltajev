"""Loader for LocalLLaMA/typed-decisions (community benchmark).

400 test cases x 5 questions across 4 workflows, each question carrying a
gold label + annotator-consensus probability distribution. Adapted into
deltajev Records + gold tuples for the eval harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schema import Record, Question

DEFAULT_PARQUET = Path(__file__).resolve().parents[2] / "data" / "test-00000-of-00001.parquet"


def _load(path: str | Path) -> list[dict]:
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "id": row["id"],
                "workflow": row["workflow"],
                "state": row["state"],
                "questions": json.loads(row["questions"]),
                "gold": json.loads(row["gold"]),
                "label_agreement": json.loads(row["label_agreement"]),
            }
        )
    return out


def to_record(item: dict) -> tuple[Record, list[tuple[str | None, dict | None]]]:
    qs, golds = [], []
    for qname in item["questions"]:
        q = item["questions"][qname]
        criteria = q.get("criteria")
        if criteria is None:  # bare noul: implicit true/false poles
            options = {"true": "yes", "false": "no"}
        elif isinstance(criteria, list):  # score: label is its index
            options = {str(i): str(v) for i, v in enumerate(criteria)}
        else:
            options = {str(k): str(v) for k, v in criteria.items()}
        qs.append(Question(q["type"], q["instructions"], options))
        g = item["gold"].get(qname, {})
        golds.append((g.get("label"), g.get("probabilities")))
    return Record(item["state"], qs, item["id"]), golds


def load_typed_decisions(path: str | Path = DEFAULT_PARQUET, workflow: str | None = None) -> list[dict]:
    items = _load(path)
    return [it for it in items if workflow is None or it["workflow"] == workflow]


def load_train_items(workflow: str | None = None, data_dir: str | Path | None = None) -> list[dict]:
    """Load TRAIN-split records (one parquet per workflow)."""
    d = Path(data_dir or Path(__file__).resolve().parents[2] / "data")
    items: list[dict] = []
    for wf in WORKFLOWS:
        if workflow and wf != workflow:
            continue
        items.extend(_load(d / f"{wf}_train.parquet"))
    return items


WORKFLOWS = [
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
]
