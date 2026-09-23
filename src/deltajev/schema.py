"""Typed question schema: Jev-style `noul` / `choice` / `score` primitives.

Schema is *runtime data*, never baked into weights. This module is the single
source of truth for option ordering and letter assignment (A-P, up to 16
options per question in direct mode).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOP"  # option slots; must each encode to a single token


class SchemaError(ValueError):
    pass


@dataclass
class Question:
    qtype: str  # "noul" | "choice" | "score"
    prompt: str
    options: dict[str, str]  # label -> description, insertion order is stable
    group: int = -1  # filled by Engine

    def __post_init__(self) -> None:
        if self.qtype not in ("noul", "choice", "score"):
            raise SchemaError(f"unknown qtype: {self.qtype!r}")
        if not (2 <= len(self.options) <= len(LETTERS)):
            raise SchemaError(
                f"{self.qtype}: need 2..{len(LETTERS)} options, got {len(self.options)}"
            )

    @property
    def letters(self) -> list[str]:
        return list(LETTERS[: len(self.options)])

    @property
    def labels(self) -> list[str]:
        return list(self.options.keys())


@dataclass
class Decision:
    """Typed answer for one question: value + per-option probability distribution."""

    qtype: str
    prompt: str
    probabilities: dict[str, float]  # label -> prob
    value: Any = None  # bool | label | score index
    confidence: float = 0.0

    def to_json(self) -> dict:
        return {
            "qtype": self.qtype,
            "prompt": self.prompt,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


@dataclass
class Record:
    """One decision request: unstructured state + typed questions."""

    state: str
    questions: list[Question]
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            payload = json.dumps(
                {"state": self.state, "questions": [q.options for q in self.questions]},
                sort_keys=True,
                ensure_ascii=False,
            )
            self.record_id = hashlib.sha256(payload.encode()).hexdigest()[:16]

    def max_options(self) -> int:
        return max(len(q.options) for q in self.questions)


def noul(prompt: str, true_desc: str = "yes", false_desc: str = "no") -> Question:
    """Boolean question. 10% of Jev-style data uses bare noul; we always
    materialise both poles so ordering is explicit and stable."""
    return Question("noul", prompt, {"true": true_desc, "false": false_desc})


def choice(prompt: str, options: dict[str, str]) -> Question:
    return Question("choice", prompt, dict(options))


def score(prompt: str, levels: list[str]) -> Question:
    """Ordinal grading; label is the level name, value is its index."""
    if not levels:
        raise SchemaError("score needs at least one level")
    return Question("score", prompt, {str(i): lv for i, lv in enumerate(levels)})
