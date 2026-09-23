from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deltajev.schema import Record, choice, noul, score  # noqa: E402


class MockModel:
    """Tiny stub exposing the HF causal-LM surface the engine touches:
    .parameters(), .eval(), forward(input_ids, ...) -> logits [B, T, V]."""

    def __init__(self, vocab_size: int = 200_000, pick_letter: str = "A"):
        import torch

        self.torch = torch
        self.vocab_size = vocab_size
        self.letter_a = ord("A")
        self.pick_letter = pick_letter
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.param])

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        torch = self.torch
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab_size)
        # put all probability mass on the token id of 'A'
        logits[:, -1, self.letter_a] = 10.0
        class Out:
            pass
        out = Out()
        out.logits = logits
        if kwargs.get("use_cache"):
            past = tuple(
                (torch.zeros(1, 1, t, 2), torch.zeros(1, 1, t, 2)) for _ in range(2)
            )
            out.past_key_values = past
        return out


class MockTokenizer:
    """Good-enough stand-in: letters and prompts are whitespace-ish tokens."""

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        ids = []
        for ch in text:
            if "A" <= ch <= "P" or ch in ":.":
                ids.append(ord(ch))
            else:
                ids.append(97)  # filler token
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids])
        return ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def make_record() -> Record:
    return Record(
        "Customer complains the invoice was charged twice after plan upgrade.",
        [
            noul("Is this ticket urgent?"),
            choice(
                "Which team should handle this?",
                {"billing": "payment issues", "technical": "product bugs"},
            ),
            score("Severity grade?", ["low", "mid", "high"]),
        ],
    )


def test_schema_validation():
    rec = make_record()
    assert len(rec.questions) == 3
    assert rec.questions[0].letters == ["A", "B"]
    try:
        choice("x", {str(i): "d" for i in range(20)})
        assert False, "should reject > 16 options"
    except Exception:
        pass


def test_engine_end_to_end():
    from deltajev.engine import DecisionEngine

    eng = DecisionEngine(MockModel(), MockTokenizer(), device="cpu", use_chat=False)
    rec = make_record()
    fresh = eng.score_record(rec)
    assert len(fresh.decisions) == 3
    # mock always answers 'A' -> true / billing / 0 with prob ~1
    assert fresh.decisions[0].value is True
    assert fresh.decisions[1].value == "billing"
    assert fresh.decisions[2].value == 0
    shared = eng.score_record_shared(rec)
    assert len(shared.decisions) == 3
    assert shared.timings["decisions_per_second"] > 0


def test_distribution_sums_to_one():
    from deltajev.engine import DecisionEngine

    eng = DecisionEngine(MockModel(), MockTokenizer(), device="cpu", use_chat=False)
    rec = make_record()
    for d in eng.score_record(rec).decisions:
        assert abs(sum(d.probabilities.values()) - 1.0) < 1e-4
