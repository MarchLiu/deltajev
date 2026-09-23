"""Decision-native inference: zero text generation, one forward pass per record.

Pipeline (adapted from the SemIf / JEV-CPU calling convention):
  1. validate record (strict schema, no silent truncation)
  2. encode options as single-token letters A-P (verify round-trip + pinning)
  3. one forward pass, logits_to_keep=1 (last position only)
  4. gather letter slots -> per-question softmax over that question's options

shared-state mode: when many criteria judge the same state, prefill once into
the KV cache, then run all questions as one batched scoring pass. On Qwen3.8
this is where GDN should shine: 48 of 64 layers carry only a fixed-size
recurrent state, so branch cost is dominated by the 16 attention layers.

The model is ANY causal LM with a standard HF interface; the point of this
repo is the Qwen3.8-27B hybrid, but the engine is deliberately model-agnostic.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch

from .schema import LETTERS, Decision, Question, Record, SchemaError

SYSTEM_INSTRUCTION = (
    "You are a decision engine. Read the state, then for each question output "
    "exactly one uppercase letter corresponding to the best option. "
    "No explanation. Answer with a single letter."
)


@dataclass
class ScoredRecord:
    record_id: str
    decisions: list[Decision]
    timings: dict = field(default_factory=dict)


class DecisionEngine:
    def __init__(self, model, tokenizer, device: str | None = None, use_chat: bool = True):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device
        self.use_chat = use_chat
        self._verify_slots()

    # ---------- chat templating ----------

    def _chat_prompt(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    # ---------- slot pinning ----------

    def _verify_slots(self) -> None:
        """Each letter must round-trip to exactly one token, and the answer
        boundary must be stable: appending a letter may add one token or
        deterministically merge the boundary tail with the letter (BPE), but
        the prefix before that must be untouched."""
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise SchemaError(f"letter {letter!r} is not a single token: {ids}")
            if self.tok.decode(ids) != letter:
                raise SchemaError(f"letter {letter!r} round-trip failed")
        self._slot_ids_for("Decide now. Answer:")

    def _slot_ids_for(self, boundary: str) -> dict[str, int]:
        """Resolve the token id each letter takes as the continuation of
        `boundary` (context-dependent under BPE merges like ':A' or ' A')."""
        base = self.tok.encode(boundary, add_special_tokens=False)
        slots: dict[str, int] = {}
        for letter in LETTERS:
            ext = self.tok.encode(boundary + letter, add_special_tokens=False)
            if len(ext) == len(base) + 1 and ext[:-1] == base:
                sid = ext[-1]
            elif (
                len(ext) == len(base)
                and ext[:-1] == base[:-1]
                and ext[-1] != base[-1]
            ):
                sid = ext[-1]  # BPE merged boundary tail + letter (e.g. ':A')
            else:
                raise SchemaError(
                    f"answer boundary unstable for {letter!r}: {base} -> {ext}"
                )
            slots[letter] = sid
        if len(set(slots.values())) != len(slots):
            raise SchemaError(f"letter slots are not distinct at boundary {boundary!r}")
        return slots

    # ---------- encoding ----------

    def _question_block(self, q: Question) -> str:
        if q.qtype == "noul":
            # Instruction is a proposition: frame it as an explicit true/false
            # judgment, otherwise instruction-tuned models just agree with it.
            lines = [f"Q: Judge this statement about the state. Statement: {q.prompt}"]
        else:
            lines = [f"Q: {q.prompt}"]
        for letter, (label, desc) in zip(q.letters, q.options.items()):
            lines.append(f"{letter}. {label}: {desc}")
        return "\n".join(lines)

    def encode_prompt(self, rec: Record, question_idx: int | None = None) -> str:
        parts = [SYSTEM_INSTRUCTION, f"STATE:\n{rec.state}"]
        if question_idx is None:
            for q in rec.questions:
                parts.append(self._question_block(q))
            parts.append("Answer with one letter per question.")
        else:
            parts.append(self._question_block(rec.questions[question_idx]))
            parts.append("Answer:")
        prompt = "\n\n".join(parts)
        return prompt

    def encode_shared_prefix(self, rec: Record) -> str:
        """Prefix for shared-state mode: all questions once, ending on the
        first answer anchor. Branch i continues with tail_shared(rec, i)."""
        parts = [SYSTEM_INSTRUCTION, f"STATE:\n{rec.state}"]
        for q in rec.questions:
            parts.append(self._question_block(q))
        parts.append("Answer with one letter per question.")
        parts.append("ANSWERS:\nA1:")
        return "\n\n".join(parts)

    @staticmethod
    def tail_shared(rec: Record, question_idx: int) -> str:
        """Continuation that reaches the answer slot of question question_idx
        from the shared prefix end ('' for the first question)."""
        return "" if question_idx == 0 else f"\nA{question_idx + 1}:"

    def _question_content(self, rec: Record, question_idx: int) -> str:
        return f"STATE:\n{rec.state}\n\n{self._question_block(rec.questions[question_idx])}\n\nAnswer with a single letter."

    def _boundary_prompt(self, rec: Record, question_idx: int) -> str:
        """Full prompt whose next-token distribution holds the letter slots."""
        raw = self.encode_prompt(rec, question_idx)
        return self._chat_prompt(self._question_content(rec, question_idx)) if self.use_chat else raw

    def _forward_last_logits(self, prompt: str) -> torch.Tensor:
        ids = self.tok.encode(
            prompt,
            add_special_tokens=not self.use_chat,  # template carries its own specials
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(ids)
        return out.logits[0, -1, :]

    # ---------- scoring ----------

    def score_question(self, rec: Record, question_idx: int) -> Decision:
        q = rec.questions[question_idx]
        prompt = self._boundary_prompt(rec, question_idx)
        logits = self._forward_last_logits(prompt)
        slot_ids = self._slot_ids_for(prompt)
        slots = torch.tensor(
            [slot_ids[l] for l in q.letters], device=logits.device
        )
        probs = torch.softmax(logits[slots], dim=-1).tolist()
        return self._to_decision(q, probs)

    def score_record(self, rec: Record) -> ScoredRecord:
        """Fresh scoring: one forward per question. Baseline mode."""
        t0 = time.perf_counter()
        decisions = [self.score_question(rec, i) for i in range(len(rec.questions))]
        return ScoredRecord(
            rec.record_id,
            decisions,
            {"mode": "fresh", "seconds": round(time.perf_counter() - t0, 4)},
        )

    def score_record_shared(self, rec: Record) -> ScoredRecord:
        """Shared-state scoring: encode state ONCE, then score every question's
        answer boundary from the same prefill. Implementation detail: we build
        the shared prefix (system + state + all question blocks WITHOUT answer
        slot), then for each question append only the answer boundary tokens.
        With HF this is done via explicit past_key_values reuse."""
        t0 = time.perf_counter()
        if not rec.questions:
            return ScoredRecord(rec.record_id, [], {"mode": "shared", "seconds": 0.0})

        # shared prefix: system + state + all question blocks + "ANSWERS:\nA1:"
        if self.use_chat:
            content = (
                f"STATE:\n{rec.state}\n\n"
                + "\n\n".join(self._question_block(q) for q in rec.questions)
                + "\n\nAnswer with one letter per question.\n\nANSWERS:\nA1:"
            )
            prefix = self._chat_prompt(content)
        else:
            prefix = self.encode_shared_prefix(rec)
        prefix_ids = self.tok.encode(prefix, add_special_tokens=not self.use_chat)
        input_ids = torch.tensor([prefix_ids], device=self.device)
        attention = torch.ones_like(input_ids)
        prefix_len = len(prefix_ids)

        with torch.inference_mode():
            out = self.model(input_ids, attention_mask=attention, use_cache=True)
            past = out.past_key_values
            decisions = []
            for i, q in enumerate(rec.questions):
                tail = self.tail_shared(rec, i)
                tail_ids = (
                    self.tok.encode(tail, add_special_tokens=False) if tail else []
                )
                tail_t = (
                    torch.tensor([tail_ids], device=self.device)
                    if tail_ids
                    else None
                )
                n_ctx = prefix_len + (tail_t.shape[1] if tail_t is not None else 0)
                if tail_t is not None:
                    out2 = self.model(
                        tail_t,
                        attention_mask=torch.ones(
                            (1, n_ctx), dtype=torch.long, device=self.device
                        ),
                        past_key_values=past,
                        use_cache=False,
                    )
                    logits = out2.logits[0, -1, :]
                else:
                    logits = out.logits[0, -1, :]
                boundary = tail if tail else "ANSWERS:\nA1:"
                slot_ids = self._slot_ids_for(boundary)
                slots = torch.tensor(
                    [slot_ids[l] for l in q.letters], device=logits.device
                )
                probs = torch.softmax(logits[slots], dim=-1).tolist()
                decisions.append(self._to_decision(q, probs))

        return ScoredRecord(
            rec.record_id,
            decisions,
            {
                "mode": "shared",
                "seconds": round(time.perf_counter() - t0, 4),
                "decisions_per_second": round(
                    len(decisions) / max(time.perf_counter() - t0, 1e-6), 3
                ),
            },
        )

    def _to_decision(self, q: Question, probs: list[float]) -> Decision:
        pairs = list(zip(q.labels, probs))
        best_label, best_p = max(pairs, key=lambda kv: kv[1])
        value: object
        if q.qtype == "noul":
            value = best_label == "true"
        elif q.qtype == "score":
            value = int(best_label)
        else:
            value = best_label
        return Decision(q.qtype, q.prompt, dict(pairs), value, best_p)


def load_engine(model_name: str, device: str | None = None, dtype=None) -> DecisionEngine:
    """Load any HF causal LM. Default target: Qwen/Qwen3.8-27B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": dtype or "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device:
        model = model.to(device)
    return DecisionEngine(model, tok, device)
