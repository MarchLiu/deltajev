# deltajev — Semantic ifs from Qwen3.8-27B

**Typed decisions + per-option probabilities from open weights, zero text generation. Does a Gated DeltaNet hybrid make decision-native inference cheaper?**

> Not affiliated with TypeSafe AI. This project reproduces the *interface
> pattern* of a typed decision engine (runtime criteria, `noul`/`choice`/`score`
> primitives, per-option probabilities, no decoding) on open weights. It does
> **not** reproduce any undisclosed proprietary model or training method.
> Method lineage: [SemIf](https://github.com/TheoLeeCJ/SemIf) /
> [JEV-CPU](https://github.com/leesk212/JEV-CPU); see `docs/experiment-design.md`.

## The hypothesis

Every existing open reproduction of the "System One" pattern scores options
with a *plain causal LM*. The cost driver is prefill over the state.
**Qwen3.8-27B** is a hybrid: 16×[3×GatedDeltaNet + 1×GatedAttention]. Only 16
of 64 layers keep a growing KV cache; the 48 GDN layers carry a fixed-size
recurrent state. Therefore:

- **H1 (latency):** fresh-mode per-decision latency scales better with state
  length than a dense causal LM of comparable quality.
- **H2 (shared-state):** one prefill → many decisions amplifies the GDN
  advantage, because branching adds KV cache only for 25% of layers.
- **H3 (quality):** 27B ≥ 4B on decision benchmarks, approaching published
  frontier-agreement numbers from the same protocol.

## Quick start

```bash
pip install -e ".[eval]"
# 27B needs ~55GB (BF16) — a 64GB+ Mac or one 80GB GPU works.
python -m deltajev.cli examples/records.jsonl --model Qwen/Qwen3.8-27B --mode shared
pytest -q   # mock-model smoke tests run without any weights
```

Input is JSONL: `{"state": "...", "questions": [{"qtype": "...", "prompt": "...", "options": {...}}]}`.
Output is JSONL of typed decisions with per-option probability distributions.

## Layout

```
src/deltajev/
  schema.py        noul / choice / score; letters A-P; runtime schema
  engine.py        slot pinning, single-forward scoring, shared-state mode
  cli.py           JSONL in, typed decisions + timings out
  eval/harness.py  balanced accuracy, TVD, decisions/s; frozen prompts, sha256 audit
tests/test_smoke.py  runs with a mock model — no weights needed
docs/experiment-design.md  protocols, baselines, ablations
configs/eval.yaml  frozen evaluation matrix
```

## Results

LocalLLaMA/typed-decisions, official 400-case test split (2,000 decisions), zero-shot
logit reading, fresh chat mode, no fine-tuning, no calibration (details in
`results/LOG.md`):

| System | Accuracy | Mean TVD | Source |
|---|---:|---:|---|
| random (option-count weighted) | 0.318 | — | computed |
| Qwen3.5-4B (deltajev v0.1) | 0.574 | 0.365 | measured |
| **Qwen3.8-27B BF16 (deltajev v0.1)** | **0.726** | **0.255** | measured |
| TypeSafe Jev (published) | 0.727 | — | [model card](https://typesafe.ai) |
| TypeSafe Jev (independent re-measure) | 0.626 | — | [laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark) |
| fine-tuned specialists (Laya / Verdict 2.0) | 0.73–0.77 | — | in-domain, not comparable |

Caveats: gold labels are 3-annotator consensus spreads (±0.02 argmax noise); our
noul prompting fix was tuned on a subset of this benchmark (mild dev leakage
until a held-out rerun); latency claims deferred until the GDN-optimized probe.
Latency (MPS, unoptimized GDN kernels): 27B 0.21 decisions/s, 4B 0.91.

## License

MIT. See `LICENSE`.
