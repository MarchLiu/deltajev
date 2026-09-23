"""deltajev: typed decisions from open models, zero text generation.

Not affiliated with TypeSafe AI. We reproduce the *interface pattern*
(runtime criteria, typed options, per-option calibrated-ish probabilities,
no decoding) on open weights — specifically Qwen3.8-27B's GDN hybrid —
and measure whether linear attention makes decision-native inference
cheaper. We do NOT reproduce any undisclosed model or training method.
"""

__version__ = "0.1.0"
