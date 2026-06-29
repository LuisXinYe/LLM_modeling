"""Analytical matmul-only compute-cost model for low-precision recipes.

Mirrors the accounting in Zhou et al. 2025 (arXiv:2502.11458), Fig 1a / Table 2:
count forward + backward matmul FLOPs per transformer block, scale each by a
per-compute-class speed, and report cost as a percentage of the all-FP16 cost.
The MHA core (QK^T + softmax*V) is kept at FP16 (FlashAttention) in the denominator.

NOTE: under the paper's stated speeds (FP16=1, FP8=2, FP4=4) the all-FP4-linear
recipe is ~36%, NOT the paper's Table-2 57.1% (the cost-% is invariant to the
fwd/bwd multiplier, so this is not a backward-counting artifact). The paper's
Table-2 metric implies an effective FP4 ~2x; pass `speed_map` to match it.
"""
from __future__ import annotations

from llm_perf.config import ModelConfig
from llm_perf.precision import PrecisionConfig, compute_class

SPEED_MAP_PAPER = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0}


def _speed(dtype: str, speed_map: dict) -> float:
    cc = compute_class(dtype)            # "bf16" | "fp8" | "fp4"
    key = "fp16" if cc == "bf16" else cc  # bf16 class == FP16 baseline
    return speed_map[key]


def theoretical_compute_cost(
    model_cfg: ModelConfig, precision_cfg: PrecisionConfig,
    speed_map: dict | None = None, seq_len: int = 4096,
) -> dict:
    """Matmul-only theoretical compute cost as % of all-FP16. See module docstring."""
    if speed_map is None:
        speed_map = SPEED_MAP_PAPER
    layer = model_cfg.get_layers()[0]
    d = model_cfg.hidden_size
    d_qo = layer.num_heads * layer.head_dim
    d_kv = layer.num_kv_heads * layer.head_dim
    d_ff = layer.intermediate_size

    # Per-token forward FLOPs (matmuls only).
    attn_linear_fwd = (2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d)  # QKV + O
    mha_core_fwd = 4 * d_qo * seq_len                                             # QK^T + softmax*V
    ffn_fwd = 6 * d * d_ff                                                        # SwiGLU 3 matmuls

    fwd_total = attn_linear_fwd + mha_core_fwd + ffn_fwd
    forward_split = {
        "ffn": ffn_fwd / fwd_total,
        "attn_linear": attn_linear_fwd / fwd_total,
        "mha_core": mha_core_fwd / fwd_total,
    }

    # Forward + backward (backward = 2x forward) scaled by speed.
    def t(flops_fwd: float, fwd_dtype: str, bwd_dtype: str) -> float:
        return flops_fwd / _speed(fwd_dtype, speed_map) + 2 * flops_fwd / _speed(bwd_dtype, speed_map)

    attn_fwd_dt = precision_cfg.linear_fwd("attn").dtype
    attn_bwd_dt = precision_cfg.linear_bwd("attn").dtype
    ffn_fwd_dt = precision_cfg.linear_fwd("ffn").dtype
    ffn_bwd_dt = precision_cfg.linear_bwd("ffn").dtype

    t_assigned = (
        t(attn_linear_fwd, attn_fwd_dt, attn_bwd_dt)
        + t(ffn_fwd, ffn_fwd_dt, ffn_bwd_dt)
        + t(mha_core_fwd, "fp16", "fp16")  # core always FP16
    )
    t_fp16 = t(attn_linear_fwd, "fp16", "fp16") + t(ffn_fwd, "fp16", "fp16") + t(mha_core_fwd, "fp16", "fp16")

    return {
        "cost_pct": 100.0 * t_assigned / t_fp16,
        "forward_split": forward_split,
        "breakdown": {
            "attn_linear": (attn_fwd_dt, attn_bwd_dt),
            "ffn": (ffn_fwd_dt, ffn_bwd_dt),
        },
    }
