"""Per-tensor-role precision resolution for low-precision training modeling.

Pure functions + pydantic config. No dependency on builder/simulator/ops so it
can be unit-tested in isolation. See
docs/superpowers/specs/2026-06-27-low-precision-training-modeling-design.md.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# Single source of truth for element sizes. fp4 stores two values per byte.
# Ref: NVIDIA OCP MX spec (mxfp4 = 4-bit element + shared e8m0 scale);
# FP8 formats per "FP8 Formats for Deep Learning" (Micikevicius et al. 2022).
_DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,          # ModelConfig.dtype alias (defaults to e4m3)
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp4": 0.5,        # generic 4-bit alias
    "fp4_e2m1": 0.5,
    "mxfp4": 0.5,
}

_COMPUTE_CLASS = {
    "fp32": "bf16",
    "bf16": "bf16",
    "fp16": "bf16",
    "fp8": "fp8",      # ModelConfig.dtype alias
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp4": "fp4",      # generic 4-bit alias
    "fp4_e2m1": "fp4",
    "mxfp4": "fp4",
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per element for a dtype (may be fractional, e.g. fp4 = 0.5)."""
    return _DTYPE_BYTES[dtype]


def compute_class(dtype: str) -> str:
    """Map a dtype to its matmul compute pipe class: 'bf16' | 'fp8' | 'fp4'."""
    return _COMPUTE_CLASS[dtype]


def scale_overhead_bytes(numel: float, block_size: int, scale_bytes: int = 4) -> float:
    """Quantization scale-metadata bytes.

    block_size=0 → one scale per tensor. block_size=B → ceil(numel/B) scales.
    Ref: fine-grained block scaling (DeepSeek-V3 tech report, per-128 blocks).
    """
    if block_size <= 0:
        return float(scale_bytes)
    num_blocks = -(-int(numel) // block_size)  # ceil division
    return float(num_blocks * scale_bytes)


class TensorPrecision(BaseModel):
    dtype: str = "bf16"
    block_size: int = 0          # 0 = per-tensor scale; >0 = fine-grained block
    hadamard: bool = False       # stochastic Hadamard transform before quant
    hadamard_block: int = 0      # rotation size; 0 → resolver default (128)
    scale_bytes: int = 4         # bytes per scale (4=fp32, 1=e8m0 for mxfp4)


class ModuleLinearPrecision(BaseModel):
    """Precision for one linear module's forward vs backward GEMMs.

    Ref: Zhou et al. 2025 (arXiv:2502.11458) — forward FP4 / backward FP8 asymmetry.
    """
    fwd: TensorPrecision = TensorPrecision()
    bwd: TensorPrecision = TensorPrecision()


class PrecisionConfig(BaseModel):
    weights: TensorPrecision = TensorPrecision()
    activations: TensorPrecision = TensorPrecision()
    gradients: TensorPrecision = TensorPrecision()
    comm: TensorPrecision = TensorPrecision()
    attn_linear: Optional[ModuleLinearPrecision] = None  # QKV + O projections
    ffn_linear: Optional[ModuleLinearPrecision] = None   # FFN gate/up/down
    master_dtype: str = "fp32"
    error_feedback: bool = False
    ef_dtype: str = "fp16"
    high_precision_layers: list[str] = []
    high_precision_period: int = 0
    high_precision_dtype: str = "bf16"

    @classmethod
    def bf16_default(cls) -> "PrecisionConfig":
        """All-bf16 recipe — reproduces today's (single-dtype) behavior."""
        return cls()

    @classmethod
    def from_model_dtype(cls, dtype: str) -> "PrecisionConfig":
        """Default recipe whose four tensor roles follow the model storage dtype.

        Spec §A: when no explicit PrecisionConfig is given, every tensor role
        (weights/activations/gradients/comm) defaults to the model's own dtype,
        not a hardcoded bf16. Master weights stay fp32 (mixed-precision Adam).

        For bf16/fp16 models this is a no-op equal to bf16_default() (2 B == 2 B),
        so existing single-dtype behavior is preserved bit-for-bit.
        """
        return cls(
            weights=TensorPrecision(dtype=dtype),
            activations=TensorPrecision(dtype=dtype),
            gradients=TensorPrecision(dtype=dtype),
            comm=TensorPrecision(dtype=dtype),
        )

    def linear_fwd(self, module: str) -> TensorPrecision:
        """Forward-GEMM precision for module 'attn'|'ffn'; falls back to global activations."""
        mp = self.attn_linear if module == "attn" else self.ffn_linear
        return mp.fwd if mp is not None else self.activations

    def linear_bwd(self, module: str) -> TensorPrecision:
        """Backward-GEMM precision for module 'attn'|'ffn'; falls back to global gradients."""
        mp = self.attn_linear if module == "attn" else self.ffn_linear
        return mp.bwd if mp is not None else self.gradients

    @classmethod
    def fp4_paper(cls) -> "PrecisionConfig":
        """Zhou et al. 2025 FP4 recipe: attn-proj FP8, FFN-fwd FP4, linear-bwd FP8."""
        fp8 = TensorPrecision(dtype="fp8_e4m3", block_size=128)
        fp4 = TensorPrecision(dtype="fp4_e2m1", block_size=128)
        return cls(
            attn_linear=ModuleLinearPrecision(fwd=fp8, bwd=fp8),
            ffn_linear=ModuleLinearPrecision(fwd=fp4, bwd=fp8),
        )
