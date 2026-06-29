import pytest
from llm_perf.precision import (
    dtype_bytes, compute_class, scale_overhead_bytes,
    TensorPrecision, PrecisionConfig, ModuleLinearPrecision,
)


def test_dtype_bytes_map():
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp16") == 2
    assert dtype_bytes("fp8_e4m3") == 1
    assert dtype_bytes("fp8_e5m2") == 1
    assert dtype_bytes("fp4_e2m1") == 0.5
    assert dtype_bytes("mxfp4") == 0.5


def test_compute_class_map():
    assert compute_class("bf16") == "bf16"
    assert compute_class("fp16") == "bf16"
    assert compute_class("fp32") == "bf16"
    assert compute_class("fp8_e4m3") == "fp8"
    assert compute_class("fp8_e5m2") == "fp8"
    assert compute_class("fp4_e2m1") == "fp4"
    assert compute_class("mxfp4") == "fp4"


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        dtype_bytes("int3")


def test_scale_overhead_per_tensor_is_negligible():
    # block_size=0 → one scale per tensor
    assert scale_overhead_bytes(1024, block_size=0, scale_bytes=4) == 4


def test_scale_overhead_fine_grained():
    # 1024 elements, block 128 → 8 scales * 4 bytes
    assert scale_overhead_bytes(1024, block_size=128, scale_bytes=4) == 32


def test_precision_config_default_is_all_bf16():
    cfg = PrecisionConfig.bf16_default()
    assert cfg.weights.dtype == "bf16"
    assert cfg.activations.dtype == "bf16"
    assert cfg.gradients.dtype == "bf16"
    assert cfg.comm.dtype == "bf16"
    assert cfg.master_dtype == "fp32"


def test_tensor_precision_fields():
    tp = TensorPrecision(dtype="fp4_e2m1", block_size=128, hadamard=True, hadamard_block=128)
    assert tp.block_size == 128
    assert tp.hadamard is True


def test_module_linear_precision_defaults():
    m = ModuleLinearPrecision()
    assert m.fwd.dtype == "bf16" and m.bwd.dtype == "bf16"


def test_resolver_falls_back_to_global_roles_when_unset():
    pc = PrecisionConfig(
        activations=TensorPrecision(dtype="fp8_e4m3"),
        gradients=TensorPrecision(dtype="fp16"),
    )
    # attn_linear/ffn_linear unset → fall back to global activations/gradients
    assert pc.linear_fwd("attn").dtype == "fp8_e4m3"
    assert pc.linear_bwd("ffn").dtype == "fp16"


def test_resolver_uses_module_precision_when_set():
    pc = PrecisionConfig(
        ffn_linear=ModuleLinearPrecision(
            fwd=TensorPrecision(dtype="fp4_e2m1"), bwd=TensorPrecision(dtype="fp8_e4m3")
        )
    )
    assert pc.linear_fwd("ffn").dtype == "fp4_e2m1"
    assert pc.linear_bwd("ffn").dtype == "fp8_e4m3"
    # attn still falls back (unset) to global activations default bf16
    assert pc.linear_fwd("attn").dtype == "bf16"


def test_fp4_paper_recipe():
    pc = PrecisionConfig.fp4_paper()
    assert pc.linear_fwd("attn").dtype == "fp8_e4m3"   # attention protected
    assert pc.linear_fwd("ffn").dtype == "fp4_e2m1"    # FFN forward FP4
    assert pc.linear_bwd("attn").dtype == "fp8_e4m3"   # backward FP8
    assert pc.linear_bwd("ffn").dtype == "fp8_e4m3"
    assert pc.master_dtype == "fp32"
