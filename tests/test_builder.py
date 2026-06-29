"""Tests for builder.py — SimOp generation from config."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_perf.builder import SimOp, build_generation_step, build_layer_ops, build_training_step
from llm_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.precision import PrecisionConfig, TensorPrecision

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def model_cfg() -> ModelConfig:
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))


@pytest.fixture
def hw() -> HardwareConfig:
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))


@pytest.fixture
def parallel_cfg_tp1() -> ParallelismConfig:
    return ParallelismConfig(tp=1, pp=1, dp=1, ep=1)


@pytest.fixture
def parallel_cfg_tp4() -> ParallelismConfig:
    return ParallelismConfig(tp=4, pp=1, dp=1, ep=1)


@pytest.fixture
def parallel_cfg_dp2() -> ParallelismConfig:
    return ParallelismConfig(tp=1, pp=1, dp=2, ep=1)


@pytest.fixture
def rl_cfg() -> WorkloadConfig:
    return WorkloadConfig(
        total_prompts=100,
        group_size=8,
        avg_prompt_len=512,
        avg_response_len=512,
        train_micro_batch_size=2,
        gen_batch_size=8,
    )


@pytest.fixture
def single_layer(model_cfg: ModelConfig) -> LayerConfig:
    return model_cfg.get_layers()[0]


# ---------------------------------------------------------------------------
# Test 1: build_layer_ops returns SimOps
# ---------------------------------------------------------------------------


def test_build_layer_ops_returns_simops(single_layer, model_cfg, parallel_cfg_tp1, hw):
    result = build_layer_ops(
        layer_cfg=single_layer,
        model_cfg=model_cfg,
        parallel_cfg=parallel_cfg_tp1,
        hw=hw,
        batch=2,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    assert isinstance(result, list)
    assert len(result) > 0
    for op in result:
        assert isinstance(op, SimOp)


# ---------------------------------------------------------------------------
# Test 2: build_layer_ops has compute stream
# ---------------------------------------------------------------------------


def test_build_layer_ops_has_compute_stream(single_layer, model_cfg, parallel_cfg_tp1, hw):
    result = build_layer_ops(
        layer_cfg=single_layer,
        model_cfg=model_cfg,
        parallel_cfg=parallel_cfg_tp1,
        hw=hw,
        batch=2,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "compute" in streams


# ---------------------------------------------------------------------------
# Test 3: TP > 1 produces tp_comm ops
# ---------------------------------------------------------------------------


def test_build_layer_ops_tp_comm(single_layer, model_cfg, parallel_cfg_tp4, hw):
    result = build_layer_ops(
        layer_cfg=single_layer,
        model_cfg=model_cfg,
        parallel_cfg=parallel_cfg_tp4,
        hw=hw,
        batch=2,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "tp_comm" in streams


# ---------------------------------------------------------------------------
# Test 4: TP = 1 has no tp_comm ops
# ---------------------------------------------------------------------------


def test_build_layer_ops_no_tp_comm_when_tp1(single_layer, model_cfg, parallel_cfg_tp1, hw):
    result = build_layer_ops(
        layer_cfg=single_layer,
        model_cfg=model_cfg,
        parallel_cfg=parallel_cfg_tp1,
        hw=hw,
        batch=2,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "tp_comm" not in streams


# ---------------------------------------------------------------------------
# Test 5: build_training_step has forward and backward ops
# ---------------------------------------------------------------------------


def test_build_training_step_has_fwd_bwd(model_cfg, hw, parallel_cfg_tp1, rl_cfg):
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg)
    assert len(all_ops) > 0

    names = [op.name for op in all_ops]
    # Forward ops should have TRAIN_FWD characteristics — we check attention and FFN exist
    # There are 2 * num_layers layers worth of attention ops (fwd + bwd)
    attn_ops = [n for n in names if "attention" in n]
    ffn_ops = [n for n in names if "ffn" in n]
    # Each layer appears twice (fwd + bwd), with 32 layers
    assert len(attn_ops) >= 2  # at least one fwd + one bwd
    assert len(ffn_ops) >= 2


# ---------------------------------------------------------------------------
# Test 6: dp > 1 has dp_comm ops
# ---------------------------------------------------------------------------


def test_build_training_step_dp_sync(model_cfg, hw, parallel_cfg_dp2, rl_cfg):
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_dp2, rl_cfg)
    streams = {op.stream for op in all_ops}
    assert "dp_comm" in streams


# ---------------------------------------------------------------------------
# Test 7: build_generation_step returns (prefill, decode) both non-empty
# ---------------------------------------------------------------------------


def test_build_generation_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg):
    prefill_ops, decode_ops = build_generation_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg)
    assert isinstance(prefill_ops, list)
    assert isinstance(decode_ops, list)
    assert len(prefill_ops) > 0
    assert len(decode_ops) > 0


# ---------------------------------------------------------------------------
# Test 8: depends_on indices are all < current op index
# ---------------------------------------------------------------------------


def test_simop_depends_on_valid(model_cfg, hw, parallel_cfg_tp4, rl_cfg):
    """All depends_on indices must reference a prior op in the sequence."""
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_tp4, rl_cfg)
    for i, op in enumerate(all_ops):
        for dep in op.depends_on:
            assert dep < i, (
                f"Op[{i}] '{op.name}' has depends_on={dep} which is >= {i}"
            )


# ---------------------------------------------------------------------------
# Test 9: training step has weight_bytes > 0 for compute ops
# ---------------------------------------------------------------------------


def test_weight_bytes_nonzero(model_cfg, hw, parallel_cfg_tp1, rl_cfg):
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg)
    # At least some compute ops (attention, FFN) should carry weight_bytes
    weight_ops = [op for op in all_ops if op.weight_bytes > 0 and op.stream == "compute"]
    assert len(weight_ops) > 0, "Expected at least some ops with weight_bytes > 0"


# ---------------------------------------------------------------------------
# Test 10: MoE + EP > 1 produces ep_comm ops
# ---------------------------------------------------------------------------


def test_moe_layer_has_ep_comm(model_cfg, hw, rl_cfg):
    """A MoE layer with EP > 1 should emit ep_alltoall on ep_comm stream."""
    moe_layer = LayerConfig(
        attention="GQA",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        ffn="MoE",
        num_experts=8,
        num_shared_experts=0,
        top_k=2,
        expert_intermediate_size=2048,
        shared_intermediate_size=0,
    )
    parallel_ep2 = ParallelismConfig(tp=1, pp=1, dp=1, ep=2)

    result = build_layer_ops(
        layer_cfg=moe_layer,
        model_cfg=model_cfg,
        parallel_cfg=parallel_ep2,
        hw=hw,
        batch=2,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "ep_comm" in streams, f"Expected ep_comm stream, got: {streams}"


# ---------------------------------------------------------------------------
# Test 11: SP replaces allreduce with allgather + reducescatter
# ---------------------------------------------------------------------------


def test_sp_replaces_allreduce_with_ag_rs(single_layer, model_cfg, hw):
    """sp=True + tp>1 should produce allgather + reducescatter instead of allreduce."""
    parallel_sp = ParallelismConfig(tp=4, pp=1, dp=1, ep=1, sp=True)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    names = [op.name for op in result]
    assert not any("allreduce" in n for n in names), f"SP should replace allreduce: {names}"
    assert any("allgather" in n for n in names), f"SP should add allgather: {names}"
    assert any("reducescatter" in n for n in names), f"SP should add reducescatter: {names}"


# ---------------------------------------------------------------------------
# Test 12: SP=False keeps allreduce
# ---------------------------------------------------------------------------


def test_sp_false_keeps_allreduce(single_layer, model_cfg, hw):
    """sp=False + tp>1 should still use allreduce."""
    parallel_no_sp = ParallelismConfig(tp=4, pp=1, dp=1, ep=1, sp=False)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    names = [op.name for op in result]
    assert any("allreduce" in n for n in names)
    assert not any("allgather" in n for n in names)


# ---------------------------------------------------------------------------
# Test 13: SP comm volume matches allreduce
# ---------------------------------------------------------------------------


def test_sp_comm_volume_matches_allreduce(single_layer, model_cfg, hw):
    """SP (AG+RS) total comm duration should equal AllReduce duration.

    AllGather and ReduceScatter are each sized by the full (gathered) sequence,
    so AG+RS == AllReduce for the same tensor (ring: (N-1)/N + (N-1)/N = 2(N-1)/N).
    """
    tp = 4
    parallel_no_sp = ParallelismConfig(tp=tp, pp=1, dp=1, ep=1, sp=False)
    parallel_sp = ParallelismConfig(tp=tp, pp=1, dp=1, ep=1, sp=True)

    ops_no_sp = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    ops_sp = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )

    t_allreduce = sum(op.duration for op in ops_no_sp if "comm" in op.stream)
    t_sp = sum(op.duration for op in ops_sp if "comm" in op.stream)
    assert t_sp == pytest.approx(t_allreduce, rel=0.01)


# ---------------------------------------------------------------------------
# Test 14: CP > 1 inserts cp_comm stream ops
# ---------------------------------------------------------------------------


def test_cp_inserts_ring_comm(single_layer, model_cfg, hw):
    """cp > 1 should insert cp_comm stream ops."""
    parallel_cp = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=4)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_cp,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "cp_comm" in streams, f"CP should add cp_comm stream: {streams}"


# ---------------------------------------------------------------------------
# Test 15: CP = 1 has no cp_comm ops
# ---------------------------------------------------------------------------


def test_cp_1_no_comm(single_layer, model_cfg, hw):
    """cp=1 should not insert any cp_comm ops."""
    parallel_no_cp = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=1)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_cp,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "cp_comm" not in streams


# ---------------------------------------------------------------------------
# Test 16: CP > 1 reduces attention compute time (seq_len / cp)
# ---------------------------------------------------------------------------


def test_cp_reduces_attention_seq_len(model_cfg, hw):
    """cp > 1 should result in less compute time (seq_len / cp for attention)."""
    layer = model_cfg.get_layers()[0]
    parallel_1 = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=1)
    parallel_4 = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=4)

    ops_1 = build_layer_ops(
        layer_cfg=layer, model_cfg=model_cfg, parallel_cfg=parallel_1,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    ops_4 = build_layer_ops(
        layer_cfg=layer, model_cfg=model_cfg, parallel_cfg=parallel_4,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )

    compute_time_1 = sum(op.duration for op in ops_1 if op.stream == "compute")
    compute_time_4 = sum(op.duration for op in ops_4 if op.stream == "compute")
    assert compute_time_4 < compute_time_1


# ---------------------------------------------------------------------------
# Test 17: MTP head ops inserted when mtp_depth > 0
# ---------------------------------------------------------------------------


def test_mtp_inserts_head_ops(hw, rl_cfg):
    """Models with mtp_depth > 0 should have mtp_head ops in training."""
    mc = ModelConfig(
        name="test_mtp", hidden_size=4096, vocab_size=32000, num_layers=2, dtype="bf16",
        default_layer=LayerConfig(
            attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
            ffn="SwiGLU", intermediate_size=11008,
        ),
        auxiliary={"mtp_depth": 1},
    )
    parallel = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)
    all_ops = build_training_step(mc, hw, parallel, rl_cfg)
    names = [op.name for op in all_ops]
    assert any("mtp" in n for n in names), f"Should have MTP ops: {names}"
    mtp_ops = [n for n in names if "mtp" in n]
    assert len(mtp_ops) == 2  # fwd + bwd


# ---------------------------------------------------------------------------
# Test 18: No MTP ops without auxiliary
# ---------------------------------------------------------------------------


def test_no_mtp_without_auxiliary(model_cfg, hw, parallel_cfg_tp1, rl_cfg):
    """Models without auxiliary.mtp_depth should have no mtp ops."""
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg)
    names = [op.name for op in all_ops]
    assert not any("mtp" in n for n in names)


# ---------------------------------------------------------------------------
# Test 19: Hybrid architecture with per-layer configs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: Validation — bad TP / PP rejected at entry points
# ---------------------------------------------------------------------------


def test_build_training_step_rejects_bad_tp():
    """TP that doesn't divide num_heads should raise ValueError."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = WorkloadConfig(total_prompts=100, group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=6, pp=1, dp=1)
    with pytest.raises(ValueError, match="num_heads.*divisible.*tp"):
        build_training_step(mc, hw, parallel, rl)


def test_build_generation_step_rejects_bad_tp():
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = WorkloadConfig(total_prompts=100, group_size=4, gen_batch_size=8)
    parallel = ParallelismConfig(tp=6, pp=1, dp=1)
    with pytest.raises(ValueError, match="num_heads.*divisible.*tp"):
        build_generation_step(mc, hw, parallel, rl)


def test_build_training_step_rejects_pp_gt_layers():
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = WorkloadConfig(total_prompts=100, group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=1, pp=64, dp=1)
    with pytest.raises(ValueError, match="pp.*layers"):
        build_training_step(mc, hw, parallel, rl)


def test_build_training_step_allows_uneven_pp():
    """Uneven PP splits are now supported (remainder distributed across stages).

    Llama 3.1 8B has 32 layers; pp=3 splits as 11/11/10 rather than raising.
    """
    from llm_perf.builder import _split_stages

    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = WorkloadConfig(group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=1, pp=3, dp=1)

    ops = build_training_step(mc, hw, parallel, rl)
    assert len(ops) > 0
    # 32 layers across 3 stages → 11 + 11 + 10
    stage_sizes = [len(s) for s in _split_stages(mc.get_layers(), 3)]
    assert sum(stage_sizes) == mc.num_layers
    assert max(stage_sizes) - min(stage_sizes) <= 1


def test_dp_bucketed_allreduce_per_layer(hw, rl_cfg):
    """dp > 1: DP gradient sync is bucketed PER LAYER on the dp_comm stream,
    overlapping with backward (DDP/Megatron-style gradient bucketing) —
    NOT a single AllReduce issued after the whole backward pass.

    With num_layers=4 (shrunk from llama3_1_8b for speed) and dp=8, the
    builder must emit exactly 4 dp_comm ops (one bucket per layer), each
    depending directly on that layer's own backward (a compute-stream op),
    confirming the overlap wiring. With dp=1, there must be zero dp_comm ops.
    """
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    mc.num_layers = 4
    num_layers = len(mc.get_layers())
    assert num_layers == 4

    parallel_dp8 = ParallelismConfig(tp=1, pp=1, dp=8, ep=1)
    all_ops = build_training_step(mc, hw, parallel_dp8, rl_cfg)

    dp_comm_ops = [(i, op) for i, op in enumerate(all_ops) if op.stream == "dp_comm"]
    # One bucket per transformer layer in the stage — NOT a single post-backward AllReduce.
    assert len(dp_comm_ops) == num_layers == 4, (
        f"Expected one dp_comm bucket per layer ({num_layers}), got {len(dp_comm_ops)}"
    )

    # Each dp_comm bucket must depend (directly) on a backward compute op of its
    # own layer, proving it overlaps with backward instead of waiting for all of it.
    for i, op in dp_comm_ops:
        assert len(op.depends_on) == 1
        dep_idx = op.depends_on[0]
        dep_op = all_ops[dep_idx]
        assert dep_op.stream == "compute", (
            f"dp_comm op[{i}] should depend on a backward compute op, got stream={dep_op.stream}"
        )

    # Sanity: dp=1 emits no dp_comm ops at all.
    parallel_dp1 = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)
    all_ops_dp1 = build_training_step(mc, hw, parallel_dp1, rl_cfg)
    assert sum(1 for op in all_ops_dp1 if op.stream == "dp_comm") == 0


def test_hybrid_layers_different_types(hw, rl_cfg):
    """Model with per-layer configs should build ops for each layer type."""
    mc = ModelConfig(
        name="hybrid", hidden_size=4096, vocab_size=32000, num_layers=4, dtype="bf16",
        layers=[
            LayerConfig(attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
                       ffn="SwiGLU", intermediate_size=11008),
            LayerConfig(attention="SWA", num_heads=32, num_kv_heads=8, head_dim=128,
                       ffn="SwiGLU", intermediate_size=11008, window_size=4096),
            LayerConfig(attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
                       ffn="SwiGLU", intermediate_size=11008),
            LayerConfig(attention="SWA", num_heads=32, num_kv_heads=8, head_dim=128,
                       ffn="SwiGLU", intermediate_size=11008, window_size=4096),
        ],
    )
    parallel = ParallelismConfig(tp=1, pp=1, dp=1)
    all_ops = build_training_step(mc, hw, parallel, rl_cfg)
    assert len(all_ops) > 0
    names = [op.name for op in all_ops]
    assert any("gqa" in n for n in names)
    assert any("swa" in n for n in names)


def test_comm_ops_are_fabric_tagged():
    """Every comm SimOp must carry a fabric tag; DP inter-node ops must be 'nic'."""
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    # devices_per_node=8; tp=4 intra-node (nvlink), dp=16 inter-node (nic)
    pc = ParallelismConfig(tp=4, dp=16)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc, rl)
    comm_ops = [o for o in ops if o.stream.endswith("_comm")]
    assert comm_ops, "expected some comm ops"
    assert all(o.fabric in ("nvlink", "nic") for o in comm_ops)
    dp_ops = [o for o in ops if o.stream == "dp_comm"]
    assert dp_ops and all(o.fabric == "nic" for o in dp_ops)  # dp=16 inter-node


# ---------------------------------------------------------------------------
# Test: low-precision quant chain injection
# ---------------------------------------------------------------------------


def _fp8_precision():
    return PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
        activations=TensorPrecision(dtype="fp8_e4m3", block_size=128, hadamard=True, hadamard_block=128),
    )


def test_lowprec_gemm_injects_quant_chain():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc, rl, precision_cfg=_fp8_precision())
    names = [o.name for o in ops]
    assert any(n.startswith("quantize") for n in names)
    assert any(n.startswith("hadamard") for n in names)
    assert any(n.startswith("dequant") for n in names)


def test_bf16_default_injects_no_quant_ops():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops_default = build_training_step(model, hw, pc, rl)  # no precision_cfg
    assert not any(o.name.startswith(("quantize", "hadamard", "dequant")) for o in ops_default)


# ---------------------------------------------------------------------------
# Test: fp8 comm halves dp_grad_sync bytes and injects quant/dequant
# ---------------------------------------------------------------------------


def test_fp8_comm_halves_dp_grad_bytes_and_adds_quant():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)

    bf16_comm = PrecisionConfig.bf16_default()
    fp8_comm = PrecisionConfig(comm=TensorPrecision(dtype="fp8_e4m3"))

    ops_bf16 = build_training_step(model, hw, pc, rl, precision_cfg=bf16_comm)
    ops_fp8 = build_training_step(model, hw, pc, rl, precision_cfg=fp8_comm)

    dp_bytes_bf16 = sum(o.comm_bytes for o in ops_bf16 if o.name == "dp_grad_sync")
    dp_bytes_fp8 = sum(o.comm_bytes for o in ops_fp8 if o.name == "dp_grad_sync")
    assert dp_bytes_fp8 == pytest.approx(dp_bytes_bf16 / 2, rel=1e-6)
    assert any(o.name.startswith("quantize_grad") for o in ops_fp8)
    assert any(o.name.startswith("dequant_grad") for o in ops_fp8)


# ---------------------------------------------------------------------------
# Test: error-feedback buffer memory and compensation_add op
# ---------------------------------------------------------------------------


def test_error_feedback_adds_buffer_and_compensation_op():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    no_ef = PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3"))
    with_ef = PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3"),
        error_feedback=True,
        ef_dtype="fp16",
    )
    w_no = sum(o.weight_bytes for o in build_training_step(model, hw, pc, rl, precision_cfg=no_ef))
    w_ef = sum(o.weight_bytes for o in build_training_step(model, hw, pc, rl, precision_cfg=with_ef))
    assert w_ef > w_no
    assert any(
        o.name.startswith("compensation_add")
        for o in build_training_step(model, hw, pc, rl, precision_cfg=with_ef)
    )


# ---------------------------------------------------------------------------
# Tests: attention split (Task 4) — GQA/MHA proj + core SimOps
# ---------------------------------------------------------------------------


def test_gqa_attention_splits_when_attn_linear_set():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops_list = build_training_step(model, hw, pc_par, rl, precision_cfg=PrecisionConfig.fp4_paper())
    names = [o.name for o in ops_list]
    assert any(n == "attention_proj" for n in names)
    assert any(n == "attention_core" for n in names)


def test_no_attention_split_without_attn_linear():
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops_list = build_training_step(model, hw, pc_par, rl)  # default, no module precision
    names = [o.name for o in ops_list]
    assert not any(n in ("attention_proj", "attention_core") for n in names)
    assert any(n.startswith("attention_") for n in names)  # monolithic preserved


def test_attention_split_equal_precision_sums_to_monolithic_duration():
    # With proj+core both at bf16, the two SimOp durations sum to the old single
    # attention duration (no time created/lost by the split).
    from llm_perf.precision import ModuleLinearPrecision
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    bf16_split = PrecisionConfig(attn_linear=ModuleLinearPrecision())  # both bf16
    ops_mono = build_training_step(model, hw, pc_par, rl)
    ops_split = build_training_step(model, hw, pc_par, rl, precision_cfg=bf16_split)
    mono = sum(o.duration for o in ops_mono if o.name.startswith("attention_"))
    split = sum(o.duration for o in ops_split if o.name in ("attention_proj", "attention_core"))
    assert split == pytest.approx(mono, rel=1e-6)


# ---------------------------------------------------------------------------
# Test: direction-aware FFN precision (Task 5)
# ---------------------------------------------------------------------------


def test_ffn_backward_uses_bwd_precision():
    # fp4 forward, fp8 backward (the paper): the TRAIN_BWD ffn GEMM runs at fp8,
    # the TRAIN_FWD ffn GEMM at fp4.
    from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    pc = PrecisionConfig(ffn_linear=ModuleLinearPrecision(
        fwd=TensorPrecision(dtype="fp4_e2m1"), bwd=TensorPrecision(dtype="fp8_e4m3")))
    ops = build_training_step(model, hw, pc_par, rl, precision_cfg=pc)
    ffn = [o for o in ops if o.name == "ffn_swiglu"]
    classes = {o.op_class for o in ffn}
    assert "fp4" in classes  # forward
    assert "fp8" in classes  # backward


# ---------------------------------------------------------------------------
# Test: asymmetric weights/activations — FFN weight bytes follow weights role
# (regression guard for the ffn_linear=None per-role fallback fix)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: fp4_paper() emits quant overhead (FIX 1 regression guard)
# ---------------------------------------------------------------------------


def test_fp4_paper_emits_quant_overhead():
    """fp4_paper() must charge quantize/dequant overhead for FFN (fp4) and
    attn-proj (fp8) chains. Before the fix, pc.activations stayed bf16, so
    _inject_quant_chain skipped the chain entirely — no overhead was charged.

    Also guard the no-regression direction: bf16_default() must emit NONE of
    these ops.
    """
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)

    ops_fp4 = build_training_step(model, hw, pc_par, rl, precision_cfg=PrecisionConfig.fp4_paper())
    names_fp4 = [o.name for o in ops_fp4]

    assert any(n.startswith("quantize") for n in names_fp4), (
        "fp4_paper() must emit quantize_act ops; got none — overhead is not charged"
    )
    assert any(n.startswith("dequant") for n in names_fp4), (
        "fp4_paper() must emit dequant_out ops; got none — overhead is not charged"
    )

    # bf16_default() must NOT emit any quant/dequant/hadamard ops (no-regression guard)
    ops_bf16 = build_training_step(model, hw, pc_par, rl)
    assert not any(o.name.startswith(("quantize", "hadamard", "dequant")) for o in ops_bf16), (
        "bf16_default() must not emit quant overhead ops"
    )


def test_ffn_weight_bytes_follow_weights_role_not_activations():
    """When ffn_linear is None and weights != activations dtype, forward ffn_swiglu
    weight_bytes must be sized by pc.weights (fp8=1B/elem), NOT pc.activations (bf16=2B/elem).

    Concretely: asymmetric config (fp8 weights, bf16 activations) must produce
    HALF the ffn_swiglu weight_bytes of a pure-bf16 config, because fp8=1B vs bf16=2B.
    """
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)

    # Asymmetric: fp8 weights, bf16 activations, ffn_linear=None (unset)
    asymmetric_pc = PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3"),
        activations=TensorPrecision(dtype="bf16"),
    )
    # Pure bf16: both weights and activations at bf16
    bf16_pc = PrecisionConfig.bf16_default()

    ops_asym = build_training_step(model, hw, pc_par, rl, precision_cfg=asymmetric_pc)
    ops_bf16 = build_training_step(model, hw, pc_par, rl, precision_cfg=bf16_pc)

    # Collect forward ffn_swiglu weight_bytes (backward sets weight_bytes=0 by design)
    # Forward ops appear before backward ops in the training step sequence; since
    # weight_bytes is zeroed for BWD ops, summing is equivalent to summing FWD only.
    asym_w = sum(o.weight_bytes for o in ops_asym if o.name == "ffn_swiglu")
    bf16_w = sum(o.weight_bytes for o in ops_bf16 if o.name == "ffn_swiglu")

    assert bf16_w > 0, "baseline bf16 ffn_swiglu weight_bytes must be > 0"
    assert asym_w == pytest.approx(bf16_w / 2, rel=1e-6), (
        f"fp8 weights should be half of bf16 weight bytes; "
        f"got asym={asym_w}, bf16={bf16_w}"
    )
