"""Tests for builder.py — SimOp generation from config."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_perf.builder import SimOp, build_generation_step, build_layer_ops, build_training_step
from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    RLConfig,
    load_hardware_config,
    load_model_config,
)

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
def rl_cfg() -> RLConfig:
    return RLConfig(
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
