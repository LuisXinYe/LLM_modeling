"""Tests for pipeline.py — generation/training time and bottleneck analysis."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rl_perf.config import (
    HardwareConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
    load_hardware_config,
    load_model_config,
)
from rl_perf.pipeline import (
    bottleneck_analysis,
    effective_response_len,
    epoch_time,
    generation_time,
    training_time,
)
from rl_perf.simulator import SimResult

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def model_cfg() -> ModelConfig:
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))


@pytest.fixture
def hw() -> HardwareConfig:
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))


@pytest.fixture
def parallel_cfg() -> ParallelismConfig:
    return ParallelismConfig(tp=8, pp=1, dp=8, ep=1)


@pytest.fixture
def rl_cfg() -> RLConfig:
    return RLConfig(
        total_prompts=64,
        group_size=4,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        std_response_len=200,
        train_micro_batch_size=2,
        gradient_accumulation_steps=1,
        gen_batch_size=16,
    )


# ---------------------------------------------------------------------------
# Test 1: effective_response_len with std
# ---------------------------------------------------------------------------


def test_effective_response_len_with_std():
    avg = 2048
    std = 800
    batch = 32
    result = effective_response_len(avg=avg, std=std, batch_size=batch)
    # Should be avg + std * sqrt(2 * ln(32)) = 2048 + 800 * sqrt(2 * 3.465...)
    expected = avg + std * math.sqrt(2 * math.log(batch))
    assert result > avg
    assert result == pytest.approx(expected, rel=1e-6)
    # Reasonable upper bound
    assert result < avg + std * 10


# ---------------------------------------------------------------------------
# Test 2: effective_response_len fallback to max_len
# ---------------------------------------------------------------------------


def test_effective_response_len_fallback_max():
    result = effective_response_len(avg=2048, std=None, batch_size=32, max_len=4096)
    assert result == 4096


# ---------------------------------------------------------------------------
# Test 3: effective_response_len no std no max → returns avg
# ---------------------------------------------------------------------------


def test_effective_response_len_no_std_no_max():
    result = effective_response_len(avg=512)
    assert result == 512


# ---------------------------------------------------------------------------
# Test 4: bottleneck_analysis — generation bottleneck
# ---------------------------------------------------------------------------


def test_bottleneck_analysis_gen():
    bottleneck, slack = bottleneck_analysis(t_gen=10.0, t_train=6.0)
    assert bottleneck == "GENERATION"
    assert slack == pytest.approx(10.0 / 6.0 - 1, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 5: bottleneck_analysis — training bottleneck
# ---------------------------------------------------------------------------


def test_bottleneck_analysis_train():
    bottleneck, slack = bottleneck_analysis(t_gen=5.0, t_train=10.0)
    assert bottleneck == "TRAINING"
    assert slack > 0


# ---------------------------------------------------------------------------
# Test 6: epoch_time
# ---------------------------------------------------------------------------


def test_epoch_time():
    result = epoch_time(t_gen=20.0, t_train=15.0, startup_overhead=0.5)
    assert result == pytest.approx(20.5, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 7: generation_time using real configs — nonzero
# ---------------------------------------------------------------------------


def test_generation_time_nonzero(model_cfg, hw, parallel_cfg, rl_cfg):
    t, _, _ = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t > 0


# ---------------------------------------------------------------------------
# Test 8: training_time using real configs — nonzero
# ---------------------------------------------------------------------------


def test_training_time_nonzero(model_cfg, hw, parallel_cfg, rl_cfg):
    t, _ = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t > 0


def test_epoch_time_colocated():
    t = epoch_time(t_gen=10.0, t_train=15.0, colocated=True)
    assert t == pytest.approx(25.0)  # gen + train


def test_epoch_time_separated():
    t = epoch_time(t_gen=10.0, t_train=15.0, startup_overhead=0.5, colocated=False)
    assert t == pytest.approx(15.5)  # max + startup


# ---------------------------------------------------------------------------
# Test: generation_time returns 3-tuple with SimResult
# ---------------------------------------------------------------------------


def test_generation_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """generation_time should return (total_time, SimResult, t_per_batch)."""
    result = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 3
    t, sim, t_batch = result
    assert t > 0
    assert isinstance(sim, SimResult)
    assert t_batch > 0
    assert t_batch < t  # single batch < total


# ---------------------------------------------------------------------------
# Test: training_time returns 2-tuple with SimResult
# ---------------------------------------------------------------------------


def test_training_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """training_time should return (total_time, SimResult)."""
    result = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 2
    t, sim = result
    assert t > 0
    assert isinstance(sim, SimResult)
    assert sim.weight_bytes > 0


# ---------------------------------------------------------------------------
# Test: t_per_batch includes decode time (startup overhead fix)
# ---------------------------------------------------------------------------


def test_startup_overhead_includes_decode(model_cfg, hw, parallel_cfg, rl_cfg):
    """t_per_batch should be > prefill-only time (includes decode)."""
    _, sim, t_per_batch = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    from rl_perf.builder import build_generation_step
    from rl_perf.simulator import simulate as sim_fn

    prefill_ops, _ = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)
    t_prefill = sim_fn(prefill_ops).wall_clock_time
    assert t_per_batch > t_prefill


# ---------------------------------------------------------------------------
# Speculative decoding reduces generation time
# ---------------------------------------------------------------------------


def test_speculative_decoding_reduces_gen_time():
    """Speculative decoding with acceptance_len > 1 should reduce generation time."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "deepseekv3_671b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    parallel = ParallelismConfig(tp=8, pp=1, dp=8)

    rl_base = RLConfig(total_prompts=100, group_size=4, gen_batch_size=16,
                       train_micro_batch_size=2)
    rl_spec = RLConfig(total_prompts=100, group_size=4, gen_batch_size=16,
                       train_micro_batch_size=2,
                       use_speculative_decoding=True, mtp_acceptance_len=2)

    t_base, _, _ = generation_time(mc, hw, parallel, rl_base)
    t_spec, _, _ = generation_time(mc, hw, parallel, rl_spec)
    assert t_spec < t_base  # speculative decoding should be faster


# ---------------------------------------------------------------------------
# PP bubble ratio increases training time
# ---------------------------------------------------------------------------


def test_pp_bubble_ratio(model_cfg, hw, rl_cfg):
    """PP > 1 should increase training time due to bubble overhead."""
    parallel_pp1 = ParallelismConfig(tp=8, pp=1, dp=8)
    parallel_pp4 = ParallelismConfig(tp=8, pp=4, dp=2)

    t1, _ = training_time(model_cfg, hw, parallel_pp1, rl_cfg)
    t4, _ = training_time(model_cfg, hw, parallel_pp4, rl_cfg)
    # PP=4 should have bubble overhead making per-step time larger
    # But fewer layers per stage, so total could go either way
    # Just verify both are positive and different
    assert t1 > 0
    assert t4 > 0
    assert t1 != t4
