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
    t = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t > 0


# ---------------------------------------------------------------------------
# Test 8: training_time using real configs — nonzero
# ---------------------------------------------------------------------------


def test_training_time_nonzero(model_cfg, hw, parallel_cfg, rl_cfg):
    t = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert t > 0


def test_epoch_time_colocated():
    t = epoch_time(t_gen=10.0, t_train=15.0, colocated=True)
    assert t == pytest.approx(25.0)  # gen + train


def test_epoch_time_separated():
    t = epoch_time(t_gen=10.0, t_train=15.0, startup_overhead=0.5, colocated=False)
    assert t == pytest.approx(15.5)  # max + startup
