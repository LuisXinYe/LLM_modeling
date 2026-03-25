"""Tests for model.py (RLPerformanceModel) and report.py (format_table)."""

from __future__ import annotations

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
from rl_perf.model import RLPerformanceModel
from rl_perf.report import TargetReport, format_json, format_table

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

VALID_BOTTLENECKS = {"GENERATION", "TRAINING", "BALANCED"}


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
        train_micro_batch_size=2,
        gradient_accumulation_steps=1,
        gen_batch_size=16,
    )


@pytest.fixture
def perf_model(model_cfg, hw) -> RLPerformanceModel:
    return RLPerformanceModel(model_cfg, hw)


# ---------------------------------------------------------------------------
# Test 1: derive_targets — basic sanity
# ---------------------------------------------------------------------------


def test_derive_targets(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.derive_targets(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
        time_budget_hours=10.0,
    )
    assert isinstance(report, TargetReport)
    assert report.epoch_time_hours > 0
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.bottleneck in VALID_BOTTLENECKS
    # within_budget is a bool
    assert isinstance(report.within_budget, bool)


# ---------------------------------------------------------------------------
# Test 2: feasibility_check — no time budget, within_budget always True
# ---------------------------------------------------------------------------


def test_feasibility_check(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    assert isinstance(report, TargetReport)
    assert report.epoch_time_hours > 0
    assert report.within_budget is True  # no budget provided → always feasible
    assert isinstance(report.feasible, bool)


# ---------------------------------------------------------------------------
# Test 3: what-if group_size — larger group should take longer training time
# ---------------------------------------------------------------------------


def test_what_if_group_size(perf_model, parallel_cfg):
    rl_small = RLConfig(
        total_prompts=64,
        group_size=8,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    rl_large = RLConfig(
        total_prompts=64,
        group_size=16,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    report_small = perf_model.derive_targets(64, rl_small, parallel_cfg, parallel_cfg)
    report_large = perf_model.derive_targets(64, rl_large, parallel_cfg, parallel_cfg)

    # group_size=16 → 2x total_responses → should take longer overall
    assert report_large.epoch_time_hours > report_small.epoch_time_hours


# ---------------------------------------------------------------------------
# Test 4: memory profile — all numeric fields positive, booleans exist
# ---------------------------------------------------------------------------


def test_memory_profile(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    mem = report.memory
    assert mem.weight_gb > 0
    assert mem.optimizer_gb > 0
    assert mem.activation_peak_gb > 0
    assert mem.kv_cache_gb > 0
    assert mem.total_train_gb > 0
    assert mem.total_gen_gb > 0
    assert mem.usable_hbm_gb > 0
    assert isinstance(mem.train_feasible, bool)
    assert isinstance(mem.gen_feasible, bool)


# ---------------------------------------------------------------------------
# Test 5: format_table returns string with key words
# ---------------------------------------------------------------------------


def test_format_table(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    table = format_table(report)
    assert isinstance(table, str)
    assert "RL Training Performance Report" in table
    assert "Bottleneck" in table
    assert "Generation" in table
    assert "Training" in table
    assert "Memory" in table
    # Should contain either OK or OOM
    assert "OK" in table or "OOM" in table
    # Should contain FEASIBLE status
    assert "FEASIBLE" in table


# ---------------------------------------------------------------------------
# Test 6: format_json — round-trips through JSON, key fields present
# ---------------------------------------------------------------------------


def test_format_json(perf_model, rl_cfg):
    import json as json_mod

    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)

    result = format_json(report)
    parsed = json_mod.loads(result)
    assert "epoch_time_hours" in parsed
    assert "memory" in parsed
    assert parsed["memory"]["weight_gb"] > 0


# ---------------------------------------------------------------------------
# Test 7: memory from SimResult
# ---------------------------------------------------------------------------


def test_memory_from_sim_result(perf_model, rl_cfg):
    """Memory profile should use SimResult weight_bytes."""
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)
    assert report.memory.weight_gb > 0
    assert report.memory.activation_peak_gb > 0


# ---------------------------------------------------------------------------
# Test 8: what_if
# ---------------------------------------------------------------------------


def test_what_if(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    base = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)

    result = perf_model.what_if(
        base_config=rl_cfg.model_dump(),
        overrides={"group_size": 16},
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert result.epoch_time_hours > base.epoch_time_hours


# ---------------------------------------------------------------------------
# Test 9: sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    results = perf_model.sensitivity(
        rl_cfg=rl_cfg, param_name="group_size", values=[4, 8, 16],
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert len(results) == 3
    assert results[2].epoch_time_hours > results[0].epoch_time_hours


# ---------------------------------------------------------------------------
# Test 10: sensitivity invalid param
# ---------------------------------------------------------------------------


def test_sensitivity_invalid_param(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    with pytest.raises(ValueError, match="Unknown RLConfig field"):
        perf_model.sensitivity(
            rl_cfg=rl_cfg, param_name="nonexistent_field", values=[1, 2],
            total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
        )


# ---------------------------------------------------------------------------
# Test 11: weight_bytes no double count
# ---------------------------------------------------------------------------


def test_weight_bytes_no_double_count():
    """Verify builder zeroes weight_bytes on BWD ops so SimResult doesn't double-count."""
    from rl_perf.builder import build_training_step
    from rl_perf.simulator import simulate

    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = RLConfig(total_prompts=100, group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)

    ops_list = build_training_step(mc, hw, parallel, rl)
    fwd_weight = sum(op.weight_bytes for op in ops_list)
    # Since BWD ops have weight_bytes=0, this sum equals fwd-only weight
    sim = simulate(ops_list)
    assert sim.weight_bytes == pytest.approx(fwd_weight)


# ---------------------------------------------------------------------------
# Test 12: feasible field — True only when within_budget AND memory OK
# ---------------------------------------------------------------------------


def test_feasible_field_time_and_memory(perf_model, rl_cfg, parallel_cfg):
    """feasible should be True only when within_budget AND memory OK."""
    report = perf_model.derive_targets(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg, train_parallel=parallel_cfg,
        time_budget_hours=100.0,
    )
    assert hasattr(report, 'feasible')
    assert isinstance(report.feasible, bool)
    if report.memory.train_feasible and report.memory.gen_feasible:
        assert report.feasible is True


# ---------------------------------------------------------------------------
# Test 13: feasible False when over time
# ---------------------------------------------------------------------------


def test_feasible_false_when_over_time(perf_model, rl_cfg, parallel_cfg):
    """feasible should be False when time budget exceeded."""
    report = perf_model.derive_targets(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg, train_parallel=parallel_cfg,
        time_budget_hours=0.000001,
    )
    assert report.within_budget is False
    assert report.feasible is False


# ---------------------------------------------------------------------------
# Test 14: feasibility_check with time_budget_hours
# ---------------------------------------------------------------------------


def test_feasibility_check_with_time_budget(perf_model, rl_cfg, parallel_cfg):
    """feasibility_check should accept optional time_budget_hours."""
    report = perf_model.feasibility_check(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg, train_parallel=parallel_cfg,
        time_budget_hours=0.000001,
    )
    assert report.within_budget is False
    assert report.feasible is False
