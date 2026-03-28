"""Tests for the Pareto search and sensitivity sweep engine (search.py).

Tests are written FIRST following TDD: RED → GREEN → REFACTOR.
"""

import pytest

from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
)
from rl_perf.model import RLPerformanceModel
from rl_perf.search import SearchResult, pareto_search, sensitivity_sweep


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_cfg():
    """Small dense model for fast tests."""
    layer = LayerConfig(
        attention="GQA",
        num_heads=16,
        num_kv_heads=4,
        head_dim=128,
        ffn="SwiGLU",
        intermediate_size=5504,
    )
    return ModelConfig(
        name="test-model",
        hidden_size=2048,
        vocab_size=32000,
        num_layers=24,
        dtype="bf16",
        default_layer=layer,
    )


@pytest.fixture
def test_hw_cfg():
    """Test hardware with 8 devices per node."""
    return HardwareConfig(
        name="test-gpu",
        peak_tflops_bf16=400.0,
        hbm_capacity_gb=80.0,
        hbm_bandwidth_tb_s=2.0,
        devices_per_node=8,
    )


@pytest.fixture
def small_rl_cfg():
    """Small RL config for quick iteration."""
    return RLConfig(
        total_prompts=64,
        group_size=4,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        gen_batch_size=32,
        train_micro_batch_size=2,
        colocated=False,
        reference_model=False,
    )


@pytest.fixture
def perf_model(small_model_cfg, test_hw_cfg):
    return RLPerformanceModel(small_model_cfg, test_hw_cfg)


# ---------------------------------------------------------------------------
# Test SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResultDataclass:
    def test_fields_present(self, perf_model, small_rl_cfg):
        """SearchResult must have all required fields."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        report = perf_model.derive_targets(8, small_rl_cfg, gen_par, train_par)
        result = SearchResult(
            devices=8,
            gen_parallel=gen_par,
            train_parallel=train_par,
            report=report,
            is_feasible=report.feasible,
            is_oom=not (report.memory.train_feasible and report.memory.gen_feasible),
        )
        assert hasattr(result, "devices")
        assert hasattr(result, "gen_parallel")
        assert hasattr(result, "train_parallel")
        assert hasattr(result, "report")
        assert hasattr(result, "is_feasible")
        assert hasattr(result, "is_oom")
        assert hasattr(result, "is_pareto")

    def test_is_pareto_defaults_false(self, perf_model, small_rl_cfg):
        """is_pareto should default to False."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        report = perf_model.derive_targets(8, small_rl_cfg, gen_par, train_par)
        result = SearchResult(
            devices=8,
            gen_parallel=gen_par,
            train_parallel=train_par,
            report=report,
            is_feasible=report.feasible,
            is_oom=not (report.memory.train_feasible and report.memory.gen_feasible),
        )
        assert result.is_pareto is False


# ---------------------------------------------------------------------------
# Tests for pareto_search
# ---------------------------------------------------------------------------


class TestParetoSearch:
    def test_returns_list(self, perf_model, test_hw_cfg, small_rl_cfg):
        """pareto_search must return a list."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8])
        assert isinstance(results, list)

    def test_all_results_are_search_result(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Every item in the list must be a SearchResult."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16])
        for r in results:
            assert isinstance(r, SearchResult)

    def test_devices_field_matches_input(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Each result's devices field should be one of the requested counts."""
        device_counts = [8, 16]
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, device_counts)
        for r in results:
            assert r.devices in device_counts

    def test_nonempty_for_valid_counts(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Should return at least one result for valid device counts."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16, 32, 64])
        assert len(results) > 0

    def test_at_least_one_pareto_point(self, perf_model, test_hw_cfg, small_rl_cfg):
        """There must be at least one Pareto point among feasible results."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16, 32, 64])
        feasible = [r for r in results if r.is_feasible]
        if feasible:
            pareto_points = [r for r in feasible if r.is_pareto]
            assert len(pareto_points) >= 1

    def test_pareto_correctness_no_dominance(
        self, perf_model, test_hw_cfg, small_rl_cfg
    ):
        """No Pareto point should be dominated by another Pareto point."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16, 32, 64])
        pareto = [r for r in results if r.is_pareto]
        for i, a in enumerate(pareto):
            for j, b in enumerate(pareto):
                if i == j:
                    continue
                # b should not strictly dominate a
                b_better_devices = b.devices <= a.devices
                b_better_time = b.report.epoch_time_hours <= a.report.epoch_time_hours
                b_strictly_better = (
                    b.devices < a.devices
                    or b.report.epoch_time_hours < a.report.epoch_time_hours
                )
                dominated = b_better_devices and b_better_time and b_strictly_better
                assert not dominated, (
                    f"Pareto point {a.devices}d/{a.report.epoch_time_hours:.2f}h is dominated by {b.devices}d/{b.report.epoch_time_hours:.2f}h"
                )

    def test_non_pareto_points_are_dominated(
        self, perf_model, test_hw_cfg, small_rl_cfg
    ):
        """Every non-Pareto feasible point must be dominated by at least one Pareto point."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16, 32, 64])
        pareto = [r for r in results if r.is_pareto]
        non_pareto_feasible = [r for r in results if r.is_feasible and not r.is_pareto]
        for nr in non_pareto_feasible:
            dominated = any(
                p.devices <= nr.devices
                and p.report.epoch_time_hours <= nr.report.epoch_time_hours
                and (
                    p.devices < nr.devices
                    or p.report.epoch_time_hours < nr.report.epoch_time_hours
                )
                for p in pareto
            )
            assert dominated, (
                f"Non-Pareto point {nr.devices}d/{nr.report.epoch_time_hours:.2f}h not dominated"
            )

    def test_is_oom_flag(self, perf_model, test_hw_cfg, small_rl_cfg):
        """is_oom must be True when memory is infeasible."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16])
        for r in results:
            expected_oom = not (
                r.report.memory.train_feasible and r.report.memory.gen_feasible
            )
            assert r.is_oom == expected_oom

    def test_gen_parallel_devices_le_total(self, perf_model, test_hw_cfg, small_rl_cfg):
        """gen_parallel total_devices must not exceed the device count."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16])
        for r in results:
            assert r.gen_parallel.total_devices <= r.devices

    def test_train_parallel_devices_eq_total(
        self, perf_model, test_hw_cfg, small_rl_cfg
    ):
        """train_parallel total_devices must equal the device count."""
        results = pareto_search(perf_model, test_hw_cfg, small_rl_cfg, [8, 16])
        for r in results:
            assert r.train_parallel.total_devices == r.devices


# ---------------------------------------------------------------------------
# Tests for sensitivity_sweep
# ---------------------------------------------------------------------------


class TestSensitivitySweep:
    def test_returns_list(self, perf_model, test_hw_cfg, small_rl_cfg):
        """sensitivity_sweep must return a list."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            [256, 512, 1024],
            8,
            gen_par,
            train_par,
        )
        assert isinstance(results, list)

    def test_returns_correct_count(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Output list length must equal number of swept values."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        values = [128, 256, 512, 1024, 2048]
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            values,
            8,
            gen_par,
            train_par,
        )
        assert len(results) == len(values)

    def test_all_are_search_result(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Every item must be a SearchResult."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            [256, 512],
            8,
            gen_par,
            train_par,
        )
        for r in results:
            assert isinstance(r, SearchResult)

    def test_longer_response_longer_epoch(self, perf_model, test_hw_cfg, small_rl_cfg):
        """More tokens to generate should result in longer epoch time."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        values = [128, 512, 2048]
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            values,
            8,
            gen_par,
            train_par,
        )
        times = [r.report.epoch_time_hours for r in results]
        # Epoch time should be monotonically non-decreasing with response length
        assert times[0] <= times[1] <= times[2], (
            f"Expected non-decreasing times: {times}"
        )

    def test_devices_field_is_total_devices(
        self, perf_model, test_hw_cfg, small_rl_cfg
    ):
        """devices field in each result must equal total_devices arg."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            [256, 512],
            8,
            gen_par,
            train_par,
        )
        for r in results:
            assert r.devices == 8

    def test_sweep_integer_param(self, perf_model, test_hw_cfg, small_rl_cfg):
        """Should work for total_prompts sweep."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "total_prompts",
            [32, 64, 128],
            8,
            gen_par,
            train_par,
        )
        assert len(results) == 3

    def test_time_budget_none_by_default(self, perf_model, test_hw_cfg, small_rl_cfg):
        """When time_budget_hours is None, within_budget should always be True."""
        gen_par = ParallelismConfig(tp=1, pp=1, dp=8)
        train_par = ParallelismConfig(tp=1, pp=1, dp=8)
        results = sensitivity_sweep(
            perf_model,
            test_hw_cfg,
            small_rl_cfg,
            "avg_response_len",
            [256, 512],
            8,
            gen_par,
            train_par,
            time_budget_hours=None,
        )
        for r in results:
            assert r.report.within_budget is True
