"""Tests for Plotly visualization builders in rl_perf.ui.plots."""

from typing import List

import plotly.graph_objects as go
import pytest

from rl_perf.config import ParallelismConfig
from rl_perf.report import MemoryProfile, TargetReport
from rl_perf.search import SearchResult
from rl_perf.ui.plots import (
    build_memory_figure,
    build_pareto_figure,
    build_sensitivity_figure,
    build_timeline_figure,
)


def _make_memory(**overrides) -> MemoryProfile:
    defaults = dict(
        weight_gb=20.0,
        optimizer_gb=40.0,
        activation_peak_gb=10.0,
        kv_cache_gb=5.0,
        ref_model_gb=20.0,
        total_train_gb=90.0,
        total_gen_gb=25.0,
        usable_hbm_gb=80.0,
        train_feasible=True,
        gen_feasible=True,
    )
    defaults.update(overrides)
    return MemoryProfile(**defaults)


def _make_report(**overrides) -> TargetReport:
    defaults = dict(
        epoch_time_hours=2.0,
        within_budget=True,
        bottleneck="generation",
        bottleneck_slack=0.1,
        gen_tps_target=10000.0,
        train_tps_target=5000.0,
        gen_samples_per_sec=100.0,
        train_samples_per_sec=50.0,
        gen_time_hours=1.5,
        train_time_hours=0.5,
        memory=_make_memory(),
        feasible=True,
    )
    defaults.update(overrides)
    return TargetReport(**defaults)


def _make_search_result(**overrides) -> SearchResult:
    gen_parallel = ParallelismConfig(tp=2, pp=1, dp=4)
    train_parallel = ParallelismConfig(tp=2, pp=2, dp=2)
    report = _make_report()
    defaults = dict(
        devices=8,
        gen_parallel=gen_parallel,
        train_parallel=train_parallel,
        report=report,
        is_feasible=True,
        is_oom=False,
        is_pareto=True,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


# --- build_timeline_figure ---


def test_timeline_colocated_returns_figure():
    report = _make_report()
    fig = build_timeline_figure(report, colocated=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_timeline_separated_returns_figure():
    report = _make_report()
    fig = build_timeline_figure(report, colocated=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_timeline_colocated_has_two_bars():
    """Colocated mode: Gen and Train stacked in one row → 2 traces."""
    report = _make_report()
    fig = build_timeline_figure(report, colocated=True)
    assert len(fig.data) >= 2


def test_timeline_separated_has_two_bars():
    """Separated mode: Gen and Train in separate rows → 2 traces."""
    report = _make_report()
    fig = build_timeline_figure(report, colocated=False)
    assert len(fig.data) >= 2


# --- build_memory_figure ---


def test_memory_figure_returns_figure():
    report = _make_report()
    fig = build_memory_figure(report)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_memory_figure_has_stacked_bars():
    """Training bar: 4 components; Generation bar: 2 components → at least 4 traces."""
    report = _make_report()
    fig = build_memory_figure(report)
    assert len(fig.data) >= 4


def test_memory_figure_has_hbm_line():
    """Should include a scatter/shape trace for HBM limit."""
    report = _make_report()
    fig = build_memory_figure(report)
    # HBM line can be a Scatter trace or layout shape — check for at least one Scatter
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    has_hbm_shape = len(fig.layout.shapes) > 0
    assert len(scatter_traces) > 0 or has_hbm_shape


# --- build_pareto_figure ---


def test_pareto_figure_returns_figure():
    results = [_make_search_result()]
    fig = build_pareto_figure(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_pareto_figure_with_budget_line():
    results = [_make_search_result()]
    fig = build_pareto_figure(results, time_budget_hours=3.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_pareto_figure_multiple_results():
    results = [
        _make_search_result(devices=8, is_pareto=True, is_feasible=True, is_oom=False),
        _make_search_result(
            devices=16,
            is_pareto=False,
            is_feasible=True,
            is_oom=False,
            report=_make_report(epoch_time_hours=1.0),
        ),
        _make_search_result(
            devices=4,
            is_pareto=False,
            is_feasible=False,
            is_oom=True,
            report=_make_report(epoch_time_hours=5.0, feasible=False),
        ),
    ]
    fig = build_pareto_figure(results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# --- build_sensitivity_figure ---


def test_sensitivity_figure_returns_figure():
    reports = [
        _make_report(epoch_time_hours=1.0, within_budget=True),
        _make_report(epoch_time_hours=2.0, within_budget=True),
        _make_report(epoch_time_hours=3.0, within_budget=False),
    ]
    fig = build_sensitivity_figure("num_samples", [100, 200, 300], reports)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_sensitivity_figure_with_budget_line():
    reports = [
        _make_report(epoch_time_hours=1.0, within_budget=True),
        _make_report(epoch_time_hours=4.0, within_budget=False),
    ]
    fig = build_sensitivity_figure("num_samples", [100, 200], reports, time_budget_hours=2.0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_sensitivity_figure_infeasible_report():
    """Reports with feasible=False should still produce a valid figure."""
    reports = [
        _make_report(
            epoch_time_hours=10.0,
            within_budget=False,
            feasible=False,
            memory=_make_memory(train_feasible=False),
        ),
    ]
    fig = build_sensitivity_figure("batch_size", [128], reports)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
