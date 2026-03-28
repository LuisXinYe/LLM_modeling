"""Tests for device topology visualization (TDD — written first)."""

from __future__ import annotations

import pytest
import plotly.graph_objects as go

from rl_perf.config import HardwareConfig, ParallelismConfig
from rl_perf.ui.topology import RankInfo, compute_rank_mapping, build_logical_mesh_figure


@pytest.fixture
def hw():
    return HardwareConfig(
        name="test",
        peak_tflops_bf16=400,
        hbm_capacity_gb=80,
        hbm_bandwidth_tb_s=2.0,
        devices_per_node=8,
    )


# ---------------------------------------------------------------------------
# compute_rank_mapping tests
# ---------------------------------------------------------------------------

def test_rank_mapping_tp8_pp1_dp8(hw):
    """TP=8, PP=1, DP=8 → 64 ranks, all RankInfo instances."""
    par = ParallelismConfig(tp=8, pp=1, dp=8)
    result = compute_rank_mapping(par, hw, num_layers=32)

    assert len(result) == 64
    assert all(isinstance(r, RankInfo) for r in result)
    # Rank 0: tp=0, pp=0, dp=0
    r0 = result[0]
    assert r0.global_rank == 0
    assert r0.tp_rank == 0
    assert r0.pp_stage == 0
    assert r0.dp_rank == 0
    # All layers on single stage
    assert r0.layer_start == 0
    assert r0.layer_end == 31


def test_rank_mapping_tp8_pp2_dp4(hw):
    """TP=8, PP=2, DP=4 → 64 ranks, r0 has tp=0, pp=0, dp=0."""
    par = ParallelismConfig(tp=8, pp=2, dp=4)
    result = compute_rank_mapping(par, hw, num_layers=32)

    assert len(result) == 64
    r0 = result[0]
    assert r0.tp_rank == 0
    assert r0.pp_stage == 0
    assert r0.dp_rank == 0
    # Layer range for pp_stage=0 with 2 stages, 32 layers → 16 per stage
    assert r0.layer_start == 0
    assert r0.layer_end == 15

    # Verify pp_stage=1 appears in mapping
    stage1_ranks = [r for r in result if r.pp_stage == 1]
    assert len(stage1_ranks) == 8 * 4  # tp * dp
    assert stage1_ranks[0].layer_start == 16
    assert stage1_ranks[0].layer_end == 31


def test_rank_mapping_with_ep(hw):
    """TP=4, PP=2, DP=2, EP=4 → 64 ranks, 4 distinct ep_ranks."""
    par = ParallelismConfig(tp=4, pp=2, dp=2, ep=4)
    result = compute_rank_mapping(par, hw, num_layers=32)

    assert len(result) == 64  # 4*2*2*4
    ep_ranks = {r.ep_rank for r in result}
    assert ep_ranks == {0, 1, 2, 3}


def test_rank_mapping_total_ranks(hw):
    """TP=4, PP=1, DP=4, EP=2 → exactly tp*pp*dp*ep == 32 ranks."""
    par = ParallelismConfig(tp=4, pp=1, dp=4, ep=2)
    result = compute_rank_mapping(par, hw, num_layers=16)

    expected = 4 * 1 * 4 * 2
    assert len(result) == expected == 32

    # All global ranks are unique
    global_ranks = [r.global_rank for r in result]
    assert len(set(global_ranks)) == 32

    # Node and local_gpu are correct
    for r in result:
        assert r.node == r.global_rank // hw.devices_per_node
        assert r.local_gpu == r.global_rank % hw.devices_per_node


def test_rank_mapping_decomposition_order(hw):
    """Verify innermost→outermost order: TP, EP, PP, DP."""
    par = ParallelismConfig(tp=2, pp=2, dp=2, ep=2)
    result = compute_rank_mapping(par, hw, num_layers=4)

    # rank 0: tp=0, ep=0, pp=0, dp=0
    r = result[0]
    assert (r.tp_rank, r.ep_rank, r.pp_stage, r.dp_rank) == (0, 0, 0, 0)

    # rank 1: tp=1, ep=0, pp=0, dp=0
    r = result[1]
    assert (r.tp_rank, r.ep_rank, r.pp_stage, r.dp_rank) == (1, 0, 0, 0)

    # rank 2: tp=0, ep=1, pp=0, dp=0
    r = result[2]
    assert (r.tp_rank, r.ep_rank, r.pp_stage, r.dp_rank) == (0, 1, 0, 0)

    # rank 4: tp=0, ep=0, pp=1, dp=0
    r = result[4]
    assert (r.tp_rank, r.ep_rank, r.pp_stage, r.dp_rank) == (0, 0, 1, 0)

    # rank 8: tp=0, ep=0, pp=0, dp=1
    r = result[8]
    assert (r.tp_rank, r.ep_rank, r.pp_stage, r.dp_rank) == (0, 0, 0, 1)


# ---------------------------------------------------------------------------
# build_logical_mesh_figure tests
# ---------------------------------------------------------------------------

def test_build_logical_mesh_figure_returns_figure(hw):
    """build_logical_mesh_figure returns a go.Figure with traces."""
    par = ParallelismConfig(tp=4, pp=2, dp=2)
    fig = build_logical_mesh_figure(par, hw, num_layers=32)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # Has at least one trace


def test_build_logical_mesh_figure_simple(hw):
    """Simple TP=2, PP=2, DP=1 → figure is valid."""
    par = ParallelismConfig(tp=2, pp=2, dp=1)
    fig = build_logical_mesh_figure(par, hw, num_layers=8)

    assert isinstance(fig, go.Figure)
    # Should have traces for PP stages (2 stages × 1 DP group)
    assert len(fig.data) >= 2


def test_build_logical_mesh_figure_with_ep(hw):
    """EP>1 should produce valid figure."""
    par = ParallelismConfig(tp=2, pp=2, dp=2, ep=2)
    fig = build_logical_mesh_figure(par, hw, num_layers=8, max_dp_shown=2)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_build_logical_mesh_figure_max_dp_shown(hw):
    """max_dp_shown limits DP groups displayed."""
    par = ParallelismConfig(tp=2, pp=1, dp=4)
    # max_dp_shown=1: only first DP group shown
    fig1 = build_logical_mesh_figure(par, hw, num_layers=8, max_dp_shown=1)
    # max_dp_shown=2: first two DP groups shown
    fig2 = build_logical_mesh_figure(par, hw, num_layers=8, max_dp_shown=2)

    # fig2 should have more or equal traces than fig1
    assert len(fig2.data) >= len(fig1.data)
