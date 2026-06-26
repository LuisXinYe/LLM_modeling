"""Tests for dynamic context parallelism (dynamic-CP) analytical modeling.

Compares two recipes — static CP and dynamic CP, each with packing + PP bubble.
"""

from pathlib import Path

import pytest

from llm_perf.config import (
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.dynamic_cp import (
    assign_cp,
    compare_cp_strategies,
    lognormal_buckets,
    packing_efficiency,
)

CONFIGS = Path(__file__).parent.parent / "configs"


def test_assign_cp_powers_of_two():
    # quota = 4096, max_cp = 8 → cp = smallest pow2 with len/cp <= quota
    assert assign_cp(4096, 4096, 8) == 1
    assert assign_cp(8192, 4096, 8) == 2
    assert assign_cp(16384, 4096, 8) == 4
    assert assign_cp(32768, 4096, 8) == 8
    assert assign_cp(65536, 4096, 8) == 8   # clamped to max_cp
    assert assign_cp(100, 4096, 8) == 1     # short → no sharding


def test_lognormal_buckets_normalized():
    b = lognormal_buckets(4096, 8192, 65536, n_buckets=8)
    assert abs(sum(f for _, f in b) - 1.0) < 1e-6
    assert all(length > 0 and frac > 0 for length, frac in b)


def test_lognormal_buckets_degenerate_without_std():
    # no std → single fixed-length bucket (no variable-length distribution)
    assert lognormal_buckets(4096, None, 65536) == [(4096.0, 1.0)]
    assert lognormal_buckets(4096, 0, 65536) == [(4096.0, 1.0)]


def test_packing_efficiency_bounds_and_fill():
    # a single bucket exactly equal to the budget packs perfectly
    assert packing_efficiency([(4096, 1.0)], 4096) == pytest.approx(1.0)
    # a length that divides the budget evenly → full fill
    assert packing_efficiency([(2048, 1.0)], 4096) == pytest.approx(1.0)
    # a length that does NOT divide the budget → fragmentation < 1
    eta = packing_efficiency([(3000, 1.0)], 4096)  # floor(4096/3000)=1 → 3000/4096
    assert eta == pytest.approx(3000 / 4096)
    # always in (0, 1]
    eta2 = packing_efficiency(lognormal_buckets(4096, 8192, 65536), 65536)
    assert 0 < eta2 <= 1.0


@pytest.fixture
def mc():
    return load_model_config(str(CONFIGS / "models" / "llama3_1_8b.yaml"))


@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS / "hardware" / "ascend_910c.yaml"))


def test_compare_speedup_variable_length(mc, hw):
    par = ParallelismConfig(tp=8, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(4096, 8192, 65536, n_buckets=8)
    r = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=64)
    # variable-length with short-sequence mass → dynamic is faster per step
    assert r["speedup"] > 1.0
    assert r["dynamic"]["step_s"] < r["static"]["step_s"]
    # dynamic achieves higher MFU / TFLOPS-per-GPU
    assert r["dynamic"]["tflops_per_gpu"] > r["static"]["tflops_per_gpu"]
    assert r["tflops_ratio"] > 1.0
    # short bucket → small cp; longest bucket → max_cp (dynamic)
    assert r["dynamic"]["buckets"][0]["cp"] < r["max_cp"]
    assert r["dynamic"]["buckets"][-1]["cp"] == r["max_cp"]
    # static forces max_cp everywhere
    assert all(b["cp"] == r["max_cp"] for b in r["static"]["buckets"])


def test_compare_no_gain_uniform_length(mc, hw):
    par = ParallelismConfig(tp=8, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    # every sequence at max length → both recipes assign max_cp → no gain
    r = compare_cp_strategies(mc, hw, par, wl, [(32768, 1.0)], total_ranks=64)
    assert r["speedup"] == pytest.approx(1.0)
    assert r["dynamic"]["buckets"][0]["cp"] == r["max_cp"]


def test_pp_bubble_shrinks_with_more_micro_batches(mc, hw):
    par = ParallelismConfig(tp=8, cp=8, dp=1, pp=4)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(4096, 8192, 65536, n_buckets=8)
    few = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=64, num_micro_batches=4)
    many = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=64, num_micro_batches=64)
    # more micro-batches amortize the (pp-1) warmup → smaller bubble fraction
    assert many["dynamic"]["bubble_ratio"] < few["dynamic"]["bubble_ratio"]
    assert few["dynamic"]["bubble_ratio"] > 0.0
    # pp=1 → no bubble at all
    par1 = ParallelismConfig(tp=8, cp=8, dp=1, pp=1)
    r1 = compare_cp_strategies(mc, hw, par1, wl, buckets, total_ranks=64)
    assert r1["dynamic"]["bubble_ratio"] == pytest.approx(0.0)


def test_packing_inflates_step_time(mc, hw):
    par = ParallelismConfig(tp=8, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(4096, 8192, 65536, n_buckets=8)
    perfect = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=64, packing_eff=1.0)
    lossy = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=64, packing_eff=0.5)
    # worse packing → longer step, lower MFU; the speedup ratio is unchanged
    assert lossy["dynamic"]["step_s"] > perfect["dynamic"]["step_s"]
    assert lossy["dynamic"]["mfu"] < perfect["dynamic"]["mfu"]
    assert lossy["speedup"] == pytest.approx(perfect["speedup"])
