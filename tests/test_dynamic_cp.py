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
    assign_bin_cp,
    assign_cp,
    compare_cp_strategies,
    lognormal_buckets,
    pack_units,
    packing_efficiency,
)
from llm_perf.pp_pipeline import PoolUnit

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


def test_compare_pipeline_speedup_variable_length(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=8)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(4096, 8192, 65536, n_buckets=6)
    r = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=8, pp=8, v=1)
    assert r["speedup"] > 1.0
    assert r["dynamic"]["step_s"] < r["static"]["step_s"]
    assert 0.0 <= r["dynamic"]["bubble_ratio"] < 1.0
    assert r["static"]["m"] >= 1 and r["dynamic"]["m"] >= 1
    # static forces max_cp on every unit
    assert all(u["cp"] == r["max_cp"] for u in r["static"]["units"])


def test_compare_pipeline_uniform_no_gain(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=8)
    wl = WorkloadConfig(group_size=1)
    r = compare_cp_strategies(mc, hw, par, wl, [(32768, 1.0)], total_ranks=8, pp=8, v=1)
    assert r["speedup"] == pytest.approx(1.0, abs=0.05)


def test_assign_bin_cp_workload_vs_memory(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    quota = 4096
    big_hbm = hw.usable_hbm_gb
    # short seq, ample memory → workload-driven (cp=1)
    assert assign_bin_cp(mc, hw, par, wl, 4096, quota, 8, big_hbm) == 1
    # long seq → workload pushes cp up
    assert assign_bin_cp(mc, hw, par, wl, 32768, quota, 8, big_hbm) >= 4
    # tiny memory budget forces cp up even for a short seq (memory-driven)
    cp_mem = assign_bin_cp(mc, hw, par, wl, 4096, quota, 8, usable_hbm_gb=0.05)
    assert cp_mem > 1


def test_pack_units_homogeneous_and_counts():
    def cp_of(L):
        return 1 if L <= 4096 else 8

    def packing_eff_of(L):
        return 1.0

    buckets = [(4096.0, 0.5), (32768.0, 0.5)]
    R, B = 8, 4096
    global_batch_seqs = 64
    units = pack_units(buckets, R, B, cp_of, packing_eff_of=packing_eff_of,
                       global_batch_seqs=global_batch_seqs)
    assert all(isinstance(u, PoolUnit) for u in units)
    # each bin contributes >=1 unit; units in a bin share cp + seq_len
    cps = {u.seq_len: u.cp for u in units}
    assert cps[4096.0] == 1 and cps[32768.0] == 8
    assert all(u.packed_tokens == R * B for u in units)
    # dp-aware unit count: bin_seqs = w_b * global_batch_seqs, dp_b = R // cp_b,
    # n_b = ceil(bin_seqs / dp_b).
    # short bin: cp=1 -> dp=8, bin_seqs=32 -> n_b=ceil(32/8)=4
    # long bin:  cp=8 -> dp=1, bin_seqs=32 -> n_b=ceil(32/1)=32
    counts = {}
    for u in units:
        counts[u.seq_len] = counts.get(u.seq_len, 0) + 1
    assert counts[4096] == 4
    assert counts[32768] == 32
    # the cp=1 (dp=R) bin needs R x fewer units than the cp=max_cp (dp=1) bin
    # for the same number of routed sequences.
    assert counts[32768] == R * counts[4096]


def test_order_units_balanced_spreads_slow():
    from llm_perf.dynamic_cp import order_units
    units = [PoolUnit(cp=1, seq_len=4096, packed_tokens=1, bin_index=0) for _ in range(6)]
    units += [PoolUnit(cp=8, seq_len=32768, packed_tokens=1, bin_index=1) for _ in range(2)]
    out = order_units(units, order="balanced")
    # the two slow (cp=8) units should not be adjacent at the very end
    slow_positions = [i for i, u in enumerate(out) if u.cp == 8]
    assert slow_positions[1] - slow_positions[0] >= 2


def test_run_pipeline_smoke(mc, hw):
    from llm_perf.dynamic_cp import run_pipeline
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=8)
    wl = WorkloadConfig(group_size=1)
    units = [PoolUnit(cp=1, seq_len=4096, packed_tokens=8 * 4096, bin_index=0) for _ in range(4)]
    units += [PoolUnit(cp=8, seq_len=32768, packed_tokens=8 * 4096, bin_index=1) for _ in range(2)]
    res = run_pipeline(mc, hw, par, wl, units, p=8, v=1)
    assert res.step_time > 0
    assert 0.0 <= res.bubble_ratio < 1.0


def test_compare_mfu_bounded(mc, hw):
    # MFU numerator/denominator must use a consistent backward-cost model, so
    # MFU (achieved/peak) is bounded by the hardware compute_eff ceiling
    # (<= 0.5 for ascend_910c) for BOTH static and dynamic recipes — dynamic-CP
    # raises utilization but cannot exceed the GEMM efficiency ceiling.
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=8)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(4096, 8192, 65536, n_buckets=6)
    r = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=8, pp=8, v=1)
    compute_eff = hw.calibration.compute_eff_large_gemm
    assert 0 < r["static"]["mfu"] <= compute_eff
    assert 0 < r["dynamic"]["mfu"] <= compute_eff
    assert r["dynamic"]["mfu"] > r["static"]["mfu"]
