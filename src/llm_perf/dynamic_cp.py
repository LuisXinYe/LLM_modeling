"""Dynamic context parallelism (dynamic-CP) — simplified analytical model.

Background (NVIDIA, "Speeding Up Variable-Length Training with Dynamic Context
Parallelism"): with variable-length training, a static CP degree is sized for
the *longest* sequence in a batch, which over-shards shorter sequences and
exposes CP communication that compute can no longer hide. Dynamic-CP instead
picks a CP degree per workload so that long sequences get a large CP and short
ones a small CP (down to 1).

This module compares two end-to-end recipes over a sequence-length distribution:

    * **Static CP  + packing + PP bubble** — every sequence sharded by max_cp.
    * **Dynamic CP + packing + PP bubble** — per-bucket CP by sequence length.

Both recipes share the same two realism layers on top of the raw per-sample
compute, so the comparison reflects achievable numbers rather than an idealized
upper bound:

    1. **Packing efficiency η** — variable-length sequences are packed into
       fixed-size micro-batches (token budget B); fragmentation leaves η < 1 of
       the budget doing useful work. Modeled as a fill factor from the length
       distribution (see ``packing_efficiency``).
    2. **PP bubble** — with pp pipeline stages and M micro-batches, the
       warmup/cooldown leaves a (pp-1)/(M+pp-1) idle fraction (reuses
       ``_stage_utils._pp_bubble_time``).

The static-vs-dynamic *gap* comes from the CP assignment (resource cost in
rank-seconds); packing and the PP bubble are overheads layered on both. We also
report MFU / achieved TFLOPS-per-GPU and the per-rank peak memory (the O(S)
memory axis), so the bi-objective FLOPs-vs-memory tension is visible.

It deliberately does NOT replicate the full packing solver + pipeline simulator
from the paper — it is an analytical estimate, not a scheduler.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from llm_perf.builder import build_forward_pass, build_training_step, _split_stages
from llm_perf.pp_pipeline import PoolUnit, stage_unit_time, simulate_pipeline
from llm_perf.simulator import SimResult, simulate


def assign_cp(seq_len: float, quota: float, max_cp: int) -> int:
    """Power-of-two CP degree for a sequence.

    Returns the smallest power of two cp such that seq_len / cp <= quota,
    clamped to [1, max_cp]. Short sequences map to cp=1 (no CP sharding),
    long ones up to max_cp.
    """
    if seq_len <= 0 or quota <= 0:
        return 1
    cp = 2 ** math.ceil(math.log2(max(1.0, seq_len / quota)))
    return max(1, min(max_cp, int(cp)))


def lognormal_buckets(
    avg: float, std: Optional[float], max_len: float, n_buckets: int = 8
) -> List[Tuple[float, float]]:
    """Discretize a clipped log-normal length distribution into buckets.

    Returns [(representative_len, fraction)] over n_buckets equal-width bins in
    (0, max_len], with fractions from the log-normal CDF matched to (avg, std).
    If std is missing/zero, returns a single bucket at `avg` (degenerate, i.e.
    fixed-length — dynamic-CP then has no variable-length gain).
    """
    if not std or std <= 0 or avg <= 0:
        return [(float(avg), 1.0)]

    var = std * std
    mu = math.log(avg * avg / math.sqrt(var + avg * avg))
    sigma = math.sqrt(math.log(1 + var / (avg * avg)))

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))

    buckets: List[Tuple[float, float]] = []
    prev_edge = 0.0
    prev_cdf = 0.0
    for i in range(n_buckets):
        edge = max_len * (i + 1) / n_buckets
        c = cdf(edge)
        frac = c - prev_cdf
        center = (prev_edge + edge) / 2
        if frac > 1e-6:
            buckets.append((center, frac))
        prev_edge, prev_cdf = edge, c

    # Renormalize (tail beyond max_len is clipped into the last bucket).
    total = sum(f for _, f in buckets) or 1.0
    return [(length, f / total) for length, f in buckets]


def packing_efficiency(
    seq_buckets: List[Tuple[float, float]], token_budget: float
) -> float:
    """Length-aware packing fill factor η ∈ (0, 1].

    Sequences are packed into micro-batches of `token_budget` tokens. Packing
    same-length sequences greedily, a bucket of length L fits ``floor(B/L)``
    sequences per micro-batch, leaving ``B - floor(B/L)*L`` tokens of
    fragmentation. The distribution-weighted fill is

        η = Σ_b  w_b · floor(B/L_b) · L_b / B          (L_b <= B)
        η_b = 1                                         (L_b  > B; one seq spans
                                                          the budget, CP-sharded)

    A wider budget B packs short sequences more tightly (η → 1); a budget equal
    to the mean length leaves large gaps for the short tail.
    """
    B = token_budget
    if B <= 0:
        return 1.0
    total = sum(f for _, f in seq_buckets) or 1.0
    eta = 0.0
    for length, frac in seq_buckets:
        w = frac / total
        if length <= 0:
            eta += w
        elif length > B:
            eta += w  # long sequence fills the budget (then CP-sharded)
        else:
            fit = math.floor(B / length)
            eta += w * (fit * length / B)
    return max(1e-6, min(1.0, eta))


def _sample_sim(model_cfg, hw, base_par, rl_cfg, seq_len: float, cp: int) -> SimResult:
    """Simulate ONE sample of length seq_len at CP degree cp (batch=1, pp=dp=1).

    Returns the full SimResult so callers can read wall-clock (with CP-ring comm
    exposed), pure compute time (AI-core busy → useful FLOPs), and peak memory.
    """
    par = base_par.model_copy(update={"cp": int(cp), "dp": 1, "pp": 1})
    cfg = rl_cfg.model_copy(update={
        "avg_prompt_len": int(seq_len),
        "avg_response_len": 0,
        "train_micro_batch_size": 1,
    })
    return simulate(build_training_step(model_cfg, hw, par, cfg))


def _fwd_compute_cp1(model_cfg, hw, base_par, wl, seq_len: float) -> float:
    """Forward-only compute time of the full model at cp=1 (pp=1, dp=1).

    Used as the MFU numerator's per-sequence unit so the numerator's implied
    backward cost (via ``1 + bwd_factor`` in the caller) matches the pipeline
    simulator's heuristic backward model (``bwd_t = bwd_factor * fwd_t``),
    instead of ``build_training_step``'s real forward+backward (~6x forward).
    """
    par = base_par.model_copy(update={"cp": 1, "pp": 1, "dp": 1})
    cfg = wl.model_copy(update={
        "avg_prompt_len": int(seq_len),
        "avg_response_len": 0,
        "train_micro_batch_size": 1,
    })
    return simulate(build_forward_pass(model_cfg, hw, par, cfg)).compute_time


def _recipe(model_cfg, hw, parallel_cfg, rl_cfg, buckets, total_ranks, quota,
            token_budget, max_cp, p, v, bwd_factor, order, cp_of,
            global_batch_seqs):
    """Cost of one recipe (CP assignment given by ``cp_of``) through the
    variable-length 1F1B(+V) pipeline simulator.

    Pipeline: ``pack_units`` (bins → homogeneous pool units) → ``order_units``
    (inflow order) → ``run_pipeline`` (simulate). MFU is reported against the
    cp=1 irreducible-compute baseline (same convention as before): CP sharding
    replicates fixed attention/small-op cost, which is overhead, not useful
    FLOPs.
    """
    usable = hw.usable_hbm_gb

    def eta_of(L):
        return packing_efficiency([(L, 1.0)], token_budget)

    units = pack_units(buckets, total_ranks, token_budget, cp_of, eta_of,
                       global_batch_seqs)
    units = order_units(units, order=order)
    res = run_pipeline(model_cfg, hw, parallel_cfg, rl_cfg, units, p, v, bwd_factor)
    # MFU vs irreducible cp=1 compute (existing convention). Each pool-wide
    # unit at CP degree u.cp actually runs dp_u = total_ranks // u.cp sequences
    # of useful compute concurrently, so the numerator must be scaled by the
    # dp multiplier; the denominator is the GPU-time available over the whole
    # rank pool (total_ranks * p stages * wall-clock step).
    #
    # The numerator must assume the SAME backward-cost model as the pipeline
    # (denominator): bwd_t = bwd_factor * fwd_t. Using build_training_step's
    # real forward+backward (~6x forward) here would inflate the numerator
    # relative to the simulator's heuristic denominator and push MFU above
    # the compute_eff ceiling. So useful = forward(cp=1) * (1 + bwd_factor).
    fwd_cache: dict = {}

    def fwd_cp1(seq_len):
        if seq_len not in fwd_cache:
            fwd_cache[seq_len] = _fwd_compute_cp1(model_cfg, hw, parallel_cfg, rl_cfg, seq_len)
        return fwd_cache[seq_len]

    useful = sum((total_ranks // max(1, u.cp)) *
                 fwd_cp1(u.seq_len) * (1.0 + bwd_factor)
                 for u in units)
    rank_seconds = sum(u.cp * _sample_sim(model_cfg, hw, parallel_cfg, rl_cfg,
                                          u.seq_len, u.cp).wall_clock_time for u in units)
    compute_eff = hw.calibration.compute_eff_large_gemm
    denom = total_ranks * p * res.step_time
    mfu = compute_eff * useful / denom if denom > 0 else 0.0
    weight_gb = (_sample_sim(model_cfg, hw, parallel_cfg, rl_cfg, units[0].seq_len,
                             units[0].cp).weight_bytes / 1e9) if units else 0.0
    peak_mem_gb = res.peak_activation_bytes / 1e9 + weight_gb
    busy = res.per_device_busy
    imbalance = (max(busy) / (sum(busy) / len(busy))) if busy and sum(busy) > 0 else 1.0
    return {
        "m": len(units),
        "step_s": res.step_time,
        "bubble_ratio": res.bubble_ratio,
        "mfu": mfu,
        "tflops_per_gpu": hw.peak_tflops_bf16 * mfu,
        "peak_mem_gb": peak_mem_gb,
        "feasible": peak_mem_gb <= usable,
        "imbalance": imbalance,
        "rank_seconds_per_sample": rank_seconds / len(units) if units else 0.0,
        "units": [{"cp": u.cp, "seq_len": u.seq_len} for u in units],
    }


def compare_cp_strategies(
    model_cfg,
    hw,
    parallel_cfg,
    rl_cfg,
    seq_buckets: List[Tuple[float, float]],
    total_ranks: int,
    quota: Optional[float] = None,
    token_budget: Optional[float] = None,
    num_micro_batches: Optional[int] = None,
    packing_eff: Optional[float] = None,
    pp: Optional[int] = None,
    v: int = 1,
    bwd_factor: float = 2.0,
    order: str = "balanced",
    global_batch_seqs: int = 64,
) -> dict:
    """Compare *static CP* vs *dynamic CP*, each through the variable-length
    1F1B(+V) pipeline simulator (``pack_units`` → ``order_units`` →
    ``run_pipeline``).

    Args:
        model_cfg, hw: model + hardware configs.
        parallel_cfg: base parallelism. ``cp`` is the max CP available; ``pp``
            is the pipeline depth (overridable via the ``pp`` kwarg); ``tp``
            is folded into the per-unit stage sim.
        rl_cfg: workload (layer/recompute settings; lengths come from buckets).
        seq_buckets: [(seq_len, fraction)] length distribution (renormalized).
        total_ranks: size of the CP/DP rank pool work is packed/amortized over.
        quota: target per-rank sequence length for dynamic CP. Defaults to
            max_len / max_cp (longest bucket → full CP).
        token_budget: micro-batch packing budget in tokens. Defaults to max_len.
        num_micro_batches: unused by the pipeline simulator; kept for backward
            compatibility with callers (the simulator derives bubble from the
            packed unit count and pipeline depth).
        packing_eff: unused (kept for backward-compatible signature); packing
            efficiency is derived per-bucket internally via ``packing_efficiency``.
        pp: pipeline depth. Defaults to ``parallel_cfg.pp``.
        v: virtual pipeline-parallel multiplier (interleaved schedule). Only
            v=1 is currently supported by the underlying simulator.
        bwd_factor: backward/forward time ratio passed to the per-stage sim.
        order: unit inflow order for ``order_units`` ("balanced" by default).
        global_batch_seqs: total sequences in the global batch routed across
            bins; drives the dp-aware unit count in ``pack_units`` (each bin's
            unit count is ``ceil(bin_seqs / dp_b)`` where ``dp_b = R // cp_b``).

    Returns a dict with ``static`` and ``dynamic`` recipe breakdowns (see
    ``_recipe``) plus the top-level ``speedup`` (static_step / dynamic_step)
    and ``tflops_ratio``.
    """
    max_cp = max(1, int(parallel_cfg.cp))
    max_len = max(length for length, _ in seq_buckets)
    if quota is None:
        quota = max_len / max_cp
    if token_budget is None:
        token_budget = max_len
    p = int(pp) if pp is not None else int(parallel_cfg.pp)
    usable = hw.usable_hbm_gb

    def static_cp_of(L):
        return max_cp

    def dynamic_cp_of(L):
        return assign_bin_cp(model_cfg, hw, parallel_cfg, rl_cfg, L, quota, max_cp, usable)

    static = _recipe(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets, total_ranks,
                     quota, token_budget, max_cp, p, v, bwd_factor, order, static_cp_of,
                     global_batch_seqs)
    dynamic = _recipe(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets, total_ranks,
                      quota, token_budget, max_cp, p, v, bwd_factor, order, dynamic_cp_of,
                      global_batch_seqs)

    speedup = static["step_s"] / dynamic["step_s"] if dynamic["step_s"] > 0 else 0.0
    tflops_ratio = (dynamic["tflops_per_gpu"] / static["tflops_per_gpu"]
                    if static["tflops_per_gpu"] > 0 else 0.0)
    return {
        "max_cp": max_cp, "quota": quota, "token_budget": token_budget,
        "total_ranks": total_ranks, "p": p, "v": v,
        "static": static, "dynamic": dynamic,
        "speedup": speedup, "tflops_ratio": tflops_ratio,
    }


def assign_bin_cp(model_cfg, hw, base_par, wl, seq_len, quota, max_cp,
                  usable_hbm_gb) -> int:
    """Per-bin CP = clamp(max(cp_workload, cp_memory), 1, max_cp).

    cp_workload keeps per-rank sequence ≤ quota; cp_memory is the smallest cp
    whose per-rank (weight+activation) fits usable_hbm_gb. Doubles cp until it
    fits (memory repair); returns max_cp if it never fits (caller flags OOM).
    """
    cp = assign_cp(seq_len, quota, max_cp)
    while cp <= max_cp:
        sim = _sample_sim(model_cfg, hw, base_par, wl, seq_len, cp)
        mem_gb = (sim.weight_bytes + sim.peak_activation_bytes) / 1e9
        if mem_gb <= usable_hbm_gb or cp >= max_cp:
            return cp
        cp = min(max_cp, cp * 2)
    return max_cp


def pack_units(buckets, total_ranks, token_budget, cp_of, packing_eff_of,
               global_batch_seqs):
    """Pack a length distribution into homogeneous pool-wide units.

    Encodes the pool-wide microbatch abstraction (spec Section A): a pool of
    ``R = total_ranks`` ranks running a unit at context-parallel degree ``cp``
    uses data-parallel ``dp = R // cp``, so the unit processes ``dp`` sequences
    CONCURRENTLY in one wall-time slot. A cp=1 unit (dp=R) thus needs R× fewer
    units to drain the same number of sequences than a cp=max_cp unit (dp=1) —
    this is the dynamic-CP throughput gain that the (now-removed) cp-agnostic
    unit count failed to capture.

    For each bin b: w_b = renormalized fraction, cp_b = cp_of(L_b),
    dp_b = max(1, R // cp_b), bin_seqs = w_b * global_batch_seqs (sequences
    routed to this bin), n_b = ceil(bin_seqs / dp_b) for a non-empty bin (0 if
    bin_seqs == 0). Each emitted unit shares the bin's cp and representative
    seq_len; ``packing_eff_of`` is accepted for signature compatibility with
    callers but packing efficiency does not change the dp-driven unit count.
    """
    total_frac = sum(f for _, f in buckets) or 1.0
    R, B = total_ranks, token_budget
    units = []
    for bi, (length, frac) in enumerate(buckets):
        w = frac / total_frac
        cp_b = cp_of(length)
        dp_b = max(1, R // max(1, int(cp_b)))
        bin_seqs = w * global_batch_seqs
        n_b = math.ceil(bin_seqs / dp_b) if bin_seqs > 0 else 0
        for _ in range(n_b):
            units.append(PoolUnit(cp=int(cp_b), seq_len=int(length),
                                  packed_tokens=int(R * B), bin_index=bi))
    return units


def order_units(units, order: str = "balanced"):
    """Order pool units for pipeline inflow. 'balanced' interleaves slow units."""
    if order == "as_packed":
        return list(units)
    keyed = sorted(units, key=lambda u: (u.cp, u.seq_len), reverse=True)  # slow first
    if order == "descending":
        return keyed
    # balanced: deal slow→fast round-robin into `stride` buckets then concatenate
    n = len(keyed)
    if n <= 1:
        return keyed
    stride = max(2, int(round(n ** 0.5)))
    buckets = [[] for _ in range(stride)]
    for i, u in enumerate(keyed):
        buckets[i % stride].append(u)
    out = []
    for b in buckets:
        out.extend(b)
    return out


def run_pipeline(model_cfg, hw, base_par, wl, units, p, v=1, bwd_factor=2.0):
    """Time a list of pool units through the variable-length 1F1B(+V) pipeline."""
    S = p * v
    chunks = _split_stages(model_cfg.get_layers(), S)
    cache: dict = {}
    unit_stage_times = []
    unit_act_bytes = []
    for u in units:
        per_stage = []
        for vs in range(S):
            fwd_t, bwd_t = stage_unit_time(
                model_cfg, hw, base_par, wl, chunks[vs], chunk_id=vs,
                cp=u.cp, seq_len=u.seq_len, bwd_factor=bwd_factor, cache=cache,
            )
            per_stage.append((fwd_t, bwd_t))
        unit_stage_times.append(per_stage)
        # activation footprint of the unit at one stage (proxy: inner-sim peak)
        sim = _sample_sim(model_cfg, hw, base_par, wl, u.seq_len, u.cp)
        unit_act_bytes.append(sim.peak_activation_bytes)
    return simulate_pipeline(unit_stage_times, unit_act_bytes, p, v=v)
