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

from llm_perf._stage_utils import _pp_bubble_time
from llm_perf.builder import build_training_step
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


def _strategy_cost(
    model_cfg,
    hw,
    parallel_cfg,
    rl_cfg,
    seq_buckets,
    *,
    cp_of,
    max_cp,
    total_ranks: int,
    token_budget: float,
    num_micro_batches: int,
    packing_eff: Optional[float],
    sim_cache: dict,
) -> dict:
    """Cost of one recipe (CP assignment given by ``cp_of``) over the length dist.

    Layers, in order:
      1. per-sample wall + compute time + memory at the assigned CP (sim_cache);
      2. rank-seconds resource cost  Σ w·cp·t  (the static-vs-dynamic axis);
      3. packing efficiency η  (fragmentation fill, divides effective work);
      4. PP bubble  (idle GPU-seconds from pipeline warmup/cooldown);
      5. amortized step time, MFU / achieved TFLOPS-per-GPU, peak memory.
    """
    total_frac = sum(f for _, f in seq_buckets) or 1.0

    def sim(length: float, cp: int) -> SimResult:
        key = (round(length), cp)
        if key not in sim_cache:
            sim_cache[key] = _sample_sim(model_cfg, hw, parallel_cfg, rl_cfg, length, cp)
        return sim_cache[key]

    rank_s = 0.0          # Σ w·cp·t : wall busy GPU-seconds (comm + CP replication)
    useful_rank_s = 0.0   # Σ w·compute(cp=1) : irreducible algorithmic FLOPs (GPU-s)
    peak_mem_gb = 0.0
    bucket_rows = []
    for length, frac in seq_buckets:
        w = frac / total_frac
        cp_b = cp_of(length)
        r = sim(length, cp_b)
        rank_s += w * cp_b * r.wall_clock_time
        # Useful work is the IRREDUCIBLE compute (cp=1). CP sharding replicates
        # fixed attention/small-op cost — that replication is overhead, not useful
        # FLOPs, so MFU is measured against the cp=1 baseline (same for both recipes).
        useful_rank_s += w * sim(length, 1).compute_time
        mem_gb = (r.weight_bytes + r.peak_activation_bytes) / 1e9
        peak_mem_gb = max(peak_mem_gb, mem_gb)
        bucket_rows.append({
            "seq_len": float(length),
            "fraction": w,
            "cp": cp_b,
            "t_s": r.wall_clock_time,
            "cost_rank_s": cp_b * r.wall_clock_time,
            "mem_gb": mem_gb,
        })

    # Packing: fragmentation inflates the effective work by 1/η.
    eta = packing_eff if packing_eff is not None else packing_efficiency(seq_buckets, token_budget)
    packed_rank_s = rank_s / eta

    # PP bubble: idle GPU-seconds layered on the packed work.
    bubble_rank_s = _pp_bubble_time(packed_rank_s, parallel_cfg, num_micro_batches)
    occupied_rank_s = packed_rank_s + bubble_rank_s
    bubble_ratio = bubble_rank_s / occupied_rank_s if occupied_rank_s > 0 else 0.0

    # Amortized wall-clock per sample over the rank pool (throughput proxy).
    step_s = occupied_rank_s / total_ranks if total_ranks > 0 else float("inf")

    # MFU / achieved TFLOPS-per-GPU, self-consistent with the cost model:
    #   compute_time = useful_FLOPs / (peak · compute_eff)  ⇒
    #   MFU = compute_eff · (useful AI-core GPU-s) / (occupied GPU-s)
    compute_eff = hw.calibration.compute_eff_large_gemm
    mfu = compute_eff * useful_rank_s / occupied_rank_s if occupied_rank_s > 0 else 0.0
    tflops_per_gpu = hw.peak_tflops_bf16 * mfu

    usable_hbm = hw.usable_hbm_gb
    return {
        "rank_seconds_per_sample": rank_s,
        "packing_eff": eta,
        "num_micro_batches": num_micro_batches,
        "bubble_ratio": bubble_ratio,
        "occupied_rank_s": occupied_rank_s,
        "step_s": step_s,
        "throughput_samples_s": 1.0 / step_s if step_s > 0 else 0.0,
        "mfu": mfu,
        "tflops_per_gpu": tflops_per_gpu,
        "peak_mem_gb": peak_mem_gb,
        "usable_hbm_gb": usable_hbm,
        "feasible": peak_mem_gb <= usable_hbm,
        "buckets": bucket_rows,
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
) -> dict:
    """Compare *static CP* vs *dynamic CP*, each with packing + PP bubble.

    Args:
        model_cfg, hw: model + hardware configs.
        parallel_cfg: base parallelism. ``cp`` is the max CP available; ``pp`` is
            the pipeline depth used for the bubble; ``tp`` is folded into the
            per-sample sim.
        rl_cfg: workload (layer/recompute settings; lengths come from buckets).
        seq_buckets: [(seq_len, fraction)] length distribution (renormalized).
        total_ranks: size of the CP/DP rank pool the work is amortized over.
        quota: target per-rank sequence length for dynamic CP. Defaults to
            max_len / max_cp (longest bucket → full CP).
        token_budget: micro-batch packing budget in tokens. Defaults to max_len.
        num_micro_batches: micro-batches per pipeline fill (for the bubble).
            Defaults to max(pp, gradient_accumulation_steps).
        packing_eff: override the modeled packing efficiency η for both recipes.

    Returns a dict with ``static`` and ``dynamic`` cost breakdowns plus the
    top-level ``speedup`` (static_step / dynamic_step), ``tflops_ratio`` and
    ``mfu`` gain. ``speedup`` is driven by the CP assignment; packing and the
    PP bubble shift the absolute step time / MFU of both recipes.
    """
    max_cp = max(1, int(parallel_cfg.cp))
    max_len = max(length for length, _ in seq_buckets)
    if quota is None:
        quota = max_len / max_cp
    if token_budget is None:
        token_budget = max_len
    if num_micro_batches is None:
        num_micro_batches = max(int(parallel_cfg.pp), int(rl_cfg.gradient_accumulation_steps))

    sim_cache: dict = {}  # shared (len, cp) cache across both strategies

    def common(cp_of):
        return _strategy_cost(
            model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets,
            cp_of=cp_of, max_cp=max_cp, total_ranks=total_ranks,
            token_budget=token_budget, num_micro_batches=num_micro_batches,
            packing_eff=packing_eff, sim_cache=sim_cache,
        )

    static = common(lambda length: max_cp)
    dynamic = common(lambda length: assign_cp(length, quota, max_cp))

    speedup = (
        static["step_s"] / dynamic["step_s"] if dynamic["step_s"] > 0 else 0.0
    )
    tflops_ratio = (
        dynamic["tflops_per_gpu"] / static["tflops_per_gpu"]
        if static["tflops_per_gpu"] > 0 else 0.0
    )
    return {
        "max_cp": max_cp,
        "quota": quota,
        "token_budget": token_budget,
        "num_micro_batches": num_micro_batches,
        "total_ranks": total_ranks,
        "static": static,
        "dynamic": dynamic,
        "speedup": speedup,
        "tflops_ratio": tflops_ratio,
    }
