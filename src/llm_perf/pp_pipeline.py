"""Variable-length 1F1B(+V) pipeline simulator over the pool-wide microbatch
abstraction. See docs/superpowers/specs/2026-06-26-dynamic-cp-pipeline-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from llm_perf.builder import build_forward_pass
from llm_perf.simulator import simulate


@dataclass
class PoolUnit:
    """One pool-wide pipeline microbatch (fills the whole CP/DP pool).

    cp: context-parallel degree used for this unit.
    seq_len: full sequence length L processed (per-rank tokens = L // cp).
    packed_tokens: total tokens carried by the unit (≈ R·B), for accounting.
    bin_index: source length-bin index (diagnostics / ordering).
    """
    cp: int
    seq_len: int
    packed_tokens: int
    bin_index: int


def stage_unit_time(
    model_cfg, hw, base_par, wl, chunk_layers, chunk_id: int, cp: int,
    seq_len: int, bwd_factor: float = 2.0, cache: Optional[dict] = None,
) -> Tuple[float, float]:
    """Forward/backward time (s) for one pool unit at one virtual stage.

    fwd_t = max(compute, cp_ring_comm) + tp_comm   (CP-ring overlaps compute).
    bwd_t = bwd_factor * fwd_t.
    Cached by (chunk_id, cp, seq_len) — identical bins/stages reuse the sim.
    """
    key = (chunk_id, int(cp), int(seq_len))
    if cache is not None and key in cache:
        return cache[key]
    par = base_par.model_copy(update={"cp": int(cp), "pp": 1, "dp": 1})
    cfg = wl.model_copy(update={
        "avg_prompt_len": int(seq_len), "avg_response_len": 0,
        "train_micro_batch_size": 1,
    })
    sim = simulate(build_forward_pass(model_cfg, hw, par, cfg, stage_layers=chunk_layers))
    fwd_t = max(sim.compute_time, sim.cp_comm_time) + sim.tp_comm_time
    bwd_t = bwd_factor * fwd_t
    result = (fwd_t, bwd_t)
    if cache is not None:
        cache[key] = result
    return result
