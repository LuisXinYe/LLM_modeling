# Dynamic-CP Pipeline Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare static-CP vs dynamic-CP on the same global batch using a 3-step dynamic-CP solver and a variable-length 1F1B(+V) pipeline simulator with CP-ring comm/compute overlap.

**Architecture:** A "pool-wide microbatch" abstraction turns the 2-D (stage × intra-stage-rank) schedule into a 1-D pipeline over the R=8 CP/DP pool. Inner `simulate()` calls give per-(unit, virtual-stage) fwd/bwd scalars; an outer `simulate()` over a meta-op DAG (schedule encoded as dependencies) yields the real variable-length step time and bubble. A heuristic solver packs sequences into homogeneous bins, picks memory-aware per-bin CP, and chooses `m` by searching the token budget against the simulator.

**Tech Stack:** Python 3.10+, pydantic configs, existing `llm_perf.builder`/`simulator` multi-stream engine, pytest, ruff.

## Global Constraints

- Reference scenario: Llama-3.1-8B, 128 GPUs, `tp=2 × pp=8 × (dp·cp pool R=8)`, `max_cp=8`, 1F1B with `V≥1`.
- Comm-overlap model lives ONLY in the new `pp_pipeline.py` path. Do NOT modify `builder.py`, `inference.py`, `training.py`, `post_training.py` — existing 193 tests must stay green.
- MFU uses the irreducible-FLOPs (cp=1) baseline convention already in `dynamic_cp.py`.
- `bwd_t = bwd_factor × fwd_t`, `bwd_factor` default `2.0`.
- Overlap rule: per-stage `fwd_t = max(compute_time, cp_comm_time) + tp_comm_time` (CP-ring overlaps compute; TP comm does not).
- Virtual stages `S = p × V` must satisfy `S ≤ num_layers` (Llama-8B = 32 layers → V ≤ 4). `_split_stages(all_layers, S)` produces the chunks.
- Inner sim sets sequence length via `WorkloadConfig(avg_prompt_len=L, avg_response_len=0, train_micro_batch_size=1)`; pass full `L`, the builder shards by cp internally (`local_seq = L // cp`).
- Remove `length-routed DP`: it is already gone from `dynamic_cp.py`. Static-CP sweep is out of scope.
- Reply to the user in Chinese (project convention).

---

## File Structure

- **Create** `src/llm_perf/pp_pipeline.py` — `PoolUnit`, inner stage timing (`stage_unit_time`), schedule generator (`pipeline_schedule`), outer simulator (`simulate_pipeline`, `PipelineResult`). One responsibility: turn a list of pool units into a pipeline step time / bubble / memory.
- **Modify** `src/llm_perf/dynamic_cp.py` — add the solver (`pack_units`, `assign_bin_cp`, `solve_units`) and rewire `compare_cp_strategies` through solver + `simulate_pipeline`. Keep `assign_cp`, `lognormal_buckets`, `packing_efficiency`, `_sample_sim`.
- **Create** `tests/test_pp_pipeline.py` — simulator unit tests (anchor + variable-length + overlap + memory).
- **Modify** `tests/test_dynamic_cp.py` — solver + integration tests.
- **Modify** `examples/demo_dynamic_cp.py` — Llama 128-GPU / tp2·pp8 demo with the new output table.

---

## Task 1: PoolUnit + inner stage timing with comm overlap

**Files:**
- Create: `src/llm_perf/pp_pipeline.py`
- Test: `tests/test_pp_pipeline.py`

**Interfaces:**
- Consumes: `llm_perf.builder.build_forward_pass`, `llm_perf.builder._split_stages`, `llm_perf.simulator.simulate`, `llm_perf.config.ParallelismConfig/WorkloadConfig`.
- Produces:
  - `@dataclass PoolUnit: cp:int; seq_len:int; packed_tokens:int; bin_index:int`
  - `stage_unit_time(model_cfg, hw, base_par, wl, chunk_layers, chunk_id, cp, seq_len, bwd_factor=2.0, cache=None) -> tuple[float,float]` returns `(fwd_t, bwd_t)` seconds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_pipeline.py
from pathlib import Path
import pytest
from llm_perf.config import (
    ParallelismConfig, WorkloadConfig, load_hardware_config, load_model_config,
)
from llm_perf.builder import _split_stages
from llm_perf.pp_pipeline import PoolUnit, stage_unit_time

CONFIGS = Path(__file__).parent.parent / "configs"

@pytest.fixture
def mc():
    return load_model_config(str(CONFIGS / "models" / "llama3_1_8b.yaml"))

@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS / "hardware" / "ascend_910c.yaml"))

def test_stage_unit_time_overlap_and_bwd(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    chunk = _split_stages(mc.get_layers(), 8)[0]  # one PP stage worth of layers
    fwd_t, bwd_t = stage_unit_time(mc, hw, par, wl, chunk, chunk_id=0, cp=8, seq_len=32768)
    assert fwd_t > 0
    assert bwd_t == pytest.approx(2.0 * fwd_t)  # default bwd_factor

def test_stage_unit_time_cp1_no_ring_overlap_noop(mc, hw):
    # cp=1 has no CP ring comm, so overlap max(compute, cp_comm) == compute
    par = ParallelismConfig(tp=2, cp=1, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    chunk = _split_stages(mc.get_layers(), 8)[0]
    fwd_t, _ = stage_unit_time(mc, hw, par, wl, chunk, chunk_id=0, cp=1, seq_len=4096)
    # fwd_t equals compute + tp_comm (cp_comm == 0): recompute the raw sim to compare
    from llm_perf.builder import build_forward_pass
    from llm_perf.simulator import simulate
    p1 = par.model_copy(update={"cp": 1, "pp": 1})
    w1 = wl.model_copy(update={"avg_prompt_len": 4096, "avg_response_len": 0,
                               "train_micro_batch_size": 1})
    sim = simulate(build_forward_pass(mc, hw, p1, w1, stage_layers=chunk))
    assert fwd_t == pytest.approx(sim.compute_time + sim.tp_comm_time)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pp_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_perf.pp_pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/pp_pipeline.py
"""Variable-length 1F1B(+V) pipeline simulator over the pool-wide microbatch
abstraction. See docs/superpowers/specs/2026-06-26-dynamic-cp-pipeline-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pp_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/pp_pipeline.py tests/test_pp_pipeline.py
git commit -m "feat(pp_pipeline): PoolUnit + inner stage timing with CP-ring comm overlap"
```

---

## Task 2: 1F1B(+V) schedule generator

**Files:**
- Modify: `src/llm_perf/pp_pipeline.py`
- Test: `tests/test_pp_pipeline.py`

**Interfaces:**
- Produces: `pipeline_schedule(m:int, p:int, v:int=1) -> list[list[tuple[int,int,str]]]`. Returns `per_device[d]` = ordered list of events `(unit_idx, vstage, phase)` with `phase in {'F','B'}`, where `vstage % p == d`. For `v=1` this is standard 1F1B; for `v>1` it is the round-robin-interleaved flattening over `S = p*v` virtual stages.

Schedule construction: build the standard 1F1B event order for each of the `S = p*v` virtual stages, then for each physical device round-robin-interleave the event lists of its `v` virtual stages (`vstage = d, d+p, …`). Device serialization in the outer simulator then reproduces the interleaved bubble.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_pipeline.py  (append)
from llm_perf.pp_pipeline import pipeline_schedule

def test_schedule_counts_v1():
    m, p = 6, 4
    sched = pipeline_schedule(m, p, v=1)
    assert len(sched) == p
    for d in range(p):
        evs = sched[d]
        # each device runs m forwards and m backwards
        assert sum(1 for _, _, ph in evs if ph == "F") == m
        assert sum(1 for _, _, ph in evs if ph == "B") == m
        # all events on device d belong to vstage d (V=1)
        assert all(vs == d for _, vs, _ in evs)
        # 1F1B warmup: device d issues (p-1-d) forwards before its first backward
        first_b = next(i for i, (_, _, ph) in enumerate(evs) if ph == "B")
        assert first_b == (p - 1 - d)

def test_schedule_counts_v2():
    m, p, v = 4, 4, 2
    sched = pipeline_schedule(m, p, v)
    assert len(sched) == p
    for d in range(p):
        evs = sched[d]
        # v virtual stages per device, each with m F and m B
        assert sum(1 for _, _, ph in evs if ph == "F") == m * v
        assert sum(1 for _, _, ph in evs if ph == "B") == m * v
        assert all(vs % p == d for _, vs, _ in evs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pp_pipeline.py -k schedule -v`
Expected: FAIL with `ImportError: cannot import name 'pipeline_schedule'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/pp_pipeline.py  (append)
def _stage_1f1b_order(stage_idx: int, num_stages: int, m: int) -> List[Tuple[int, str]]:
    """Standard 1F1B event order for one stage: (microbatch_idx, phase)."""
    warmup = num_stages - 1 - stage_idx
    warmup = max(0, min(warmup, m))
    events: List[Tuple[int, str]] = []
    f_idx = 0
    b_idx = 0
    for _ in range(warmup):
        events.append((f_idx, "F")); f_idx += 1
    while f_idx < m:
        events.append((f_idx, "F")); f_idx += 1
        events.append((b_idx, "B")); b_idx += 1
    while b_idx < m:
        events.append((b_idx, "B")); b_idx += 1
    return events


def pipeline_schedule(m: int, p: int, v: int = 1) -> List[List[Tuple[int, int, str]]]:
    """Per-device ordered events for 1F1B(+V interleaved).

    S = p*v virtual stages; vstage s lives on device s % p. For each device,
    round-robin-interleave the standard-1F1B event lists of its v virtual stages.
    """
    S = p * v
    # standard 1F1B order per virtual stage: list of (unit_idx, phase)
    vstage_orders = [_stage_1f1b_order(s, S, m) for s in range(S)]
    per_device: List[List[Tuple[int, int, str]]] = [[] for _ in range(p)]
    for d in range(p):
        my_vstages = list(range(d, S, p))  # d, d+p, d+2p, ...
        cursors = {vs: 0 for vs in my_vstages}
        remaining = sum(len(vstage_orders[vs]) for vs in my_vstages)
        while remaining > 0:
            for vs in my_vstages:
                c = cursors[vs]
                if c < len(vstage_orders[vs]):
                    unit_idx, phase = vstage_orders[vs][c]
                    per_device[d].append((unit_idx, vs, phase))
                    cursors[vs] = c + 1
                    remaining -= 1
    return per_device
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pp_pipeline.py -k schedule -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/pp_pipeline.py tests/test_pp_pipeline.py
git commit -m "feat(pp_pipeline): 1F1B(+V) schedule generator"
```

---

## Task 3: Outer pipeline simulator + closed-form anchor

**Files:**
- Modify: `src/llm_perf/pp_pipeline.py`
- Test: `tests/test_pp_pipeline.py`

**Interfaces:**
- Produces:
  - `@dataclass PipelineResult: step_time:float; bubble_ratio:float; peak_activation_bytes:float; per_device_busy:list[float]`
  - `simulate_pipeline(unit_stage_times, unit_act_bytes, p, v=1, p2p_time=0.0) -> PipelineResult`, where `unit_stage_times[unit_idx][vstage] == (fwd_t, bwd_t)` (length `p*v` per unit) and `unit_act_bytes[unit_idx]` is the per-unit activation footprint (bytes). `bubble_ratio = 1 - sum(per_device_busy) / (p * step_time)`.

Builds a meta-op DAG: one `SimOp` per `(unit, vstage, phase)` on stream `dev{vstage % p}`, with data-flow deps (`F(j,v)←F(j,v-1)`, `B(j,v)←B(j,v+1)` and `B(j,last)←F(j,last)`) plus the per-device schedule-order chain from `pipeline_schedule`. Runs `simulate()`; `step_time = wall_clock_time`.

- [ ] **Step 1: Write the failing test (equal-length anchor)**

```python
# tests/test_pp_pipeline.py  (append)
from llm_perf.pp_pipeline import simulate_pipeline, PipelineResult

def _equal_times(m, p, v, fwd=1.0, bwd=2.0):
    return [[(fwd, bwd)] * (p * v) for _ in range(m)]

def test_anchor_bubble_v1():
    m, p = 8, 4
    res = simulate_pipeline(_equal_times(m, p, 1), [1.0] * m, p, v=1)
    expected = (p - 1) / (m + p - 1)         # standard 1F1B bubble
    assert res.bubble_ratio == pytest.approx(expected, abs=0.02)

def test_anchor_bubble_v2():
    m, p, v = 8, 4, 2
    res = simulate_pipeline(_equal_times(m, p, v), [1.0] * m, p, v=v)
    expected = (p - 1) / (m * v + p - 1)     # interleaved bubble
    assert res.bubble_ratio == pytest.approx(expected, abs=0.04)

def test_pp1_no_bubble():
    m, p = 5, 1
    res = simulate_pipeline(_equal_times(m, p, 1, fwd=1.0, bwd=2.0), [1.0] * m, p, v=1)
    assert res.bubble_ratio == pytest.approx(0.0, abs=1e-9)
    assert res.step_time == pytest.approx(m * (1.0 + 2.0))

def test_more_microbatches_smaller_bubble():
    p = 4
    b_few = simulate_pipeline(_equal_times(4, p, 1), [1.0] * 4, p, v=1).bubble_ratio
    b_many = simulate_pipeline(_equal_times(16, p, 1), [1.0] * 16, p, v=1).bubble_ratio
    assert b_many < b_few
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pp_pipeline.py -k "anchor or bubble or pp1" -v`
Expected: FAIL with `ImportError: cannot import name 'simulate_pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/pp_pipeline.py  (append)
from llm_perf.builder import SimOp


@dataclass
class PipelineResult:
    step_time: float
    bubble_ratio: float
    peak_activation_bytes: float
    per_device_busy: List[float]


def simulate_pipeline(
    unit_stage_times: List[List[Tuple[float, float]]],
    unit_act_bytes: List[float],
    p: int,
    v: int = 1,
    p2p_time: float = 0.0,
) -> PipelineResult:
    """Outer 1F1B(+V) pipeline simulation over pool-wide units.

    unit_stage_times[j][vs] = (fwd_t, bwd_t) for unit j at virtual stage vs.
    Returns step time, measured bubble ratio, peak in-flight activation, busy/dev.
    """
    m = len(unit_stage_times)
    S = p * v
    if m == 0:
        return PipelineResult(0.0, 0.0, 0.0, [0.0] * p)

    ops: List[SimOp] = []
    # index of the op for (unit j, vstage vs, phase) once created
    op_index: Dict[Tuple[int, int, str], int] = {}

    def add_op(j: int, vs: int, phase: str, duration: float, out_bytes: float) -> int:
        idx = len(ops)
        ops.append(SimOp(
            name=f"u{j}_v{vs}_{phase}", stream=f"dev{vs % p}",
            duration=duration, depends_on=[], output_bytes=out_bytes,
        ))
        op_index[(j, vs, phase)] = idx
        return idx

    # 1) create all ops (forward then backward virtual-stage chains)
    for j in range(m):
        for vs in range(S):
            fwd_t, _ = unit_stage_times[j][vs]
            add_op(j, vs, "F", fwd_t, unit_act_bytes[j])
        for vs in range(S):
            _, bwd_t = unit_stage_times[j][vs]
            add_op(j, vs, "B", bwd_t, 0.0)

    # 2) data-flow dependencies
    for j in range(m):
        for vs in range(1, S):                       # forward chain
            ops[op_index[(j, vs, "F")]].depends_on.append(op_index[(j, vs - 1, "F")])
        # backward starts after the last forward of the unit
        ops[op_index[(j, S - 1, "B")]].depends_on.append(op_index[(j, S - 1, "F")])
        for vs in range(S - 2, -1, -1):              # backward chain
            ops[op_index[(j, vs, "B")]].depends_on.append(op_index[(j, vs + 1, "B")])

    # 3) per-device schedule-order chain (encodes 1F1B order)
    sched = pipeline_schedule(m, p, v)
    for d in range(p):
        prev = None
        for (j, vs, phase) in sched[d]:
            cur = op_index[(j, vs, phase)]
            if prev is not None:
                ops[cur].depends_on.append(prev)
            prev = cur

    sim = simulate(ops)
    # busy time per device = sum of its op durations
    busy = [0.0] * p
    for op in ops:
        busy[int(op.stream[3:])] += op.duration
    step = sim.wall_clock_time
    total_capacity = p * step
    bubble = 1.0 - (sum(busy) / total_capacity) if total_capacity > 0 else 0.0
    return PipelineResult(
        step_time=step,
        bubble_ratio=max(0.0, bubble),
        peak_activation_bytes=sim.peak_activation_bytes,
        per_device_busy=busy,
    )
```

Note: `p2p_time` is accepted now for interface stability; it is wired into per-stage durations in Task 4 (kept 0.0 here so the anchor matches the closed form exactly).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pp_pipeline.py -k "anchor or bubble or pp1 or microbatches" -v`
Expected: PASS (4 tests). If `test_anchor_bubble_v2` exceeds tolerance, the round-robin interleave is the documented approximation — keep `abs=0.04`.

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/pp_pipeline.py tests/test_pp_pipeline.py
git commit -m "feat(pp_pipeline): outer 1F1B(+V) simulator; reproduces closed-form bubble"
```

---

## Task 4: Variable-length bubble, ordering sensitivity, memory ∝ p

**Files:**
- Test: `tests/test_pp_pipeline.py`
- Modify: `src/llm_perf/pp_pipeline.py` (add `peak_activation_bytes` semantics check only; no new API)

**Interfaces:**
- Consumes: `simulate_pipeline`, `PipelineResult` from Task 3.
- Produces: no new public symbol — validates variable-length behavior and the `unit_act_bytes`→memory path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pp_pipeline.py  (append)
def test_variable_length_bubble_differs_from_closed_form():
    # mix of fast (short) and slow (long) units → bubble != equal-length formula
    p, v = 4, 1
    times = []
    for j in range(8):
        t = 1.0 if j % 2 == 0 else 4.0   # alternating fast/slow units
        times.append([(t, 2.0 * t)] * (p * v))
    res = simulate_pipeline(times, [1.0] * 8, p, v=v)
    closed = (p - 1) / (8 + p - 1)
    assert res.bubble_ratio != pytest.approx(closed, abs=1e-6)
    assert 0.0 <= res.bubble_ratio < 1.0

def test_ordering_affects_bubble():
    # clustering all slow units last vs interleaving changes the bubble
    p, v = 4, 1
    def mk(order):
        return [[(t, 2.0 * t)] * (p * v) for t in order]
    clustered = mk([1, 1, 1, 1, 4, 4, 4, 4])
    interleaved = mk([1, 4, 1, 4, 1, 4, 1, 4])
    b_cluster = simulate_pipeline(clustered, [1.0] * 8, p, v=v).bubble_ratio
    b_inter = simulate_pipeline(interleaved, [1.0] * 8, p, v=v).bubble_ratio
    assert b_cluster != pytest.approx(b_inter, abs=1e-6)

def test_peak_memory_scales_with_p_not_m():
    # in-flight depth ≈ p, so doubling m must NOT double peak activation
    p, v = 4, 1
    r_small = simulate_pipeline(_equal_times(6, p, v), [1.0] * 6, p, v=v)
    r_big = simulate_pipeline(_equal_times(24, p, v), [1.0] * 24, p, v=v)
    assert r_big.peak_activation_bytes == pytest.approx(r_small.peak_activation_bytes, rel=0.5)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_pp_pipeline.py -k "variable or ordering or peak_memory" -v`
Expected: PASS if Task 3 is correct (these assert behavior already produced). If `test_peak_memory_scales_with_p_not_m` FAILS (peak grows with m), fix the activation free timing in `simulate()` usage: ensure each `F` op's `output_bytes` is freed when its matching `B` completes — already encoded via `depends_on`, so verify the `B(j,vs)` ops depend on the corresponding `F(j,vs)` so ref-counting frees activation. Add that dependency:

```python
# src/llm_perf/pp_pipeline.py — inside simulate_pipeline, section (2), after backward chain
        for vs in range(S):                          # free fwd activation at its bwd
            ops[op_index[(j, vs, "B")]].depends_on.append(op_index[(j, vs, "F")])
```

- [ ] **Step 3: Re-run to verify pass**

Run: `pytest tests/test_pp_pipeline.py -v`
Expected: PASS (all pp_pipeline tests)

- [ ] **Step 4: Commit**

```bash
git add src/llm_perf/pp_pipeline.py tests/test_pp_pipeline.py
git commit -m "test(pp_pipeline): variable-length bubble, ordering sensitivity, memory∝p"
```

---

## Task 5: Solver — pack into bins + memory-aware per-bin CP

**Files:**
- Modify: `src/llm_perf/dynamic_cp.py`
- Test: `tests/test_dynamic_cp.py`

**Interfaces:**
- Consumes: `assign_cp`, `_sample_sim` (existing), `PoolUnit` (from `pp_pipeline`).
- Produces:
  - `assign_bin_cp(model_cfg, hw, base_par, wl, seq_len, quota, max_cp, usable_hbm_gb) -> int` — `clamp(max(cp_workload, cp_memory), 1, max_cp)`; doubles cp until per-rank `weight+activation` mem ≤ `usable_hbm_gb` (memory repair). Returns `max_cp` even if still OOM (caller flags feasibility).
  - `pack_units(buckets, total_ranks, token_budget, cp_of, packing_eff_of) -> list[PoolUnit]` — homogeneous pool-wide units per bin; `n_b = bin_tokens/(R·B·η_b)` (≥1 per non-empty bin); `seq_len = L_bin`, `packed_tokens = R·token_budget`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_cp.py  (append imports + tests)
from llm_perf.dynamic_cp import assign_bin_cp, pack_units
from llm_perf.pp_pipeline import PoolUnit

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
    buckets = [(4096.0, 0.5), (32768.0, 0.5)]
    R, B = 8, 4096
    cp_of = lambda L: 1 if L <= 4096 else 8
    units = pack_units(buckets, R, B, cp_of, packing_eff_of=lambda L: 1.0)
    assert all(isinstance(u, PoolUnit) for u in units)
    # each bin contributes >=1 unit; units in a bin share cp + seq_len
    cps = {u.seq_len: u.cp for u in units}
    assert cps[4096.0] == 1 and cps[32768.0] == 8
    assert all(u.packed_tokens == R * B for u in units)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_cp.py -k "assign_bin_cp or pack_units" -v`
Expected: FAIL with `ImportError: cannot import name 'assign_bin_cp'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/dynamic_cp.py  (add import near top)
from llm_perf.pp_pipeline import PoolUnit

# src/llm_perf/dynamic_cp.py  (append functions)
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


def pack_units(buckets, total_ranks, token_budget, cp_of, packing_eff_of):
    """Pack a length distribution into homogeneous pool-wide units.

    Each non-empty bin yields ceil(bin_tokens/(R·B·η)) units (≥1), all sharing
    the bin's cp and representative seq_len. Fractions are renormalized.
    """
    import math
    total_frac = sum(f for _, f in buckets) or 1.0
    R, B = total_ranks, token_budget
    units = []
    for bi, (length, frac) in enumerate(buckets):
        w = frac / total_frac
        bin_tokens = w * R * B  # tokens of this bin per pool-wide batch slot scale
        eta = packing_eff_of(length)
        n_b = max(1, math.ceil(bin_tokens / (R * B * eta))) if bin_tokens > 0 else 0
        cp_b = cp_of(length)
        for _ in range(n_b):
            units.append(PoolUnit(cp=int(cp_b), seq_len=int(length),
                                  packed_tokens=int(R * B), bin_index=bi))
    return units
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynamic_cp.py -k "assign_bin_cp or pack_units" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/dynamic_cp.py tests/test_dynamic_cp.py
git commit -m "feat(dynamic_cp): solver pack + memory-aware per-bin CP"
```

---

## Task 6: Solver — assemble units, order, and drive the simulator

**Files:**
- Modify: `src/llm_perf/dynamic_cp.py`
- Test: `tests/test_dynamic_cp.py`

**Interfaces:**
- Consumes: `pack_units`, `assign_bin_cp`, `stage_unit_time`, `simulate_pipeline`, `_split_stages`.
- Produces:
  - `order_units(units, order='balanced') -> list[PoolUnit]` — `balanced` spreads slow (large `cp`/`seq_len`) units evenly; `as_packed` keeps order; `descending` slow-first.
  - `run_pipeline(model_cfg, hw, base_par, wl, units, p, v, bwd_factor=2.0) -> PipelineResult` — builds `unit_stage_times` via cached `stage_unit_time` over `_split_stages(layers, p*v)` chunks, `unit_act_bytes` from the inner sim peak, then calls `simulate_pipeline`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_cp.py  (append)
from llm_perf.dynamic_cp import order_units, run_pipeline

def test_order_units_balanced_spreads_slow():
    units = [PoolUnit(cp=1, seq_len=4096, packed_tokens=1, bin_index=0) for _ in range(6)]
    units += [PoolUnit(cp=8, seq_len=32768, packed_tokens=1, bin_index=1) for _ in range(2)]
    out = order_units(units, order="balanced")
    # the two slow (cp=8) units should not be adjacent at the very end
    slow_positions = [i for i, u in enumerate(out) if u.cp == 8]
    assert slow_positions[1] - slow_positions[0] >= 2

def test_run_pipeline_smoke(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=8)
    wl = WorkloadConfig(group_size=1)
    units = [PoolUnit(cp=1, seq_len=4096, packed_tokens=8 * 4096, bin_index=0) for _ in range(4)]
    units += [PoolUnit(cp=8, seq_len=32768, packed_tokens=8 * 4096, bin_index=1) for _ in range(2)]
    res = run_pipeline(mc, hw, par, wl, units, p=8, v=1)
    assert res.step_time > 0
    assert 0.0 <= res.bubble_ratio < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_cp.py -k "order_units or run_pipeline" -v`
Expected: FAIL with `ImportError: cannot import name 'order_units'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/dynamic_cp.py  (add imports)
from llm_perf.builder import _split_stages
from llm_perf.pp_pipeline import run_pipeline as _unused  # noqa: F401  (placeholder removed below)
```

Then implement (do NOT import `run_pipeline` from pp_pipeline — define it here):

```python
# src/llm_perf/dynamic_cp.py  (append; remove the placeholder import line above)
from llm_perf.pp_pipeline import stage_unit_time, simulate_pipeline


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynamic_cp.py -k "order_units or run_pipeline" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/dynamic_cp.py tests/test_dynamic_cp.py
git commit -m "feat(dynamic_cp): unit ordering + run_pipeline driver"
```

---

## Task 7: Rewire compare_cp_strategies through solver + pipeline

**Files:**
- Modify: `src/llm_perf/dynamic_cp.py`
- Test: `tests/test_dynamic_cp.py`

**Interfaces:**
- Consumes: `pack_units`, `assign_bin_cp`, `order_units`, `run_pipeline`, `packing_efficiency`.
- Produces: rewritten `compare_cp_strategies(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets, total_ranks, quota=None, token_budget=None, num_micro_batches=None, packing_eff=None, pp=None, v=1, bwd_factor=2.0, order='balanced') -> dict` with keys: `max_cp, quota, token_budget, total_ranks, p, v, static, dynamic, speedup, tflops_ratio`. Each of `static`/`dynamic` is a dict: `{m, step_s, bubble_ratio, mfu, tflops_per_gpu, peak_mem_gb, feasible, imbalance, rank_seconds_per_sample, units:[{cp,seq_len}...]}`.

For each recipe: choose `cp_of` (static → `max_cp`; dynamic → `assign_bin_cp`), `pack_units`, `order_units`, `run_pipeline`. Compute `mfu` via the cp=1 irreducible-compute baseline (reuse `_sample_sim(...,1).compute_time` summed over units, divided by `p × step × peak_tflops`-equivalent), keeping the existing convention. `speedup = static.step_s / dynamic.step_s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dynamic_cp.py  (replace the old compare tests with these)
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
```

Delete the now-obsolete `test_compare_speedup_variable_length`, `test_compare_no_gain_uniform_length`, `test_pp_bubble_shrinks_with_more_micro_batches`, `test_packing_inflates_step_time` (they assert the old analytic API). Keep `test_assign_cp_*`, `test_lognormal_*`, `test_packing_efficiency_*`, and the Task-5/6 tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dynamic_cp.py -k compare_pipeline -v`
Expected: FAIL (old `compare_cp_strategies` signature lacks `pp`/`v` and returns old keys)

- [ ] **Step 3: Write minimal implementation** — replace `compare_cp_strategies` and its helper `_strategy_cost`:

```python
# src/llm_perf/dynamic_cp.py  (replace compare_cp_strategies; delete _strategy_cost)
def _recipe(model_cfg, hw, parallel_cfg, rl_cfg, buckets, total_ranks, quota,
            token_budget, max_cp, p, v, bwd_factor, order, cp_of):
    usable = hw.usable_hbm_gb
    eta_of = lambda L: packing_efficiency([(L, 1.0)], token_budget)
    units = pack_units(buckets, total_ranks, token_budget, cp_of, eta_of)
    units = order_units(units, order=order)
    res = run_pipeline(model_cfg, hw, parallel_cfg, rl_cfg, units, p, v, bwd_factor)
    # MFU vs irreducible cp=1 compute (existing convention)
    useful = sum(_sample_sim(model_cfg, hw, parallel_cfg, rl_cfg, u.seq_len, 1).compute_time
                 for u in units)
    rank_seconds = sum(u.cp * _sample_sim(model_cfg, hw, parallel_cfg, rl_cfg,
                                          u.seq_len, u.cp).wall_clock_time for u in units)
    compute_eff = hw.calibration.compute_eff_large_gemm
    denom = p * res.step_time
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


def compare_cp_strategies(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets,
                          total_ranks, quota=None, token_budget=None,
                          num_micro_batches=None, packing_eff=None, pp=None,
                          v=1, bwd_factor=2.0, order="balanced") -> dict:
    """Static-CP vs dynamic-CP, each through the variable-length 1F1B(+V) pipeline."""
    max_cp = max(1, int(parallel_cfg.cp))
    max_len = max(length for length, _ in seq_buckets)
    if quota is None:
        quota = max_len / max_cp
    if token_budget is None:
        token_budget = max_len
    p = int(pp) if pp is not None else int(parallel_cfg.pp)
    usable = hw.usable_hbm_gb

    static_cp_of = lambda L: max_cp
    dynamic_cp_of = lambda L: assign_bin_cp(model_cfg, hw, parallel_cfg, rl_cfg,
                                            L, quota, max_cp, usable)

    static = _recipe(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets, total_ranks,
                     quota, token_budget, max_cp, p, v, bwd_factor, order, static_cp_of)
    dynamic = _recipe(model_cfg, hw, parallel_cfg, rl_cfg, seq_buckets, total_ranks,
                      quota, token_budget, max_cp, p, v, bwd_factor, order, dynamic_cp_of)

    speedup = static["step_s"] / dynamic["step_s"] if dynamic["step_s"] > 0 else 0.0
    tflops_ratio = (dynamic["tflops_per_gpu"] / static["tflops_per_gpu"]
                    if static["tflops_per_gpu"] > 0 else 0.0)
    return {
        "max_cp": max_cp, "quota": quota, "token_budget": token_budget,
        "total_ranks": total_ranks, "p": p, "v": v,
        "static": static, "dynamic": dynamic,
        "speedup": speedup, "tflops_ratio": tflops_ratio,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dynamic_cp.py -v`
Expected: PASS (all dynamic_cp tests, old compare tests removed)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/dynamic_cp.py tests/test_dynamic_cp.py
git commit -m "feat(dynamic_cp): compare via solver + variable-length pipeline simulator"
```

---

## Task 8: Demo update + full regression + lint

**Files:**
- Modify: `examples/demo_dynamic_cp.py`
- Verify: whole suite + ruff

**Interfaces:**
- Consumes: new `compare_cp_strategies(..., pp=, v=)` return shape.

- [ ] **Step 1: Update the demo to the Llama 128-GPU scenario and new keys**

```python
# examples/demo_dynamic_cp.py — replace defaults + the printing block
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--cp", type=int, default=8, help="max CP degree available")
    p.add_argument("--pp", type=int, default=8, help="pipeline depth")
    p.add_argument("--vstages", type=int, default=1, help="virtual stages V")
    p.add_argument("--total-ranks", type=int, default=8, help="CP/DP pool size R")
    # ... (avg/std/max-len/buckets as before) ...
```

```python
# examples/demo_dynamic_cp.py — replace the analysis call + output
    par = ParallelismConfig(tp=args.tp, cp=args.cp, dp=1, pp=args.pp)
    wl = WorkloadConfig(group_size=1)
    buckets = lognormal_buckets(args.avg, args.std, args.max_len, args.buckets)
    r = compare_cp_strategies(mc, hw, par, wl, buckets, total_ranks=args.total_ranks,
                              pp=args.pp, v=args.vstages)

    print(f"Model: {mc.name} | tp={args.tp} pp={args.pp} V={args.vstages} "
          f"| max CP={r['max_cp']} | pool R={r['total_ranks']}")
    print("=" * 80)
    def show(name, s):
        print(f"  {name:<8} step={s['step_s']*1000:>8.2f}ms  m={s['m']:>3}  "
              f"bubble={s['bubble_ratio']*100:>4.1f}%  MFU={s['mfu']*100:>4.1f}%  "
              f"TFLOPS/GPU={s['tflops_per_gpu']:>6.1f}  peakmem={s['peak_mem_gb']:>5.1f}GB "
              f"{'OK' if s['feasible'] else 'OOM'}  imbal={s['imbalance']:.2f}")
    show("Static", r["static"])
    show("Dynamic", r["dynamic"])
    print("-" * 80)
    print(f"  Speedup (static/dynamic step): {r['speedup']:.2f}x   "
          f"TFLOPS/GPU ratio: {r['tflops_ratio']:.2f}x")
```

Remove all references to the deleted `length_routed_dp_analysis`, `dynamic_cp_analysis`, and the old per-bucket cost table loop.

- [ ] **Step 2: Run the demo**

Run: `python examples/demo_dynamic_cp.py`
Expected: prints a Static/Dynamic table with `speedup > 1`, both `OK`, sane MFU.

- [ ] **Step 3: Full regression + lint**

Run: `python -m pytest tests/ -q`
Expected: all green (193 prior + new pp_pipeline/dynamic_cp tests).

Run: `ruff check src/ tests/ examples/ && ruff format --check src/llm_perf/pp_pipeline.py src/llm_perf/dynamic_cp.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add examples/demo_dynamic_cp.py
git commit -m "feat(demo): Llama 128-GPU tp2·pp8 dynamic-CP pipeline comparison"
```

---

## Self-Review

**Spec coverage:**
- A (pool-wide abstraction, two-level simulate) → Tasks 1,3,6. ✓
- B (solver: pack / per-bin memory-aware CP / determine m) → Tasks 5,6. Note: `m` is currently fixed by `pack_units` (one budget); the B-grid search for `m` is reduced to the default budget = `max_len`. **Gap:** the spec's step-3 B-grid search is simplified to a single budget. Acceptable for first cut; documented here as a follow-up (vary `token_budget`, pick min feasible step) — NOT implemented to keep the plan bounded.
- C (1F1B+V simulator, comm overlap, bwd split, pipeline memory) → Tasks 1,2,3,4. ✓
- D (metrics + ratios) → Task 7. Three-source decomposition reduced to keeping `rank_seconds_per_sample`; full multiplicative reconstruction is a diagnostic, not asserted. ✓ (partial, noted)
- E (tests: anchor, variable, overlap, memory, regression) → Tasks 3,4,7,8. ✓

**Placeholder scan:** Task 6 Step 3 had a placeholder import line; it is explicitly removed in the same step. No TBD/TODO remain in code steps.

**Type consistency:** `PoolUnit(cp, seq_len, packed_tokens, bin_index)` used consistently (Tasks 1,5,6). `stage_unit_time(...chunk_id...,cache)` signature matches its call in `run_pipeline`. `simulate_pipeline(unit_stage_times, unit_act_bytes, p, v, p2p_time)` matches caller. `compare_cp_strategies(..., pp, v)` matches demo + tests.

**Known simplifications (intentional, bounded scope):**
1. Step-3 `m` search reduced to a single `token_budget=max_len` (no B-grid). Follow-up.
2. P2P time defaults to 0.0 in `simulate_pipeline` (interface present, not yet fed). Follow-up.
3. Interleaved (V>1) uses the round-robin flattening approximation (anchor tolerance 4%).
