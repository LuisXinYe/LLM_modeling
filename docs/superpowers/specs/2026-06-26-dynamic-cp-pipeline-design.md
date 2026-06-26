# Design: Dynamic-CP vs Static-CP with packing, solver & variable-length PP-bubble simulation

Date: 2026-06-26
Status: Approved (brainstorming) → ready for implementation plan

## Goal

Make the dynamic-CP vs static-CP comparison reflect achievable numbers by adding
a faithful variable-length pipeline model. Two recipes are compared on the **same
global batch**:

- **Static CP + packing + PP bubble** — every sequence sharded by `max_cp`.
- **Dynamic CP + packing + PP bubble** — per-bin CP chosen by a 3-step solver
  (pack → per-bin CP → determine `m`) iterating against a cost model until load
  is balanced and memory is feasible.

Both run through the same variable-length pipeline simulator with CP-ring
comm/compute overlap. The length-routed-DP scheme is removed (out of scope).

Reference scenario: **Llama-3.1-8B on 128 GPUs**, `tp=2 × pp=8 × (dp·cp pool R=8)`,
`max_cp=8`, interleaved 1F1B with `V≥1` virtual stages.

## Non-goals

- No ILP / exact bin-packing solver (heuristic one-pass + memory repair only).
- No zero-bubble schedule (textbook 1F1B+V only).
- No changes to the shared `builder`/inference/training/post-training paths — the
  comm-overlap model lives only in the new pipeline path (isolation, no regression).
- No static-CP sweep for now (single static = `max_cp`).

---

## Section A — Core abstraction: the "pool-wide microbatch"

Each PP stage owns `tp×(dp·cp) = 2×8 = 16` GPUs; 8 stages = 128 GPUs. Under
dynamic CP the intra-stage width varies (a `cp=1` microbatch occupies 1 of the 8
pool ranks so 8 can run `dp=8`; a `cp=8` microbatch fills all 8, `dp=1`).
Simulating exact (stage × rank) packing is a 2-D scheduling problem.

**Abstraction:** a *pipeline microbatch* = one work unit that occupies the **whole
R=8 pool**. Its composition is the solver's output:

- **Short-sequence unit:** pool runs `dp=8 × cp=1`, packs `R·B` tokens of short
  sequences; per-stage time = per-rank time for `B` tokens at `cp=1` (fast, wide).
- **Long-sequence unit:** pool runs `dp=1 × cp=8`, one long sequence; per-stage
  time = per-rank time at `cp=8` (slow, narrow).

Every pool-wide unit carries `R·B` tokens regardless of `cp`. Because each unit
fills the pool, units **serialize within a stage** → one compute stream per stage
suffices and `simulate()` models the pipeline directly.

**Honest framing:** with total tokens fixed, `m` (unit count) is roughly common
to both recipes — it is the solver's step-3 knob, not the main static-vs-dynamic
differentiator. Dynamic CP's gain comes from (1) faster short units (cp=1 avoids
ring comm), (2) better bubble shape (more uniform unit times), and possibly
(3) packing efficiency η. We do **not** claim "fewer units".

**Two-level `simulate()` reuse:**
- Inner: per `(unit, stage)` call `build_forward_pass`/`simulate()` → scalar
  compute/cp_comm/tp_comm times.
- Outer: per `(unit, stage, phase)` use those scalars as `SimOp.duration`, encode
  1F1B+V as dependency edges, run `simulate()` again → real variable-length step
  time and bubble.

**Modules:**
- New `src/llm_perf/pp_pipeline.py` — pool-unit stage timing, 1F1B+V scheduler,
  outer simulate, pipeline metrics.
- `src/llm_perf/dynamic_cp.py` — add the 3-step solver; `compare_cp_strategies`
  drives static & dynamic through the pipeline simulator. Reuse `_sample_sim`.

---

## Section B — Dynamic-CP solver (heuristic one-pass + memory repair)

Inputs: global batch (buckets × `global_batch_tokens`), pool `R=8`, `p=8`, `V`,
`max_cp=8`, per-rank token budget `B`, usable HBM.

**Step 1 — length-aware pack.** Group sequences into homogeneous length bins
(reuse `lognormal_buckets`). Each bin forms pool-wide units at its CP using
`dp=R/cp`, each unit carrying `R·B` tokens (`B` per rank). Units per bin
`n_b = bin_tokens / (R·B·η_b)`. Homogeneous bins → uniform unit time → tight bubble.

**Step 2 — per-bin CP (memory-aware):**
```
cp_workload = assign_cp(L_bin, quota≈B, max_cp)              # per-rank seq ≤ budget
cp_memory   = min cp s.t. inner-sim peak_mem(L_bin/cp) ≤ usable_HBM
cp_bin      = clamp(max(cp_workload, cp_memory), 1, max_cp)
```
- Static: `cp_bin ≡ max_cp` for all bins (`dp=1`).
- Dynamic: per-bin as above.
- Memory repair (the bounded solver↔cost-model loop): if a unit OOMs at the
  chosen cp, `cp_bin *= 2` and repack; if it hits `max_cp` and still OOMs, mark
  `feasible=False`.

**Step 3 — determine `m`.** All bins' units concatenate into the microbatch
sequence, `m = Σ n_b`. Tune `B` over a small grid (larger `B` → bigger units,
higher GEMM efficiency, smaller `m`, bigger bubble; smaller `B` → opposite). For
each candidate `B`, rebuild units → run the outer pipeline simulator → pick the
`B` (hence `m`) minimizing step time subject to memory feasibility. This realizes
the `W·(mV+p−1)` trade-off via measured cost instead of the closed form.

**Unit ordering (1F1B inflow, affects bubble):** default `balanced interleave`
(spread slow/long units to avoid warmup-bubble clustering); knob `order ∈
{balanced, descending, as-packed}`.

**Load balance (two levels):** (a) inter-unit — guaranteed by homogeneous-bin
packing; (b) inter-stage — `_split_stages` handles split, simulator absorbs
residual imbalance for mixed-layer models. Report `imbalance = max_stage/mean_stage`.

**Output to simulator:** `List[PoolUnit]`, each `{cp, packed_tokens,
per_stage→(fwd_t, bwd_t)}`; one list for static, one for dynamic.

---

## Section C — Variable-length pipeline simulator (two-level simulate)

**Inner — per `(unit, virtual stage)` fwd/bwd scalars.** Virtual stages = `p×V`;
`_split_stages(all_layers, p×V)` → `p×V` chunks; device `d` holds chunks
`d, d+p, d+2p, …` (interleaved). For each chunk × unit:
```
# unit sequence length L: long unit L = cp·B (sharded to B/rank); short unit L = B, cp=1.
# Pass the FULL L — build_forward_pass shards internally via local_seq = L // cp.
sim   = simulate(build_forward_pass(model, hw, par(cp=unit.cp, pp=1),
                                    wl(seq_len=L), stage_layers=chunk))
fwd_t = max(sim.compute_time, sim.cp_comm_time) + sim.tp_comm_time   # CP-ring overlap
bwd_t = bwd_factor × fwd_t                                           # default 2.0, calibratable
```
No separate backward builder, so backward uses `bwd_factor×fwd` (backward FLOPs
≈ 2× forward; comm scales similarly). **Cache** key `(chunk signature, cp,
tokens)` collapses the heavy work from `O(m×pV)` to `O(bins × pV × cp_candidates)`.

**Outer — schedule as a meta-op DAG, run `simulate()` again:**
- Streams: one compute stream `dev{d}` per physical device (`d=0..p−1`); the V
  chunks on a device serialize naturally. P2P on a separate `p2p_comm` stream
  (overlaps compute), duration from `_compute_pp_p2p_time` logic (activation bytes
  ∝ unit per-rank tokens × hidden).
- Meta-op per `(unit j, vstage v, phase∈{F,B})`:
  `SimOp(stream=dev{v%p}, duration=fwd_t/bwd_t)`.
- Two dependency classes:
  1. Data flow: `F(j,v)` ← `F(j,v−1)`; `B(j,v)` ← `B(j,v+1)` and `F(j, last)`.
  2. Schedule order (1F1B+V): on each `dev{d}` stream, op `k` ← op `k−1` in the
     textbook warmup→steady→cooldown order. This pins the exact schedule.

`simulate()` topo-sorts + per-stream clocks → `wall_clock_time` = real
variable-length step time; bubble emerges from per-device stream gaps (no closed
form).

**Pipeline-level memory (free 1F1B-vs-GPipe difference):** set each F meta-op's
`output_bytes = unit activation footprint`, freed when its `B(j,v)` completes →
outer `peak_activation_bytes` = in-flight units × activation (1F1B ≈ p, GPipe ≈ m).
Compare vs usable HBM → pipeline feasibility.

**Assumptions (explicit):** ① `bwd = bwd_factor×fwd`; ② P2P via existing
approximation; ③ textbook 1F1B+V order (not zero-bubble); ④ overlap applies to
CP-ring only, TP comm not overlapped.

---

## Section D — Comparison metrics & output

`compare_cp_strategies` runs `solver → pp_pipeline_simulate` for **static** and
**dynamic**, returning per-recipe metrics + top-level ratios.

Per recipe:

| field | meaning / convention |
|---|---|
| `step_time_s` | outer `wall_clock`; real step time for one global batch (incl. variable-length bubble) |
| `m` | pool-wide unit count (solver step-3) |
| `bubble_ratio` | `1 − Σ_device useful_busy / (p × wall)` — measured, not closed form |
| `mfu` | `compute_eff × Σ useful_compute(cp=1 baseline) / (p × wall)` — irreducible-FLOPs convention |
| `tflops_per_gpu` | `peak_tflops × mfu` |
| `peak_pipeline_mem_gb` | outer in-flight activation peak (1F1B ≈ p units) + weights/optimizer |
| `feasible` | `peak_pipeline_mem ≤ usable_HBM` and solver did not OOM at `max_cp` |
| `imbalance` | `max_stage / mean_stage` |
| `units` | per-unit `{cp, tokens, fwd_t, bwd_t}` detail |

Top-level:
- `speedup = static.step_time / dynamic.step_time` — main metric; now includes
  real variable-length bubble and comm-overlap differences (no longer equal to the
  pure rank-seconds ratio).
- `tflops_ratio = dynamic.tflops_per_gpu / static.tflops_per_gpu`.
- `bubble_delta`, `mem_delta` — dynamic's bubble/memory gains and costs made
  visible (dynamic short units at cp=1 are denser but hold larger activation, so
  memory may rise — this must be visible).

**Three-source decomposition (honesty):** keep the legacy `rank_seconds`
component in the output so users see how much of `speedup` is pure CP assignment
vs the bubble/overlap corrections.

**Demo:** `demo_dynamic_cp.py` defaults to Llama / 128 GPUs / `tp2·pp8·R8` /
`max_cp8`, printing the per-recipe table + unit composition + three-source split.

---

## Section E — Testing strategy

**Anchor principle:** the simulator must reproduce the closed-form bubble in the
equal-length degenerate case before anything else; variable-length then departs
from it.

**1. Solver (`tests/test_dynamic_cp.py` extended):**
- per-bin CP = `max(cp_workload, cp_memory)`: long seq → memory-driven, short seq
  → workload-driven, each verified.
- memory repair: shrink usable HBM → cp forced to double until feasible; hits
  `max_cp` still OOM → `feasible=False`.
- static: all bins `cp≡max_cp`, `dp=1`.
- homogeneous packing: units in one bin share `fwd_t`.

**2. Pipeline simulator (new `tests/test_pp_pipeline.py`):**
- **Equal-length anchor:** m equal units, `pp=p`, `V=1` → `bubble_ratio ≈
  (p−1)/(m+p−1)` (~1% tol); `V>1` → `≈ (p−1)/(mV+p−1)`.
- `m↑` → bubble↓; `pp=1` → bubble=0 and `step = Σ unit times`.
- **Variable-length:** mixed long/short units → bubble ≠ closed form, and is
  order-sensitive (`balanced` < `as-packed`).
- **comm overlap:** same units, `max(compute,cp_comm)` vs `compute+cp_comm` →
  overlap step smaller; cp=1 units equal (no ring).
- **memory:** `peak_pipeline_mem` scales with `p` (in-flight depth), not `m`.

**3. Integration (`compare_cp_strategies`):**
- variable dist → `speedup>1`, `dynamic.tflops_per_gpu > static`; uniform max-len
  → `speedup≈1`.
- three-source decomposition self-consistent: `rank_seconds × bubble × overlap`
  corrections reconstruct measured step within tolerance.
- Llama/128/`tp2·pp8` end-to-end runs; numbers sane (reasonable MFU, feasible).

**4. Regression:**
- existing **193 tests stay green** (comm overlap only in the new path).
- `ruff check` clean.
