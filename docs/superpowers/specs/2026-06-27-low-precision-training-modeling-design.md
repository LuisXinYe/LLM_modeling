# Design: Low-precision training modeling (FP8/FP4 with fine-grained quant + stochastic Hadamard)

Date: 2026-06-27
Status: Approved (brainstorming) → ready for implementation plan

## Goal

Give llm-perf the ability to model **low-precision training recipes** and predict
their impact on the resource profile — primarily to support two decisions:

1. **Compare precision recipes** (BF16 vs FP8 vs FP4-fine-grained+Hadamard …)
   end-to-end: step/epoch time, exposed communication, memory, feasibility.
2. **Network / interconnect sizing**: quantify how low-precision changes
   communication volume and the **compute–communication overlap margin**, so we
   can tell whether reduced comm stays hidden under backward or becomes exposed,
   and infer the interconnect bandwidth/topology actually required.

The model is a **performance** model: it captures only what changes FLOPs, bytes,
or communication volume. Numerical accuracy / convergence is explicitly out of
scope.

## Scope decisions (from brainstorming)

- **Granularity:** precision is specified **per tensor-role** (weights /
  activations / gradients / comm), each with a dtype + block size + optional
  Hadamard flag. Not per-GEMM-path (TE-style) and not a fixed recipe enum.
- **Overhead modeling:** **explicit overhead ops** (scale-compute, stochastic
  Hadamard, dequant, compensation add) enter the DAG — not folded into a single
  calibration coefficient. This is required so we can show whether overhead eats
  the low-precision speedup and whether it crowds out the DP-overlap margin.
- **Performance footprints in scope (all four selected):** quantization overhead
  ops; high-precision compensation buffers (error feedback / residual); mixed &
  periodic high-precision steps; fp32 master weights + optimizer states.
- **Overlap fix bundled in:** the simulator's missing **fabric-level bandwidth
  contention** is fixed here (same root cause as the low-precision→comm→overlap
  causal chain, so done once).

## Non-goals

- No numerical accuracy / convergence / loss-scaling-stability modeling.
- No per-GEMM-path precision config (per-role only).
- No new collective algorithms; reuse existing ring/tree/alltoall comm model.
- No SM/compute slowdown from overlapped comm in v1 — left as an **optional,
  default-off coefficient hook** (see Section E).
- No change to the generation/decode precision path beyond reusing the same
  per-role bytes (generation is weight+KV-cache bound; training is the focus).

---

## Current-state findings (baseline this design corrects)

- `ModelConfig.dtype` is a **single model-wide dtype**; `dtype_bytes` is used
  uniformly for weight, activation, and comm bytes. No per-role precision.
- `roofline_time()` always uses `hw.peak_tflops_bf16` → low-precision compute
  speedup is not modeled.
- No quantization overhead ops exist (scale, Hadamard, dequant).
- **DP overlap already works.** `build_training_step` (builder.py:905-959) buckets
  the DP grad sync per layer on the `dp_comm` stream, depending only on that
  layer's backward, so later-layer backward overlaps it — matching DDP/Megatron.
  The `docs/architecture.md` "No overlapping communication" note is **stale**.
- **Real gap:** the simulator gives each stream its own clock with **full
  bandwidth**, so two collectives that overlap (e.g. inter-node DP grad sync +
  inter-node EP all-to-all, or intra-node TP + intra-node CP) each get full
  bandwidth — physically impossible on a shared fabric. NVLink (intra) and NIC
  (inter) are genuinely separate, so they should *not* contend; same-fabric
  concurrency *should*. Current model ignores both, biasing exposed comm
  optimistically — exactly the quantity network sizing depends on.

---

## Section A — Config: `PrecisionConfig`

A new pydantic model carried on the training/workload path (not on `ModelConfig`,
so the same model can be run under multiple recipes).

```python
class TensorPrecision(BaseModel):
    dtype: str = "bf16"          # bf16|fp16|fp32|fp8_e4m3|fp8_e5m2|fp4_e2m1|mxfp4
    block_size: int = 0          # 0 = per-tensor scale; >0 = fine-grained block
    hadamard: bool = False       # stochastic Hadamard transform before quant
    hadamard_block: int = 0      # Hadamard rotation size (0 → default, e.g. 128)

class PrecisionConfig(BaseModel):
    weights: TensorPrecision = TensorPrecision()
    activations: TensorPrecision = TensorPrecision()
    gradients: TensorPrecision = TensorPrecision()
    comm: TensorPrecision = TensorPrecision()        # gradients/activations on the wire
    master_dtype: str = "fp32"                        # master weights + optimizer states
    error_feedback: bool = False                      # residual/error-feedback buffers
    ef_dtype: str = "fp16"
    high_precision_layers: list[str] = []             # e.g. ["attention"] kept high
    high_precision_period: int = 0                    # every N steps a high-prec step; 0=off
    high_precision_dtype: str = "bf16"                # the dtype used for high-prec layers/steps
```

**Byte sizing.** Extend a single `dtype_bytes(dtype)` helper:
`fp32=4, bf16/fp16=2, fp8_*=1, fp4_*/mxfp4=0.5`. For a quantized tensor with
`block_size=B`, scale metadata adds `numel/B * scale_bytes` (scale_bytes default
4 for fp32 scales; mxfp4 uses 1-byte e8m0 → expose as a field). A per-tensor
scale (`B=0`) adds a negligible constant.

**Backward compat.** When no `PrecisionConfig` is supplied, every role defaults to
`ModelConfig.dtype` and behavior is identical to today (regression-free).

---

## Section B — Hardware: per-precision peak TFLOPS

```python
class HardwareConfig(BaseModel):
    peak_tflops: dict[str, float] = {}   # {"bf16":…, "fp8":…, "fp4":…}
    peak_tflops_bf16: float              # kept; mirrored into peak_tflops["bf16"]
```

`peak_tflops` is keyed by a **compute class** (`bf16`/`fp8`/`fp4`), not the full
dtype string — e4m3 and e5m2 share the fp8 pipe. A resolver maps
`dtype → compute_class`. If a class is missing, fall back to `bf16` peak with a
warning (conservative: no speedup credited).

`roofline_time(cost, hw, is_large_gemm, compute_class="bf16")` selects the peak by
`compute_class`. Existing callers keep `"bf16"` default → unchanged.

---

## Section C — ops.py: precision-aware costs + overhead ops

**Precision-aware existing ops.** Compute ops gain optional per-role dtype args
(default 2 bytes → today's behavior). For a GEMM:
- `weight_bytes` uses `weights.dtype`; activation `output_bytes` uses
  `activations.dtype`; `mem_rw` mixes per operand.
- the matmul `compute_class` is passed to `roofline_time` → low-precision speedup.
- block-scale metadata bytes added to `weight_bytes`/`output_bytes` as applicable.

**New overhead ops** (each returns an `OpCost`, costed via `roofline_time` with
`is_large_gemm=False` unless noted):

| Op | Models | Cost shape |
|----|--------|-----------|
| `op_quantize(numel, in_dtype, out_dtype, block_size)` | scale reduction + cast | memory-bound: read `numel*in_bytes` + write `numel*out_bytes` + scale writes; tiny FLOPs |
| `op_hadamard(tokens, d, block)` | stochastic Hadamard (FWHT) | `flops ≈ tokens*d*log2(block)`; mem `tokens*d*(in+out)` — small GEMM tier |
| `op_dequant(numel, in_dtype, out_dtype, block_size)` | upcast + scale apply | memory-bound, mirror of quantize |
| `op_compensation_add(numel, dtype)` | error-feedback residual add | memory-bound element-wise over the EF buffer |

All FLOP/byte formulas carry a docstring citing the source (repo convention,
CLAUDE.md).

---

## Section D — builder.py: inject overhead + tag precision

**Around each low-precision GEMM** (when the operand dtype is below the compute
accumulation precision): inject, on the `compute` stream in dependency order:
`op_quantize(activation)` → `op_hadamard` (if `activations.hadamard`) →
low-precision matmul (`compute_class` from operands) → `op_dequant(output)` to the
accumulation dtype. Weight quantization is amortized **once per step** (weights
are static within a step), modeled as a single pre-step `op_quantize` over the
sharded weights, not per-GEMM.

**Around communication:** if `comm.dtype` is below the producing tensor's dtype,
inject `op_quantize` before the collective and `op_dequant` after — on the
`compute` stream (real cost of fp8 gradient all-reduce). The collective's
`comm_bytes` is recomputed with `comm.dtype` → reduced volume.

**Compensation buffers:** when `error_feedback=True`, add a persistent buffer
(`numel * ef_bytes`, weight-like, counted in memory) per quantized
weight/gradient tensor, plus one `op_compensation_add` per affected tensor per
step.

**Mixed precision (`high_precision_layers`):** per-layer precision resolution —
listed components (e.g. `"attention"`, `"router"`) use `high_precision_dtype`
instead of the low-precision roles; their GEMMs skip quant/Hadamard/dequant.

**Periodic high-precision steps (`high_precision_period=N`):** modeled at the
pipeline level (Section F) as a weighted blend of a low-prec step and a
high-prec step — **not** by building both DAGs every call.

**Fabric tag:** every comm SimOp gets `fabric ∈ {"nvlink","nic"}` derived from
`_is_intra_node(group, hw)` (intra → nvlink, inter → nic). This feeds Section E.

---

## Section E — simulator.py: fabric-level bandwidth contention

Add a **per-fabric shared clock** alongside the existing per-stream clocks. A comm
op now starts at:

```
start = max(stream_clock[op.stream], dep_max, fabric_clock[op.fabric])
finish = start + op.duration
stream_clock[op.stream] = finish
fabric_clock[op.fabric] = finish      # only for comm ops
```

Effect: collectives **on the same fabric** time-share (serialize) — a conservative
approximation of bandwidth sharing — while still overlapping freely with
`compute`. Collectives on **different fabrics** (NVLink vs NIC) do not contend.
Non-comm ops are unaffected (no `fabric`), so DP-under-backward overlap is
preserved exactly as today, but a concurrent same-fabric collective now correctly
extends the exposed tail.

**Approximation choice (confirmed):** same-fabric time-sharing, not
bandwidth-split-by-concurrency. Simpler, deterministic, and conservative (never
optimistic about comm). A `# NOTE` documents that true bandwidth-split would
shorten back-to-back same-fabric transfers slightly.

**Optional contention hook (default off):** a calibration coefficient
`overlap_compute_slowdown` (default 1.0) that, when >1.0, lengthens compute ops
that run concurrently with comm — reserved for later, not wired into v1 results.

`SimResult` gains a per-fabric exposed-comm breakdown (see Section F).

---

## Section F — Reporting: serve recipe-compare + network-sizing

Extend `SimResult` / `TargetReport`:

- **Compute time split by precision class** (`bf16`/`fp8`/`fp4`) and a separate
  **quant-overhead time** line (quantize+Hadamard+dequant+compensation), so the
  "did overhead eat the speedup" question is answerable directly.
- **Exposed communication by fabric** (`nvlink` vs `nic`): wall-time on each
  fabric that is *not* hidden under compute. This is the primary network-sizing
  output.
- **Memory breakdown** including scale metadata and compensation buffers, plus
  the fp32 master+optimizer term (already analytical in `model.py`).
- **Recipe comparison helper** `compare_precision(model, hw, recipes)` → table of
  {step time, speedup vs bf16, comm volume & reduction %, exposed-comm delta per
  fabric, peak memory delta, feasibility}.

**Periodic-step blend:** epoch/step time = `(1 - 1/N)·t_lowprec + (1/N)·t_highprec`
when `high_precision_period=N>0`, where each `t_*` is a full simulated step.

---

## Component boundaries (isolation)

- `precision.py` (new): `PrecisionConfig`, dtype/byte/compute-class resolvers.
  Pure functions, no deps on builder/simulator. Independently testable.
- `ops.py`: gains overhead ops + optional dtype args; still pure cost functions.
- `builder.py`: consults `PrecisionConfig` to inject ops and tag fabric; no
  precision logic of its own beyond resolution calls.
- `simulator.py`: fabric clocks only; no knowledge of precision.
- `report.py`/`model.py`: aggregation + comparison helper.

Each unit answers: *what it does / how you call it / what it depends on* without
reading another's internals.

---

## Testing strategy

- **Regression:** with no `PrecisionConfig`, every existing test passes unchanged
  (defaults reproduce today's numbers).
- **Byte/precision unit tests:** fp8 halves weight bytes vs bf16; fp4 quarters;
  block scale adds the expected metadata bytes.
- **Speedup:** an fp8 GEMM with `peak_tflops["fp8"]=2×bf16` is ~2× faster minus
  overhead; assert overhead ops appear in the DAG.
- **Fabric contention (the key fix):** construct two inter-node collectives that
  overlap in time → exposed `nic` time = sum (serialized), not max; an intra-node
  TP + inter-node DP pair → no mutual extension. Pin DP-under-backward overlap
  still holds (extends existing builder overlap test).
- **Comm reduction:** fp8 `comm.dtype` halves `comm_bytes` and shrinks exposed
  `nic` time; quant/dequant ops appear around the collective.
- **End-to-end recipe compare:** bf16 vs fp8 vs fp4 on a reference model produces
  monotonic speedup and memory ordering, with exposed-comm crossover surfaced.

Reference scenario for examples: a dense model (e.g. Llama-3.1-8B) on a multi-node
cluster where `dp` spans nodes (so DP grad sync lands on `nic` and fabric
contention is observable).
