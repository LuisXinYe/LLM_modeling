# llm-perf Architecture

## Overview

llm-perf predicts RL training epoch time through a three-layer architecture.
Each layer builds on the one below it:

```
                        ┌─────────────────────────────┐
  L3  pipeline.py       │  Two-stage epoch pipeline    │
                        │  generation ──► training     │
                        └──────────┬──────────────────┘
                                   │ SimOp DAGs
                        ┌──────────▼──────────────────┐
  L2  builder.py        │  Model+Parallelism → DAG    │
      simulator.py      │  Multi-stream simulation    │
                        └──────────┬──────────────────┘
                                   │ OpCost
                        ┌──────────▼──────────────────┐
  L1  ops.py            │  Roofline cost per operator  │
                        └─────────────────────────────┘
```

**Data flow:** YAML configs (`config.py`) define model, hardware, parallelism,
and RL parameters. `builder.py` calls `ops.py` to cost each operator, wraps
results into `SimOp` nodes with stream and dependency info, and hands the DAG
to `simulator.py`. `pipeline.py` orchestrates generation and training phases,
calling builder+simulator for each, then combines them into an epoch time.
`model.py:LLMPerformanceModel` is the top-level API that drives everything and
produces a `TargetReport`.


## L1: Roofline Cost Model (ops.py)

Each operator function returns an `OpCost(flops, mem_rw, weight_bytes,
output_bytes, comm_bytes)`. The roofline model converts this to wall-clock
time:

```
time = max(flops / (peak_tflops * eff), mem_rw / (hbm_bw * mem_eff))
```

When `flops / mem_rw` exceeds the hardware's arithmetic intensity, the op is
**compute-bound** (large GEMMs like attention projections, FFN). Otherwise it
is **memory-bound** (RMSNorm, small element-wise ops).

Two-tier calibration separates large-GEMM efficiency
(`compute_eff_large_gemm`) from small-op efficiency (`compute_eff_small_op`).
`roofline_time()` selects the tier via the `is_large_gemm` flag.

**Compute ops:** `op_gqa_attention()` models GQA/MHA with per-head FLOPs
(projections + QK^T + softmax*V). `op_swa_attention()` caps the KV length by
`window_size`. `op_mla_attention()` models DeepSeek-V2 style latent attention
with weight absorption during inference. `op_swiglu_ffn()` costs the 3-linear
SwiGLU (6*d*d_ff FLOPs). `op_moe_ffn()` adds routed + shared expert costs.
`op_mtp_head()` costs multi-token prediction heads.

**Communication ops:** `comm_time()` supports ring, tree, alltoall, and p2p
algorithms with latency-aware modeling (per-step latency for inter-node).
`op_allreduce`, `op_allgather`, `op_reducescatter`, `op_alltoall`, and
`op_ring_cp` compute the transferred bytes using standard collective formulas
(e.g., ring AllReduce = `2 * msg * (N-1)/N`).


## L2: DAG Builder + Multi-stream Simulator

### builder.py

`build_layer_ops()` converts one transformer layer into a sequence of `SimOp`
nodes. It applies parallelism **before** calling ops.py: head counts are
divided by TP, expert counts by EP, sequence length by CP. Each `SimOp`
carries a `stream` tag and `depends_on` indices for cross-stream
synchronization.

A single layer produces (in order): `rmsnorm_pre_attn` -> [CP ring KV comm]
-> [SP AllGather] -> `attention_{type}` -> TP AllReduce/ReduceScatter ->
`rmsnorm_pre_ffn` -> [SP AllGather] -> `ffn_{type}` -> [EP AllToAll] ->
TP AllReduce/ReduceScatter -> [mHC residual].

When `sp=True` (sequence parallelism), AllReduce is replaced by
AllGather + ReduceScatter pairs on the `tp_comm` stream.

`build_training_step()` chains forward layers -> MTP head fwd -> MTP head bwd
-> backward layers (reversed) -> DP gradient sync -> optimizer step.
`build_generation_step()` returns separate prefill and per-token decode DAGs.

**Streams used:** `compute`, `tp_comm`, `ep_comm`, `cp_comm`, `dp_comm`.

### simulator.py

`simulate()` runs Kahn's topological sort then executes ops with per-stream
clocks. Each op starts at `max(stream_clock, max(dep_finish_times))` and
advances its stream clock by `duration`. Wall-clock time is
`max(all stream clocks)`.

Activation memory is tracked via ref-counting: `output_bytes` are allocated
when an op finishes and freed when all consumers complete. This gives a peak
activation estimate without running a real allocator.

`SimResult` returns: `wall_clock_time`, `weight_bytes`, `peak_activation_bytes`,
`total_comm_bytes`.


## L3: Two-stage Pipeline

### Generation phase

`generation_time()` builds prefill and decode DAGs via `build_generation_step`,
simulates each, then computes:

```
t_per_batch = t_prefill + effective_response_len * t_decode_per_token
```

`effective_response_len` uses a Gumbel approximation
(`avg + std * sqrt(2*ln(B))`) to estimate the expected max response length in
a batch -- the real bottleneck since all sequences in a batch must complete.

When speculative decoding is enabled with MTP, the decode tokens are divided
by a throughput multiplier: `acceptance_len / (1 + draft_overhead)`.

Total generation time = `ceil(total_responses / (batch * gen_dp)) * t_per_batch`.

### Training phase

`training_time()` simulates one micro-step via `build_training_step`, then
adjusts for PP bubbles (`(pp-1)/(M+pp-1)` ratio) and recomputation/offload
penalties (30% for full recomputation, 5% for attention-only, 15%/10% for
optimizer/activation offload).

Total training time = `num_steps * t_step`, where
`num_steps = ceil(total_responses / effective_batch)`.

### Epoch time

```python
epoch_time = t_gen + t_train              # colocated (same GPUs)
epoch_time = max(t_gen, t_train) + startup # separated (different GPU pools)
```

`bottleneck_analysis()` reports which phase dominates and the slack ratio.


## Memory Model

`model.py:_compute_memory_profile()` builds a `MemoryProfile` with these
components:

| Component | Source | Formula |
|-----------|--------|---------|
| **Weights** | SimResult.weight_bytes | Sum of all fwd-pass op weight_bytes |
| **Optimizer** | Analytical | `param_count * 12` bytes (Adam: fp32 master + momentum + variance). Divided by DP when `zero_stage >= 1` |
| **Activation peak** | SimResult.peak_activation_bytes | Ref-counted peak from simulator |
| **KV cache** | Analytical per-layer | GQA/MHA: `2 * kv_heads/tp * head_dim * dtype * batch * max_seq`. MLA: `(d_c + rope_dim) * dtype * batch * max_seq`. SWA: same as GQA but capped by `window_size` |
| **Reference model** | Conditional | Equal to weight_gb when `reference_model=True` and `ref_offload_cpu=False` |

**Training total** = weight + optimizer + activation_peak + ref_model.
**Generation total** = weight + kv_cache.
Feasibility: each total must fit within `usable_hbm_gb` (capacity * 0.85).


## Calibration

`CalibrationConfig` (in `config.py`) holds four efficiency coefficients that
bridge theoretical peak to measured performance:

| Field | Default | Meaning |
|-------|---------|---------|
| `compute_eff_large_gemm` | 0.50 | GEMM utilization (attention, FFN matmuls) |
| `compute_eff_small_op` | 0.20 | Small/element-wise op utilization (norms) |
| `memory_efficiency` | 0.70 | Effective HBM bandwidth fraction |
| `comm_efficiency` | 0.70 | Effective interconnect bandwidth fraction |

These are applied multiplicatively in `roofline_time()` and `comm_time()`.
Tuning them against profiling data is the primary way to match predictions to
reality. See [docs/calibration-guide.md](calibration-guide.md) for the
measurement and fitting workflow.


## Current Limitations

- **PP modeling is stage-0 only.** `build_training_step` and
  `build_generation_step` take the first `num_layers // pp` layers. Inter-stage
  P2P communication and pipeline scheduling (1F1B, interleaved) are not
  simulated -- only the bubble ratio penalty is applied analytically.
- **Reward model not modeled.** RL reward computation is outside the cost model.
- **Optimizer step is a placeholder.** Duration is `param_count * 1e-10`,
  not a roofline-based estimate.
- **EP dispatch is coarse.** MoE AllToAll cost assumes uniform expert routing;
  load imbalance is not modeled.
- **Activation checkpointing is a flat penalty.** Full recomputation adds 30%
  and attention recomputation adds 5%, rather than re-simulating the DAG with
  checkpointed ops.
- **Attention projections are bf16-only under low precision.** Attention is a
  monolithic SimOp (projection + softmax fused), so the quant chain and
  precision-aware memory sizing apply to FFN GEMMs only; attention stays at
  bf16 roofline and bf16 bytes.


## Communication overlap and fabric contention

Communication is *not* serialized with compute. Each collective runs on its own
stream (`tp_comm`, `ep_comm`, `dp_comm`, `cp_comm`) with an independent clock, so
it overlaps compute wherever the DAG permits. In particular, **DP gradient sync
is bucketed per layer** and issued as soon as that layer's backward finishes
(depending only on that layer's last backward op), so it overlaps the backward
compute of earlier layers — matching PyTorch DDP / Megatron gradient bucketing.
TP/EP collectives sit on the critical path (the next compute op depends on them),
so they are exposed, as in vanilla tensor/expert parallelism.

Concurrent collectives on the **same physical fabric** share bandwidth: each comm
op also respects a per-fabric clock (`nvlink` for intra-node groups, `nic` for
inter-node), so same-fabric transfers time-share (a conservative approximation of
bandwidth splitting) while NVLink and NIC run independently. `SimResult` reports
`exposed_comm_by_fabric` — the per-fabric time not hidden under compute — which
is the primary signal for interconnect sizing.


## Low-precision training cost path

`PrecisionConfig` specifies precision per tensor role (weights / activations /
gradients / comm), each with a dtype, optional fine-grained block size, and an
optional stochastic Hadamard flag. When a role is below bf16, `builder.py` injects
a quantize → (Hadamard) → low-precision matmul → dequant chain around the FFN
GEMM, sizes the resident weight copy by `weights.dtype` and the saved activation
by `activations.dtype`, and reduces gradient-comm volume (with quantize/dequant
around the collective). `roofline_time` selects the per-precision peak TFLOPS via
a `compute_class`. fp32 master weights remain in the optimizer term; error-feedback
buffers are a separate resident term. With no `PrecisionConfig`, every role
defaults to `ModelConfig.dtype`, reproducing the single-dtype behavior exactly.
`model.compare_precision()` runs recipes side by side (step time, speedup, comm
reduction, exposed comm by fabric, peak memory, feasibility).
