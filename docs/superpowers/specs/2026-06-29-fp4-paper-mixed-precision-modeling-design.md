# Design: Modeling the FP4 mixed-precision pretraining scheme (Zhou et al. 2025)

Date: 2026-06-29
Status: Approved (brainstorming) → ready for implementation plan
Paper: *Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models*, Zhou et al. (Shanghai AI Lab / USTC), arXiv:2502.11458v1.

## Goal

Model the paper's **module- and direction-wise FP4 mixed-precision pretraining
scheme** in `llm-perf`, with two concrete deliverables:

1. **Validation anchor** — reproduce the paper's **Table 2 theoretical
   computation-cost percentages** (57.1 / 60.7 / 66.1 / 69.6 / 100 %) within a
   small tolerance, proving the method is modeled faithfully.
2. **Reusable recipe** — extend `PrecisionConfig` so the paper's scheme is a
   first-class recipe that flows through the full roofline path (compute, quant
   overhead, memory, comm) for what-if analysis (e.g. apply it to Llama-8B / real
   FP4 hardware via `compare_precision`).

## The paper's method (modeling target)

Precision is assigned **per module × per direction**, not globally:

| Computation | Precision | Rationale |
|---|---|---|
| FFN linear, **forward** GEMM | **FP4** | FFN ≈ 57 % of block compute; main win |
| Attention QKV + O projections ("neighbor linear"), forward | **FP8** | protect attention (FP4 flattens attention scores) |
| MHA core (QK·Kᵀ, softmax, score·V) | **FP16** | FlashAttention; never quantized |
| Linear **backward** GEMMs (wgrad + dgrad) | **FP8** | backward more error-sensitive; grads ≈0.02 underflow in FP4 |
| Activation fn, LayerNorm/RMSNorm | FP16; master weights FP32 | small, unquantized |

- Fine-grained quantization: per-token+per-channel → per-block as model grows
  (already expressible via existing `TensorPrecision.block_size`).
- **Cost metric (Table 2):** matmul-only theoretical time as % of FP16, assuming
  **FP16=1×, FP8=2×, FP4=4×** speed. The FP16 MHA core stays in the denominator,
  which is why all-FP4-linears is 57 %, not 25 %.

## Scope decisions (from brainstorming)

- **Granularity:** targeted **module × direction** (attn_linear / ffn_linear,
  each with forward/backward precision). NOT full TE per-GEMM-path; NOT a
  standalone paper-replica.
- **Approach A:** a dedicated **analytical cost calculator** for the Table-2
  anchor (matmul-only, ideal 1/2/4× speed) **plus** integration into the existing
  roofline path for what-if. Two units, two cost ledgers (the analytical one
  deliberately excludes the norms/element-wise/overhead that the full roofline
  includes).
- **Attention split:** decompose attention into **projection (precision-aware) +
  core (FP16)** — closes the previously-deferred attention-quantization gap.
  Implemented for **GQA/MHA** (the paper's GPT-2/Llama architectures); MLA / SWA /
  DSA fall back to current monolithic behavior (paper precision not applied).
- **Table 2 reproduction uses a ± tolerance** (target ≤ ~3 pp absolute), with any
  residual documented against the paper's stated FLOP split (Fig 1a).

## Non-goals

- No 2-stage target-precision **schedule** modeling (the existing periodic/blend
  mechanism can approximate later if wanted).
- No per-block vs per-token/channel **quant-overhead** differentiation beyond the
  generic `block_size` already present.
- No change to MLA/SWA/DSA attention internals.
- No numerical-accuracy / convergence modeling (perf model only).

---

## Section A — Config: module × direction precision

New optional models on `PrecisionConfig` (backward-compatible — when unset, the
existing global-role behavior and the bf16/dtype defaults are bit-identical):

```python
class ModuleLinearPrecision(BaseModel):
    fwd: TensorPrecision = TensorPrecision()   # forward GEMM operand precision
    bwd: TensorPrecision = TensorPrecision()   # backward (wgrad+dgrad) GEMM precision

class PrecisionConfig(BaseModel):
    # ... existing roles/fields unchanged ...
    attn_linear: Optional[ModuleLinearPrecision] = None   # QKV + O projections
    ffn_linear:  Optional[ModuleLinearPrecision] = None   # FFN gate/up/down
```

**Resolution rule** (a small helper, e.g. `pc.fwd_dtype(module)` /
`pc.bwd_dtype(module)`): if the module's `ModuleLinearPrecision` is set, use its
`fwd`/`bwd`; else fall back to the existing global roles (`activations`/`weights`
for forward, `gradients` for backward). This keeps every current recipe and the
bf16 default untouched.

A convenience constructor `PrecisionConfig.fp4_paper()` materializes the paper's
recipe: `attn_linear.fwd=fp8`, `ffn_linear.fwd=fp4`, both `.bwd=fp8`, block-scaled,
master fp32.

---

## Section B — Attention projection / core split

`op_gqa_attention` already computes `proj_flops` (QKV+O) and `attn_flops` (core)
separately. Expose them so the builder can cost the two parts at different
precisions:

- ops.py: `op_gqa_attention` (and `op_mha_attention`) gain a way to return the
  **projection** cost and the **core** cost as distinct `OpCost`s (e.g. a sibling
  `op_attention_split(...) -> (proj_cost, core_cost)`, or a flag). `weight_bytes`
  and the projection `mem_rw`/`output_bytes` stay with the projection part; core
  carries only its FLOPs + activation traffic.
- builder.py `build_layer_ops`: for GQA/MHA layers, emit **two compute SimOps** —
  `attn_proj` (compute_class from `attn_linear` fwd/bwd; eligible for the quant
  chain, precision-aware weight/activation bytes) and `attn_core` (compute_class
  FP16, never quantized). The core depends on the projection; the post-attention
  op depends on the core (preserves the existing dependency chain).
- MLA/SWA/DSA: keep the single monolithic attention SimOp (no split); `attn_linear`
  precision is ignored with a one-line note.

This is the structural change that lets attention projections be FP8 while the
core stays FP16 — required by the paper and by precise FFN/attention ratios.

---

## Section C — Direction-aware precision (forward vs backward)

`build_layer_ops` is called per phase (`TRAIN_FWD`, `TRAIN_BWD`). For the
quantized GEMMs (FFN, attention projection):

- In `TRAIN_FWD`, select `compute_class` from the module's **fwd** precision.
- In `TRAIN_BWD`, select `compute_class` from the module's **bwd** precision.

The backward layer pass (already built with `phase=TRAIN_BWD`) thus runs the
GEMM at the backward precision (e.g. FP8) while the forward pass ran FP4 — the
paper's asymmetry. The quant chain / precision-aware bytes use the corresponding
direction's `TensorPrecision`.

---

## Section D — Analytical theoretical-cost calculator (Table-2 anchor)

A pure function, isolated from the roofline path, mirroring the paper's
accounting:

```python
def theoretical_compute_cost(model_cfg, precision_cfg: PrecisionConfig) -> dict
# reads precision_cfg.attn_linear / ffn_linear (fwd/bwd) per (module, direction)
# returns {"cost_pct": float, "breakdown": {...}}
```

Method (matmul-only, per the paper):
1. Enumerate the block's matmuls: **attn QKV proj, attn O proj, FFN up, FFN gate,
   FFN down** (from `model_cfg` dims). Count **forward** FLOPs and **backward**
   FLOPs (= 2× forward) for each.
2. Include **MHA-core** FLOPs (QK·Kᵀ + softmax·V, fwd+bwd) fixed at FP16.
3. Speed map **FP16=1, FP8=2, FP4=4**; `time(matmul) = flops / speed`.
4. `cost_pct = 100 · Σ time(assigned) / Σ time(all-FP16)`.

The function takes a `PrecisionConfig` (reading `attn_linear`/`ffn_linear` fwd/bwd)
so the same recipe object drives both the analytical anchor and the roofline path.

**Calibration to Table 2:** the five Table-2 rows are encoded as test cases; the
sequence length / config follows the paper (LLaMA-2-125M ablation, or the stated
LLaMA-7B-4K split in Fig 1a). Residual vs the paper's exact numbers is documented
in the spec/test (target ≤ ~3 pp). If a systematic gap appears, record the FLOP
assumption that differs (e.g. whether O-proj counts toward "attention linear",
whether GQA KV-head sharing is in the paper's split).

---

## Section E — Integration into roofline + reporting

- The module×direction precision flows through `build_layer_ops` (Sections B/C)
  into the existing roofline/quant-chain/memory path. `compare_precision` accepts
  a `fp4_paper()` recipe and reports its step time, speedup, comm reduction,
  exposed comm, peak memory, feasibility — i.e. the paper's scheme applied to any
  model/hardware/parallelism.
- No new report fields required; the existing per-precision breakdown
  (`compute_seconds_by_class`) now also reflects attn-proj vs ffn classes.

---

## Section F — Validation & demo

- **Table-2 test:** `theoretical_compute_cost` reproduces the 5 rows within
  tolerance (assert each within ±3 pp; print residuals).
- **Backward-compat:** with no module precision set, builder output and the full
  suite are unchanged (bf16 default bit-identical); attention split with equal
  precision on both parts reproduces today's single-op timing (sum of the two
  SimOps == old monolithic duration when same compute_class).
- **Integration test:** `fp4_paper()` recipe through `compare_precision` on a
  small model yields speedup ∈ (1, 4), attention-proj at FP8 and FFN at FP4 in
  `compute_seconds_by_class`, and lower step time than bf16.
- **Demo** `examples/demo_fp4_paper.py`: print the Table-2 reproduction (model vs
  paper, with residuals) and apply the paper recipe to Llama-3.1-8B (step time,
  memory, feasibility).

---

## Component boundaries (isolation)

- `precision.py`: `ModuleLinearPrecision`, resolver helpers, `fp4_paper()`.
  Pure, independently testable.
- `ops.py`: attention split (proj/core costs); unchanged for other ops.
- `cost_analysis.py` (new, small): `theoretical_compute_cost` — pure, no roofline
  dependency, the Table-2 anchor lives here.
- `builder.py`: emit attn_proj/attn_core SimOps; direction-aware compute_class on
  FFN + attn-proj GEMMs. No precision logic of its own beyond resolver calls.
- `model.py`/demo: wiring + comparison.

Each unit answers *what it does / how to call it / what it depends on* in isolation.

## Testing strategy

- Unit: resolver fallback (module unset → global role); `fp4_paper()` field values.
- ops: `op_attention_split` proj+core FLOPs sum to the monolithic `op_gqa_attention`
  FLOPs (no double count / no loss).
- builder: GQA layer emits attn_proj + attn_core; with equal precision the two
  durations sum to the prior single attention duration (regression); MLA layer
  still emits one attention op.
- cost_analysis: 5 Table-2 rows within ±3 pp; all-FP16 == 100 % exactly.
- Regression: full suite green with no module precision set (bf16 default
  bit-identical).

Reference configs: GPT-2 / LLaMA per paper Table 4 for the analytical anchor;
Llama-3.1-8B for the roofline what-if demo.
