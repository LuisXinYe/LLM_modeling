# FP4 Paper Mixed-Precision Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model Zhou et al.'s FP4 mixed-precision pretraining scheme (module × direction precision; attention projection in FP8, FFN forward FP4, linear backward FP8) in `llm-perf`, with an analytical cost anchor that exactly reproduces the paper's Fig-1a forward FLOP split, and the scheme integrated as a reusable `fp4_paper()` recipe for full roofline what-if.

**Architecture:** Extend `PrecisionConfig` with optional per-module (`attn_linear`, `ffn_linear`) × per-direction (`fwd`/`bwd`) precision (Task 1). Add a pure analytical `cost_analysis.theoretical_compute_cost` reproducing Fig 1a + Table-2 ordering with a parameterized speed map (Task 2). Split attention into precision-aware projection + FP16 core in `ops.py` (Task 3) and `builder.py` (Task 4). Make FFN GEMM precision direction-aware (Task 5). Integrate + demo (Task 6).

**Tech Stack:** Python 3.10+, numpy, pydantic v2, pytest. Pure-Python cost model, no ML framework deps. venv at `.venv`.

## Global Constraints

- Python >= 3.10; pure Python + numpy only — no torch/ML framework imports.
- All FLOP/byte formulas carry a docstring citing their source.
- Backward compatibility is MANDATORY: with no `attn_linear`/`ffn_linear` set, the builder DAG and the full suite are bit-identical to today (the bf16/`from_model_dtype` default path is untouched). Current suite is green (run `pytest tests/ -q` to confirm the baseline before starting).
- Run tests/lint in the venv: `source .venv/bin/activate`, then `pytest tests/ -v` and `ruff check src/`. There are 4 pre-existing ruff errors in unrelated files (`_stage_utils.py`, `post_training.py`, `training.py`) and ~2 pre-existing pytest warnings — do not add new ones.
- Speed map (single source of truth, Task 2): `SPEED_MAP_PAPER = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0}`. Compute classes are `"bf16"|"fp8"|"fp4"` (note: `compute_class("fp16")=="bf16"`; the cost calculator treats the bf16 class as the FP16 baseline, speed 1.0).
- Fig-1a forward split target (LLaMA-7B-4K: d=4096, heads=32, head_dim=128, d_ff=11008, seq=4096, MHA so d_kv=d_qo=4096): FFN ≈ 57%, attn-linear ≈ 28.7%, MHA-core ≈ 14.3%.
- Existing helpers to reuse: `precision.dtype_bytes`, `precision.compute_class`; `builder._compute_class` (alias of `compute_class`), `builder._prec_dtype_bytes` (alias of `dtype_bytes`), `builder._inject_quant_chain(result, gemm_op, numel, tokens, d, pc, hw, index_offset, dep_idx, component_name="")`, `builder._idx`, `ops.roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16")`, `ops.op_gqa_attention(...)`.

---

### Task 1: `PrecisionConfig` module × direction precision

**Files:**
- Modify: `src/llm_perf/precision.py` (add `ModuleLinearPrecision`, fields, resolver, `fp4_paper()`)
- Test: `tests/test_precision.py` (append)

**Interfaces:**
- Consumes: existing `TensorPrecision`, `compute_class`.
- Produces:
  - `class ModuleLinearPrecision(BaseModel)`: `fwd: TensorPrecision = TensorPrecision()`, `bwd: TensorPrecision = TensorPrecision()`.
  - `PrecisionConfig.attn_linear: Optional[ModuleLinearPrecision] = None`, `PrecisionConfig.ffn_linear: Optional[ModuleLinearPrecision] = None`.
  - `PrecisionConfig.linear_fwd(module: str) -> TensorPrecision` and `PrecisionConfig.linear_bwd(module: str) -> TensorPrecision`, where `module in {"attn", "ffn"}`. Falls back to the global roles when the module precision is unset: fwd → `self.activations`, bwd → `self.gradients`.
  - `PrecisionConfig.fp4_paper() -> PrecisionConfig` classmethod.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precision.py  (append)
from llm_perf.precision import ModuleLinearPrecision


def test_module_linear_precision_defaults():
    m = ModuleLinearPrecision()
    assert m.fwd.dtype == "bf16" and m.bwd.dtype == "bf16"


def test_resolver_falls_back_to_global_roles_when_unset():
    pc = PrecisionConfig(
        activations=TensorPrecision(dtype="fp8_e4m3"),
        gradients=TensorPrecision(dtype="fp16"),
    )
    # attn_linear/ffn_linear unset → fall back to global activations/gradients
    assert pc.linear_fwd("attn").dtype == "fp8_e4m3"
    assert pc.linear_bwd("ffn").dtype == "fp16"


def test_resolver_uses_module_precision_when_set():
    pc = PrecisionConfig(
        ffn_linear=ModuleLinearPrecision(
            fwd=TensorPrecision(dtype="fp4_e2m1"), bwd=TensorPrecision(dtype="fp8_e4m3")
        )
    )
    assert pc.linear_fwd("ffn").dtype == "fp4_e2m1"
    assert pc.linear_bwd("ffn").dtype == "fp8_e4m3"
    # attn still falls back (unset) to global activations default bf16
    assert pc.linear_fwd("attn").dtype == "bf16"


def test_fp4_paper_recipe():
    pc = PrecisionConfig.fp4_paper()
    assert pc.linear_fwd("attn").dtype == "fp8_e4m3"   # attention protected
    assert pc.linear_fwd("ffn").dtype == "fp4_e2m1"    # FFN forward FP4
    assert pc.linear_bwd("attn").dtype == "fp8_e4m3"   # backward FP8
    assert pc.linear_bwd("ffn").dtype == "fp8_e4m3"
    assert pc.master_dtype == "fp32"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_precision.py -k "module_linear or resolver or fp4_paper" -v`
Expected: FAIL — `ImportError: cannot import name 'ModuleLinearPrecision'`.

- [ ] **Step 3: Write minimal implementation**

In `src/llm_perf/precision.py`, add the class before `PrecisionConfig` (reuse `Optional` — add `from typing import Optional` if missing):

```python
class ModuleLinearPrecision(BaseModel):
    """Precision for one linear module's forward vs backward GEMMs.

    Ref: Zhou et al. 2025 (arXiv:2502.11458) — forward FP4 / backward FP8 asymmetry.
    """
    fwd: TensorPrecision = TensorPrecision()
    bwd: TensorPrecision = TensorPrecision()
```

Add the two fields to `PrecisionConfig` (after `comm`):

```python
    attn_linear: Optional[ModuleLinearPrecision] = None  # QKV + O projections
    ffn_linear: Optional[ModuleLinearPrecision] = None   # FFN gate/up/down
```

Add resolver methods + recipe to `PrecisionConfig`:

```python
    def linear_fwd(self, module: str) -> TensorPrecision:
        """Forward-GEMM precision for module 'attn'|'ffn'; falls back to global activations."""
        mp = self.attn_linear if module == "attn" else self.ffn_linear
        return mp.fwd if mp is not None else self.activations

    def linear_bwd(self, module: str) -> TensorPrecision:
        """Backward-GEMM precision for module 'attn'|'ffn'; falls back to global gradients."""
        mp = self.attn_linear if module == "attn" else self.ffn_linear
        return mp.bwd if mp is not None else self.gradients

    @classmethod
    def fp4_paper(cls) -> "PrecisionConfig":
        """Zhou et al. 2025 FP4 recipe: attn-proj FP8, FFN-fwd FP4, linear-bwd FP8."""
        fp8 = TensorPrecision(dtype="fp8_e4m3", block_size=128)
        fp4 = TensorPrecision(dtype="fp4_e2m1", block_size=128)
        return cls(
            attn_linear=ModuleLinearPrecision(fwd=fp8, bwd=fp8),
            ffn_linear=ModuleLinearPrecision(fwd=fp4, bwd=fp8),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_precision.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/precision.py tests/test_precision.py
git commit -m "feat(precision): module x direction linear precision + fp4_paper recipe"
```

---

### Task 2: Analytical `theoretical_compute_cost` (Fig-1a anchor + Table-2 ordering)

**Files:**
- Create: `src/llm_perf/cost_analysis.py`
- Test: `tests/test_cost_analysis.py`

**Interfaces:**
- Consumes: `ModelConfig`, `PrecisionConfig` (Task 1), `precision.compute_class`.
- Produces:
  - `SPEED_MAP_PAPER = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0}`.
  - `theoretical_compute_cost(model_cfg, precision_cfg, speed_map=SPEED_MAP_PAPER) -> dict` with keys `cost_pct: float`, `forward_split: dict` (`{"ffn","attn_linear","mha_core"}` → fraction), `breakdown: dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_analysis.py
import pytest
from llm_perf.config import LayerConfig, ModelConfig
from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
from llm_perf.cost_analysis import theoretical_compute_cost, SPEED_MAP_PAPER


def llama7b_4k() -> ModelConfig:
    # MHA (num_kv_heads == num_heads), d=4096, d_ff=11008, 32 heads x 128, seq 4096
    layer = LayerConfig(
        attention="MHA", num_heads=32, num_kv_heads=32, head_dim=128,
        ffn="SwiGLU", intermediate_size=11008,
    )
    return ModelConfig(name="llama7b", hidden_size=4096, vocab_size=32000,
                       num_layers=32, default_layer=layer)


SEQ = 4096


def test_forward_split_matches_fig1a():
    pc = PrecisionConfig.bf16_default()
    out = theoretical_compute_cost(llama7b_4k(), pc, seq_len=SEQ)
    fs = out["forward_split"]
    assert fs["ffn"] == pytest.approx(0.573, abs=0.01)        # ~57%
    assert fs["attn_linear"] == pytest.approx(0.284, abs=0.01)  # ~28.7%
    assert fs["mha_core"] == pytest.approx(0.142, abs=0.01)     # ~14.3%
    assert sum(fs.values()) == pytest.approx(1.0, abs=1e-6)


def test_all_fp16_is_100pct():
    out = theoretical_compute_cost(llama7b_4k(), PrecisionConfig.bf16_default(), seq_len=SEQ)
    assert out["cost_pct"] == pytest.approx(100.0, abs=1e-6)


def test_all_fp4_under_stated_speeds_is_flop_honest_not_paper():
    # Documented finding: paper's stated FP4=4x yields ~36% for all-FP4, NOT 57.1%.
    all_fp4 = PrecisionConfig(
        attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                          bwd=TensorPrecision(dtype="fp4_e2m1")),
        ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                         bwd=TensorPrecision(dtype="fp4_e2m1")),
    )
    out = theoretical_compute_cost(llama7b_4k(), all_fp4, seq_len=SEQ)
    assert out["cost_pct"] == pytest.approx(35.7, abs=2.0)  # FLOP-honest


def test_all_fp4_under_paper_implied_speeds_matches_57():
    all_fp4 = PrecisionConfig(
        attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                          bwd=TensorPrecision(dtype="fp4_e2m1")),
        ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype="fp4_e2m1"),
                                         bwd=TensorPrecision(dtype="fp4_e2m1")),
    )
    out = theoretical_compute_cost(llama7b_4k(), all_fp4, seq_len=SEQ,
                                   speed_map={"fp16": 1.0, "fp8": 1.4, "fp4": 2.0})
    assert out["cost_pct"] == pytest.approx(57.1, abs=3.0)


def test_table2_ordering_attn_fp8_ffn_fp4_cheaper_than_attn_fp4_ffn_fp8():
    def recipe(attn, ffn):
        return PrecisionConfig(
            attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=attn),
                                              bwd=TensorPrecision(dtype="fp4_e2m1")),
            ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=ffn),
                                             bwd=TensorPrecision(dtype="fp4_e2m1")),
        )
    m = llama7b_4k()
    c_attn8_ffn4 = theoretical_compute_cost(m, recipe("fp8_e4m3", "fp4_e2m1"), seq_len=SEQ)["cost_pct"]
    c_attn4_ffn8 = theoretical_compute_cost(m, recipe("fp4_e2m1", "fp8_e4m3"), seq_len=SEQ)["cost_pct"]
    # FFN is the bigger matmul, so quantizing it harder (FP4) is cheaper overall
    assert c_attn8_ffn4 < c_attn4_ffn8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_cost_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_perf.cost_analysis'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/cost_analysis.py
"""Analytical matmul-only compute-cost model for low-precision recipes.

Mirrors the accounting in Zhou et al. 2025 (arXiv:2502.11458), Fig 1a / Table 2:
count forward + backward matmul FLOPs per transformer block, scale each by a
per-compute-class speed, and report cost as a percentage of the all-FP16 cost.
The MHA core (QK^T + softmax*V) is kept at FP16 (FlashAttention) in the denominator.

NOTE: under the paper's stated speeds (FP16=1, FP8=2, FP4=4) the all-FP4-linear
recipe is ~36%, NOT the paper's Table-2 57.1% (the cost-% is invariant to the
fwd/bwd multiplier, so this is not a backward-counting artifact). The paper's
Table-2 metric implies an effective FP4 ~2x; pass `speed_map` to match it.
"""
from __future__ import annotations

from llm_perf.config import ModelConfig
from llm_perf.precision import PrecisionConfig, compute_class

SPEED_MAP_PAPER = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0}


def _speed(dtype: str, speed_map: dict) -> float:
    cc = compute_class(dtype)            # "bf16" | "fp8" | "fp4"
    key = "fp16" if cc == "bf16" else cc  # bf16 class == FP16 baseline
    return speed_map[key]


def theoretical_compute_cost(
    model_cfg: ModelConfig, precision_cfg: PrecisionConfig,
    speed_map: dict = SPEED_MAP_PAPER, seq_len: int = 4096,
) -> dict:
    """Matmul-only theoretical compute cost as % of all-FP16. See module docstring."""
    layer = model_cfg.get_layers()[0]
    d = model_cfg.hidden_size
    d_qo = layer.num_heads * layer.head_dim
    d_kv = layer.num_kv_heads * layer.head_dim
    d_ff = layer.intermediate_size

    # Per-token forward FLOPs (matmuls only).
    attn_linear_fwd = (2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d)  # QKV + O
    mha_core_fwd = 4 * d_qo * seq_len                                             # QK^T + softmax*V
    ffn_fwd = 6 * d * d_ff                                                        # SwiGLU 3 matmuls

    fwd_total = attn_linear_fwd + mha_core_fwd + ffn_fwd
    forward_split = {
        "ffn": ffn_fwd / fwd_total,
        "attn_linear": attn_linear_fwd / fwd_total,
        "mha_core": mha_core_fwd / fwd_total,
    }

    # Forward + backward (backward = 2x forward) scaled by speed.
    def t(flops_fwd: float, fwd_dtype: str, bwd_dtype: str) -> float:
        return flops_fwd / _speed(fwd_dtype, speed_map) + 2 * flops_fwd / _speed(bwd_dtype, speed_map)

    attn_fwd_dt = precision_cfg.linear_fwd("attn").dtype
    attn_bwd_dt = precision_cfg.linear_bwd("attn").dtype
    ffn_fwd_dt = precision_cfg.linear_fwd("ffn").dtype
    ffn_bwd_dt = precision_cfg.linear_bwd("ffn").dtype

    t_assigned = (
        t(attn_linear_fwd, attn_fwd_dt, attn_bwd_dt)
        + t(ffn_fwd, ffn_fwd_dt, ffn_bwd_dt)
        + t(mha_core_fwd, "fp16", "fp16")  # core always FP16
    )
    t_fp16 = t(attn_linear_fwd, "fp16", "fp16") + t(ffn_fwd, "fp16", "fp16") + t(mha_core_fwd, "fp16", "fp16")

    return {
        "cost_pct": 100.0 * t_assigned / t_fp16,
        "forward_split": forward_split,
        "breakdown": {
            "attn_linear": (attn_fwd_dt, attn_bwd_dt),
            "ffn": (ffn_fwd_dt, ffn_bwd_dt),
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_cost_analysis.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/cost_analysis.py tests/test_cost_analysis.py
git commit -m "feat(cost): analytical theoretical_compute_cost (Fig1a anchor + Table2 ordering)"
```

---

### Task 3: `op_attention_split` in `ops.py` (projection + core costs)

**Files:**
- Modify: `src/llm_perf/ops.py` (add `op_attention_split`)
- Test: `tests/test_ops.py` (append)

**Interfaces:**
- Consumes: existing `op_gqa_attention`, `OpCost`, `Phase`.
- Produces: `op_attention_split(num_heads, num_kv_heads, head_dim, hidden_size, batch, seq_len, phase, kv_len=None, dtype_bytes=2, proj_weight_dtype_bytes=None, proj_act_dtype_bytes=None) -> tuple[OpCost, OpCost]` returning `(proj_cost, core_cost)`. `proj_cost` carries QKV+O projection FLOPs + all weight_bytes + the input/output activation traffic; `core_cost` carries the QK^T+softmax*V FLOPs only (no weights). Their FLOPs sum equals `op_gqa_attention`'s total FLOPs for the same args.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ops.py  (append)
from llm_perf.ops import op_attention_split, op_gqa_attention
from llm_perf.config import Phase


def test_attention_split_flops_sum_to_monolithic():
    args = dict(num_heads=32, num_kv_heads=32, head_dim=128, hidden_size=4096,
                batch=1, seq_len=4096, phase=Phase.TRAIN_FWD)
    mono = op_gqa_attention(**args)
    proj, core = op_attention_split(**args)
    assert proj.flops + core.flops == pytest.approx(mono.flops, rel=1e-9)
    # weights all live with the projection; core has none
    assert proj.weight_bytes == pytest.approx(mono.weight_bytes)
    assert core.weight_bytes == 0


def test_attention_split_proj_precision_aware_weight_bytes():
    args = dict(num_heads=32, num_kv_heads=32, head_dim=128, hidden_size=4096,
                batch=1, seq_len=4096, phase=Phase.TRAIN_FWD)
    proj_bf16, _ = op_attention_split(**args, proj_weight_dtype_bytes=2)
    proj_fp8, _ = op_attention_split(**args, proj_weight_dtype_bytes=1)
    assert proj_fp8.weight_bytes == pytest.approx(proj_bf16.weight_bytes / 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k attention_split -v`
Expected: FAIL — `ImportError: cannot import name 'op_attention_split'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm_perf/ops.py`:

```python
def op_attention_split(
    num_heads: int, num_kv_heads: int, head_dim: int, hidden_size: int,
    batch: int, seq_len: int, phase: Phase, kv_len: int | None = None,
    dtype_bytes: int = 2,
    proj_weight_dtype_bytes: float | None = None,
    proj_act_dtype_bytes: float | None = None,
) -> tuple["OpCost", "OpCost"]:
    """Split GQA/MHA attention into (projection GEMM, core attention) costs.

    Projection = QKV + O linears (quantizable; carries all weights + activation
    traffic). Core = QK^T + softmax*V (kept FP16 per Zhou et al. 2025). FLOPs of
    the two parts sum to op_gqa_attention's total. proj_weight/act_dtype_bytes
    size the resident weight copy / saved activation for low precision (default
    dtype_bytes). Ref: GQA paper; Zhou et al. arXiv:2502.11458.
    """
    H, G, d = num_heads, num_kv_heads, hidden_size
    batch_tokens = batch * seq_len
    d_qo = H * head_dim
    d_kv = G * head_dim
    L = (kv_len if kv_len is not None else seq_len) if phase == Phase.DECODE else seq_len
    wb = dtype_bytes if proj_weight_dtype_bytes is None else proj_weight_dtype_bytes
    ab = dtype_bytes if proj_act_dtype_bytes is None else proj_act_dtype_bytes

    proj_flops = (2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d) * batch_tokens
    attn_flops = (4 * d_qo * L * batch) if phase == Phase.DECODE else (4 * d_qo * L * batch_tokens)
    bwd = 2 if phase == Phase.TRAIN_BWD else 1

    weight_b = (d * d_qo + d * d_kv + d * d_kv + d_qo * d) * wb
    proj_mem = weight_b + batch_tokens * d * dtype_bytes + batch_tokens * d_qo * dtype_bytes
    proj_out = batch_tokens * d * ab if phase == Phase.TRAIN_FWD else 0
    proj = OpCost(flops=bwd * proj_flops, mem_rw=proj_mem, weight_bytes=weight_b, output_bytes=proj_out)

    core_mem = 2 * batch_tokens * d_qo * dtype_bytes  # read scores / write context (approx)
    if phase == Phase.DECODE:
        core_mem += batch * L * 2 * d_kv * dtype_bytes
    core = OpCost(flops=bwd * attn_flops, mem_rw=core_mem, weight_bytes=0, output_bytes=0)
    return proj, core
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k attention_split -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/ops.py tests/test_ops.py
git commit -m "feat(ops): op_attention_split (projection + FP16 core costs)"
```

---

### Task 4: Builder emits `attn_proj` + `attn_core` for GQA/MHA

**Files:**
- Modify: `src/llm_perf/builder.py` (`build_layer_ops`, GQA/MHA attention branch ~360-444)
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Consumes: `op_attention_split` (Task 3); `PrecisionConfig.linear_fwd/linear_bwd`, `fp4_paper` (Task 1); `_inject_quant_chain`, `_compute_class`, `_prec_dtype_bytes`, `_idx`, `roofline_time`.
- Produces: for GQA/MHA layers, when `pc.attn_linear` is set, the layer DAG contains two compute SimOps `attention_proj` and `attention_core` (core depends on proj; the post-attention op depends on core). When `pc.attn_linear` is None, behavior is unchanged (single `attention_{type}` SimOp). MLA/SWA/DSA always emit the single op.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
from llm_perf.precision import PrecisionConfig
from llm_perf.simulator import simulate


def test_gqa_attention_splits_when_attn_linear_set():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc_par, rl, precision_cfg=PrecisionConfig.fp4_paper())
    names = [o.name for o in ops]
    assert any(n == "attention_proj" for n in names)
    assert any(n == "attention_core" for n in names)


def test_no_attention_split_without_attn_linear():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc_par, rl)  # default, no module precision
    names = [o.name for o in ops]
    assert not any(n in ("attention_proj", "attention_core") for n in names)
    assert any(n.startswith("attention_") for n in names)  # monolithic preserved


def test_attention_split_equal_precision_sums_to_monolithic_duration():
    # With proj+core both at bf16, the two SimOp durations sum to the old single
    # attention duration (no time created/lost by the split).
    from llm_perf.precision import ModuleLinearPrecision, TensorPrecision
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    bf16_split = PrecisionConfig(attn_linear=ModuleLinearPrecision())  # both bf16
    ops_mono = build_training_step(model, hw, pc_par, rl)
    ops_split = build_training_step(model, hw, pc_par, rl, precision_cfg=bf16_split)
    mono = sum(o.duration for o in ops_mono if o.name.startswith("attention_"))
    split = sum(o.duration for o in ops_split if o.name in ("attention_proj", "attention_core"))
    assert split == pytest.approx(mono, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k "attention_split or attention_splits or no_attention_split" -v`
Expected: FAIL — no `attention_proj`/`attention_core` ops emitted.

- [ ] **Step 3: Write minimal implementation**

In `build_layer_ops`, replace the GQA/MHA attention SimOp construction so that when `pc.attn_linear is not None`, it emits a split. Determine the module direction from `phase` (`Phase.TRAIN_BWD` → use `pc.linear_bwd("attn")`; else `pc.linear_fwd("attn")`). Keep the existing single-op path for the unset case and for non-GQA/MHA types.

```python
    # inside build_layer_ops, GQA/MHA branch — replace the single attn_op build:
    if attn_type in ("GQA", "MHA") and pc.attn_linear is not None:
        attn_prec = pc.linear_bwd("attn") if phase == Phase.TRAIN_BWD else pc.linear_fwd("attn")
        proj_cost, core_cost = ops.op_attention_split(
            num_heads=tp_num_heads, num_kv_heads=tp_kv_heads, head_dim=layer_cfg.head_dim,
            hidden_size=d, batch=batch, seq_len=local_seq_len, phase=phase, kv_len=kv_len,
            dtype_bytes=dtype_bytes,
            proj_weight_dtype_bytes=_prec_dtype_bytes(attn_prec.dtype),
            proj_act_dtype_bytes=_prec_dtype_bytes(attn_prec.dtype),
        )
        proj_op = SimOp(
            name="attention_proj", stream="compute",
            duration=ops.roofline_time(proj_cost, hw, is_large_gemm=True,
                                       compute_class=_compute_class(attn_prec.dtype)),
            depends_on=[attn_dep_idx],
            weight_bytes=proj_cost.weight_bytes, output_bytes=proj_cost.output_bytes,
            op_class=_compute_class(attn_prec.dtype),
        )
        proj_tail = _inject_quant_chain(
            result=result, gemm_op=proj_op, numel=batch_tokens * d, tokens=batch_tokens,
            d=d, pc=pc, hw=hw, index_offset=index_offset, dep_idx=attn_dep_idx,
            component_name="attn",
        )
        core_op = SimOp(
            name="attention_core", stream="compute",
            duration=ops.roofline_time(core_cost, hw, is_large_gemm=True),  # FP16 core
            depends_on=[proj_tail],
            weight_bytes=0, output_bytes=core_cost.output_bytes,
        )
        result.append(core_op)
    else:
        attn_op = SimOp(
            name=f"attention_{attn_type.lower()}", stream="compute",
            duration=ops.roofline_time(attn_cost, hw, is_large_gemm=attn_is_large_gemm),
            depends_on=[attn_dep_idx],
            weight_bytes=attn_cost.weight_bytes, output_bytes=attn_cost.output_bytes,
        )
        result.append(attn_op)
```

Note: `_inject_quant_chain` requires `component_name="attn"` NOT be in `pc.high_precision_layers` to inject; for the bf16-split regression test (both bf16) the chain's bf16 branch appends the proj op unchanged, so proj+core durations still sum to the monolithic value (the quant chain adds zero-cost nothing when compute_class is bf16). Verify the downstream `last_compute_local`/dependency wiring still points at the core op (the post-attention residual/norm should depend on `attention_core`). Adjust the index that the next op consumes to the core op's index.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS; existing builder tests unaffected (single-op path unchanged when `attn_linear` is None).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): split GQA/MHA attention into precision-aware proj + FP16 core"
```

---

### Task 5: Direction-aware FFN precision in the builder

**Files:**
- Modify: `src/llm_perf/builder.py` (FFN SwiGLU + MoE GEMM construction ~553-660)
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Consumes: `PrecisionConfig.linear_fwd/linear_bwd` (Task 1).
- Produces: the FFN GEMM `compute_class` and the quant-chain precision are selected by direction — `Phase.TRAIN_BWD` uses `pc.linear_bwd("ffn")`, otherwise `pc.linear_fwd("ffn")`. When `ffn_linear` is None this resolves to the existing global activations/gradients, so behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
def test_ffn_backward_uses_bwd_precision():
    # fp4 forward, fp8 backward (the paper): the TRAIN_BWD ffn GEMM runs at fp8,
    # the TRAIN_FWD ffn GEMM at fp4.
    from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc_par = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    pc = PrecisionConfig(ffn_linear=ModuleLinearPrecision(
        fwd=TensorPrecision(dtype="fp4_e2m1"), bwd=TensorPrecision(dtype="fp8_e4m3")))
    ops = build_training_step(model, hw, pc_par, rl, precision_cfg=pc)
    ffn = [o for o in ops if o.name == "ffn_swiglu"]
    classes = {o.op_class for o in ffn}
    assert "fp4" in classes  # forward
    assert "fp8" in classes  # backward
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k ffn_backward_uses_bwd -v`
Expected: FAIL — both ffn GEMMs share one class (currently `pc.activations.dtype`).

- [ ] **Step 3: Write minimal implementation**

In the SwiGLU and MoE FFN branches of `build_layer_ops`, replace the `pc.activations.dtype` used for the FFN GEMM's `compute_class`/`op_class`/bytes with a direction-resolved precision:

```python
        ffn_prec = pc.linear_bwd("ffn") if phase == Phase.TRAIN_BWD else pc.linear_fwd("ffn")
        # ... in the SimOp:
        #   duration=ops.roofline_time(ffn_cost, hw, is_large_gemm=True,
        #                              compute_class=_compute_class(ffn_prec.dtype)),
        #   op_class=_compute_class(ffn_prec.dtype),
        # ... and pass ffn_prec.dtype's bytes to op_swiglu_ffn/op_moe_ffn
        #     weight_dtype_bytes/act_dtype_bytes (replacing pc.weights/activations there).
```

Apply the same `ffn_prec`-based selection to the `weight_dtype_bytes`/`act_dtype_bytes` passed into `op_swiglu_ffn`/`op_moe_ffn` and to the `_inject_quant_chain` precision (the chain reads `pc.activations`; to keep direction-correct quant, the simplest faithful change is to leave the chain keyed on `pc.activations` for forward and accept that backward quant overhead uses the forward block size — note this as a known simplification in the report). When `ffn_linear` is None, `linear_fwd/bwd("ffn")` return the global `activations`/`gradients`, preserving today's behavior.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS; bf16/default path unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): direction-aware FFN precision (fwd vs bwd compute_class)"
```

---

### Task 6: Integration — `compare_precision` with `fp4_paper` + demo

**Files:**
- Create: `examples/demo_fp4_paper.py`
- Test: `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `compare_precision` (existing), `PrecisionConfig.fp4_paper` (Task 1), `theoretical_compute_cost` (Task 2).
- Produces: a demo that (a) prints the Fig-1a split + Table-2 ordering from `theoretical_compute_cost`, and (b) applies `fp4_paper()` to Llama-3.1-8B via `compare_precision`. An integration test asserting the paper recipe is faster than bf16 with attn-proj FP8 + FFN FP4 in the class breakdown.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (append)
def test_fp4_paper_recipe_through_compare_precision():
    from llm_perf.model import compare_precision
    from llm_perf.precision import PrecisionConfig
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    rows = compare_precision(model, hw, pc, rl, {
        "bf16": PrecisionConfig.bf16_default(),
        "fp4_paper": PrecisionConfig.fp4_paper(),
    })
    by = {r["name"]: r for r in rows}
    assert by["fp4_paper"]["speedup_vs_bf16"] > 1.0
    assert by["fp4_paper"]["speedup_vs_bf16"] <= 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_model.py -k fp4_paper_recipe_through -v`
Expected: FAIL if any earlier wiring is incomplete; otherwise this is the integration gate.

- [ ] **Step 3: Write minimal implementation**

Create `examples/demo_fp4_paper.py` mirroring `examples/demo_low_precision.py`:

```python
"""Demo: model Zhou et al. 2025 FP4 mixed-precision pretraining scheme."""
from llm_perf.config import load_model_config, load_hardware_config, ParallelismConfig, WorkloadConfig
from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
from llm_perf.cost_analysis import theoretical_compute_cost
from llm_perf.model import compare_precision


def main():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")

    print("== Fig 1a forward FLOP split (LLaMA-7B-4K) ==")
    fs = theoretical_compute_cost(model, PrecisionConfig.bf16_default(), seq_len=4096)["forward_split"]
    for k, v in fs.items():
        print(f"  {k:12s} {v*100:5.1f}%")

    print("\n== Table-2 recipes — theoretical compute cost % ==")
    def rec(a, f, b):
        return PrecisionConfig(
            attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=a), bwd=TensorPrecision(dtype=b)),
            ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=f), bwd=TensorPrecision(dtype=b)))
    rows = {
        "FP4/FP4/FP4": rec("fp4_e2m1", "fp4_e2m1", "fp4_e2m1"),
        "FP8/FP4/FP8": rec("fp8_e4m3", "fp4_e2m1", "fp8_e4m3"),
        "FP16 (base)": PrecisionConfig.bf16_default(),
    }
    for name, pc in rows.items():
        stated = theoretical_compute_cost(model, pc, seq_len=4096)["cost_pct"]
        implied = theoretical_compute_cost(model, pc, seq_len=4096,
                                           speed_map={"fp16": 1.0, "fp8": 1.4, "fp4": 2.0})["cost_pct"]
        print(f"  {name:12s} stated(1/2/4x)={stated:5.1f}%  paper-implied={implied:5.1f}%")
    print("  NOTE: paper Table 2 all-FP4 = 57.1%; FLOP-honest 1/2/4x gives ~36% (implies FP4~2x).")

    print("\n== Apply fp4_paper() recipe to Llama-3.1-8B (full roofline) ==")
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    for r in compare_precision(model, hw, pc, rl,
                               {"bf16": PrecisionConfig.bf16_default(), "fp4_paper": PrecisionConfig.fp4_paper()}):
        print(f"  {r['name']:10s} speedup={r['speedup_vs_bf16']:.3f} peakMem={r['peak_memory_gb']:.1f}GB feasible={r['feasible']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + demo to verify**

Run: `source .venv/bin/activate && pytest tests/ -q && python examples/demo_fp4_paper.py`
Expected: full suite PASS (final regression gate); demo prints the Fig-1a split (~57/28.7/14.3), the stated-vs-implied Table-2 costs, and the Llama-8B what-if.

- [ ] **Step 5: Commit**

```bash
git add examples/demo_fp4_paper.py tests/test_model.py
git commit -m "feat: fp4_paper recipe integration + demo (Fig1a split, Table2, Llama-8B what-if)"
```

---

## Final verification

- [ ] Full suite + lint:

```bash
source .venv/bin/activate && pytest tests/ -v && ruff check src/
```

Expected: all tests pass; no new ruff errors (4 pre-existing in unrelated files).

- [ ] Update `docs/low-precision-findings.md` (or a new `docs/fp4-paper-modeling.md`) with the Fig-1a reproduction, the Table-2 FP4≈2× finding, and the Llama-8B what-if numbers. Commit.

---

## Self-review notes (coverage map)

- Spec §A (module×direction config) → Task 1. Spec §D (analytical cost, Fig1a anchor, parameterized speed map) → Task 2. Spec §B (attention proj/core split) → Tasks 3 (ops) + 4 (builder). Spec §C (direction-aware) → Tasks 4 (attn) + 5 (ffn). Spec §E (integration) + §F (validation/demo) → Task 6.
- Backward-compat asserted in Tasks 4 (no-split when unset; bf16-split sums to monolithic), 5 (default unchanged), and the final full-suite gate.
- The Table-2 absolute-% limitation is encoded honestly in Task 2 tests (stated 1/2/4× → ~36%; paper-implied map → ~57%) and surfaced in the Task 6 demo.
- Known simplification (flagged in Task 5): the FFN quant-chain overhead stays keyed on `pc.activations` block size even for the backward direction — acceptable (overhead is small/second-order), noted in the implementer report.
