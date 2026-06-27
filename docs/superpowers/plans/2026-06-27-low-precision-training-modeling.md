# Low-precision Training Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make llm-perf model low-precision training recipes (FP8/FP4 with per-role precision, fine-grained block scaling, stochastic Hadamard) and predict their effect on compute time, exposed communication, and memory — with a corrected fabric-level bandwidth-contention model so network-sizing numbers are honest.

**Architecture:** A new pure `precision.py` resolves per-role dtypes → bytes and compute classes. `ops.py` gains a `compute_class` arg on `roofline_time` plus four overhead ops. `simulator.py` adds a per-fabric shared clock so same-fabric collectives time-share while NVLink/NIC stay independent. `builder.py` consults a `PrecisionConfig` to inject the quant chain around low-precision GEMMs and collectives, tag each comm op's fabric, and add compensation/mixed-precision behavior. `report.py`/`model.py` aggregate per-precision and per-fabric breakdowns and expose a recipe-comparison helper.

**Tech Stack:** Python 3.10+, numpy, pydantic v2, pytest. No ML framework deps (pure-Python cost model). venv at `.venv`.

## Global Constraints

- Python >= 3.10; pure Python + numpy only — no torch/ML framework imports.
- All FLOP/byte formulas must carry a docstring citing their source (CLAUDE.md convention).
- Hardware constants live in `configs/hardware/*.yaml`, never hardcoded.
- Backward compatibility is mandatory: with no `PrecisionConfig` and the existing single-value `peak_tflops_bf16`, every current test in `tests/` must still pass unchanged.
- Run tests/lint inside the venv: `source .venv/bin/activate` then `pytest tests/ -v` and `ruff check src/ && ruff format src/`.
- One op function per concern in `ops.py`; pure functions returning `OpCost`.
- New dtype byte map (single source of truth, Task 1): `fp32=4, bf16=2, fp16=2, fp8_e4m3=1, fp8_e5m2=1, fp4_e2m1=0.5, mxfp4=0.5`. Compute-class map: `bf16/fp16/fp32→"bf16"`, `fp8_*→"fp8"`, `fp4_*/mxfp4→"fp4"`.

---

### Task 1: `precision.py` — config + dtype/byte/compute-class resolvers

**Files:**
- Create: `src/llm_perf/precision.py`
- Test: `tests/test_precision.py`

**Interfaces:**
- Consumes: nothing (pure, foundational).
- Produces:
  - `dtype_bytes(dtype: str) -> float` — bytes per element (may be 0.5).
  - `compute_class(dtype: str) -> str` — `"bf16" | "fp8" | "fp4"`.
  - `scale_overhead_bytes(numel: float, block_size: int, scale_bytes: int = 4) -> float`.
  - `class TensorPrecision(BaseModel)`: `dtype="bf16"`, `block_size=0`, `hadamard=False`, `hadamard_block=0`, `scale_bytes=4`.
  - `class PrecisionConfig(BaseModel)`: `weights`, `activations`, `gradients`, `comm` (all `TensorPrecision`), `master_dtype="fp32"`, `error_feedback=False`, `ef_dtype="fp16"`, `high_precision_layers: list[str]=[]`, `high_precision_period=0`, `high_precision_dtype="bf16"`.
  - `PrecisionConfig.bf16_default() -> PrecisionConfig` classmethod returning an all-bf16 instance (the regression-safe default).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precision.py
import pytest
from llm_perf.precision import (
    dtype_bytes, compute_class, scale_overhead_bytes,
    TensorPrecision, PrecisionConfig,
)


def test_dtype_bytes_map():
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp16") == 2
    assert dtype_bytes("fp8_e4m3") == 1
    assert dtype_bytes("fp8_e5m2") == 1
    assert dtype_bytes("fp4_e2m1") == 0.5
    assert dtype_bytes("mxfp4") == 0.5


def test_compute_class_map():
    assert compute_class("bf16") == "bf16"
    assert compute_class("fp16") == "bf16"
    assert compute_class("fp32") == "bf16"
    assert compute_class("fp8_e4m3") == "fp8"
    assert compute_class("fp8_e5m2") == "fp8"
    assert compute_class("fp4_e2m1") == "fp4"
    assert compute_class("mxfp4") == "fp4"


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        dtype_bytes("int3")


def test_scale_overhead_per_tensor_is_negligible():
    # block_size=0 → one scale per tensor
    assert scale_overhead_bytes(1024, block_size=0, scale_bytes=4) == 4


def test_scale_overhead_fine_grained():
    # 1024 elements, block 128 → 8 scales * 4 bytes
    assert scale_overhead_bytes(1024, block_size=128, scale_bytes=4) == 32


def test_precision_config_default_is_all_bf16():
    cfg = PrecisionConfig.bf16_default()
    assert cfg.weights.dtype == "bf16"
    assert cfg.activations.dtype == "bf16"
    assert cfg.gradients.dtype == "bf16"
    assert cfg.comm.dtype == "bf16"
    assert cfg.master_dtype == "fp32"


def test_tensor_precision_fields():
    tp = TensorPrecision(dtype="fp4_e2m1", block_size=128, hadamard=True, hadamard_block=128)
    assert tp.block_size == 128
    assert tp.hadamard is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_precision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_perf.precision'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/llm_perf/precision.py
"""Per-tensor-role precision resolution for low-precision training modeling.

Pure functions + pydantic config. No dependency on builder/simulator/ops so it
can be unit-tested in isolation. See
docs/superpowers/specs/2026-06-27-low-precision-training-modeling-design.md.
"""

from __future__ import annotations

from pydantic import BaseModel

# Single source of truth for element sizes. fp4 stores two values per byte.
# Ref: NVIDIA OCP MX spec (mxfp4 = 4-bit element + shared e8m0 scale);
# FP8 formats per "FP8 Formats for Deep Learning" (Micikevicius et al. 2022).
_DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp4_e2m1": 0.5,
    "mxfp4": 0.5,
}

_COMPUTE_CLASS = {
    "fp32": "bf16",
    "bf16": "bf16",
    "fp16": "bf16",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp4_e2m1": "fp4",
    "mxfp4": "fp4",
}


def dtype_bytes(dtype: str) -> float:
    """Bytes per element for a dtype (may be fractional, e.g. fp4 = 0.5)."""
    return _DTYPE_BYTES[dtype]


def compute_class(dtype: str) -> str:
    """Map a dtype to its matmul compute pipe class: 'bf16' | 'fp8' | 'fp4'."""
    return _COMPUTE_CLASS[dtype]


def scale_overhead_bytes(numel: float, block_size: int, scale_bytes: int = 4) -> float:
    """Quantization scale-metadata bytes.

    block_size=0 → one scale per tensor. block_size=B → ceil(numel/B) scales.
    Ref: fine-grained block scaling (DeepSeek-V3 tech report, per-128 blocks).
    """
    if block_size <= 0:
        return float(scale_bytes)
    num_blocks = -(-int(numel) // block_size)  # ceil division
    return float(num_blocks * scale_bytes)


class TensorPrecision(BaseModel):
    dtype: str = "bf16"
    block_size: int = 0          # 0 = per-tensor scale; >0 = fine-grained block
    hadamard: bool = False       # stochastic Hadamard transform before quant
    hadamard_block: int = 0      # rotation size; 0 → resolver default (128)
    scale_bytes: int = 4         # bytes per scale (4=fp32, 1=e8m0 for mxfp4)


class PrecisionConfig(BaseModel):
    weights: TensorPrecision = TensorPrecision()
    activations: TensorPrecision = TensorPrecision()
    gradients: TensorPrecision = TensorPrecision()
    comm: TensorPrecision = TensorPrecision()
    master_dtype: str = "fp32"
    error_feedback: bool = False
    ef_dtype: str = "fp16"
    high_precision_layers: list[str] = []
    high_precision_period: int = 0
    high_precision_dtype: str = "bf16"

    @classmethod
    def bf16_default(cls) -> "PrecisionConfig":
        """All-bf16 recipe — reproduces today's (single-dtype) behavior."""
        return cls()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_precision.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/precision.py tests/test_precision.py
git commit -m "feat(precision): per-role precision config + dtype/byte/compute-class resolvers"
```

---

### Task 2: Hardware per-precision peak TFLOPS + `roofline_time(compute_class=...)`

**Files:**
- Modify: `src/llm_perf/config.py` (`HardwareConfig`, add `peak_tflops` + resolver)
- Modify: `src/llm_perf/ops.py:24-44` (`roofline_time` signature)
- Test: `tests/test_ops.py` (append), `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `compute_class` semantics from Task 1 (string keys `"bf16"|"fp8"|"fp4"`).
- Produces:
  - `HardwareConfig.peak_tflops: dict[str, float]` and `HardwareConfig.tflops_for(compute_class: str) -> float` (falls back to bf16 peak with a warning when the class is absent).
  - `roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16") -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ops.py  (append)
from llm_perf.ops import roofline_time, OpCost
from llm_perf.config import HardwareConfig


def _hw_with_fp8():
    return HardwareConfig(
        name="t", peak_tflops_bf16=100.0,
        peak_tflops={"bf16": 100.0, "fp8": 200.0, "fp4": 400.0},
        hbm_capacity_gb=80, hbm_bandwidth_tb_s=3.0,
    )


def test_roofline_uses_compute_class_peak():
    hw = _hw_with_fp8()
    cost = OpCost(flops=1e12, mem_rw=0)
    t_bf16 = roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16")
    t_fp8 = roofline_time(cost, hw, is_large_gemm=True, compute_class="fp8")
    t_fp4 = roofline_time(cost, hw, is_large_gemm=True, compute_class="fp4")
    assert t_fp8 == pytest.approx(t_bf16 / 2, rel=1e-6)
    assert t_fp4 == pytest.approx(t_bf16 / 4, rel=1e-6)


def test_roofline_default_compute_class_is_bf16():
    hw = _hw_with_fp8()
    cost = OpCost(flops=1e12, mem_rw=0)
    assert roofline_time(cost, hw, is_large_gemm=True) == pytest.approx(
        roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16")
    )


def test_tflops_for_missing_class_falls_back_to_bf16():
    hw = HardwareConfig(
        name="t", peak_tflops_bf16=100.0,
        hbm_capacity_gb=80, hbm_bandwidth_tb_s=3.0,
    )  # no peak_tflops given
    assert hw.tflops_for("fp8") == 100.0  # conservative fallback
    assert hw.tflops_for("bf16") == 100.0
```

Ensure `import pytest` exists at the top of `tests/test_ops.py` (add if missing).

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k "compute_class or tflops_for" -v`
Expected: FAIL — `TypeError: roofline_time() got an unexpected keyword argument 'compute_class'` / `AttributeError: 'HardwareConfig' object has no attribute 'tflops_for'`

- [ ] **Step 3: Write minimal implementation**

In `src/llm_perf/config.py`, add to `HardwareConfig` (after `peak_tflops_bf16` field and before `calibration`):

```python
    peak_tflops: dict[str, float] = {}  # {"bf16":…, "fp8":…, "fp4":…}; bf16 mirrored from peak_tflops_bf16 if absent
```

And add a model-validator + resolver method to `HardwareConfig`:

```python
    def model_post_init(self, __context) -> None:
        # Always mirror the legacy scalar into the dict so callers can use either.
        self.peak_tflops.setdefault("bf16", self.peak_tflops_bf16)

    def tflops_for(self, compute_class: str) -> float:
        """Peak TFLOPS for a compute class; falls back to bf16 peak if absent."""
        if compute_class in self.peak_tflops:
            return self.peak_tflops[compute_class]
        import warnings
        warnings.warn(
            f"peak_tflops['{compute_class}'] missing; falling back to bf16 peak "
            f"(no low-precision speedup credited)",
            stacklevel=2,
        )
        return self.peak_tflops_bf16
```

In `src/llm_perf/ops.py`, change `roofline_time`:

```python
def roofline_time(
    cost: OpCost, hw: HardwareConfig, is_large_gemm: bool = True,
    compute_class: str = "bf16",
) -> float:
    """Roofline model: time = max(compute_time, memory_time).

    Uses two-tier calibration (large GEMM vs small ops) and per-precision peak
    TFLOPS selected by compute_class ("bf16"|"fp8"|"fp4").
    """
    eff = (
        hw.calibration.compute_eff_large_gemm
        if is_large_gemm
        else hw.calibration.compute_eff_small_op
    )
    peak = hw.tflops_for(compute_class)
    compute_time = (
        cost.flops / (peak * 1e12 * eff) if cost.flops > 0 else 0
    )
    memory_time = (
        cost.mem_rw / (hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency)
        if cost.mem_rw > 0
        else 0
    )
    return max(compute_time, memory_time)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_ops.py tests/test_config.py -v`
Expected: PASS, including pre-existing tests (regression check).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/config.py src/llm_perf/ops.py tests/test_ops.py
git commit -m "feat(ops): per-precision peak TFLOPS + roofline compute_class selection"
```

---

### Task 3: Overhead ops in `ops.py` (quantize / Hadamard / dequant / compensation-add)

**Files:**
- Modify: `src/llm_perf/ops.py` (append four op functions)
- Test: `tests/test_ops.py` (append)

**Interfaces:**
- Consumes: `dtype_bytes` from Task 1; `OpCost`, `roofline_time` from `ops.py`.
- Produces (all return `OpCost`):
  - `op_quantize(numel: float, in_dtype: str, out_dtype: str, block_size: int = 0, scale_bytes: int = 4) -> OpCost`
  - `op_dequant(numel: float, in_dtype: str, out_dtype: str, block_size: int = 0, scale_bytes: int = 4) -> OpCost`
  - `op_hadamard(tokens: float, d: int, block: int) -> OpCost`
  - `op_compensation_add(numel: float, dtype: str) -> OpCost`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ops.py  (append)
from llm_perf.ops import op_quantize, op_dequant, op_hadamard, op_compensation_add


def test_quantize_is_memory_bound_and_adds_scale_bytes():
    # bf16 (2B) -> fp8 (1B), 1024 elems, block 128 -> 8 scales*4B
    cost = op_quantize(1024, "bf16", "fp8_e4m3", block_size=128, scale_bytes=4)
    # read 1024*2 + write 1024*1 + scale writes 8*4
    assert cost.mem_rw == pytest.approx(1024 * 2 + 1024 * 1 + 32)
    assert cost.flops > 0  # small reduction work, but tiny
    # memory-bound: compute time must not dominate at typical hw
    assert cost.mem_rw > cost.flops


def test_dequant_mirrors_quantize_direction():
    cost = op_dequant(1024, "fp8_e4m3", "bf16", block_size=128, scale_bytes=4)
    # read 1024*1 + write 1024*2 + read scales 8*4
    assert cost.mem_rw == pytest.approx(1024 * 1 + 1024 * 2 + 32)


def test_hadamard_fwht_flops_scale_with_log_block():
    # FWHT: tokens*d*log2(block) MACs *2 flops
    cost = op_hadamard(tokens=100, d=256, block=128)
    assert cost.flops == pytest.approx(2 * 100 * 256 * 7)  # log2(128)=7
    assert cost.mem_rw > 0


def test_compensation_add_is_elementwise_memory():
    cost = op_compensation_add(2048, "fp16")
    # read EF buffer + read grad + write EF buffer = 3 * numel * 2B
    assert cost.mem_rw == pytest.approx(3 * 2048 * 2)
    assert cost.weight_bytes == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k "quantize or dequant or hadamard or compensation" -v`
Expected: FAIL — `ImportError: cannot import name 'op_quantize'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/llm_perf/ops.py` (import the resolver at top: `from llm_perf.precision import dtype_bytes as _dtype_bytes`):

```python
def op_quantize(
    numel: float, in_dtype: str, out_dtype: str,
    block_size: int = 0, scale_bytes: int = 4,
) -> OpCost:
    """Quantize: per-block scale reduction + cast. Memory-bound element-wise.

    Reads numel*in_bytes, writes numel*out_bytes, plus scale-metadata writes.
    FLOPs ~ a few per element (max-abs reduce + divide). Ref: fine-grained
    block quantization (DeepSeek-V3); cost dominated by HBM traffic.
    """
    in_b = _dtype_bytes(in_dtype)
    out_b = _dtype_bytes(out_dtype)
    if block_size <= 0:
        scale_b = scale_bytes
    else:
        scale_b = (-(-int(numel) // block_size)) * scale_bytes
    mem_rw = numel * in_b + numel * out_b + scale_b
    flops = 3 * numel  # abs, compare/reduce, divide
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_dequant(
    numel: float, in_dtype: str, out_dtype: str,
    block_size: int = 0, scale_bytes: int = 4,
) -> OpCost:
    """Dequantize: upcast + apply scale. Memory-bound mirror of op_quantize."""
    in_b = _dtype_bytes(in_dtype)
    out_b = _dtype_bytes(out_dtype)
    if block_size <= 0:
        scale_b = scale_bytes
    else:
        scale_b = (-(-int(numel) // block_size)) * scale_bytes
    mem_rw = numel * in_b + numel * out_b + scale_b
    flops = numel  # one multiply per element
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_hadamard(tokens: float, d: int, block: int) -> OpCost:
    """Stochastic Hadamard via Fast Walsh-Hadamard Transform (FWHT).

    FWHT over a block of size B costs B*log2(B) MACs; applied to tokens*d/B
    blocks => tokens*d*log2(B) MACs. Ref: QuaRot / stochastic Hadamard rotation
    for outlier-free low-precision training.
    """
    import math
    b = block if block and block > 1 else 128
    macs = tokens * d * math.log2(b)
    flops = 2 * macs
    in_b = 2  # operates on bf16/fp16 activations before quant
    mem_rw = 2 * tokens * d * in_b  # read + write the rotated tensor
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_compensation_add(numel: float, dtype: str) -> OpCost:
    """Error-feedback residual add: ef_buf += (x - dequant(quant(x))).

    Memory-bound: read EF buffer + read residual + write EF buffer.
    Ref: error feedback / residual accumulation in low-precision training.
    """
    b = _dtype_bytes(dtype)
    mem_rw = 3 * numel * b
    flops = numel
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -v`
Expected: PASS (overhead-op tests + existing ops tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/ops.py tests/test_ops.py
git commit -m "feat(ops): quantize/dequant/Hadamard/compensation overhead ops"
```

---

### Task 4: Simulator fabric-level bandwidth contention

**Files:**
- Modify: `src/llm_perf/builder.py:82-91` (add `fabric` field to `SimOp`)
- Modify: `src/llm_perf/simulator.py:90-106` (per-fabric clock; per-fabric exposed breakdown)
- Test: `tests/test_simulator.py` (append)

**Interfaces:**
- Consumes: `SimOp` (existing).
- Produces:
  - `SimOp.fabric: Optional[str] = None` (`"nvlink"` | `"nic"` | None for compute).
  - `simulate()` respects a per-fabric shared clock for ops whose `fabric is not None`.
  - `SimResult.exposed_comm_by_fabric: dict[str, float]` — wall-time on each fabric not hidden under compute.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulator.py  (append)
from llm_perf.builder import SimOp
from llm_perf.simulator import simulate


def test_same_fabric_collectives_serialize():
    # Two independent NIC collectives (no dep) must time-share the NIC:
    # wall on nic = sum of durations, not max.
    ops = [
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
        SimOp(name="ep", stream="ep_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(2.0)


def test_different_fabric_collectives_overlap():
    # NVLink (intra) + NIC (inter), independent → run in parallel.
    ops = [
        SimOp(name="tp", stream="tp_comm", duration=1.0, depends_on=[], fabric="nvlink"),
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(1.0)


def test_comm_still_overlaps_compute():
    # A NIC collective independent of a compute op overlaps it (fabric != compute).
    ops = [
        SimOp(name="bwd", stream="compute", duration=2.0, depends_on=[], fabric=None),
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(2.0)  # comm hidden under compute


def test_exposed_comm_by_fabric_reported():
    ops = [
        SimOp(name="bwd", stream="compute", duration=0.5, depends_on=[], fabric=None),
        SimOp(name="dp", stream="dp_comm", duration=2.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    # 2.0 nic comm, only 0.5 hidden → 1.5 exposed on nic
    assert res.exposed_comm_by_fabric["nic"] == pytest.approx(1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_simulator.py -k "fabric or overlaps_compute" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'fabric'`

- [ ] **Step 3: Write minimal implementation**

In `src/llm_perf/builder.py`, add to the `SimOp` dataclass (after `consumers`):

```python
    fabric: Optional[str] = None  # "nvlink" (intra-node) | "nic" (inter-node) | None for compute
```

In `src/llm_perf/simulator.py`, extend `SimResult` with a field:

```python
    exposed_comm_by_fabric: Dict[str, float] = field(default_factory=dict)
```

(add `from dataclasses import dataclass, field` import if `field` is missing.)

Replace the multi-clock loop (lines ~90-106) with a fabric-aware version:

```python
    stream_clock: Dict[str, float] = defaultdict(float)
    fabric_clock: Dict[str, float] = defaultdict(float)
    stream_durations: Dict[str, float] = defaultdict(float)
    fabric_durations: Dict[str, float] = defaultdict(float)
    finish_time = [0.0] * n

    for idx in topo_order:
        op = ops[idx]
        dep_max = max((finish_time[d] for d in op.depends_on), default=0.0)
        floor = max(stream_clock[op.stream], dep_max)
        if op.fabric is not None:
            floor = max(floor, fabric_clock[op.fabric])
        start = floor
        finish_time[idx] = start + op.duration
        stream_clock[op.stream] = finish_time[idx]
        stream_durations[op.stream] += op.duration
        if op.fabric is not None:
            fabric_clock[op.fabric] = finish_time[idx]
            fabric_durations[op.fabric] += op.duration

    wall_clock = max(stream_clock.values()) if stream_clock else 0.0
    compute_time = stream_durations.get("compute", 0.0)
    tp_comm_time = stream_durations.get("tp_comm", 0.0)
    ep_comm_time = stream_durations.get("ep_comm", 0.0)
    dp_comm_time = stream_durations.get("dp_comm", 0.0)
    cp_comm_time = stream_durations.get("cp_comm", 0.0)

    # Exposed comm per fabric = fabric busy-time not hidden under compute.
    exposed_comm_by_fabric: Dict[str, float] = {}
    for fab, busy in fabric_durations.items():
        exposed_comm_by_fabric[fab] = max(0.0, fabric_clock[fab] - compute_time) \
            if fabric_clock[fab] > compute_time else max(0.0, busy - compute_time)
```

Then pass `exposed_comm_by_fabric=exposed_comm_by_fabric` into the returned `SimResult(...)`.

> Implementation note for the exposed-comm formula: the intent is "fabric busy-time that is not covered by concurrent compute." Use `exposed = max(0.0, fabric_clock[fab] - compute_time)` as the primary estimate (fabric finishes after compute → the tail is exposed). The test `test_exposed_comm_by_fabric_reported` pins the 1.5 case; keep the formula that satisfies it: `exposed_comm_by_fabric[fab] = max(0.0, fabric_durations[fab] - compute_time)`.

Simplify the block to exactly:

```python
    exposed_comm_by_fabric = {
        fab: max(0.0, busy - compute_time)
        for fab, busy in fabric_durations.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_simulator.py -v`
Expected: PASS, including the existing DP-overlap test (regression).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py src/llm_perf/simulator.py tests/test_simulator.py
git commit -m "feat(sim): per-fabric bandwidth contention + exposed-comm-by-fabric"
```

---

### Task 5: Tag existing comm ops with fabric in builder

**Files:**
- Modify: `src/llm_perf/builder.py` (`_build_tp_comm`, CP/EP/DP comm SimOp construction)
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Consumes: `SimOp.fabric` (Task 4), `_is_intra_node` (existing).
- Produces: every comm `SimOp` emitted by the builder carries `fabric="nvlink"` when its group is intra-node and `fabric="nic"` when inter-node. Add a helper `_fabric(group_size, hw) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
from llm_perf.builder import build_training_step
from llm_perf.config import load_model_config, load_hardware_config, ParallelismConfig, WorkloadConfig


def test_comm_ops_are_fabric_tagged():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    # dp spans nodes (dp=16 > devices_per_node) → DP grad sync on nic
    pc = ParallelismConfig(tp=4, dp=16)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc, rl)
    comm_ops = [o for o in ops if o.stream.endswith("_comm")]
    assert comm_ops, "expected some comm ops"
    assert all(o.fabric in ("nvlink", "nic") for o in comm_ops)
    dp_ops = [o for o in ops if o.stream == "dp_comm"]
    assert dp_ops and all(o.fabric == "nic" for o in dp_ops)  # dp=16 inter-node
```

(Confirm `devices_per_node` in `configs/hardware/ascend_910c.yaml`; if it is ≥16, raise `dp` in the test so the group is genuinely inter-node. Adjust the `tp`/`dp` values to match the real `devices_per_node` while keeping `dp > devices_per_node`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k fabric_tagged -v`
Expected: FAIL — `assert all(... o.fabric in (...))` fails because comm ops have `fabric=None`.

- [ ] **Step 3: Write minimal implementation**

Add helper near `_is_intra_node` in `builder.py`:

```python
def _fabric(group_size: int, hw: HardwareConfig) -> str:
    """Physical fabric a collective traverses: intra-node NVLink/HCCS vs inter-node NIC."""
    return "nvlink" if _is_intra_node(group_size, hw) else "nic"
```

In `_build_tp_comm`, set `fabric=_fabric(tp, hw)` on the returned `SimOp`. For each other comm `SimOp` construction (CP ring → `_fabric(cp, hw)`; EP all-to-all → `_fabric(ep, hw)`; DP grad sync at builder.py:949 → `_fabric(dp, hw)`; the DSA index allreduce at builder.py:363 → `_fabric(tp, hw)`), add the matching `fabric=...` kwarg.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS (regression-safe; only adds a field value).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): tag TP/CP/EP/DP comm ops with physical fabric"
```

---

### Task 6: Inject quant chain around low-precision GEMMs

**Files:**
- Modify: `src/llm_perf/builder.py` (`build_layer_ops` — thread `precision_cfg`; add `_inject_quant_chain` helper)
- Modify: `src/llm_perf/builder.py` (`build_training_step` signature — accept `precision_cfg`)
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Consumes: `PrecisionConfig`, `compute_class`, `dtype_bytes` (Task 1); overhead ops (Task 3); `roofline_time(compute_class=...)` (Task 2).
- Produces:
  - `build_layer_ops(..., precision_cfg: Optional[PrecisionConfig] = None)` — default `None` resolves to `PrecisionConfig.bf16_default()`.
  - `build_training_step(..., precision_cfg: Optional[PrecisionConfig] = None)`.
  - When a GEMM runs below bf16 (activations or weights compute_class != "bf16"), the DAG contains, before the matmul: a `quantize_*` op (+ `hadamard_*` if `activations.hadamard`), and after it a `dequant_*` op — all on the `compute` stream, correctly chained.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
from llm_perf.precision import PrecisionConfig, TensorPrecision


def _fp8_precision():
    return PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
        activations=TensorPrecision(dtype="fp8_e4m3", block_size=128, hadamard=True, hadamard_block=128),
    )


def test_lowprec_gemm_injects_quant_chain():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops = build_training_step(model, hw, pc, rl, precision_cfg=_fp8_precision())
    names = [o.name for o in ops]
    assert any(n.startswith("quantize") for n in names)
    assert any(n.startswith("hadamard") for n in names)
    assert any(n.startswith("dequant") for n in names)


def test_bf16_default_injects_no_quant_ops():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    ops_default = build_training_step(model, hw, pc, rl)  # no precision_cfg
    assert not any(o.name.startswith(("quantize", "hadamard", "dequant")) for o in ops_default)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k "quant_chain or no_quant" -v`
Expected: FAIL — `build_training_step() got an unexpected keyword argument 'precision_cfg'`.

- [ ] **Step 3: Write minimal implementation**

Thread `precision_cfg: Optional[PrecisionConfig] = None` through `build_training_step` and `build_layer_ops`; at the top of each, `pc = precision_cfg or PrecisionConfig.bf16_default()`. Add a helper that wraps a compute GEMM SimOp with its quant chain:

```python
def _inject_quant_chain(
    result: List[SimOp], gemm_op: SimOp, numel: float, tokens: float, d: int,
    pc: PrecisionConfig, hw: HardwareConfig, index_offset: int, dep_idx: int,
) -> int:
    """Prepend quantize(+hadamard) and append dequant around gemm_op.

    Returns the global index of the dequant op (the new chain tail). When the
    activation compute_class is bf16 (no low precision), appends gemm_op as-is
    and returns its index.
    """
    from llm_perf.precision import compute_class, dtype_bytes
    act = pc.activations
    if compute_class(act.dtype) == "bf16":
        gemm_op.depends_on = [dep_idx] if dep_idx is not None else []
        result.append(gemm_op)
        return index_offset + len(result) - 1

    prev = dep_idx
    q = SimOp(
        name="quantize_act",
        stream="compute",
        duration=ops.roofline_time(
            ops.op_quantize(numel, "bf16", act.dtype, act.block_size, act.scale_bytes),
            hw, is_large_gemm=False),
        depends_on=[prev] if prev is not None else [],
    )
    result.append(q)
    prev = index_offset + len(result) - 1

    if act.hadamard:
        h = SimOp(
            name="hadamard_act",
            stream="compute",
            duration=ops.roofline_time(
                ops.op_hadamard(tokens, d, act.hadamard_block), hw, is_large_gemm=False),
            depends_on=[prev],
        )
        result.append(h)
        prev = index_offset + len(result) - 1

    gemm_op.depends_on = [prev]
    result.append(gemm_op)
    prev = index_offset + len(result) - 1

    dq = SimOp(
        name="dequant_out",
        stream="compute",
        duration=ops.roofline_time(
            ops.op_dequant(numel, act.dtype, "bf16", act.block_size, act.scale_bytes),
            hw, is_large_gemm=False),
        depends_on=[prev],
    )
    result.append(dq)
    return index_offset + len(result) - 1
```

In `build_layer_ops`, route the attention-projection and FFN GEMMs through `_inject_quant_chain` (use the existing per-op `batch_tokens` for `tokens`, `d` for hidden, and `batch_tokens * d` for `numel`). Pass each GEMM's `compute_class` to its `roofline_time` call. Keep RMSNorm/residual untouched (they are not quantized GEMMs). Preserve the existing `depends_on` chaining by using the returned tail index as the next op's dependency.

> Keep the change surgical: the simplest correct integration wraps the FFN GEMM (the dominant matmul) first to satisfy the test, then extends to attention projections. Do FFN in this task; attention projections fold in here too if straightforward, otherwise leave a follow-up note (no separate task needed — same pattern).

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS; existing builder tests unaffected (bf16 default path unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): inject quant/Hadamard/dequant chain around low-precision GEMMs"
```

---

### Task 7: Low-precision communication (reduced bytes + quant/dequant around collectives)

**Files:**
- Modify: `src/llm_perf/builder.py` (DP grad sync at ~949; comm-byte sizing uses `comm.dtype`)
- Test: `tests/test_builder.py` (append)

**Interfaces:**
- Consumes: `PrecisionConfig.comm`, `dtype_bytes` (Task 1); overhead ops (Task 3).
- Produces: when `pc.comm.dtype` has fewer bytes than the gradient dtype, the DP grad-sync `comm_bytes` is scaled by `dtype_bytes(comm.dtype)/dtype_bytes(grad_dtype)`, and a `quantize_grad` op precedes + `dequant_grad` follows the collective on the `compute` stream.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
def test_fp8_comm_halves_dp_grad_bytes_and_adds_quant():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)

    bf16_comm = PrecisionConfig.bf16_default()
    fp8_comm = PrecisionConfig(comm=TensorPrecision(dtype="fp8_e4m3"))

    ops_bf16 = build_training_step(model, hw, pc, rl, precision_cfg=bf16_comm)
    ops_fp8 = build_training_step(model, hw, pc, rl, precision_cfg=fp8_comm)

    dp_bytes_bf16 = sum(o.comm_bytes for o in ops_bf16 if o.name == "dp_grad_sync")
    dp_bytes_fp8 = sum(o.comm_bytes for o in ops_fp8 if o.name == "dp_grad_sync")
    assert dp_bytes_fp8 == pytest.approx(dp_bytes_bf16 / 2, rel=1e-6)
    assert any(o.name.startswith("quantize_grad") for o in ops_fp8)
    assert any(o.name.startswith("dequant_grad") for o in ops_fp8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k fp8_comm -v`
Expected: FAIL — fp8 dp bytes equal bf16 (no scaling) and no `quantize_grad` op.

- [ ] **Step 3: Write minimal implementation**

In `build_training_step`'s DP-sync block (builder.py ~930-959): compute `grad_bytes` using `dtype_bytes(pc.gradients.dtype)` for the local gradient and size the collective message with `dtype_bytes(pc.comm.dtype)`. When `compute_class(pc.comm.dtype) != "bf16"`, insert a `quantize_grad` SimOp (`op_quantize(layer_param_count, pc.gradients.dtype, pc.comm.dtype, ...)`) on the `compute` stream depending on the layer's last backward op, make the `dp_grad_sync` op depend on it, and append a `dequant_grad` SimOp after on the `compute` stream depending on the dp_sync index. The dp_sync op keeps `stream="dp_comm"` and now carries the reduced `comm_bytes` and `fabric=_fabric(dp, hw)` (from Task 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS; bf16 default path keeps full-precision comm bytes (regression).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): low-precision gradient comm (reduced bytes + quant around collective)"
```

---

### Task 8: Compensation buffers + mixed/periodic high-precision

**Files:**
- Modify: `src/llm_perf/builder.py` (error-feedback buffer memory + compensation-add op; mixed-precision per-layer skip)
- Modify: `src/llm_perf/model.py` (`high_precision_period` step blend in training time)
- Test: `tests/test_builder.py`, `tests/test_model.py` (append)

**Interfaces:**
- Consumes: `PrecisionConfig` fields `error_feedback`, `ef_dtype`, `high_precision_layers`, `high_precision_period`, `high_precision_dtype`; `op_compensation_add` (Task 3).
- Produces:
  - When `error_feedback=True`, each quantized weight tensor gets a persistent EF buffer counted in `weight_bytes` (`numel * dtype_bytes(ef_dtype)`) plus one `compensation_add` op per step.
  - When a layer component name is in `high_precision_layers`, that component's GEMM uses `high_precision_dtype` (compute_class bf16) and skips the quant chain.
  - `LLMPerformanceModel` training-time blends low/high-precision step when `high_precision_period=N>0`: `t = (1-1/N)*t_low + (1/N)*t_high`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_builder.py  (append)
def test_error_feedback_adds_buffer_and_compensation_op():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
    pc = ParallelismConfig(tp=1, dp=1)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    no_ef = PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3"))
    with_ef = PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3"),
                              error_feedback=True, ef_dtype="fp16")
    w_no = sum(o.weight_bytes for o in build_training_step(model, hw, pc, rl, precision_cfg=no_ef))
    w_ef = sum(o.weight_bytes for o in build_training_step(model, hw, pc, rl, precision_cfg=with_ef))
    assert w_ef > w_no
    assert any(o.name.startswith("compensation_add")
               for o in build_training_step(model, hw, pc, rl, precision_cfg=with_ef))
```

```python
# tests/test_model.py  (append)
def test_periodic_high_precision_blends_step_time():
    # With period N=4, training time is between all-low and all-high.
    # Pin via the model API; see existing test_model.py setup for fixtures.
    ...  # construct LLMPerformanceModel with a fp8 recipe + high_precision_period=4
    # assert t_blend == pytest.approx(0.75*t_low + 0.25*t_high, rel=1e-3)
```

Fill the `test_model.py` body using the same fixture pattern already in that file (load model+hw, build `LLMPerformanceModel`, call the training-time path with and without `high_precision_period`).

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k error_feedback tests/test_model.py -k periodic -v`
Expected: FAIL — no EF buffer/compensation op; no period blend.

- [ ] **Step 3: Write minimal implementation**

In `build_training_step`: after building backward layers, when `pc.error_feedback` and weights are low-precision, append per-stage a persistent EF buffer by adding `weight_bytes += param_count * dtype_bytes(pc.ef_dtype)` on a dedicated zero-duration `SimOp(name="ef_buffer", stream="compute", duration=0, weight_bytes=...)`, plus one `SimOp(name="compensation_add", ...)` with duration from `op_compensation_add(param_count, pc.ef_dtype)`. In `_inject_quant_chain`/layer build, when the component name ∈ `pc.high_precision_layers`, take the bf16 branch regardless of role dtype. In `model.py` training-time computation, when `pc.high_precision_period == N > 0`, simulate a high-precision step (precision_cfg with all roles set to `high_precision_dtype`) and a low-precision step, and return `(1 - 1/N)*t_low + (1/N)*t_high`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_builder.py tests/test_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/builder.py src/llm_perf/model.py tests/test_builder.py tests/test_model.py
git commit -m "feat(builder/model): error-feedback buffers, mixed-layer + periodic high-precision"
```

---

### Task 9: Reporting — per-precision/per-fabric breakdown + `compare_precision` helper

**Files:**
- Modify: `src/llm_perf/model.py` (thread `precision_cfg` into `LLMPerformanceModel`; surface fabric/precision breakdown in `TargetReport`)
- Modify: `src/llm_perf/report.py` (render the new lines)
- Create: `examples/demo_low_precision.py`
- Test: `tests/test_model.py`, `tests/test_report.py` (append)

**Interfaces:**
- Consumes: `SimResult.exposed_comm_by_fabric` (Task 4); per-precision compute split (Task 6); `PrecisionConfig` (Task 1).
- Produces:
  - `LLMPerformanceModel(..., precision_cfg: Optional[PrecisionConfig] = None)`.
  - `TargetReport` gains `exposed_comm_by_fabric: dict[str, float]`, `quant_overhead_seconds: float`, `compute_seconds_by_class: dict[str, float]`.
  - `compare_precision(model_cfg, hw, parallel_cfg, rl_cfg, recipes: dict[str, PrecisionConfig]) -> list[dict]` returning per-recipe `{name, step_seconds, speedup_vs_bf16, comm_bytes, comm_reduction_pct, exposed_comm_by_fabric, peak_memory_gb, feasible}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (append)
from llm_perf.model import compare_precision
from llm_perf.precision import PrecisionConfig, TensorPrecision


def test_compare_precision_orders_speedup_and_memory():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")  # must define peak_tflops fp8/fp4
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    recipes = {
        "bf16": PrecisionConfig.bf16_default(),
        "fp8": PrecisionConfig(
            weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
            activations=TensorPrecision(dtype="fp8_e4m3", block_size=128),
            comm=TensorPrecision(dtype="fp8_e4m3"),
        ),
    }
    rows = compare_precision(model, hw, pc, rl, recipes)
    by_name = {r["name"]: r for r in rows}
    assert by_name["bf16"]["speedup_vs_bf16"] == pytest.approx(1.0)
    assert by_name["fp8"]["speedup_vs_bf16"] >= 1.0
    assert by_name["fp8"]["comm_reduction_pct"] > 0
    assert "nic" in by_name["fp8"]["exposed_comm_by_fabric"] or \
           "nvlink" in by_name["fp8"]["exposed_comm_by_fabric"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_model.py -k compare_precision -v`
Expected: FAIL — `ImportError: cannot import name 'compare_precision'`.

- [ ] **Step 3: Write minimal implementation**

Add `compare_precision(...)` to `model.py` that runs the existing training-step simulation per recipe (passing `precision_cfg`), pulls `wall_clock_time`, `total_comm_bytes`, `exposed_comm_by_fabric`, and peak memory, computes `speedup_vs_bf16 = t_bf16 / t_recipe` and `comm_reduction_pct = (1 - bytes/bytes_bf16)*100`, and returns the list of dicts. Surface `quant_overhead_seconds` by summing durations of ops whose name starts with `("quantize","hadamard","dequant","compensation")`, and `compute_seconds_by_class` by tagging each compute GEMM SimOp with its class (add an optional `op_class` field to `SimOp` or derive from name). Extend `TargetReport` and `report.py` rendering with the three new fields. Write `examples/demo_low_precision.py` mirroring `examples/demo.py` but calling `compare_precision` and printing the table.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: PASS (full suite — final regression gate).

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/model.py src/llm_perf/report.py examples/demo_low_precision.py tests/test_model.py tests/test_report.py
git commit -m "feat(report): per-precision/per-fabric breakdown + compare_precision helper + demo"
```

---

## Final verification

- [ ] Run the full suite and lint:

```bash
source .venv/bin/activate && pytest tests/ -v && ruff check src/ && ruff format --check src/
```

Expected: all tests pass; ruff clean.

- [ ] Update `docs/architecture.md`: remove the stale "No overlapping communication" limitation; document the fabric-contention model and the precision-aware cost path. Add a `CHANGELOG.md` entry. Commit:

```bash
git add docs/architecture.md CHANGELOG.md
git commit -m "docs: fabric-contention + low-precision cost model; drop stale overlap limitation"
```

---

## Self-review notes (coverage map)

- Spec A (PrecisionConfig) → Task 1. Spec B (per-precision TFLOPS) → Task 2.
  Spec C (overhead ops + precision-aware bytes) → Tasks 3, 6. Spec D (builder
  injection + comm precision + fabric tags) → Tasks 5, 6, 7, 8. Spec E (fabric
  contention) → Task 4. Spec F (reporting + compare_precision + periodic blend)
  → Tasks 8, 9.
- Regression safety asserted in Tasks 2, 4, 6, 7 (bf16-default path unchanged)
  and the final full-suite gate.
- A hardware config with `peak_tflops` for fp8/fp4 must exist for Tasks 2 and 9;
  if `configs/hardware/ascend_910c.yaml` lacks them, add the keys (Task 2,
  Step 3) — keep `peak_tflops_bf16` as the bf16 mirror.
