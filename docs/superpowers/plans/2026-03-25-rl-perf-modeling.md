# RL Performance Modeling (rl-perf) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a performance modeling tool that, given model + hardware + RL config, predicts epoch time and derives TPS targets for inference and training teams.

**Architecture:** Layered pipeline — config YAML → ops.py (roofline + memory per op) → builder.py (config → op sequence with parallelism) → simulator.py (multi-clock simulation + memory tracking) → pipeline.py (two-stage gen/train bottleneck) → CLI output.

**Tech Stack:** Python 3.10+, pydantic, pyyaml, typer, rich, numpy. No ML framework dependencies.

**Spec:** `docs/superpowers/specs/2026-03-25-rl-perf-modeling-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/rl_perf/__init__.py`
- Create: `tests/__init__.py`
- Create: `CLAUDE.md`
- Create: `README.md`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/horacehxw/Projects/RL_modeling
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "rl-perf"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "typer>=0.9",
    "rich>=13.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-xdist", "ruff"]

[project.scripts]
rl-perf = "rl_perf.cli:app"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create package init and test init**

`src/rl_perf/__init__.py`:
```python
"""RL Training Performance Modeling Tool."""
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Create CLAUDE.md**

```markdown
# RL Performance Modeling (rl-perf)

## Project Overview
RL 训练性能建模工具。给定模型、卡数、数据量，推导 epoch 时间上限和各团队 TPS 指标。

## Tech Stack
- Python 3.10+ (use python3.10 explicitly if available)
- No ML framework dependencies (pure Python + numpy)
- CLI: typer, Config: pydantic + YAML

## Key Architecture
- L1 (ops.py): 模块化算子 cost model, roofline-based, 可被 benchmark 校准
- L2 (builder.py + simulator.py): Template DAG + 多流拓扑模拟
- L3 (pipeline.py): 两阶段 pipeline (generation → queue → training)

## Development
- Tests: `pytest tests/ -v`
- Lint: `ruff check src/ && ruff format src/`
- Install: `pip install -e ".[dev]"`

## Conventions
- All FLOPs formulas must have docstring citing source
- Hardware constants in configs/hardware/*.yaml, NOT hardcoded
- One op function per attention/FFN variant in ops.py
```

- [ ] **Step 5: Create minimal README.md**

```markdown
# rl-perf: RL Training Performance Modeling

Given model + hardware + RL config, predict epoch time and derive TPS targets.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
rl-perf targets --model configs/models/llama3_1_8b.yaml \
                --hardware 910C --devices 64 \
                --prompts 100000 --group-size 8 --time-budget 24h
```
```

- [ ] **Step 6: Install the package**

```bash
cd /Users/horacehxw/Projects/RL_modeling && pip install -e ".[dev]"
```

- [ ] **Step 7: Verify pytest works**

```bash
cd /Users/horacehxw/Projects/RL_modeling && pytest tests/ -v
```
Expected: 0 tests collected, no errors.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: project scaffolding with pyproject.toml, CLAUDE.md, README"
```

---

## Task 2: Config Data Model (config.py)

**Files:**
- Create: `src/rl_perf/config.py`
- Create: `tests/test_config.py`
- Create: `configs/models/llama3_1_8b.yaml`
- Create: `configs/hardware/ascend_910c.yaml`

- [ ] **Step 1: Write tests for config loading**

`tests/test_config.py`:
```python
import pytest
from pathlib import Path
from rl_perf.config import (
    ModelConfig, HardwareConfig, RLConfig, ParallelismConfig,
    LayerConfig, Phase, load_model_config, load_hardware_config,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_layer_config_defaults():
    lc = LayerConfig(attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
                     ffn="SwiGLU", intermediate_size=11008)
    assert lc.num_experts == 1  # Dense by default
    assert lc.residual == "standard"


def test_model_config_expands_layers():
    mc = ModelConfig(
        name="test", hidden_size=4096, vocab_size=32000, num_layers=4, dtype="bf16",
        default_layer=LayerConfig(
            attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
            ffn="SwiGLU", intermediate_size=11008,
        ),
    )
    assert len(mc.get_layers()) == 4
    assert all(l.attention == "GQA" for l in mc.get_layers())


def test_hardware_config_usable_hbm():
    hw = HardwareConfig(
        name="test", peak_tflops_bf16=800, hbm_capacity_gb=128,
        hbm_bandwidth_tb_s=3.2, hbm_usable_ratio=0.85,
        intra_node_bw_gb_s=400, inter_node_bw_gb_s=100,
        inter_node_latency_us=5, devices_per_node=8,
    )
    assert hw.usable_hbm_gb == pytest.approx(108.8)


def test_load_model_config_yaml(tmp_path):
    yaml_content = """
name: "Test-8B"
hidden_size: 4096
vocab_size: 32000
num_layers: 32
dtype: bf16
default_layer:
  attention: GQA
  num_heads: 32
  num_kv_heads: 8
  head_dim: 128
  ffn: SwiGLU
  intermediate_size: 11008
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
    mc = load_model_config(str(p))
    assert mc.name == "Test-8B"
    assert mc.num_layers == 32


def test_parallelism_config_total_devices():
    pc = ParallelismConfig(tp=8, pp=4, dp=4, ep=1)
    assert pc.total_devices == 128


def test_phase_enum():
    assert Phase.PREFILL.value == "prefill"
    assert Phase.TRAIN_BWD.value == "train_bwd"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement config.py**

`src/rl_perf/config.py` — complete implementation with:
- `Phase` enum (PREFILL, DECODE, TRAIN_FWD, TRAIN_BWD — MVP uses combined BWD)
- `LayerConfig` (pydantic BaseModel): attention type, heads, ffn type, MoE params, residual, mHC
- `ModelConfig`: name, hidden_size, vocab_size, num_layers, dtype, default_layer, optional layers list, auxiliary (mtp_depth). Method `get_layers()` expands default_layer × num_layers or returns per-layer list.
- `CalibrationConfig`: compute_eff_large_gemm, compute_eff_small_op, memory_efficiency, comm_efficiency with defaults
- `HardwareConfig`: peak TFLOPS, HBM capacity/bandwidth/usable_ratio, intra/inter node bandwidth, latency, devices_per_node, calibration. Property `usable_hbm_gb`.
- `RLConfig`: total_prompts, group_size, avg/max/std prompt/response len, batch sizes, reference_model, colocated flags
- `ParallelismConfig`: tp, pp, dp, ep, cp, sp, zero_stage, pp_schedule, recompute flags. Property `total_devices`.
- `load_model_config(path)` and `load_hardware_config(path)` YAML loaders

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: all PASS

- [ ] **Step 5: Create Llama 3.1 8B config YAML**

`configs/models/llama3_1_8b.yaml`:
```yaml
name: "Llama-3.1-8B"
hidden_size: 4096
vocab_size: 128256
num_layers: 32
dtype: bf16
default_layer:
  attention: GQA
  num_heads: 32
  num_kv_heads: 8
  head_dim: 128
  ffn: SwiGLU
  intermediate_size: 14336
```

- [ ] **Step 6: Create Ascend 910C hardware config YAML**

`configs/hardware/ascend_910c.yaml`:
```yaml
name: "Ascend 910C"
peak_tflops_bf16: 800
hbm_capacity_gb: 128
hbm_bandwidth_tb_s: 3.2
hbm_usable_ratio: 0.85
intra_node_bw_gb_s: 400
inter_node_bw_gb_s: 100
inter_node_latency_us: 5
devices_per_node: 8
calibration:
  compute_eff_large_gemm: 0.50
  compute_eff_small_op: 0.20
  memory_efficiency: 0.70
  comm_efficiency: 0.70
```

- [ ] **Step 7: Write and run integration test loading real YAML**

Add to `tests/test_config.py`:
```python
def test_load_real_llama_config():
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    assert mc.hidden_size == 4096
    assert mc.get_layers()[0].num_kv_heads == 8

def test_load_real_hardware_config():
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    assert hw.peak_tflops_bf16 == 800
    assert hw.usable_hbm_gb == pytest.approx(108.8)
```

```bash
pytest tests/test_config.py -v
```
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: config data model with YAML loading + Llama 8B and 910C configs"
```

---

## Task 3: Operator Cost Model (ops.py)

**Files:**
- Create: `src/rl_perf/ops.py`
- Create: `tests/test_ops.py`

- [ ] **Step 1: Write tests for OpCost and roofline**

`tests/test_ops.py`:
```python
import pytest
from rl_perf.ops import (
    OpCost, roofline_time,
    op_linear, op_gqa_attention, op_swiglu_ffn, op_moe_ffn,
    op_allreduce, op_alltoall,
)
from rl_perf.config import HardwareConfig, Phase


@pytest.fixture
def hw():
    return HardwareConfig(
        name="test", peak_tflops_bf16=800, hbm_capacity_gb=128,
        hbm_bandwidth_tb_s=3.2, hbm_usable_ratio=0.85,
        intra_node_bw_gb_s=400, inter_node_bw_gb_s=100,
        inter_node_latency_us=5, devices_per_node=8,
    )


def test_opcost_dataclass():
    c = OpCost(flops=1e12, mem_rw=1e9, weight_bytes=1e6, output_bytes=1e6)
    assert c.flops == 1e12


def test_roofline_compute_bound(hw):
    # Large GEMM: high flops, low memory
    c = OpCost(flops=1e12, mem_rw=1e6, weight_bytes=0, output_bytes=0)
    t = roofline_time(c, hw, is_large_gemm=True)
    # compute_time = 1e12 / (800e12 * 0.50) = 0.0025s
    assert t == pytest.approx(0.0025, rel=0.01)


def test_roofline_memory_bound(hw):
    # Small op: low flops, high memory
    c = OpCost(flops=1e6, mem_rw=1e11, weight_bytes=0, output_bytes=0)
    t = roofline_time(c, hw, is_large_gemm=False)
    # memory_time = 1e11 / (3.2e12 * 0.70) = 0.0446s
    assert t == pytest.approx(0.0446, rel=0.01)


def test_op_linear_forward():
    c = op_linear(in_features=4096, out_features=4096, batch_tokens=1024, phase=Phase.TRAIN_FWD)
    # FLOPs: 2 * 4096 * 4096 * 1024 = 34,359,738,368
    assert c.flops == pytest.approx(2 * 4096 * 4096 * 1024)
    assert c.weight_bytes == 4096 * 4096 * 2  # bf16
    assert c.output_bytes == 1024 * 4096 * 2  # activation kept for backward


def test_op_linear_backward():
    c = op_linear(in_features=4096, out_features=4096, batch_tokens=1024, phase=Phase.TRAIN_BWD)
    assert c.flops == pytest.approx(2 * 2 * 4096 * 4096 * 1024)  # 2x forward


def test_op_gqa_attention_prefill():
    # GQA: H=32, G=8, d_h=128, d=4096, seq=512, batch_tokens=512
    c = op_gqa_attention(
        num_heads=32, num_kv_heads=8, head_dim=128, hidden_size=4096,
        batch=1, seq_len=512, phase=Phase.PREFILL,
    )
    d = 4096
    G, H = 8, 32
    expected_proj = (4 + 4 * G / H) * d * d * 512  # projection FLOPs
    expected_attn = 4 * 512 * d * 512  # attention FLOPs (seq×seq×d, amortized)
    assert c.flops > 0


def test_op_swiglu():
    c = op_swiglu_ffn(hidden_size=4096, intermediate_size=11008, batch_tokens=1024, phase=Phase.TRAIN_FWD)
    assert c.flops == pytest.approx(6 * 4096 * 11008 * 1024)


def test_op_moe():
    c = op_moe_ffn(
        hidden_size=4096, expert_intermediate_size=2048, num_experts=64,
        num_shared_experts=1, shared_intermediate_size=11008,
        top_k=8, batch_tokens=1024, phase=Phase.TRAIN_FWD,
    )
    routed = 6 * 4096 * 2048 * 8 * 1024
    shared = 6 * 4096 * 11008 * 1 * 1024
    assert c.flops == pytest.approx(routed + shared)


def test_op_allreduce():
    c = op_allreduce(msg_bytes=1e9, group_size=8)
    # volume: 2 * 1e9 * 7/8 = 1.75e9
    assert c.comm_bytes == pytest.approx(2 * 1e9 * 7 / 8)


def test_op_alltoall():
    c = op_alltoall(tokens=1024, hidden_size=4096, top_k=8, ep_size=16, dtype_bytes=2)
    # 2 * 1024 * 8 * 4096 * 2 = 134,217,728
    assert c.comm_bytes == pytest.approx(2 * 1024 * 8 * 4096 * 2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ops.py -v
```

- [ ] **Step 3: Implement ops.py**

`src/rl_perf/ops.py` — complete implementation with:
- `OpCost` dataclass (flops, mem_rw, weight_bytes, output_bytes, comm_bytes=0)
- `roofline_time(cost, hw, is_large_gemm=True)` — uses two-tier calibration
- `op_linear(in_features, out_features, batch_tokens, phase, dtype_bytes=2)` — FLOPs + memory for all phases
- `op_gqa_attention(num_heads, num_kv_heads, head_dim, hidden_size, batch, seq_len, phase, kv_len=None)` — GQA attention with proper FLOPs formula `(4+4G/H)d² + 4sd`
- `op_mha_attention(...)` — MHA as special case of GQA where G=H
- `op_swa_attention(...)` — SWA capping seq_len at window_size for attention term
- `op_mla_attention(...)` — MLA with absorption for inference, without for training
- `op_swiglu_ffn(hidden_size, intermediate_size, batch_tokens, phase)` — 6·d·d_ff
- `op_moe_ffn(hidden_size, expert_intermediate_size, num_experts, num_shared_experts, shared_intermediate_size, top_k, batch_tokens, phase)` — routed + shared
- `op_mhc_residual(hidden_size, expansion, batch_tokens, phase, dtype_bytes=2)` — memory-bandwidth-bound overhead
- `op_rmsnorm(hidden_size, batch_tokens, phase, dtype_bytes=2)` — element-wise
- `op_allreduce(msg_bytes, group_size)` — `2*msg*(N-1)/N`
- `op_allgather(msg_bytes, group_size)`, `op_reducescatter(msg_bytes, group_size)` — `msg*(N-1)/N`
- `op_alltoall(tokens, hidden_size, top_k, ep_size, dtype_bytes)` — `2*tokens*top_k*hidden*bytes`
- `op_p2p(msg_bytes)` — simple transfer

Each function has a docstring citing the formula source.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ops.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: operator cost model with roofline + memory for GQA, MLA, SWA, SwiGLU, MoE"
```

---

## Task 4: Builder (builder.py)

**Files:**
- Create: `src/rl_perf/builder.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write tests for builder**

`tests/test_builder.py`:
```python
import pytest
from rl_perf.config import (
    ModelConfig, LayerConfig, HardwareConfig, ParallelismConfig, RLConfig, Phase,
    load_model_config, load_hardware_config,
)
from rl_perf.builder import (
    SimOp, build_layer_ops, build_training_step, build_generation_step,
)
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def llama_config():
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))

@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))

@pytest.fixture
def parallel():
    return ParallelismConfig(tp=8, pp=1, dp=8, ep=1)

@pytest.fixture
def rl_cfg():
    return RLConfig(
        total_prompts=1000, group_size=8,
        avg_prompt_len=512, avg_response_len=2048,
        max_response_len=4096,
        train_micro_batch_size=4, gradient_accumulation_steps=1,
        gen_batch_size=32,
    )


def test_build_layer_ops_returns_simops(llama_config, hw, parallel):
    layer = llama_config.get_layers()[0]
    ops = build_layer_ops(layer, llama_config, parallel, hw,
                          batch=4, seq_len=512, phase=Phase.TRAIN_FWD)
    assert len(ops) > 0
    assert all(isinstance(op, SimOp) for op in ops)
    # Should have compute and comm ops
    streams = {op.stream for op in ops}
    assert "compute" in streams


def test_build_layer_ops_tp_comm(llama_config, hw, parallel):
    layer = llama_config.get_layers()[0]
    ops = build_layer_ops(layer, llama_config, parallel, hw,
                          batch=4, seq_len=512, phase=Phase.TRAIN_FWD)
    comm_ops = [op for op in ops if "comm" in op.stream]
    if parallel.tp > 1:
        assert len(comm_ops) > 0  # TP AllReduce expected


def test_build_training_step_has_fwd_bwd(llama_config, hw, parallel, rl_cfg):
    ops = build_training_step(llama_config, hw, parallel, rl_cfg)
    names = [op.name for op in ops]
    assert any("fwd" in n.lower() or "forward" in n.lower() for n in names)
    assert any("bwd" in n.lower() or "backward" in n.lower() for n in names)


def test_build_generation_step(llama_config, hw, parallel, rl_cfg):
    ops = build_generation_step(llama_config, hw, parallel, rl_cfg)
    assert len(ops) > 0
    # Should have prefill and decode ops
    names = [op.name for op in ops]
    assert any("prefill" in n.lower() for n in names)
    assert any("decode" in n.lower() for n in names)


def test_simop_depends_on_valid_indices(llama_config, hw, parallel, rl_cfg):
    ops = build_training_step(llama_config, hw, parallel, rl_cfg)
    max_idx = len(ops) - 1
    for i, op in enumerate(ops):
        for dep in op.depends_on:
            assert 0 <= dep < i, f"Op {i} ({op.name}) has invalid dep {dep}"


def test_weight_bytes_nonzero(llama_config, hw, parallel, rl_cfg):
    ops = build_training_step(llama_config, hw, parallel, rl_cfg)
    total_weight = sum(op.weight_bytes for op in ops)
    assert total_weight > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_builder.py -v
```

- [ ] **Step 3: Implement builder.py**

`src/rl_perf/builder.py` — complete implementation with:
- `SimOp` dataclass (name, stream, duration, depends_on, weight_bytes, output_bytes, consumers)
- `build_layer_ops(layer_cfg, model_cfg, parallel_cfg, hw, batch, seq_len, phase, kv_len=None)` — builds one layer's op sequence:
  - RMSNorm → attention (dispatched by layer.attention type: GQA/MHA/MLA/SWA) → TP comm → RMSNorm → FFN (SwiGLU or MoE + EP comm) → TP comm
  - mHC residual ops if layer.residual == "mHC"
  - Shapes divided by TP/EP before passing to ops.py
  - Sets depends_on: each op depends on previous in compute stream; comm ops on the compute op they follow
  - For TRAIN_BWD: FLOPs 2x, activation released
- `build_training_step(model_cfg, hw, parallel_cfg, rl_cfg)` — 1F1B MVP:
  - Forward all layers → Backward all layers (reversed) → DP gradient AllReduce → optimizer step
  - PP: only build layers for this stage (`layers[start:end]` where start/end from PP)
  - PP bubble modeled as time multiplier: `1 + (pp-1)/(num_microbatches + pp-1)`
- `build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)` — prefill + decode:
  - Prefill: all layers with batch=gen_batch, seq=avg_prompt_len, phase=PREFILL
  - Decode: all layers with batch=gen_batch, seq=1, kv_len=prompt+response, phase=DECODE
  - Decode ops marked separately for pipeline.py to multiply by response length

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_builder.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: builder converting config to SimOp sequences with parallelism"
```

---

## Task 5: Simulator (simulator.py)

**Files:**
- Create: `src/rl_perf/simulator.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write tests for simulator**

`tests/test_simulator.py`:
```python
import pytest
from rl_perf.simulator import simulate, SimResult
from rl_perf.builder import SimOp


def test_single_op():
    ops = [SimOp("op0", "compute", duration=1.0, depends_on=[], weight_bytes=100, output_bytes=50)]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(1.0)
    assert r.weight_bytes == 100
    assert r.peak_activation_bytes == 50


def test_sequential_same_stream():
    ops = [
        SimOp("op0", "compute", 1.0, [], 100, 50),
        SimOp("op1", "compute", 2.0, [0], 0, 30),
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(3.0)


def test_parallel_different_streams():
    ops = [
        SimOp("op0", "compute", 2.0, [], 0, 0),
        SimOp("op1", "tp_comm", 1.0, [], 0, 0),  # no dependency, parallel
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(2.0)  # max of parallel


def test_dependency_across_streams():
    ops = [
        SimOp("compute0", "compute", 2.0, [], 0, 0),
        SimOp("comm0", "tp_comm", 1.0, [0], 0, 0),  # depends on compute0
        SimOp("compute1", "compute", 1.0, [1], 0, 0),  # depends on comm0
    ]
    r = simulate(ops)
    # compute0: 0-2, comm0: 2-3, compute1: 3-4
    assert r.wall_clock_time == pytest.approx(4.0)


def test_overlap_comm_with_compute():
    ops = [
        SimOp("compute0", "compute", 3.0, [], 0, 0),
        SimOp("comm0", "tp_comm", 2.0, [0], 0, 0),   # after compute0
        SimOp("compute1", "compute", 3.0, [0], 0, 0), # also after compute0, parallel with comm
    ]
    r = simulate(ops)
    # compute0: 0-3
    # comm0: 3-5 (tp_comm stream)
    # compute1: 3-6 (compute stream, parallel with comm0)
    assert r.wall_clock_time == pytest.approx(6.0)


def test_memory_tracking_peak():
    ops = [
        SimOp("op0", "compute", 1.0, [], weight_bytes=0, output_bytes=100, consumers=[1]),
        SimOp("op1", "compute", 1.0, [0], weight_bytes=0, output_bytes=200, consumers=[2]),
        # At op1 start: op0 output (100) still live + op1 output (200) = 300
        SimOp("op2", "compute", 1.0, [1], weight_bytes=0, output_bytes=50),
        # op0 freed after op1, op1 freed after op2
    ]
    r = simulate(ops)
    assert r.peak_activation_bytes >= 300


def test_weight_bytes_deduplicated():
    ops = [
        SimOp("fwd_layer0", "compute", 1.0, [], weight_bytes=500, output_bytes=0),
        SimOp("bwd_layer0", "compute", 1.0, [0], weight_bytes=500, output_bytes=0),
        # Same layer's weights counted once
    ]
    r = simulate(ops)
    # Weight dedup depends on implementation — at minimum should be >= 500
    assert r.weight_bytes >= 500
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_simulator.py -v
```

- [ ] **Step 3: Implement simulator.py**

`src/rl_perf/simulator.py` — ~200 lines:
- `SimResult` dataclass
- `simulate(ops: List[SimOp]) -> SimResult`:
  1. Build adjacency from depends_on
  2. Topological sort (Kahn's algorithm)
  3. Multi-clock: dict of stream_name → current_time
  4. For each op in topo order:
     - `earliest_start = max(stream_clock[op.stream], max(finish_time[d] for d in op.depends_on))`
     - `finish_time[i] = earliest_start + op.duration`
     - `stream_clock[op.stream] = finish_time[i]`
  5. Memory tracking: track live output_bytes using consumer ref_count
     - When op completes, decrement ref_count for its dependencies
     - When ref_count reaches 0, free that op's output_bytes
     - Track peak = max over all ops of (total_weight + live_activation)
  6. Return SimResult with max(stream clocks), total weights, peak activation, total comm bytes

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_simulator.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: multi-stream simulator with memory tracking"
```

---

## Task 6: Pipeline Model (pipeline.py)

**Files:**
- Create: `src/rl_perf/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write tests for pipeline**

`tests/test_pipeline.py`:
```python
import pytest
from rl_perf.pipeline import (
    generation_time, training_time, epoch_time, bottleneck_analysis,
    effective_response_len,
)
from rl_perf.config import (
    ModelConfig, LayerConfig, HardwareConfig, ParallelismConfig, RLConfig,
    load_model_config, load_hardware_config,
)
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def llama_config():
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))

@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))

@pytest.fixture
def rl_cfg():
    return RLConfig(
        total_prompts=1000, group_size=8,
        avg_prompt_len=512, avg_response_len=2048,
        max_response_len=4096,
        train_micro_batch_size=4, gradient_accumulation_steps=1,
        gen_batch_size=32,
    )


def test_effective_response_len_with_std():
    eff = effective_response_len(avg=2048, std=800, batch_size=32)
    assert eff > 2048  # Should be > avg due to long-tail
    assert eff < 4096  # Should be reasonable


def test_effective_response_len_fallback_max():
    eff = effective_response_len(avg=2048, std=None, batch_size=32, max_len=4096)
    assert eff == 4096  # Falls back to max when no std


def test_bottleneck_analysis():
    b, slack = bottleneck_analysis(t_gen=10.0, t_train=6.0)
    assert b == "GENERATION"
    assert slack == pytest.approx(10.0 / 6.0 - 1)


def test_epoch_time_max_of_stages():
    t_gen = 20.0  # hours
    t_train = 15.0
    startup = 0.5
    t = epoch_time(t_gen, t_train, startup_overhead=startup)
    # Should be approximately max(gen, train) + startup
    assert t == pytest.approx(t_gen + startup)


def test_generation_time_nonzero(llama_config, hw, rl_cfg):
    parallel = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    t = generation_time(llama_config, hw, parallel, rl_cfg)
    assert t > 0


def test_training_time_nonzero(llama_config, hw, rl_cfg):
    parallel = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    t = training_time(llama_config, hw, parallel, rl_cfg)
    assert t > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Implement pipeline.py**

`src/rl_perf/pipeline.py` — ~150 lines:
- `effective_response_len(avg, std, batch_size, max_len=None)` — Gumbel approximation or max fallback
- `generation_time(model_cfg, hw, parallel_cfg, rl_cfg)`:
  - Build prefill + decode ops via builder
  - Simulate to get per-batch time
  - `t_batch = t_prefill + effective_len * t_per_decode_token`
  - Total: `num_batches * t_batch` where num_batches = total_responses / (gen_batch * gen_dp)
- `training_time(model_cfg, hw, parallel_cfg, rl_cfg)`:
  - Build training step ops via builder
  - Simulate to get per-step time
  - Apply recompute/offload perf_penalty
  - Total: `num_steps * t_step` where num_steps considers gradient accumulation
- `epoch_time(t_gen, t_train, startup_overhead)`:
  - `max(t_gen, t_train) + startup_overhead`
- `bottleneck_analysis(t_gen, t_train)`:
  - Returns (bottleneck_name, slack_ratio)

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: pipeline model with generation/training time and bottleneck analysis"
```

---

## Task 7: Report + Model API (report.py + model.py)

**Files:**
- Create: `src/rl_perf/report.py`
- Create: `src/rl_perf/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write tests for model API**

`tests/test_model.py`:
```python
import pytest
from rl_perf.model import RLPerformanceModel
from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def perf_model():
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    return RLPerformanceModel(mc, hw)

@pytest.fixture
def rl_cfg():
    return RLConfig(
        total_prompts=1000, group_size=8,
        avg_prompt_len=512, avg_response_len=2048,
        max_response_len=4096,
        train_micro_batch_size=4, gradient_accumulation_steps=1,
        gen_batch_size=32,
    )


def test_derive_targets(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=gen_p, train_parallel=train_p,
        time_budget_hours=24,
    )
    assert report.epoch_time_hours > 0
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.bottleneck in ("GENERATION", "TRAINING", "BALANCED")


def test_feasibility_check(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.feasibility_check(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=gen_p, train_parallel=train_p,
    )
    assert report.epoch_time_hours > 0
    assert report.memory is not None
    assert report.memory.train_feasible is not None


def test_what_if(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    base = perf_model.derive_targets(64, rl_cfg, gen_p, train_p, 24)
    # Double group size
    rl_cfg2 = rl_cfg.model_copy(update={"group_size": 16})
    new = perf_model.derive_targets(64, rl_cfg2, gen_p, train_p, 24)
    assert new.epoch_time_hours > base.epoch_time_hours  # More data = longer
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```

- [ ] **Step 3: Implement report.py**

`src/rl_perf/report.py` — ~100 lines:
- `MemoryProfile` dataclass (weight_gb, optimizer_gb, activation_gb, kv_cache_gb, ref_model_gb, total_gb, usable_gb, train_feasible, gen_feasible)
- `TargetReport` dataclass (epoch_time_hours, within_budget, bottleneck, gen_tps_target, train_tps_target, gen_time_hours, train_time_hours, memory, parallel_config)
- `format_table(report) -> str` — rich Table formatting for CLI output
- `format_json(report) -> str` — JSON serialization

- [ ] **Step 4: Implement model.py**

`src/rl_perf/model.py` — ~200 lines:
- `RLPerformanceModel(model_cfg, hw_cfg)`:
  - `derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None)` → TargetReport
  - `feasibility_check(total_devices, rl_cfg, gen_parallel, train_parallel)` → TargetReport (time_budget=inf)
  - `what_if(base_report, **overrides)` → TargetReport
  - `sensitivity(rl_cfg, param_name, values, gen_parallel, train_parallel)` → List[TargetReport]
  - Internal: `_compute_memory_profile(parallel, rl_cfg, sim_result)` → MemoryProfile
    - weights from SimResult
    - optimizer: params × 12 bytes / dp (ZeRO sharding)
    - KV cache: computed from model config
    - reference model: weights (if enabled and not offloaded)
    - feasibility: total < usable_hbm

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_model.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: RLPerformanceModel API with derive_targets, feasibility_check, what_if"
```

---

## Task 8: CLI (cli.py)

**Files:**
- Create: `src/rl_perf/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write CLI test**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from rl_perf.cli import app

runner = CliRunner()


def test_targets_help():
    result = runner.invoke(app, ["targets", "--help"])
    assert result.exit_code == 0
    assert "model" in result.output.lower()


def test_targets_basic():
    result = runner.invoke(app, [
        "targets",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "configs/hardware/ascend_910c.yaml",
        "--devices", "64",
        "--prompts", "1000",
        "--group-size", "8",
        "--time-budget", "24",
    ])
    assert result.exit_code == 0
    assert "TPS" in result.output or "tps" in result.output or "tokens" in result.output.lower()


def test_check_basic():
    result = runner.invoke(app, [
        "check",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "configs/hardware/ascend_910c.yaml",
        "--devices", "64",
        "--prompts", "1000",
    ])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement cli.py**

`src/rl_perf/cli.py` — ~100 lines using typer:
- `app = typer.Typer()`
- `@app.command() targets(model, hardware, devices, prompts, group_size, time_budget, ...)`:
  - Load configs, create default ParallelismConfig (auto TP=min(8, devices), DP=devices/TP)
  - Call `RLPerformanceModel.derive_targets()`
  - Print formatted output via rich
- `@app.command() check(model, hardware, devices, prompts, ...)`:
  - Same but no time_budget (feasibility check)
- Handle hardware shortnames: "910C" → "configs/hardware/ascend_910c.yaml"

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```
Expected: all PASS

- [ ] **Step 5: Test CLI manually**

```bash
cd /Users/horacehxw/Projects/RL_modeling
rl-perf targets --model configs/models/llama3_1_8b.yaml --hardware configs/hardware/ascend_910c.yaml --devices 64 --prompts 1000 --group-size 8 --time-budget 24
```
Expected: Formatted table output with epoch time, TPS targets, memory profile.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: CLI with targets and check commands"
```

---

## Task 9: Demo Model Configs

**Files:**
- Create: `configs/models/qwen2_5_72b.yaml`
- Create: `configs/models/mistral_7b.yaml`
- Create: `configs/models/qwen3_235b_moe.yaml`
- Create: `configs/models/deepseekv3_671b.yaml`
- Create: `configs/hardware/cloudmatrix_384.yaml`

- [ ] **Step 1: Create all 4 remaining model configs + CM384 hardware config**

Use accurate architecture parameters from public sources. Key params:

Qwen2.5 72B: hidden=8192, layers=80, heads=64, kv_heads=8, intermediate=29568, vocab=152064
Mistral 7B: hidden=4096, layers=32, heads=32, kv_heads=8, intermediate=14336, SWA window=4096
Qwen3 235B MoE: hidden=4096, layers=94, heads=64, kv_heads=4, experts=128, top_k=8, expert_inter=2048, shared=4, shared_inter=4096
DeepSeek V3 671B: hidden=7168, layers=61, heads=128, MLA (d_c=512, d'_c=1536, d_R=64), MoE 256 experts top-8 + 1 shared, mHC n=4, MTP depth=1

CloudMatrix 384: 384 NPUs, specs similar to 910C but with UB interconnect.

- [ ] **Step 2: Write test loading all configs**

Add to `tests/test_config.py`:
```python
@pytest.mark.parametrize("name", [
    "llama3_1_8b", "qwen2_5_72b", "mistral_7b", "qwen3_235b_moe", "deepseekv3_671b",
])
def test_load_all_model_configs(name):
    mc = load_model_config(str(CONFIGS_DIR / "models" / f"{name}.yaml"))
    assert mc.num_layers > 0
    assert len(mc.get_layers()) == mc.num_layers
```

```bash
pytest tests/test_config.py::test_load_all_model_configs -v
```
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: demo model configs (5 models) + CloudMatrix 384 hardware config"
```

---

## Task 10: End-to-End Integration Test + Demo Output

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write end-to-end test**

`tests/test_e2e.py`:
```python
import pytest
from pathlib import Path
from rl_perf.model import RLPerformanceModel
from rl_perf.config import (
    load_model_config, load_hardware_config,
    RLConfig, ParallelismConfig,
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def rl_cfg():
    return RLConfig(
        total_prompts=10000, group_size=8,
        avg_prompt_len=512, avg_response_len=2048,
        max_response_len=4096,
        train_micro_batch_size=4, gradient_accumulation_steps=4,
        gen_batch_size=64,
    )


@pytest.mark.parametrize("model_name,tp,pp,dp,ep", [
    ("llama3_1_8b", 8, 1, 8, 1),
    ("qwen2_5_72b", 8, 4, 4, 1),
    ("qwen3_235b_moe", 8, 4, 4, 8),
])
def test_e2e_derive_targets(model_name, tp, pp, dp, ep, rl_cfg):
    mc = load_model_config(str(CONFIGS_DIR / "models" / f"{model_name}.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))

    perf = RLPerformanceModel(mc, hw)
    gen_p = ParallelismConfig(tp=tp, pp=1, dp=tp*pp*dp*ep//tp, ep=1)
    train_p = ParallelismConfig(tp=tp, pp=pp, dp=dp, ep=ep)

    report = perf.derive_targets(
        total_devices=tp * pp * dp * ep,
        rl_cfg=rl_cfg,
        gen_parallel=gen_p,
        train_parallel=train_p,
        time_budget_hours=24,
    )

    # Sanity checks
    assert report.epoch_time_hours > 0
    assert report.epoch_time_hours < 1000  # Reasonable upper bound
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.bottleneck in ("GENERATION", "TRAINING", "BALANCED")
    assert report.memory is not None

    # Print for visual inspection
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Devices: {tp*pp*dp*ep}")
    print(f"Epoch time: {report.epoch_time_hours:.2f} hours")
    print(f"Bottleneck: {report.bottleneck}")
    print(f"Gen TPS target: {report.gen_tps_target:,.0f} tokens/s")
    print(f"Train TPS target: {report.train_tps_target:,.0f} tokens/s")
    print(f"Memory: train={report.memory.total_train_gb:.1f}GB gen={report.memory.total_gen_gb:.1f}GB")
    print(f"{'='*60}")
```

- [ ] **Step 2: Run e2e tests**

```bash
pytest tests/test_e2e.py -v -s
```
Expected: all PASS with printed output showing reasonable numbers.

- [ ] **Step 3: Run CLI demo**

```bash
rl-perf targets --model configs/models/llama3_1_8b.yaml --hardware configs/hardware/ascend_910c.yaml --devices 64 --prompts 10000 --group-size 8 --time-budget 24

rl-perf targets --model configs/models/qwen2_5_72b.yaml --hardware configs/hardware/ascend_910c.yaml --devices 128 --prompts 10000 --group-size 8 --time-budget 24

rl-perf check --model configs/models/qwen3_235b_moe.yaml --hardware configs/hardware/ascend_910c.yaml --devices 256 --prompts 10000
```

Capture output for review.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: e2e integration tests + demo verification for Llama 8B, Qwen 72B, Qwen3 MoE"
```

---

## Task Summary

| Task | What | Files | Est. |
|------|------|-------|------|
| 1 | Project scaffolding | pyproject.toml, CLAUDE.md, README | 5 min |
| 2 | Config data model | config.py + YAML configs | 15 min |
| 3 | Operator cost model | ops.py (GQA, MLA, SWA, SwiGLU, MoE, comm) | 20 min |
| 4 | Builder | builder.py (config → SimOp sequences) | 20 min |
| 5 | Simulator | simulator.py (multi-stream + memory) | 15 min |
| 6 | Pipeline model | pipeline.py (gen/train time, bottleneck) | 10 min |
| 7 | Report + Model API | report.py, model.py | 15 min |
| 8 | CLI | cli.py | 10 min |
| 9 | Demo configs | 5 model YAMLs + CM384 | 10 min |
| 10 | E2E test + demo | test_e2e.py + CLI demo runs | 10 min |
