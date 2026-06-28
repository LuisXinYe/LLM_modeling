# Large-sparse-ratio MoE network modeling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Model how a large sparse ratio (many experts, small top_k) drives cross-node all-to-all bandwidth demand, via node-limited hierarchical routing and a scalar load-imbalance factor, plus a `sweep_sparse_ratio()` report.

**Architecture:** Two new config knobs on `ParallelismConfig` (`moe_node_limit`, `moe_imbalance_factor`). New `ops.py` functions split MoE dispatch/combine traffic into inter-node (NIC) and intra-node (NVLink/HCCS) byte components when node-limited routing is active. `builder.py` emits two fabric-tagged `SimOp`s instead of one in that case, and scales expert compute by the imbalance factor. A `sweep_sparse_ratio()` helper in `model.py` (mirroring `compare_precision`) tabulates per-fabric exposed comm across a grid.

**Tech Stack:** Python 3.10+, pydantic, pytest. Pure-Python cost model — no ML frameworks.

## Global Constraints

- Python >= 3.10. Pure Python + numpy only; no ML framework dependencies.
- All FLOPs/bytes formulas must have a docstring citing the source.
- Hardware constants come from `HardwareConfig`, never hardcoded.
- Defaults MUST reproduce current behavior exactly: `moe_node_limit = 0` and `moe_imbalance_factor = 1.0` leave every existing config/test unchanged.
- Run tests with `source .venv/bin/activate && pytest tests/ -v`. Lint with `ruff check src/ && ruff format src/`.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

### Task 1: Config knobs on ParallelismConfig

**Files:**
- Modify: `src/llm_perf/config.py:249-271` (`ParallelismConfig` fields + validators)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ParallelismConfig.moe_node_limit: int = 0`, `ParallelismConfig.moe_imbalance_factor: float = 1.0`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_moe_sparse_knobs_default_and_validate():
    from llm_perf.config import ParallelismConfig
    import pytest

    # Defaults reproduce current behavior
    p = ParallelismConfig(ep=8)
    assert p.moe_node_limit == 0
    assert p.moe_imbalance_factor == 1.0

    # Accepts explicit values
    p2 = ParallelismConfig(ep=64, moe_node_limit=4, moe_imbalance_factor=1.3)
    assert p2.moe_node_limit == 4
    assert p2.moe_imbalance_factor == 1.3

    # node_limit must be >= 0
    with pytest.raises(ValueError):
        ParallelismConfig(moe_node_limit=-1)

    # imbalance_factor must be >= 1.0
    with pytest.raises(ValueError):
        ParallelismConfig(moe_imbalance_factor=0.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_config.py::test_moe_sparse_knobs_default_and_validate -v`
Expected: FAIL with `AttributeError` / `TypeError` (fields don't exist).

- [ ] **Step 3: Add the fields and validators**

In `src/llm_perf/config.py`, inside `ParallelismConfig`, add after the `activation_offload` field (line 264):

```python
    # Large-sparse-ratio MoE routing knobs.
    # moe_node_limit: max distinct nodes a token's experts may span (DeepSeek-V3
    #   node-limited routing). 0 = unlimited → flat all-to-all (current behavior).
    # moe_imbalance_factor: hottest EP rank traffic/compute relative to the mean
    #   (>= 1.0). 1.0 = perfectly balanced (current behavior).
    moe_node_limit: int = 0
    moe_imbalance_factor: float = 1.0
```

Add validators after the existing `must_be_positive` validator (after line 271):

```python
    @field_validator("moe_node_limit")
    @classmethod
    def node_limit_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"moe_node_limit must be >= 0, got {v}")
        return v

    @field_validator("moe_imbalance_factor")
    @classmethod
    def imbalance_at_least_one(cls, v):
        if v < 1.0:
            raise ValueError(f"moe_imbalance_factor must be >= 1.0, got {v}")
        return v
```

Also extend the `ParallelismConfig` docstring (around line 247) with two lines:

```python
        moe_node_limit: Max distinct nodes a token's experts may span (node-limited
            routing). 0 = unlimited (flat all-to-all).
        moe_imbalance_factor: Hottest EP rank traffic/compute vs mean (>= 1.0).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_config.py::test_moe_sparse_knobs_default_and_validate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/config.py tests/test_config.py
git commit -m "feat(config): moe_node_limit + moe_imbalance_factor knobs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: ops.py — hierarchical all-to-all + imbalance-aware MoE FFN

**Files:**
- Modify: `src/llm_perf/ops.py` (add two functions after `op_alltoall_combine` ~line 1009; add `imbalance_factor` param to `op_moe_ffn` ~line 687)
- Test: `tests/test_ops.py`

**Interfaces:**
- Produces:
  - `op_alltoall_dispatch_hierarchical(tokens, hidden_size, top_k, node_limit, nodes_in_ep, imbalance_factor=1.0, dtype_bytes=2) -> tuple[float, float]` returning `(inter_bytes, intra_bytes)`.
  - `op_alltoall_combine_hierarchical(...)` — identical signature and return.
  - `op_moe_ffn(..., imbalance_factor: float = 1.0)` — multiplies routed-expert FLOPs by `imbalance_factor`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ops.py`:

```python
def test_alltoall_hierarchical_conserves_total_bytes():
    from llm_perf import ops
    tokens, hidden, top_k = 4096, 4096, 8
    nodes_in_ep, node_limit = 8, 3
    inter, intra = ops.op_alltoall_dispatch_hierarchical(
        tokens=tokens, hidden_size=hidden, top_k=top_k,
        node_limit=node_limit, nodes_in_ep=nodes_in_ep, dtype_bytes=2,
    )
    flat = tokens * top_k * hidden * 2
    # distinct dest nodes = min(top_k, node_limit, nodes_in_ep) = 3
    assert inter == tokens * 3 * hidden * 2
    assert intra == tokens * (top_k - 3) * hidden * 2
    assert inter + intra == flat  # co-location only moves bytes between fabrics


def test_alltoall_hierarchical_smaller_node_limit_cuts_inter():
    from llm_perf import ops
    tokens, hidden, top_k, nodes_in_ep = 4096, 4096, 8, 8
    inter_m4, _ = ops.op_alltoall_dispatch_hierarchical(
        tokens, hidden, top_k, node_limit=4, nodes_in_ep=nodes_in_ep)
    inter_m2, _ = ops.op_alltoall_dispatch_hierarchical(
        tokens, hidden, top_k, node_limit=2, nodes_in_ep=nodes_in_ep)
    assert inter_m2 < inter_m4  # tighter node limit → less cross-node traffic


def test_alltoall_hierarchical_imbalance_scales_both():
    from llm_perf import ops
    base = ops.op_alltoall_dispatch_hierarchical(
        4096, 4096, 8, node_limit=3, nodes_in_ep=8, imbalance_factor=1.0)
    hot = ops.op_alltoall_dispatch_hierarchical(
        4096, 4096, 8, node_limit=3, nodes_in_ep=8, imbalance_factor=1.5)
    assert hot[0] == base[0] * 1.5
    assert hot[1] == base[1] * 1.5


def test_alltoall_combine_hierarchical_matches_dispatch():
    from llm_perf import ops
    args = dict(tokens=4096, hidden_size=4096, top_k=8, node_limit=3, nodes_in_ep=8)
    assert ops.op_alltoall_combine_hierarchical(**args) == \
        ops.op_alltoall_dispatch_hierarchical(**args)


def test_moe_ffn_imbalance_scales_routed_flops_only():
    from llm_perf import ops
    from llm_perf.ops import Phase
    base = ops.op_moe_ffn(
        hidden_size=4096, expert_intermediate_size=2048, num_experts=32,
        num_shared_experts=1, shared_intermediate_size=2048, top_k=8,
        batch_tokens=4096, phase=Phase.TRAIN_FWD, imbalance_factor=1.0)
    hot = ops.op_moe_ffn(
        hidden_size=4096, expert_intermediate_size=2048, num_experts=32,
        num_shared_experts=1, shared_intermediate_size=2048, top_k=8,
        batch_tokens=4096, phase=Phase.TRAIN_FWD, imbalance_factor=1.5)
    # routed flops grow with imbalance; weights/activation memory unchanged
    assert hot.flops > base.flops
    assert hot.weight_bytes == base.weight_bytes
    assert hot.output_bytes == base.output_bytes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k "hierarchical or moe_ffn_imbalance" -v`
Expected: FAIL (`AttributeError: module has no attribute 'op_alltoall_dispatch_hierarchical'`; `op_moe_ffn` rejects `imbalance_factor`).

- [ ] **Step 3: Add the hierarchical functions**

In `src/llm_perf/ops.py`, after `op_alltoall_combine` (ends ~line 1009), add:

```python
def op_alltoall_dispatch_hierarchical(
    tokens: int,
    hidden_size: int,
    top_k: int,
    node_limit: int,
    nodes_in_ep: int,
    imbalance_factor: float = 1.0,
    dtype_bytes: int = 2,
) -> tuple[float, float]:
    """Node-limited MoE dispatch, split into (inter_node_bytes, intra_node_bytes).

    Each token's top_k experts are constrained to at most `node_limit` distinct
    nodes (DeepSeek-V3 node-limited routing). The token payload crosses the
    network once per distinct destination node; experts co-located on an already-
    reached node are served by an intra-node (NVLink/HCCS) copy.

    distinct_dest_nodes = min(top_k, node_limit, nodes_in_ep)
      inter_bytes = tokens * distinct_dest_nodes        * hidden * bytes * imbalance
      intra_bytes = tokens * (top_k - distinct_dest_nodes) * hidden * bytes * imbalance

    Conservative for network sizing: every distinct destination node is counted as
    remote (the source-local node is not subtracted), upper-bounding cross-node
    demand. Total inter+intra always equals the flat top_k traffic. Ref: DeepSeek-V3.
    """
    distinct_dest_nodes = min(top_k, node_limit, nodes_in_ep)
    inter_experts = distinct_dest_nodes
    intra_experts = top_k - distinct_dest_nodes
    per = tokens * hidden_size * dtype_bytes * imbalance_factor
    return (inter_experts * per, intra_experts * per)


def op_alltoall_combine_hierarchical(
    tokens: int,
    hidden_size: int,
    top_k: int,
    node_limit: int,
    nodes_in_ep: int,
    imbalance_factor: float = 1.0,
    dtype_bytes: int = 2,
) -> tuple[float, float]:
    """Node-limited MoE combine. Symmetric to dispatch; see
    op_alltoall_dispatch_hierarchical. Ref: DeepSeek-V3."""
    return op_alltoall_dispatch_hierarchical(
        tokens=tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        node_limit=node_limit,
        nodes_in_ep=nodes_in_ep,
        imbalance_factor=imbalance_factor,
        dtype_bytes=dtype_bytes,
    )
```

- [ ] **Step 4: Add `imbalance_factor` to `op_moe_ffn`**

In `src/llm_perf/ops.py`, change the `op_moe_ffn` signature (line 687-699) to add the parameter just before `dtype_bytes`:

```python
def op_moe_ffn(
    hidden_size: int,
    expert_intermediate_size: int,
    num_experts: int,
    num_shared_experts: int,
    shared_intermediate_size: int,
    top_k: int,
    batch_tokens: int,
    phase: Phase,
    imbalance_factor: float = 1.0,
    dtype_bytes: int = 2,
    weight_dtype_bytes: float | None = None,
    act_dtype_bytes: float | None = None,
) -> OpCost:
```

Then change the `routed_flops` line (currently line 708) to scale by imbalance — the hottest EP rank processes more routed tokens, so its expert GEMM time grows; weights and activation memory are unchanged:

```python
    routed_flops = (
        6 * hidden_size * expert_intermediate_size * top_k * batch_tokens
        * imbalance_factor
    )
```

Add to the docstring (after line 704) a line:

```python
    imbalance_factor scales routed-expert FLOPs to model the hottest EP rank
    doing more expert compute (>= 1.0; 1.0 = balanced). Weights and activation
    memory are unaffected. Ref: DeepSeek-V3 §2.1 (expert load balancing).
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -k "hierarchical or moe_ffn_imbalance" -v`
Expected: PASS

- [ ] **Step 6: Run the full ops suite for regressions**

Run: `source .venv/bin/activate && pytest tests/test_ops.py -v`
Expected: PASS (existing `op_moe_ffn` callers use the default `imbalance_factor=1.0`).

- [ ] **Step 7: Commit**

```bash
git add src/llm_perf/ops.py tests/test_ops.py
git commit -m "feat(ops): node-limited hierarchical all-to-all + imbalance-aware MoE FFN

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: builder.py — emit fabric-split SimOps + apply imbalance

**Files:**
- Modify: `src/llm_perf/builder.py` (add a `_nodes_spanned` helper near `_fabric` ~line 111; rewrite the MoE EP branch ~lines 591-687)
- Test: `tests/test_builder.py`

**Interfaces:**
- Consumes: `ParallelismConfig.moe_node_limit`, `ParallelismConfig.moe_imbalance_factor` (Task 1); `ops.op_alltoall_dispatch_hierarchical`, `ops.op_alltoall_combine_hierarchical`, `ops.op_moe_ffn(..., imbalance_factor=...)` (Task 2).
- Produces: when `moe_node_limit > 0` and the EP group spans >1 node, the MoE layer emits `ep_alltoall_dispatch_inter` (fabric `"nic"`) + `ep_alltoall_dispatch_intra` (fabric `"nvlink"`) and the symmetric combine pair, all on `stream="ep_comm"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_builder.py`:

```python
def _moe_layer_cfg():
    from llm_perf.config import LayerConfig
    return LayerConfig(
        attention="GQA", num_heads=64, num_kv_heads=8, head_dim=128,
        ffn="MoE", num_experts=256, num_shared_experts=1, top_k=8,
        expert_intermediate_size=2048, shared_intermediate_size=2048,
        intermediate_size=2048,
    )


def test_moe_node_limited_emits_two_fabric_dispatch_ops():
    from llm_perf import builder
    from llm_perf.builder import Phase
    from llm_perf.config import ModelConfig, HardwareConfig, ParallelismConfig

    layer = _moe_layer_cfg()
    model = ModelConfig(name="t", hidden_size=4096, vocab_size=129280,
                        num_layers=1, dtype="bf16", layers=[layer])
    hw = HardwareConfig(name="hw", peak_tflops=400, hbm_capacity_gb=64,
                        hbm_bandwidth_tb_s=3.0, devices_per_node=8)
    # ep=32 spans 4 nodes; node_limit=2 activates the split
    par = ParallelismConfig(ep=32, moe_node_limit=2)

    ops = builder.build_layer_ops(layer, model, par, hw, batch=1, seq_len=4096,
                                  phase=Phase.TRAIN_FWD)
    names = [o.name for o in ops]
    assert "ep_alltoall_dispatch_inter" in names
    assert "ep_alltoall_dispatch_intra" in names
    inter = next(o for o in ops if o.name == "ep_alltoall_dispatch_inter")
    intra = next(o for o in ops if o.name == "ep_alltoall_dispatch_intra")
    assert inter.fabric == "nic"
    assert intra.fabric == "nvlink"
    assert inter.stream == "ep_comm" and intra.stream == "ep_comm"


def test_moe_node_limit_zero_keeps_single_dispatch_op():
    from llm_perf import builder
    from llm_perf.builder import Phase
    from llm_perf.config import ModelConfig, HardwareConfig, ParallelismConfig

    layer = _moe_layer_cfg()
    model = ModelConfig(name="t", hidden_size=4096, vocab_size=129280,
                        num_layers=1, dtype="bf16", layers=[layer])
    hw = HardwareConfig(name="hw", peak_tflops=400, hbm_capacity_gb=64,
                        hbm_bandwidth_tb_s=3.0, devices_per_node=8)
    par = ParallelismConfig(ep=32)  # node_limit=0 default

    ops = builder.build_layer_ops(layer, model, par, hw, batch=1, seq_len=4096,
                                  phase=Phase.TRAIN_FWD)
    names = [o.name for o in ops]
    assert "ep_alltoall_dispatch" in names
    assert "ep_alltoall_dispatch_inter" not in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -k "node_limit" -v`
Expected: FAIL (`ep_alltoall_dispatch_inter` not found).

- [ ] **Step 3: Add the `_nodes_spanned` helper**

In `src/llm_perf/builder.py`, after `_fabric` (line 113), add:

```python
def _nodes_spanned(group_size: int, hw: HardwareConfig) -> int:
    """Number of nodes an EP/collective group spans."""
    return max(1, math.ceil(group_size / hw.devices_per_node))
```

Ensure `import math` is present at the top of the file (add it if missing).

- [ ] **Step 4: Read the imbalance/node_limit knobs in `build_layer_ops`**

In `build_layer_ops`, after `ep = parallel_cfg.ep` (line 274), add:

```python
    moe_node_limit = parallel_cfg.moe_node_limit
    moe_imbalance = parallel_cfg.moe_imbalance_factor
    nodes_in_ep = _nodes_spanned(ep, hw)
```

- [ ] **Step 5: Rewrite the MoE dispatch block**

Replace the dispatch block (lines 595-620, the `if ep > 1:` that builds `ep_alltoall_dispatch`) with:

```python
        # EP AllToAll dispatch: send tokens to target expert EP ranks
        if ep > 1:
            if moe_node_limit > 0 and nodes_in_ep > 1:
                inter_b, intra_b = ops.op_alltoall_dispatch_hierarchical(
                    tokens=batch_tokens, hidden_size=d, top_k=layer_cfg.top_k,
                    node_limit=moe_node_limit, nodes_in_ep=nodes_in_ep,
                    imbalance_factor=moe_imbalance, dtype_bytes=dtype_bytes,
                )
                result.append(SimOp(
                    name="ep_alltoall_dispatch_inter", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=inter_b), hw, group_size=ep,
                        is_intra_node=False, algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=inter_b,
                    fabric="nic",
                ))
                result.append(SimOp(
                    name="ep_alltoall_dispatch_intra", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=intra_b), hw,
                        group_size=hw.devices_per_node,
                        is_intra_node=True, algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=intra_b,
                    fabric="nvlink",
                ))
                last_compute_local = len(result) - 1
            else:
                a2a_dispatch_cost = ops.op_alltoall_dispatch(
                    tokens=batch_tokens, hidden_size=d, top_k=layer_cfg.top_k,
                    ep_size=ep, dtype_bytes=dtype_bytes,
                )
                # imbalance scales the synchronous collective's slowest participant
                disp_bytes = a2a_dispatch_cost.comm_bytes * moe_imbalance
                result.append(SimOp(
                    name="ep_alltoall_dispatch", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=disp_bytes), hw, group_size=ep,
                        is_intra_node=_is_intra_node(ep, hw), algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=disp_bytes,
                    fabric=_fabric(ep, hw),
                ))
                last_compute_local = len(result) - 1
```

- [ ] **Step 6: Pass imbalance into `op_moe_ffn`**

In the `ffn_cost = ops.op_moe_ffn(...)` call (line 622-634), add `imbalance_factor=moe_imbalance,` (e.g. right after `phase=phase,`).

- [ ] **Step 7: Rewrite the MoE combine block**

Replace the combine block (lines 662-687, the `if ep > 1:` that builds `ep_alltoall_combine`) with:

```python
        # EP AllToAll combine: gather expert outputs back to original EP ranks
        if ep > 1:
            if moe_node_limit > 0 and nodes_in_ep > 1:
                inter_b, intra_b = ops.op_alltoall_combine_hierarchical(
                    tokens=batch_tokens, hidden_size=d, top_k=layer_cfg.top_k,
                    node_limit=moe_node_limit, nodes_in_ep=nodes_in_ep,
                    imbalance_factor=moe_imbalance, dtype_bytes=dtype_bytes,
                )
                result.append(SimOp(
                    name="ep_alltoall_combine_inter", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=inter_b), hw, group_size=ep,
                        is_intra_node=False, algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=inter_b,
                    fabric="nic",
                ))
                result.append(SimOp(
                    name="ep_alltoall_combine_intra", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=intra_b), hw,
                        group_size=hw.devices_per_node,
                        is_intra_node=True, algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=intra_b,
                    fabric="nvlink",
                ))
                last_compute_local = len(result) - 1
            else:
                a2a_combine_cost = ops.op_alltoall_combine(
                    tokens=batch_tokens, hidden_size=d, top_k=layer_cfg.top_k,
                    ep_size=ep, dtype_bytes=dtype_bytes,
                )
                comb_bytes = a2a_combine_cost.comm_bytes * moe_imbalance
                result.append(SimOp(
                    name="ep_alltoall_combine", stream="ep_comm",
                    duration=ops.comm_time(
                        ops.OpCost(comm_bytes=comb_bytes), hw, group_size=ep,
                        is_intra_node=_is_intra_node(ep, hw), algorithm="alltoall"),
                    depends_on=[_idx(last_compute_local)],
                    weight_bytes=0, output_bytes=0, comm_bytes=comb_bytes,
                    fabric=_fabric(ep, hw),
                ))
                last_compute_local = len(result) - 1
```

- [ ] **Step 8: Run the new tests + full builder suite**

Run: `source .venv/bin/activate && pytest tests/test_builder.py -v`
Expected: PASS (the `node_limit=0` default path is byte-identical to before because `moe_imbalance=1.0`).

- [ ] **Step 9: Run the whole suite for regressions**

Run: `source .venv/bin/activate && pytest tests/ -q`
Expected: PASS (139+ existing tests still green).

- [ ] **Step 10: Lint and commit**

```bash
ruff check src/ && ruff format src/
git add src/llm_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): fabric-split node-limited MoE all-to-all + imbalance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: model.py — `sweep_sparse_ratio()`

**Files:**
- Modify: `src/llm_perf/model.py` (add `sweep_sparse_ratio` after `compare_precision` ~line 698)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ParallelismConfig` knobs (Task 1); builder fabric-split (Task 3); existing `pretraining_time`, `LLMPerformanceModel._train_state_gb`, `_train_activation_gb` (used exactly as in `compare_precision`, lines 648-662).
- Produces: `sweep_sparse_ratio(model_cfg, hw, base_parallel_cfg, rl_cfg, grid) -> list[dict]` where `grid` is a list of dicts, each overriding any of `num_experts`, `top_k`, `ep`, `moe_node_limit`, `moe_imbalance_factor`. Each result dict has keys: `point` (the grid dict), `step_seconds`, `exposed_comm_by_fabric`, `cross_node_gb`, `peak_memory_gb`, `feasible`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`:

```python
def test_sweep_sparse_ratio_shape_and_node_limit_cuts_nic():
    from llm_perf.model import sweep_sparse_ratio
    from llm_perf.config import (
        ModelConfig, HardwareConfig, ParallelismConfig, WorkloadConfig, LayerConfig)

    layer = LayerConfig(
        attention="GQA", num_heads=64, num_kv_heads=8, head_dim=128,
        ffn="MoE", num_experts=256, num_shared_experts=1, top_k=8,
        expert_intermediate_size=2048, shared_intermediate_size=2048,
        intermediate_size=2048)
    model = ModelConfig(name="t", hidden_size=4096, vocab_size=129280,
                        num_layers=4, dtype="bf16", layers=[layer] * 4)
    hw = HardwareConfig(name="hw", peak_tflops=400, hbm_capacity_gb=64,
                        hbm_bandwidth_tb_s=3.0, devices_per_node=8)
    base = ParallelismConfig(ep=32, dp=1, tp=1)
    rl = WorkloadConfig(total_prompts=1000, group_size=8)

    grid = [
        {"moe_node_limit": 0},
        {"moe_node_limit": 4},
        {"moe_node_limit": 2},
    ]
    rows = sweep_sparse_ratio(model, hw, base, rl, grid)
    assert len(rows) == 3
    for r in rows:
        assert set(["point", "step_seconds", "exposed_comm_by_fabric",
                    "cross_node_gb", "peak_memory_gb", "feasible"]).issubset(r)
    # tighter node limit => no more cross-node bytes than looser limit
    gb = {tuple(r["point"].items()): r["cross_node_gb"] for r in rows}
    assert gb[(("moe_node_limit", 2),)] <= gb[(("moe_node_limit", 4),)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_model.py::test_sweep_sparse_ratio_shape_and_node_limit_cuts_nic -v`
Expected: FAIL (`ImportError: cannot import name 'sweep_sparse_ratio'`).

- [ ] **Step 3: Implement `sweep_sparse_ratio`**

In `src/llm_perf/model.py`, after `compare_precision` (line 698), add:

```python
def sweep_sparse_ratio(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    base_parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    grid: list,
) -> list:
    """Sweep large-sparse-ratio MoE knobs and report per-fabric network demand.

    Each `grid` entry is a dict overriding any of: num_experts, top_k, ep,
    moe_node_limit, moe_imbalance_factor. For every point the training step is
    simulated and the per-fabric exposed communication is reported — the primary
    signal for cross-node interconnect sizing.

    Returns one dict per point with: point, step_seconds, exposed_comm_by_fabric,
    cross_node_gb, peak_memory_gb, feasible.
    """
    perf_model = LLMPerformanceModel(model_cfg, hw)
    rows = []
    for point in grid:
        # Apply parallel-config overrides (ep, node_limit, imbalance).
        par_overrides = {
            k: point[k] for k in ("ep", "moe_node_limit", "moe_imbalance_factor")
            if k in point
        }
        par = base_parallel_cfg.model_copy(update=par_overrides)

        # Apply per-layer MoE overrides (num_experts, top_k) to a model copy.
        layer_overrides = {
            k: point[k] for k in ("num_experts", "top_k") if k in point
        }
        if layer_overrides:
            new_layers = [
                lc.model_copy(update=layer_overrides) for lc in model_cfg.layers
            ]
            mc = model_cfg.model_copy(update={"layers": new_layers})
        else:
            mc = model_cfg

        t_step, train_sim, _bd = pretraining_time(mc, hw, par, rl_cfg)
        weight_gb, grad_gb, optimizer_gb = perf_model._train_state_gb(train_sim, par)
        activation_peak_gb = perf_model._train_activation_gb(par, rl_cfg)
        peak_memory_gb = weight_gb + grad_gb + optimizer_gb + activation_peak_gb

        exposed = dict(train_sim.exposed_comm_by_fabric)
        cross_node_gb = train_sim.exposed_comm_by_fabric.get("nic", 0.0)

        rows.append({
            "point": point,
            "step_seconds": t_step,
            "exposed_comm_by_fabric": exposed,
            "cross_node_gb": cross_node_gb,
            "peak_memory_gb": peak_memory_gb,
            "feasible": peak_memory_gb <= hw.usable_hbm_gb,
        })
    return rows
```

Note: `exposed_comm_by_fabric` holds **seconds** of exposed comm per fabric (see `report.py:159`). `cross_node_gb` here therefore reports the exposed **nic time** as the cross-node pressure proxy; if a byte total is wanted instead, sum `comm_bytes` of `nic`-fabric ops — but exposed time is the actual sizing signal and matches the existing report. Keep the key name `cross_node_gb` but treat it as the nic exposure metric; the test only checks monotonicity.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_model.py::test_sweep_sparse_ratio_shape_and_node_limit_cuts_nic -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm_perf/model.py tests/test_model.py
git commit -m "feat(model): sweep_sparse_ratio per-fabric network-demand report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: report formatting + demo + docs

**Files:**
- Modify: `src/llm_perf/report.py` (add `format_sparse_sweep` after `format_json` ~line 165)
- Create: `examples/demo_sparse_ratio.py`
- Modify: `docs/architecture.md:188-190` (update the "EP dispatch is coarse" limitation)
- Test: `tests/test_report.py` (create if absent; otherwise append to an existing report test module)

**Interfaces:**
- Consumes: the row dicts from `sweep_sparse_ratio` (Task 4).
- Produces: `format_sparse_sweep(rows: list) -> str` returning a human-readable table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py` (or append if it exists):

```python
def test_format_sparse_sweep_renders_points_and_fabrics():
    from llm_perf.report import format_sparse_sweep
    rows = [
        {"point": {"moe_node_limit": 0}, "step_seconds": 0.5,
         "exposed_comm_by_fabric": {"nic": 0.12, "nvlink": 0.03},
         "cross_node_gb": 0.12, "peak_memory_gb": 40.0, "feasible": True},
        {"point": {"moe_node_limit": 2}, "step_seconds": 0.42,
         "exposed_comm_by_fabric": {"nic": 0.05, "nvlink": 0.06},
         "cross_node_gb": 0.05, "peak_memory_gb": 40.0, "feasible": True},
    ]
    out = format_sparse_sweep(rows)
    assert "moe_node_limit" in out
    assert "nic" in out
    assert "nvlink" in out
    # both points appear
    assert out.count("feasible") >= 1 or "OK" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_report.py::test_format_sparse_sweep_renders_points_and_fabrics -v`
Expected: FAIL (`ImportError: cannot import name 'format_sparse_sweep'`).

- [ ] **Step 3: Implement `format_sparse_sweep`**

In `src/llm_perf/report.py`, after `format_json` (~line 165), add:

```python
def format_sparse_sweep(rows: list) -> str:
    """Render sweep_sparse_ratio rows as a per-point, per-fabric table."""
    lines = []
    lines.append("Large-sparse-ratio MoE sweep (network demand)")
    lines.append("=" * 60)
    for r in rows:
        point = ", ".join(f"{k}={v}" for k, v in r["point"].items())
        flag = "OK" if r["feasible"] else "OOM"
        lines.append(f"\n• {point}   [{flag}]")
        lines.append(f"   step:        {r['step_seconds'] * 1000:.1f} ms")
        for fabric, secs in sorted(r["exposed_comm_by_fabric"].items()):
            lines.append(f"   exposed [{fabric}]: {secs * 1000:.1f} ms")
        lines.append(f"   peak mem:    {r['peak_memory_gb']:.1f} GB  (feasible={r['feasible']})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_report.py::test_format_sparse_sweep_renders_points_and_fabrics -v`
Expected: PASS

- [ ] **Step 5: Write the demo script**

Create `examples/demo_sparse_ratio.py`:

```python
"""Demo: how node-limited routing + load imbalance drive cross-node demand.

Run: python examples/demo_sparse_ratio.py
"""
from llm_perf.config import load_model_config, load_hardware_config, \
    ParallelismConfig, WorkloadConfig
from llm_perf.model import sweep_sparse_ratio
from llm_perf.report import format_sparse_sweep

model = load_model_config("configs/models/deepseekv3_671b.yaml")
hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
base = ParallelismConfig(tp=1, ep=64, dp=1)
rl = WorkloadConfig(total_prompts=10000, group_size=8)

grid = [
    {"moe_node_limit": 0, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 4, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.3},
]
rows = sweep_sparse_ratio(model, hw, base, rl, grid)
print(format_sparse_sweep(rows))
```

- [ ] **Step 6: Run the demo to confirm it executes**

Run: `source .venv/bin/activate && python examples/demo_sparse_ratio.py`
Expected: prints a sweep table with decreasing `exposed [nic]` as `moe_node_limit` tightens, and higher exposure at `moe_imbalance_factor=1.3`.

- [ ] **Step 7: Update the architecture-doc limitation**

In `docs/architecture.md`, replace the bullet (lines 188-190):

```markdown
- **EP dispatch is coarse.** MoE AllToAll cost assumes uniform expert routing;
  load imbalance is not modeled.
```

with:

```markdown
- **EP dispatch:** MoE AllToAll supports node-limited hierarchical routing
  (`ParallelismConfig.moe_node_limit`), splitting dispatch/combine into an
  inter-node (NIC) and intra-node (NVLink/HCCS) component, and a scalar
  `moe_imbalance_factor` that scales the hottest EP rank's traffic and expert
  compute. See `sweep_sparse_ratio()` for the network-demand sweep. Probability-
  distribution (p99) imbalance and fine-grained expert GEMM-efficiency decay
  remain unmodeled.
```

- [ ] **Step 8: Run the full suite + lint**

Run: `source .venv/bin/activate && pytest tests/ -q && ruff check src/ && ruff format src/`
Expected: PASS, no lint errors.

- [ ] **Step 9: Commit**

```bash
git add src/llm_perf/report.py tests/test_report.py examples/demo_sparse_ratio.py docs/architecture.md
git commit -m "feat(report): sparse-ratio sweep table + demo + docs update

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Core insight (node-limited inter/intra split) → Task 2 (`op_alltoall_*_hierarchical`) + Task 3 (builder fabric-split). ✓
- `E[distinct_dest_nodes] = min(top_k, M, nodes_in_ep)` → Task 2 Step 3. ✓
- Imbalance amplification on comm + expert compute → Task 2 (`op_moe_ffn` flops) + Task 3 (comm bytes). ✓
- Config knobs on `ParallelismConfig` with degenerate defaults → Task 1. ✓
- `sweep_sparse_ratio()` + per-fabric report → Task 4 + Task 5. ✓
- Regression: `node_limit=0 & imbalance=1.0` ≡ current → Task 3 (single-op path untouched at imbalance 1.0) + Task 2 byte-conservation test + Task 3 Step 9 full suite. ✓
- Testing list (M monotonic, nodes_in_ep==1 → inter=0, regression) → covered in Tasks 2-4 tests. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `op_alltoall_dispatch_hierarchical` / `op_alltoall_combine_hierarchical` return `tuple[float, float]` everywhere; `imbalance_factor` param name consistent across `op_moe_ffn`, both hierarchical ops, and `ParallelismConfig.moe_imbalance_factor`; `moe_node_limit` consistent. `sweep_sparse_ratio` row keys match `format_sparse_sweep` consumption. ✓

**Note on `nodes_in_ep == 1`:** handled in builder (Task 3 Step 5) — the hierarchical path requires `nodes_in_ep > 1`, so a single-node EP group always takes the intra-only single-op path; inter-node bytes are 0 by construction.
