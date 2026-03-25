# RL Performance Modeling — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix spec deviations in memory modeling, add MTP/CP/SP operator support, and implement what_if/sensitivity query API.

**Architecture:** Bottom-up changes: ops.py (new operators) → config.py (new fields) → builder.py (SP/CP/MTP insertion) → pipeline.py (return SimResult, startup fix, speculative decoding) → model.py (memory refactor, query API) → report.py (format_json). TDD throughout.

**Tech Stack:** Python 3.10+, pydantic, pyyaml, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-rl-perf-phase2-design.md`

---

## Task 1: New Operators — op_mtp_head + op_ring_cp (ops.py)

**Files:**
- Modify: `src/rl_perf/ops.py` (append two new functions)
- Modify: `tests/test_ops.py` (append new tests)

- [ ] **Step 1: Write failing tests for op_mtp_head**

Append to `tests/test_ops.py`:

```python
from rl_perf.ops import op_mtp_head, op_ring_cp


# ---------------------------------------------------------------------------
# op_mtp_head
# ---------------------------------------------------------------------------


def test_op_mtp_head_train_fwd():
    """MTP FLOPs = 2 * hidden * vocab * mtp_depth * batch_tokens."""
    hidden, vocab, depth, batch = 7168, 129280, 1, 512
    cost = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_FWD)
    expected_flops = 2 * hidden * vocab * depth * batch
    assert cost.flops == expected_flops
    assert cost.weight_bytes == hidden * vocab * 2 * depth  # bf16
    assert cost.output_bytes == batch * vocab * 2  # activation kept


def test_op_mtp_head_train_bwd():
    """Backward = 2x forward FLOPs."""
    hidden, vocab, depth, batch = 7168, 129280, 1, 512
    fwd = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_FWD)
    bwd = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_BWD)
    assert bwd.flops == pytest.approx(2 * fwd.flops)
    assert bwd.output_bytes == 0  # no activation kept


def test_op_mtp_head_inference_not_used():
    """Inference phases produce no output_bytes."""
    cost = op_mtp_head(4096, 32000, 1, 64, Phase.PREFILL)
    assert cost.output_bytes == 0
    cost_d = op_mtp_head(4096, 32000, 1, 64, Phase.DECODE)
    assert cost_d.output_bytes == 0


def test_op_mtp_head_depth_scaling():
    """Doubling mtp_depth doubles FLOPs and weight_bytes."""
    base = op_mtp_head(4096, 32000, 1, 128, Phase.TRAIN_FWD)
    doubled = op_mtp_head(4096, 32000, 2, 128, Phase.TRAIN_FWD)
    assert doubled.flops == pytest.approx(2 * base.flops)
    assert doubled.weight_bytes == pytest.approx(2 * base.weight_bytes)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ops.py::test_op_mtp_head_train_fwd -v
```

Expected: FAIL (ImportError: cannot import name 'op_mtp_head')

- [ ] **Step 3: Implement op_mtp_head in ops.py**

Append to `src/rl_perf/ops.py` before the communication section:

```python
def op_mtp_head(
    hidden_size: int,
    vocab_size: int,
    mtp_depth: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """MTP extra LM heads. FLOPs = 2·d·V·mtp_depth per token. Ref: DeepSeek-V3 §3.4."""
    fwd_flops = 2 * hidden_size * vocab_size * mtp_depth * batch_tokens
    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops
    weight_b = hidden_size * vocab_size * dtype_bytes * mtp_depth
    mem_rw = weight_b + batch_tokens * vocab_size * dtype_bytes
    output_b = batch_tokens * vocab_size * dtype_bytes if phase == Phase.TRAIN_FWD else 0
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=weight_b, output_bytes=output_b)
```

- [ ] **Step 4: Run MTP tests**

```bash
pytest tests/test_ops.py -k "mtp" -v
```

Expected: all PASS

- [ ] **Step 5: Write failing tests for op_ring_cp**

Append to `tests/test_ops.py`:

```python
# ---------------------------------------------------------------------------
# op_ring_cp
# ---------------------------------------------------------------------------


def test_op_ring_cp():
    """Ring CP comm_bytes = 2 * (S/CP) * kv_dim * bytes * (CP-1)."""
    seq_len, cp_size, kv_dim = 4096, 4, 1024
    cost = op_ring_cp(seq_len, cp_size, kv_dim, dtype_bytes=2)
    expected = 2 * (seq_len / cp_size) * kv_dim * 2 * (cp_size - 1)
    assert cost.comm_bytes == pytest.approx(expected)


def test_op_ring_cp_single_rank():
    """CP=1 means no communication."""
    cost = op_ring_cp(4096, 1, 1024, dtype_bytes=2)
    assert cost.comm_bytes == 0


def test_op_ring_cp_comm_time(hw: HardwareConfig):
    """Ring CP duration should be positive for cp > 1."""
    cost = op_ring_cp(4096, 4, 1024, dtype_bytes=2)
    t = comm_time(cost, hw, group_size=4, is_intra_node=False, algorithm="ring_half")
    assert t > 0
```

- [ ] **Step 6: Run to verify they fail**

```bash
pytest tests/test_ops.py::test_op_ring_cp -v
```

Expected: FAIL (ImportError)

- [ ] **Step 7: Implement op_ring_cp in ops.py**

Append to `src/rl_perf/ops.py` in the communication section:

```python
def op_ring_cp(
    seq_len: int,
    cp_size: int,
    kv_dim: int,
    dtype_bytes: int = 2,
) -> OpCost:
    """Ring CP. 2 × (S/CP) × d_kv × bytes × (CP-1). Ref: parent spec §4.4."""
    if cp_size <= 1:
        return OpCost()
    comm_b = 2 * (seq_len / cp_size) * kv_dim * dtype_bytes * (cp_size - 1)
    return OpCost(comm_bytes=comm_b)
```

- [ ] **Step 8: Run all ops tests**

```bash
pytest tests/test_ops.py -v
```

Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add src/rl_perf/ops.py tests/test_ops.py
git commit -m "feat(ops): add op_mtp_head and op_ring_cp operators"
```

---

## Task 2: Config — RLConfig New Fields (config.py)

**Files:**
- Modify: `src/rl_perf/config.py:88-104` (RLConfig class)
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_config.py`:

```python
def test_rlconfig_speculative_decoding_defaults():
    cfg = RLConfig(total_prompts=1000)
    assert cfg.use_speculative_decoding is False
    assert cfg.mtp_acceptance_len is None


def test_rlconfig_speculative_decoding_set():
    cfg = RLConfig(total_prompts=1000, use_speculative_decoding=True, mtp_acceptance_len=3)
    assert cfg.use_speculative_decoding is True
    assert cfg.mtp_acceptance_len == 3
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config.py::test_rlconfig_speculative_decoding_defaults -v
```

Expected: FAIL (validation error — unknown field)

- [ ] **Step 3: Add fields to RLConfig**

In `src/rl_perf/config.py`, add to `RLConfig` class after `colocated`:

```python
    use_speculative_decoding: bool = False
    mtp_acceptance_len: Optional[int] = None
```

- [ ] **Step 4: Run config tests**

```bash
pytest tests/test_config.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl_perf/config.py tests/test_config.py
git commit -m "feat(config): add speculative decoding fields to RLConfig"
```

---

## Task 3: Builder — SP AllReduce Replacement (builder.py)

**Files:**
- Modify: `src/rl_perf/builder.py:44-68` (_build_tp_allreduce → generalize), `builder.py:74-319` (build_layer_ops)
- Modify: `tests/test_builder.py`

- [ ] **Step 1: Write failing tests for SP**

Append to `tests/test_builder.py`:

```python
def test_sp_replaces_allreduce_with_ag_rs(single_layer, model_cfg, hw):
    """sp=True + tp>1 should produce allgather + reducescatter instead of allreduce."""
    parallel_sp = ParallelismConfig(tp=4, pp=1, dp=1, ep=1, sp=True)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    names = [op.name for op in result]
    # Should NOT have allreduce
    assert not any("allreduce" in n for n in names), f"SP should replace allreduce: {names}"
    # Should have allgather and reducescatter
    assert any("allgather" in n for n in names), f"SP should add allgather: {names}"
    assert any("reducescatter" in n for n in names), f"SP should add reducescatter: {names}"


def test_sp_false_keeps_allreduce(single_layer, model_cfg, hw):
    """sp=False + tp>1 should still use allreduce."""
    parallel_no_sp = ParallelismConfig(tp=4, pp=1, dp=1, ep=1, sp=False)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    names = [op.name for op in result]
    assert any("allreduce" in n for n in names)
    assert not any("allgather" in n for n in names)


def test_sp_comm_volume_matches_allreduce(single_layer, model_cfg, hw):
    """SP (AG+RS) total comm volume should equal AllReduce volume."""
    tp = 4
    parallel_no_sp = ParallelismConfig(tp=tp, pp=1, dp=1, ep=1, sp=False)
    parallel_sp = ParallelismConfig(tp=tp, pp=1, dp=1, ep=1, sp=True)

    ops_no_sp = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )
    ops_sp = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_sp,
        hw=hw, batch=2, seq_len=512, phase=Phase.TRAIN_FWD,
    )

    # Sum durations of comm ops
    t_allreduce = sum(op.duration for op in ops_no_sp if "comm" in op.stream)
    t_sp = sum(op.duration for op in ops_sp if "comm" in op.stream)
    # Should be approximately equal (same total bandwidth)
    assert t_sp == pytest.approx(t_allreduce, rel=0.01)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_builder.py::test_sp_replaces_allreduce_with_ag_rs -v
```

Expected: FAIL (still produces allreduce)

- [ ] **Step 3: Add _build_tp_sp_comm helper to builder.py**

Add after `_build_tp_allreduce`:

```python
def _build_tp_allgather(
    name: str,
    batch: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int,
    tp: int,
    hw: HardwareConfig,
    dep_idx: int,
) -> SimOp:
    """Build a TP AllGather SimOp for SP (sequence parallelism)."""
    msg = batch * seq_len * hidden_size * dtype_bytes
    cost = ops.op_allgather(msg, group_size=tp)
    duration = ops.comm_time(
        cost, hw, group_size=tp, is_intra_node=_is_intra_node(tp, hw), algorithm="ring_half"
    )
    return SimOp(
        name=name, stream="tp_comm", duration=duration,
        depends_on=[dep_idx], weight_bytes=0, output_bytes=0,
    )


def _build_tp_reducescatter(
    name: str,
    batch: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int,
    tp: int,
    hw: HardwareConfig,
    dep_idx: int,
) -> SimOp:
    """Build a TP ReduceScatter SimOp for SP (sequence parallelism)."""
    msg = batch * seq_len * hidden_size * dtype_bytes
    cost = ops.op_reducescatter(msg, group_size=tp)
    duration = ops.comm_time(
        cost, hw, group_size=tp, is_intra_node=_is_intra_node(tp, hw), algorithm="ring_half"
    )
    return SimOp(
        name=name, stream="tp_comm", duration=duration,
        depends_on=[dep_idx], weight_bytes=0, output_bytes=0,
    )
```

- [ ] **Step 4: Modify build_layer_ops to use SP comm**

The key refactoring: replace `_build_tp_allreduce` calls with conditional SP logic. Track indices carefully using `len(result)`.

**Important: `_idx(local)` converts local index to global. Local index = position within `result` list. All `depends_on` must use `_idx()`.**

Restructure the TP comm sections of `build_layer_ops` as follows. The full op sequence for `sp=True, tp>1`:

```
[0] rmsnorm_pre_attn
[1] tp_allgather_attn      (depends_on=[_idx(0)])
[2] attention              (depends_on=[_idx(1)])    ← was _idx(0)
[3] tp_reducescatter_attn  (depends_on=[_idx(2)])
[4] rmsnorm_pre_ffn        (depends_on=[_idx(3)])
[5] tp_allgather_ffn       (depends_on=[_idx(4)])
[6] ffn                    (depends_on=[_idx(5)])    ← was _idx(4)
[7] tp_reducescatter_ffn   (depends_on=[_idx(6)])
```

For `sp=False, tp>1` (unchanged):

```
[0] rmsnorm_pre_attn
[1] attention              (depends_on=[_idx(0)])
[2] tp_allreduce_attn      (depends_on=[_idx(1)])
[3] rmsnorm_pre_ffn        (depends_on=[_idx(2)])
[4] ffn                    (depends_on=[_idx(3)])
[5] tp_allreduce_ffn       (depends_on=[_idx(4)])
```

Implementation approach: after each op is appended to `result`, compute `last_idx = _idx(len(result) - 1)` and pass it as `depends_on` to the next op. This is already the pattern used for `last_compute_idx` / `last_compute_local` — extend it to SP ops.

Replace the attention TP comm block (between attention and pre-FFN norm):

```python
    # ---- 1b. SP AllGather before attention (if sp=True) ----------------------
    if tp > 1 and parallel_cfg.sp:
        tp_ag_attn = _build_tp_allgather(
            name="tp_allgather_attn",
            batch=batch, seq_len=seq_len, hidden_size=d,
            dtype_bytes=dtype_bytes, tp=tp, hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm1 (last appended)
        )
        result.append(tp_ag_attn)
        attn_dep_idx = _idx(len(result) - 1)
    else:
        attn_dep_idx = _idx(len(result) - 1)  # norm1

    # ---- 1. Attention (depends on attn_dep_idx) ----------------------------
    attn_op = SimOp(
        ...
        depends_on=[attn_dep_idx],  # was _idx(0), now dynamic
        ...
    )
    result.append(attn_op)

    # ---- 2. TP Communication (attention) ------------------------------------
    if tp > 1:
        if parallel_cfg.sp:
            tp_attn_comm = _build_tp_reducescatter(
                name="tp_reducescatter_attn",
                batch=batch, seq_len=seq_len, hidden_size=d,
                dtype_bytes=dtype_bytes, tp=tp, hw=hw,
                dep_idx=_idx(len(result) - 1),  # depends on attention
            )
        else:
            tp_attn_comm = _build_tp_allreduce(
                name="tp_allreduce_attn",
                batch=batch, seq_len=seq_len, hidden_size=d,
                dtype_bytes=dtype_bytes, tp=tp, hw=hw,
                dep_idx=_idx(len(result) - 1),
                start_idx=_idx(len(result)),
            )
        result.append(tp_attn_comm)
```

Apply the same AllGather-before / ReduceScatter-after pattern for FFN.

- [ ] **Step 5: Run SP tests**

```bash
pytest tests/test_builder.py -k "sp" -v
```

Expected: all PASS

- [ ] **Step 6: Run all builder tests**

```bash
pytest tests/test_builder.py -v
```

Expected: all PASS (existing tests unaffected — they use sp=False by default)

- [ ] **Step 7: Commit**

```bash
git add src/rl_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): SP AllGather + ReduceScatter replacing AllReduce when sp=True"
```

---

## Task 4: Builder — CP Ring Communication (builder.py)

**Files:**
- Modify: `src/rl_perf/builder.py` (build_layer_ops)
- Modify: `tests/test_builder.py`

- [ ] **Step 1: Write failing tests for CP**

Append to `tests/test_builder.py`:

```python
def test_cp_inserts_ring_comm(single_layer, model_cfg, hw):
    """cp > 1 should insert cp_comm stream ops."""
    parallel_cp = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=4)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_cp,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "cp_comm" in streams, f"CP should add cp_comm stream: {streams}"


def test_cp_1_no_comm(single_layer, model_cfg, hw):
    """cp=1 should not insert any cp_comm ops."""
    parallel_no_cp = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=1)
    result = build_layer_ops(
        layer_cfg=single_layer, model_cfg=model_cfg, parallel_cfg=parallel_no_cp,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    streams = {op.stream for op in result}
    assert "cp_comm" not in streams


def test_cp_reduces_attention_seq_len(model_cfg, hw):
    """cp > 1 should result in less attention FLOPs (seq_len / cp)."""
    layer = model_cfg.get_layers()[0]
    parallel_1 = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=1)
    parallel_4 = ParallelismConfig(tp=1, pp=1, dp=1, ep=1, cp=4)

    ops_1 = build_layer_ops(
        layer_cfg=layer, model_cfg=model_cfg, parallel_cfg=parallel_1,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )
    ops_4 = build_layer_ops(
        layer_cfg=layer, model_cfg=model_cfg, parallel_cfg=parallel_4,
        hw=hw, batch=2, seq_len=4096, phase=Phase.TRAIN_FWD,
    )

    # Compute-only time should be less with CP (shorter seq_len for attention)
    compute_time_1 = sum(op.duration for op in ops_1 if op.stream == "compute")
    compute_time_4 = sum(op.duration for op in ops_4 if op.stream == "compute")
    assert compute_time_4 < compute_time_1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_builder.py::test_cp_inserts_ring_comm -v
```

Expected: FAIL (no cp_comm stream)

- [ ] **Step 3: Implement CP in build_layer_ops**

In `build_layer_ops`, after the attention type dispatch and before TP comm:

1. Divide `seq_len` by `cp` for attention computation:

```python
    cp = parallel_cfg.cp
    attn_seq_len = seq_len // cp if cp > 1 else seq_len
```

Use `attn_seq_len` instead of `seq_len` in the attention op calls.

2. Insert CP ring comm op before attention (after norm1):

```python
    # ---- 0c. CP Ring communication -----------------------------------------
    if cp > 1:
        # Determine kv_dim based on attention type
        if attn_type == "MLA":
            kv_dim = layer_cfg.kv_compression_dim + layer_cfg.rope_dim
        elif attn_type in ("GQA", "MHA", "SWA"):
            tp_kv_heads = max(1, layer_cfg.num_kv_heads // tp)
            kv_dim = tp_kv_heads * layer_cfg.head_dim
        else:
            kv_dim = layer_cfg.num_kv_heads * layer_cfg.head_dim

        cp_cost = ops.op_ring_cp(seq_len, cp, kv_dim, dtype_bytes)
        cp_ring = SimOp(
            name="cp_ring_kv",
            stream="cp_comm",
            duration=ops.comm_time(
                cp_cost, hw, group_size=cp,
                is_intra_node=_is_intra_node(cp, hw), algorithm="ring_half"
            ),
            depends_on=[_idx(0)],  # depends on norm1
            weight_bytes=0,
            output_bytes=0,
        )
        result.append(cp_ring)
```

- [ ] **Step 4: Run CP tests**

```bash
pytest tests/test_builder.py -k "cp" -v
```

Expected: all PASS

- [ ] **Step 5: Run all builder tests**

```bash
pytest tests/test_builder.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/rl_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): Ring CP communication for cp > 1"
```

---

## Task 5: Builder — MTP Head Insertion (builder.py)

**Files:**
- Modify: `src/rl_perf/builder.py` (build_training_step)
- Modify: `tests/test_builder.py`

- [ ] **Step 1: Write failing tests for MTP in builder**

Append to `tests/test_builder.py`:

```python
def test_mtp_inserts_head_ops(hw, rl_cfg):
    """Models with mtp_depth > 0 should have mtp_head ops in training."""
    mc = ModelConfig(
        name="test_mtp", hidden_size=4096, vocab_size=32000, num_layers=2, dtype="bf16",
        default_layer=LayerConfig(
            attention="GQA", num_heads=32, num_kv_heads=8, head_dim=128,
            ffn="SwiGLU", intermediate_size=11008,
        ),
        auxiliary={"mtp_depth": 1},
    )
    parallel = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)
    all_ops = build_training_step(mc, hw, parallel, rl_cfg)
    names = [op.name for op in all_ops]
    assert any("mtp" in n for n in names), f"Should have MTP ops: {names}"
    mtp_ops = [n for n in names if "mtp" in n]
    assert len(mtp_ops) == 2  # fwd + bwd


def test_no_mtp_without_auxiliary(model_cfg, hw, parallel_cfg_tp1, rl_cfg):
    """Models without auxiliary.mtp_depth should have no mtp ops."""
    all_ops = build_training_step(model_cfg, hw, parallel_cfg_tp1, rl_cfg)
    names = [op.name for op in all_ops]
    assert not any("mtp" in n for n in names)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_builder.py::test_mtp_inserts_head_ops -v
```

Expected: FAIL (no mtp ops)

- [ ] **Step 3: Implement MTP insertion in build_training_step**

In `build_training_step`, after the forward layer loop and before the backward loop:

```python
    # ------ MTP head (forward) ------------------------------------------------
    mtp_depth = 0
    if model_cfg.auxiliary:
        mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)

    if mtp_depth > 0:
        mtp_fwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * seq_len,
            phase=Phase.TRAIN_FWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_fwd = SimOp(
            name="mtp_head_fwd",
            stream="compute",
            duration=ops.roofline_time(mtp_fwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=mtp_fwd_cost.weight_bytes,
            output_bytes=mtp_fwd_cost.output_bytes,
        )
        all_ops.append(mtp_fwd)
        prev_dep = len(all_ops) - 1
```

After the backward layer loop and before DP grad sync:

```python
    # ------ MTP head (backward) -----------------------------------------------
    if mtp_depth > 0:
        mtp_bwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * seq_len,
            phase=Phase.TRAIN_BWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_bwd = SimOp(
            name="mtp_head_bwd",
            stream="compute",
            duration=ops.roofline_time(mtp_bwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=0,  # BWD: no weight counting
            output_bytes=0,
        )
        all_ops.append(mtp_bwd)
        prev_dep = len(all_ops) - 1
```

- [ ] **Step 4: Run MTP builder tests**

```bash
pytest tests/test_builder.py -k "mtp" -v
```

Expected: all PASS

- [ ] **Step 5: Run all builder tests**

```bash
pytest tests/test_builder.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/rl_perf/builder.py tests/test_builder.py
git commit -m "feat(builder): MTP head ops in training forward/backward"
```

---

## Task 6: Pipeline — Return SimResult + Startup Fix + Speculative Decoding (pipeline.py)

**Depends on:** Task 1 (op_mtp_head) and Task 2 (RLConfig fields) — must be completed first.

**Files:**
- Modify: `src/rl_perf/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_pipeline.py`:

```python
from rl_perf.simulator import SimResult


def test_generation_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """generation_time should return (total_time, SimResult, t_per_batch)."""
    result = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 3
    t, sim, t_batch = result
    assert t > 0
    assert isinstance(sim, SimResult)
    assert t_batch > 0
    assert t_batch < t  # single batch < total


def test_training_time_returns_tuple(model_cfg, hw, parallel_cfg, rl_cfg):
    """training_time should return (total_time, SimResult)."""
    result = training_time(model_cfg, hw, parallel_cfg, rl_cfg)
    assert isinstance(result, tuple)
    assert len(result) == 2
    t, sim = result
    assert t > 0
    assert isinstance(sim, SimResult)
    assert sim.weight_bytes > 0


def test_startup_overhead_includes_decode(model_cfg, hw, parallel_cfg, rl_cfg):
    """t_per_batch should be > prefill-only time."""
    _, sim, t_per_batch = generation_time(model_cfg, hw, parallel_cfg, rl_cfg)
    from rl_perf.builder import build_generation_step
    from rl_perf.simulator import simulate
    prefill_ops, _ = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)
    t_prefill = simulate(prefill_ops).wall_clock_time
    assert t_per_batch > t_prefill  # includes decode portion
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_pipeline.py::test_generation_time_returns_tuple -v
```

Expected: FAIL (returns float, not tuple)

- [ ] **Step 3: Modify generation_time to return 3-tuple**

First, add `Phase` to the import in `src/rl_perf/pipeline.py`:

```python
from rl_perf.config import ModelConfig, HardwareConfig, ParallelismConfig, RLConfig, Phase
```

Then change `generation_time`:

```python
def generation_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total generation time in seconds. Returns (total_time, sim_result, t_per_batch)."""
    prefill_ops, decode_ops = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)

    prefill_sim = simulate(prefill_ops)
    t_prefill = prefill_sim.wall_clock_time
    t_decode_per_token = simulate(decode_ops).wall_clock_time

    eff_len = effective_response_len(
        avg=rl_cfg.avg_response_len,
        std=rl_cfg.std_response_len,
        batch_size=rl_cfg.gen_batch_size,
        max_len=rl_cfg.max_response_len,
    )

    t_per_batch = t_prefill + eff_len * t_decode_per_token

    # Speculative decoding throughput multiplier (spec §4.3)
    if rl_cfg.use_speculative_decoding:
        mtp_depth = (model_cfg.auxiliary or {}).get("mtp_depth", 0)
        if mtp_depth > 0:
            acceptance_len = rl_cfg.mtp_acceptance_len or mtp_depth
            from rl_perf import ops
            draft_cost = ops.op_mtp_head(
                model_cfg.hidden_size, model_cfg.vocab_size, mtp_depth,
                batch_tokens=rl_cfg.gen_batch_size,
                phase=Phase.DECODE,
                dtype_bytes=model_cfg.dtype_bytes,
            )
            draft_overhead = ops.roofline_time(draft_cost, hw) / t_decode_per_token
            throughput_multiplier = acceptance_len / (1 + draft_overhead)
            t_per_batch = t_prefill + (eff_len / throughput_multiplier) * t_decode_per_token

    total_responses = rl_cfg.total_responses
    gen_dp = parallel_cfg.dp
    batches = math.ceil(total_responses / (rl_cfg.gen_batch_size * gen_dp))

    return batches * t_per_batch, prefill_sim, t_per_batch
```

- [ ] **Step 4: Modify training_time to return 2-tuple**

```python
def training_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total training time in seconds. Returns (total_time, sim_result)."""
    train_ops = build_training_step(model_cfg, hw, parallel_cfg, rl_cfg)
    train_sim = simulate(train_ops)
    t_step = train_sim.wall_clock_time

    # PP bubble ratio
    pp = parallel_cfg.pp
    if pp > 1:
        M = rl_cfg.gradient_accumulation_steps
        bubble_ratio = (pp - 1) / (M + pp - 1)
        t_step *= (1 + bubble_ratio)

    # Perf penalties
    penalty = 1.0
    if parallel_cfg.full_recomputation:
        penalty *= 1.30
    elif parallel_cfg.recompute_attention:
        penalty *= 1.05
    if parallel_cfg.optimizer_offload:
        penalty *= 1.15
    if parallel_cfg.activation_offload:
        penalty *= 1.10
    t_step *= penalty

    total_responses = rl_cfg.total_responses
    effective_batch = rl_cfg.train_micro_batch_size * rl_cfg.gradient_accumulation_steps * parallel_cfg.dp
    num_steps = math.ceil(total_responses / effective_batch)

    return num_steps * t_step, train_sim
```

- [ ] **Step 5: Run pipeline tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/rl_perf/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): return SimResult, fix startup_overhead, add speculative decoding"
```

---

## Task 7: Model API — Memory Refactor + what_if + sensitivity (model.py)

**Files:**
- Modify: `src/rl_perf/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
def test_memory_from_sim_result(perf_model, rl_cfg):
    """Memory profile should use SimResult weight_bytes."""
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)
    assert report.memory.weight_gb > 0
    assert report.memory.activation_peak_gb > 0


def test_what_if(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    base = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)

    result = perf_model.what_if(
        base_config=rl_cfg.model_dump(),
        overrides={"group_size": 16},
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert result.epoch_time_hours > base.epoch_time_hours  # more data = longer


def test_sensitivity(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    results = perf_model.sensitivity(
        rl_cfg=rl_cfg, param_name="group_size", values=[4, 8, 16],
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert len(results) == 3
    # More group_size → more responses → longer epoch
    assert results[2].epoch_time_hours > results[0].epoch_time_hours


def test_sensitivity_invalid_param(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    with pytest.raises(ValueError, match="Unknown RLConfig field"):
        perf_model.sensitivity(
            rl_cfg=rl_cfg, param_name="nonexistent_field", values=[1, 2],
            total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
        )


def test_weight_bytes_no_double_count():
    """Verify builder zeroes weight_bytes on BWD ops so SimResult doesn't double-count."""
    from rl_perf.config import load_model_config, load_hardware_config
    from rl_perf.builder import build_training_step
    from rl_perf.simulator import simulate
    from pathlib import Path

    CONFIGS_DIR = Path(__file__).parent.parent / "configs"
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = RLConfig(total_prompts=100, group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)

    ops = build_training_step(mc, hw, parallel, rl)
    # BWD ops should have weight_bytes = 0
    fwd_weight = sum(op.weight_bytes for op in ops if "bwd" not in op.name.lower())
    bwd_weight = sum(op.weight_bytes for op in ops if "bwd" in op.name.lower()
                     or op.name == "optimizer_step" or op.name == "dp_grad_sync")
    assert bwd_weight == 0, "BWD ops should have weight_bytes=0 to avoid double-counting"

    sim = simulate(ops)
    assert sim.weight_bytes == pytest.approx(fwd_weight)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_model.py::test_what_if -v
```

Expected: FAIL (no method 'what_if')

- [ ] **Step 3: Rewrite model.py — derive_targets + memory refactor**

Replace `derive_targets`, `feasibility_check`, and `_compute_memory` in `src/rl_perf/model.py`:

```python
import math
from rl_perf.config import ModelConfig, HardwareConfig, ParallelismConfig, RLConfig
from rl_perf.pipeline import generation_time, training_time, epoch_time, bottleneck_analysis
from rl_perf.simulator import SimResult
from rl_perf.report import TargetReport, MemoryProfile


class RLPerformanceModel:
    def __init__(self, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.model = model_cfg
        self.hw = hw_cfg

    def derive_targets(self, total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None):
        t_gen, gen_sim, t_per_batch = generation_time(self.model, self.hw, gen_parallel, rl_cfg)
        t_train, train_sim = training_time(self.model, self.hw, train_parallel, rl_cfg)

        startup = t_per_batch
        t_epoch = epoch_time(t_gen, t_train, startup, colocated=rl_cfg.colocated)
        bottleneck, slack = bottleneck_analysis(t_gen, t_train)

        total_responses = rl_cfg.total_responses
        avg_tokens = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len

        gen_tps = total_responses * rl_cfg.avg_response_len / t_gen if t_gen > 0 else 0
        train_tps = total_responses * avg_tokens / t_train if t_train > 0 else 0
        gen_sps = total_responses / t_gen if t_gen > 0 else 0
        train_sps = total_responses / t_train if t_train > 0 else 0

        memory = self._compute_memory_profile(train_sim, gen_sim, train_parallel, gen_parallel, rl_cfg)

        within_budget = time_budget_hours is None or (t_epoch / 3600) <= time_budget_hours

        return TargetReport(
            epoch_time_hours=t_epoch / 3600,
            within_budget=within_budget,
            bottleneck=bottleneck,
            bottleneck_slack=slack,
            gen_tps_target=gen_tps,
            train_tps_target=train_tps,
            gen_samples_per_sec=gen_sps,
            train_samples_per_sec=train_sps,
            gen_time_hours=t_gen / 3600,
            train_time_hours=t_train / 3600,
            memory=memory,
            gen_parallel=gen_parallel,
            train_parallel=train_parallel,
        )

    def feasibility_check(self, total_devices, rl_cfg, gen_parallel, train_parallel):
        return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None)

    def what_if(self, base_config: dict, overrides: dict,
                total_devices, gen_parallel, train_parallel, time_budget_hours=None) -> TargetReport:
        """base_config + overrides → TargetReport for comparison."""
        rl_cfg = RLConfig(**{**base_config, **overrides})
        return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours)

    def sensitivity(self, rl_cfg: RLConfig, param_name: str, values: list,
                    total_devices, gen_parallel, train_parallel) -> list:
        """Sweep one parameter across values."""
        if param_name not in RLConfig.model_fields:
            raise ValueError(f"Unknown RLConfig field: {param_name}")
        results = []
        for v in values:
            cfg = rl_cfg.model_copy(update={param_name: v})
            results.append(self.derive_targets(total_devices, cfg, gen_parallel, train_parallel))
        return results

    def _compute_memory_profile(self, train_sim: SimResult, gen_sim: SimResult,
                                 train_parallel, gen_parallel, rl_cfg) -> MemoryProfile:
        """Memory profile: weight/activation from SimResult, KV/optimizer/ref analytical."""
        # From SimResult (ephemeral memory)
        train_weight_gb = train_sim.weight_bytes / 1e9
        gen_weight_gb = gen_sim.weight_bytes / 1e9
        activation_peak_gb = train_sim.peak_activation_bytes / 1e9

        # Optimizer: 12 bytes per param (Adam fp32 master + momentum + variance)
        param_count = train_sim.weight_bytes / self.model.dtype_bytes
        optim_bytes = param_count * 12
        if train_parallel.zero_stage >= 1:
            optim_bytes /= train_parallel.dp
        optimizer_gb = optim_bytes / 1e9

        # KV cache for generation
        layer = self.model.get_layers()[0]
        if layer.attention == "MLA":
            kv_per_token = (layer.kv_compression_dim + layer.rope_dim) * self.model.dtype_bytes
        else:
            kv_heads_per_device = max(1, layer.num_kv_heads // gen_parallel.tp)
            kv_per_token = 2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
        layers_per_stage = self.model.num_layers / gen_parallel.pp
        kv_total = (kv_per_token * layers_per_stage * rl_cfg.gen_batch_size
                    * (rl_cfg.avg_prompt_len + rl_cfg.max_response_len))
        kv_cache_gb = kv_total / 1e9

        # Reference model (uses train weight — ref model has same sharding as train)
        ref_gb = train_weight_gb if (rl_cfg.reference_model and not rl_cfg.ref_offload_cpu) else 0

        # Totals
        total_train = train_weight_gb + optimizer_gb + activation_peak_gb + ref_gb
        total_gen = gen_weight_gb + kv_cache_gb
        usable = self.hw.usable_hbm_gb

        return MemoryProfile(
            weight_gb=train_weight_gb,
            optimizer_gb=optimizer_gb,
            activation_peak_gb=activation_peak_gb,
            kv_cache_gb=kv_cache_gb,
            ref_model_gb=ref_gb,
            total_train_gb=total_train,
            total_gen_gb=total_gen,
            usable_hbm_gb=usable,
            train_feasible=total_train < usable,
            gen_feasible=total_gen < usable,
        )
```

- [ ] **Step 4: Run model tests**

```bash
pytest tests/test_model.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl_perf/model.py tests/test_model.py
git commit -m "feat(model): memory from SimResult, what_if, sensitivity query API"
```

---

## Task 8: Report — format_json (report.py)

**Files:**
- Modify: `src/rl_perf/report.py`
- Modify: `tests/test_model.py` (add JSON test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_model.py`:

```python
import json
from rl_perf.report import format_json


def test_format_json(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p)

    result = format_json(report)
    parsed = json.loads(result)
    assert "epoch_time_hours" in parsed
    assert "memory" in parsed
    assert parsed["memory"]["weight_gb"] > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_model.py::test_format_json -v
```

Expected: FAIL (ImportError)

- [ ] **Step 3: Implement format_json**

Add to `src/rl_perf/report.py`:

```python
import json
from dataclasses import asdict


def format_json(report: TargetReport) -> str:
    """JSON serialization of TargetReport."""
    return json.dumps(asdict(report), indent=2, default=str)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_model.py::test_format_json -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rl_perf/report.py tests/test_model.py
git commit -m "feat(report): add format_json for TargetReport serialization"
```

---

## Task 9: Integration — Full Test Suite + E2E Verification

**Files:**
- Modify: `tests/test_e2e.py` (add Phase 2 tests)

- [ ] **Step 1: Add e2e test for DeepSeek V3 with MTP**

Append to `tests/test_e2e.py`:

```python
def test_e2e_deepseekv3_with_mtp(rl_cfg):
    """DeepSeek V3 with mtp_depth=1 should produce valid results."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "deepseekv3_671b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))

    perf = RLPerformanceModel(mc, hw)
    gen_p = ParallelismConfig(tp=8, pp=1, dp=48, ep=1)
    train_p = ParallelismConfig(tp=8, pp=4, dp=4, ep=8)

    report = perf.derive_targets(
        total_devices=train_p.total_devices,
        rl_cfg=rl_cfg, gen_parallel=gen_p, train_parallel=train_p,
        time_budget_hours=48,
    )
    assert report.epoch_time_hours > 0
    assert report.memory is not None

    # MTP should increase training time vs no-mtp
    mc_no_mtp = mc.model_copy(update={"auxiliary": None})
    perf_no_mtp = RLPerformanceModel(mc_no_mtp, hw)
    report_no_mtp = perf_no_mtp.derive_targets(
        total_devices=8*4*4*8, rl_cfg=rl_cfg,
        gen_parallel=gen_p, train_parallel=train_p,
    )
    assert report.train_time_hours >= report_no_mtp.train_time_hours
```

- [ ] **Step 2: Add e2e test for SP + CP**

```python
def test_e2e_sp_cp_config(rl_cfg):
    """SP and CP configurations should produce valid results."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))

    perf = RLPerformanceModel(mc, hw)
    # SP enabled with TP
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, sp=True)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, sp=True)
    report = perf.derive_targets(64, rl_cfg, gen_p, train_p)
    assert report.epoch_time_hours > 0

    # CP enabled
    gen_p_cp = ParallelismConfig(tp=8, pp=1, dp=4, cp=2)
    report_cp = perf.derive_targets(64, rl_cfg, gen_p_cp, train_p)
    assert report_cp.epoch_time_hours > 0
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: Phase 2 e2e tests for MTP, SP, CP configurations"
```

---

## Task Summary

| Task | What | Files | Est. |
|------|------|-------|------|
| 1 | op_mtp_head + op_ring_cp | ops.py, test_ops.py | 10 min |
| 2 | RLConfig new fields | config.py, test_config.py | 5 min |
| 3 | SP AllGather/ReduceScatter | builder.py, test_builder.py | 15 min |
| 4 | CP Ring communication | builder.py, test_builder.py | 10 min |
| 5 | MTP head in builder | builder.py, test_builder.py | 10 min |
| 6 | Pipeline return SimResult + fixes | pipeline.py, test_pipeline.py | 15 min |
| 7 | Memory refactor + query API | model.py, test_model.py | 15 min |
| 8 | format_json | report.py, test_model.py | 5 min |
| 9 | E2E integration tests | test_e2e.py | 10 min |
