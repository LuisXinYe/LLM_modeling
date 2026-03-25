# RL Performance Modeling — Phase 2 Design Spec

**Date:** 2026-03-25
**Status:** Draft
**Author:** horacehxw + Claude
**Parent Spec:** `docs/superpowers/specs/2026-03-25-rl-perf-modeling-design.md`

## 1. Scope

Phase 2 修复 Phase 1 (MVP) 中与 spec 的偏差，补齐缺失的底层算子建模和上层查询接口。

**包含：**
- Memory 建模重构：从 SimResult 读 weight/activation，对齐 spec §7.1
- MTP (Multi-Token Prediction) 算子建模：训练 + 推理投机解码
- CP (Context Parallelism) 通信建模：Ring CP
- SP (Sequence Parallelism) 通信建模：AllReduce → AllGather + ReduceScatter
- 上层查询接口：what_if、sensitivity
- 输出格式：format_json
- startup_overhead 修复

**不包含：**
- Phase enum 变更（保持 4 phase MVP）
- PP schedule DAG 调度（保持解析近似）
- CLI 新增命令（plan/what-if/sweep/search）
- resource_plan / search_optimal 查询方法
- SimOp comm_bytes 字段

## 2. Config Changes (config.py)

### 2.1 RLConfig 新增字段

```python
class RLConfig(BaseModel):
    # ... 现有字段不变 ...
    use_speculative_decoding: bool = False
    mtp_acceptance_len: Optional[int] = None  # None → fallback 到 mtp_depth
```

- `use_speculative_decoding`: 推理时是否启用投机解码（部署决策）
- `mtp_acceptance_len`: 平均被接受的 draft token 数（运行时特征）

### 2.2 不变的字段

- `ModelConfig.auxiliary.mtp_depth`: 已存在，builder 将开始读取
- `ParallelismConfig.cp`, `cp_type`, `sp`: 已存在，builder 将开始使用

## 3. New Operators (ops.py)

### 3.1 op_mtp_head

训练时每个 mtp_depth 额外一次 LM head forward。Ref: spec §4.3, DeepSeek-V3 §3.4.

```python
def op_mtp_head(
    hidden_size: int,
    vocab_size: int,
    mtp_depth: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """MTP extra LM heads. FLOPs = 2·d·V·mtp_depth per token."""
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

推理时（PREFILL/DECODE 标准模式）不使用 MTP，FLOPs = 0。

### 3.2 op_ring_cp

Ring CP 通信。Ref: spec §4.4.

```python
def op_ring_cp(
    seq_len: int,
    cp_size: int,
    kv_dim: int,       # num_kv_heads * head_dim（TP 后）或 MLA 的 d_c + d_R
    dtype_bytes: int = 2,
) -> OpCost:
    """Ring CP. 2 × (S/CP) × d_kv × bytes × (CP-1). Ref: spec §4.4."""
    comm_b = 2 * (seq_len / cp_size) * kv_dim * dtype_bytes * (cp_size - 1)
    return OpCost(comm_bytes=comm_b)
```

### 3.3 SP 不需要新 op

SP 使用已有的 `op_allgather` 和 `op_reducescatter`。

## 4. Builder Changes (builder.py)

### 4.1 SP: AllReduce → AllGather + ReduceScatter

当 `sp=True && tp > 1` 时，`build_layer_ops` 中的 TP 通信模式变更：

**现有逻辑:**
```
attention → tp_allreduce_attn → norm → ffn → tp_allreduce_ffn
```

**sp=True 时:**
```
tp_allgather_attn → attention → tp_reducescatter_attn →
tp_allgather_ffn → norm → ffn → tp_reducescatter_ffn
```

**通信量公式：** AllGather 和 ReduceScatter 的 `msg` 参数使用 `batch * seq_len * hidden_size * dtype_bytes`（与现有 `_build_tp_allreduce` 相同）。这是因为：
- AllGather: 每个 rank 持有 `msg / tp` 的数据，收集后为 `msg`，传输 `msg * (tp-1) / tp`
- ReduceScatter: 输入 `msg`，每个 rank 得到 `msg / tp`，传输 `msg * (tp-1) / tp`
- 总和 = `2 * msg * (tp-1) / tp`，与 AllReduce 相同

拆成两个 op 后可在不同 stream 上与 compute overlap。

### 4.2 CP: Ring CP 通信

当 `cp > 1` 时，在 `build_layer_ops` 的 attention op 前插入 Ring CP 通信 op：

```
norm → [cp_ring_kv] → attention → ...
```

CP 通信放在 `cp_comm` stream（新 stream），和 compute/tp_comm 天然 overlap。

**comm_time 参数：** CP ring 通常跨节点（cp_size 个 rank 分布在不同 node）。`is_intra_node` 判断使用与 TP 相同的逻辑：`cp_size <= hw.devices_per_node`。Duration 计算：`ops.comm_time(cp_cost, hw, is_intra_node=_is_intra_node(cp, hw))`。

**同时：** attention 的 `seq_len` 除以 `cp`（每个 CP rank 只处理 `S/CP` 长度）。

**KV dim 的计算依赖 attention type：**
- GQA/MHA/SWA: `tp_kv_heads * head_dim`
- MLA: `kv_compression_dim + rope_dim`

### 4.3 MTP: 训练阶段插入 LM head op

在 `build_training_step` 中，当 `model_cfg.auxiliary.mtp_depth > 0` 时：

- Forward pass 结束后（所有层 forward ops 之后）：插入 MTP forward op
- Backward pass 开始前：插入 MTP backward op

```python
if model_cfg.auxiliary and model_cfg.auxiliary.get("mtp_depth", 0) > 0:
    mtp_cost = ops.op_mtp_head(
        hidden_size=model_cfg.hidden_size,
        vocab_size=model_cfg.vocab_size,
        mtp_depth=model_cfg.auxiliary["mtp_depth"],
        batch_tokens=batch * seq_len,
        phase=Phase.TRAIN_FWD,
        dtype_bytes=model_cfg.dtype_bytes,
    )
    mtp_op = SimOp(
        name="mtp_head_fwd",
        stream="compute",
        duration=ops.roofline_time(mtp_cost, hw, is_large_gemm=True),
        depends_on=[prev_dep],
        weight_bytes=mtp_cost.weight_bytes,
        output_bytes=mtp_cost.output_bytes,
    )
    all_ops.append(mtp_op)
    prev_dep = len(all_ops) - 1
```

Backward 同理，使用 `Phase.TRAIN_BWD`。

推理阶段（`build_generation_step`）：不插入 MTP op。投机解码影响在 pipeline.py 层处理。

## 5. Pipeline Changes (pipeline.py)

### 5.1 返回 SimResult + t_per_batch

`generation_time` 和 `training_time` 返回值从 `float` 扩展为 tuple。

`generation_time` 返回 3-tuple `(total_time, train_sim_for_weights, t_per_batch)`：

```python
def generation_time(...) -> tuple[float, SimResult, float]:
    prefill_ops, decode_ops = build_generation_step(...)
    prefill_sim = simulate(prefill_ops)
    decode_sim = simulate(decode_ops)
    # ... 现有 batch/time 计算 ...
    # prefill_sim 用于 memory profile 中的 weight_bytes（与 decode_sim 相同）
    # prefill_sim.peak_activation_bytes 不用于 memory profile（gen memory = weight + KV cache）
    return total_time, prefill_sim, t_per_batch
```

**为什么返回 `prefill_sim`：** gen 阶段的 memory profile 不使用 `peak_activation_bytes`（gen peak = weight + KV cache，均为解析计算），`prefill_sim` 仅用于提供 `weight_bytes`（与 `decode_sim` 相同）。

`training_time` 返回 2-tuple：

```python
def training_time(...) -> tuple[float, SimResult]:
    train_ops = build_training_step(...)
    train_sim = simulate(train_ops)
    # ... bubble ratio, penalty ...
    return total_time, train_sim
```

### 5.2 startup_overhead 修复

Spec §6.1: `startup_overhead = 一个 gen batch 的完整时间`

现有: `startup = t_prefill`（缺 decode 部分）

修复: `t_per_batch = t_prefill + eff_len * t_decode_per_token`，作为 `generation_time` 返回值的第三项。`model.py` 中: `startup = t_per_batch`。

### 5.3 投机解码吞吐乘数

Spec §4.3: `effective_throughput = base × acceptance_len / (1 + draft_overhead)`

当 `rl_cfg.use_speculative_decoding=True` 时，在 `generation_time` 中应用：

```python
if rl_cfg.use_speculative_decoding:
    mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)
    acceptance_len = rl_cfg.mtp_acceptance_len or mtp_depth
    # draft overhead: mtp_depth 次 LM head forward 的时间 / 单次 decode 时间
    # batch_tokens = gen_batch_size (每步每个 sequence 产出 1 个 verify token)
    draft_cost = ops.op_mtp_head(
        model_cfg.hidden_size, model_cfg.vocab_size, mtp_depth,
        batch_tokens=rl_cfg.gen_batch_size,
        phase=Phase.DECODE,  # decode-time draft generation
        dtype_bytes=model_cfg.dtype_bytes,
    )
    draft_overhead = ops.roofline_time(draft_cost, hw) / t_decode_per_token
    throughput_multiplier = acceptance_len / (1 + draft_overhead)
    total_time /= throughput_multiplier
```

**batch_tokens 说明：** `gen_batch_size` 是 sequence 数。每步每个 sequence 产出 1 个 verify token + `mtp_depth` 个 draft token。`op_mtp_head` 的 `batch_tokens` 参数传入 `gen_batch_size` 是因为 draft head 对每个 sequence 的最后一个 hidden state 做一次 linear，共 `gen_batch_size` 个 hidden state。

## 6. Model API Changes (model.py)

### 6.1 _compute_memory_profile 重构

**核心变更:** 接收 SimResult，从中读 weight_bytes 和 peak_activation_bytes。

**前置验证：** 现有 `simulator.py` 对 `weight_bytes` 是直接 `sum(op.weight_bytes)`。`builder.py` 在 `Phase.TRAIN_BWD` 时已将 `weight_bytes` 置 0（避免 fwd+bwd 双算），因此 `train_sim.weight_bytes` = fwd 层的 weight 总量（per TP/EP shard），无需额外去重。实现前需验证此行为的测试覆盖。

```python
def _compute_memory_profile(self, train_sim: SimResult, gen_sim: SimResult,
                             train_parallel, gen_parallel, rl_cfg) -> MemoryProfile:
    # ---- 从 SimResult 读（ephemeral memory）----
    train_weight_gb = train_sim.weight_bytes / 1e9
    gen_weight_gb = gen_sim.weight_bytes / 1e9  # gen/train 可能不同 TP，weight per device 不同
    activation_peak_gb = train_sim.peak_activation_bytes / 1e9

    # ---- 解析计算（static + persistent dynamic）----

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

    # Totals (train and gen use their respective weight_gb)
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

**删除:** 现有 `_compute_memory` 中的独立 weight 计算（per-layer estimate、embed weight）和 activation 公式（Megatron 34*sbh）。

### 6.2 derive_targets 适配

```python
def derive_targets(self, total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None):
    t_gen, gen_sim, t_per_batch = generation_time(self.model, self.hw, gen_parallel, rl_cfg)
    t_train, train_sim = training_time(self.model, self.hw, train_parallel, rl_cfg)

    startup = t_per_batch
    t_epoch = epoch_time(t_gen, t_train, startup, colocated=rl_cfg.colocated)
    ...
    memory = self._compute_memory_profile(train_sim, gen_sim, train_parallel, gen_parallel, rl_cfg)
    ...
```

### 6.3 what_if — Spec §8.1

```python
def what_if(self, base_config: dict, overrides: dict,
            total_devices, gen_parallel, train_parallel, time_budget_hours=None) -> TargetReport:
    """base_config + overrides → TargetReport for comparison."""
    rl_cfg = RLConfig(**{**base_config, **overrides})
    return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours)
```

**限制：** `what_if` 的 overrides 仅作用于 `RLConfig` 字段。并行策略（gen_parallel/train_parallel）通过参数直接传入，不在 overrides 范围内。如需对比不同并行配置，应直接调用两次 `derive_targets`。

### 6.4 sensitivity — Spec §8.1

```python
def sensitivity(self, rl_cfg: RLConfig, param_name: str, values: list,
                total_devices, gen_parallel, train_parallel) -> list[TargetReport]:
    """Sweep one parameter across values."""
    if param_name not in RLConfig.model_fields:
        raise ValueError(f"Unknown RLConfig field: {param_name}")
    results = []
    for v in values:
        cfg = rl_cfg.model_copy(update={param_name: v})
        results.append(self.derive_targets(total_devices, cfg, gen_parallel, train_parallel))
    return results
```

注意：增加 `param_name` 校验，防止 Pydantic v2 `model_copy` 静默忽略未知字段导致所有结果相同。

## 7. Report Changes (report.py)

### 7.1 format_json

```python
import json
from dataclasses import asdict

def format_json(report: TargetReport) -> str:
    """JSON serialization of TargetReport."""
    return json.dumps(asdict(report), indent=2, default=str)
```

## 8. File Change Summary

| File | Change | Lines (est.) |
|------|--------|:---:|
| `config.py` | +2 fields to RLConfig | +5 |
| `ops.py` | +op_mtp_head, +op_ring_cp | +40 |
| `builder.py` | SP替换, CP插入, MTP插入, seq_len/cp | +80 |
| `pipeline.py` | 返回SimResult, startup fix, 投机解码 | +30 |
| `model.py` | memory重构, what_if, sensitivity | +40, -50 |
| `report.py` | +format_json | +10 |
| `tests/` | 各模块新增测试 | +150 |

**总计约 +350 行，-50 行。**

## 9. Spec Alignment Checklist

| Spec Section | Item | Status |
|-------------|------|:---:|
| §4.3 | MTP training cost | ✅ |
| §4.3 | MTP speculative decoding throughput | ✅ |
| §4.4 | Ring CP communication | ✅ |
| §3.4 | SP AllGather/ReduceScatter | ✅ |
| §6.1 | startup_overhead = full gen batch | ✅ |
| §7.1 | Memory from SimResult | ✅ |
| §8.1 | what_if query | ✅ |
| §8.1 | sensitivity query | ✅ |
| Plan Task 7 | format_json | ✅ |
| Plan Task 7 | _compute_memory_profile(sim_result) | ✅ |
