# RL Training Performance Modeling — Design Spec

**Date:** 2026-03-25
**Status:** Draft
**Author:** horacehxw + Claude

## 1. Problem Statement

给定模型、卡总数、数据量，给出 RL 训练完一轮所有数据的时间上限，推导训练和推理团队的 TPS（tokens/s）达成指标。

### 1.1 Core Inputs

| 输入 | 说明 |
|------|------|
| ModelConfig | 模型结构（Dense/MoE, attention type, layers, dims） |
| HardwareConfig | 硬件规格（910C/CM384, FLOPS, bandwidth, 互联） |
| RLConfig | RL 参数（prompt 数, group_size, avg/max response len） |
| ParallelismConfig | 并行策略（TP/SP/CP/PP/EP/DP, ZeRO, PP schedule） |
| TimeBudget | 时间预算（可选） |

### 1.2 Core Outputs

| 输出 | 说明 |
|------|------|
| epoch_time | 一轮 epoch 时间上限 |
| gen_tps_target | 推理团队 TPS 目标 |
| train_tps_target | 训练团队 TPS 目标 |
| bottleneck | 瓶颈阶段（GENERATION / TRAINING） |
| memory_profile | 各组件 memory 使用 + 可行性判断 |

### 1.3 Design Constraints

- **精度目标:** <30% 误差（解析公式 + 经验校准系数）
- **模型架构未定:** 工具必须参数化，改参数不需改代码
- **RL 框架:** veRL (大概率)，生成/训练通过 transfer queue 解耦
- **Reward:** Rule-based，计算量可忽略 → 两阶段 pipeline
- **硬件:** Ascend 910C / CloudMatrix 384
- **模型范围:** 7B ~ 数百B，Dense + MoE

## 2. Architecture

### 2.1 分层设计

```
ModelConfig (YAML)
    ↓
builder.py: build_*_step(config, parallel, hw, phase)
    ↓
List[SimOp]  ← 每个 op 携带 time + memory 信息
    ↓
simulator.py: simulate(ops)
    ↓
SimResult { wall_clock_time, weight_bytes, peak_activation_bytes, ... }
    ↓
pipeline.py: epoch_time(), bottleneck_analysis()
    ↓
TargetReport { gen_tps_target, train_tps_target, memory_profile, ... }
```

### 2.2 核心原则

1. **一次 op 建模，多指标输出** — roofline time 和 memory footprint 从同一个 op 模型导出，不分开建模
2. **并行切分在 builder 层体现** — ops.py 不感知并行策略，只接收切分后的 shape
3. **phase 区分训练/推理** — 5 种 phase（PREFILL, DECODE, TRAIN_FWD, TRAIN_BWD_DX, TRAIN_BWD_DW）统一在同一套 op 函数中
4. **builder 决定调度** — 不同 PP schedule（1F1B, ZeroBubble, DualPipe）是 builder 的不同 op 排列 + 依赖声明
5. **simulator 不关心策略语义** — 它只看 (stream, depends_on, duration)，拓扑排序 + 多时钟推进

## 3. Data Model

### 3.1 ModelConfig

```yaml
# configs/models/deepseekv3_671b.yaml
name: "DeepSeek-V3-671B"
hidden_size: 7168
vocab_size: 129280
num_layers: 61
dtype: bf16

# 支持 per-layer 不同配置 (Hybrid 架构)
# 如果所有层相同，用 default_layer; 否则用 layers 列表
default_layer:
  attention: MLA
  num_heads: 128
  num_kv_heads: 128
  head_dim: 128
  kv_compression_dim: 512
  query_compression_dim: 1536
  rope_dim: 64

  ffn: MoE
  num_experts: 256
  num_shared_experts: 1
  top_k: 8
  expert_intermediate_size: 2048
  intermediate_size: 18432  # shared expert

  residual: mHC
  mhc_expansion: 4

auxiliary:
  mtp_depth: 1  # Multi-Token Prediction
```

### 3.2 HardwareConfig

```yaml
# configs/hardware/ascend_910c.yaml
name: "Ascend 910C"
peak_tflops_bf16: 800
hbm_capacity_gb: 128
hbm_bandwidth_tb_s: 3.2
hbm_usable_ratio: 0.85  # 框架/HCCL/碎片预留

intra_node:
  bandwidth_gb_s: 400    # HCCS 机内互联
  devices_per_node: 8
inter_node:
  bandwidth_gb_s: 100    # RoCE/IB 机间
  latency_us: 5

# 可校准系数 (默认值，benchmark 可覆盖)
# 两级: compute-bound ops (大 GEMM) vs memory-bound ops (小算子/norm)
calibration:
  compute_eff_large_gemm: 0.50   # M*N*K > threshold
  compute_eff_small_op: 0.20     # element-wise, norm, etc.
  memory_efficiency: 0.70
  comm_efficiency: 0.70
```

### 3.3 RLConfig

```yaml
total_prompts: 100000
group_size: 8
avg_prompt_len: 512
avg_response_len: 2048
max_response_len: 4096   # long-tail 建模
std_response_len: 800     # 可选，用于 Gumbel 近似; 未提供时 fallback 到 max_response_len

train_micro_batch_size: 4
gradient_accumulation_steps: 8
gen_batch_size: 64

reference_model: true
ref_offload_cpu: false
colocated: false          # true=共卡, false=分离
```

### 3.4 ParallelismConfig

```yaml
# 生成阶段
gen:
  tp: 8
  pp: 1
  dp: 8
  ep: 1
  cp: 1
  cp_type: ring           # ring | ulysses

# 训练阶段
train:
  tp: 8
  pp: 4
  dp: 4
  ep: 8
  cp: 1
  sp: true                # Megatron-style Sequence Parallelism
  zero_stage: 1
  pp_schedule: dualpipe   # 1f1b | interleaved | zero_bubble | dualpipe
  pp_virtual_stages: 1    # interleaved 1F1B 的 virtual stage 数
  recompute_attention: true
  full_recomputation: false
  optimizer_offload: false
  activation_offload: false
```

### 3.5 SimOp

```python
@dataclass
class SimOp:
    name: str
    stream: str              # "compute" | "tp_comm" | "ep_comm" | "dp_comm"
    duration: float          # roofline time (seconds)
    depends_on: List[int]    # op index 列表

    # Memory
    weight_bytes: float = 0       # 持久 (权重)
    output_bytes: float = 0       # 瞬态 (activation 或 gradient，由 phase 决定)
    #   TRAIN_FWD: activation (保留到对应 BWD op 消费后释放)
    #   TRAIN_BWD_DX: input gradient (传递给上层 BWD)
    #   TRAIN_BWD_DW: weight gradient (optimizer 消费后释放)
    #   PREFILL/DECODE: activation (下一层消费后立即释放)
    consumers: List[int] = None   # 谁消费 output → 决定释放时机
```

### 3.6 SimResult

```python
@dataclass
class SimResult:
    wall_clock_time: float       # 多流模拟得到的总时间
    weight_bytes: float          # 去重后的权重总量
    peak_activation_bytes: float # activation 峰值
    total_comm_bytes: float      # 通信总量
```

## 4. L1: Operator Cost Model (ops.py)

### 4.1 统一接口

每个算子函数返回 `OpCost`，同时包含 roofline 和 memory 信息：

```python
@dataclass
class OpCost:
    flops: float          # FLOPs 数
    mem_rw: float         # HBM 读写字节数 (用于 roofline)
    weight_bytes: float   # 权重大小 (持久)
    output_bytes: float   # 输出 activation 大小 (瞬态)

def roofline_time(cost: OpCost, hw: HardwareConfig) -> float:
    compute_time = cost.flops / (hw.peak_tflops * 1e12 * hw.calibration.compute_efficiency)
    memory_time = cost.mem_rw / (hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency)
    return max(compute_time, memory_time)
```

### 4.2 参考实现资源

实现算子 cost model 时，可参考真实代码作为公式校准和实现细节的第一手来源：
- **vLLM / PyTorch / Triton** 算子实现（attention、MoE、comm 等）
- **内部 TensorCast**（msmodeling）和 **deepseek_v4_modeling** 的算子建模代码

### 4.3 Attention 变体公式 (forward, per token)

**MHA:**
- FLOPs: `8d² + 4sd`
- Weight: `4 × d × d × dtype_bytes`
- KV cache/token: `2 × H × d_h × dtype_bytes`

**GQA:**
- FLOPs: `(4 + 4G/H)d² + 4sd`
- Weight: `(2 + 2G/H) × d × d × dtype_bytes + d² × dtype_bytes` (Q+KV+O)
- KV cache/token: `2 × G × d_h × dtype_bytes`

**MLA (inference, with absorption):**
- Compression ratio: `r = H·d_h / d_c` (e.g., DeepSeek V3: r = 128×128/512 = 32)
- `d_c` = KV compression dim, `d'_c` = query compression dim, `d_R` = decoupled RoPE dim
- FLOPs: `6d·d_c + 4s·d_c + 2·d·H·d_R` (absorbed projections + latent-space attention + RoPE)
- Weight: `d·d_c` (W_dkv) + `d_c·d_c·H` (absorbed W_QK) + `d_c·d·H` (absorbed W_VO) + `2·d·H·d_R` (RoPE Q,K)
- KV cache/token: `(d_c + d_R) × dtype_bytes`

**MLA (training, no absorption):**
- FLOPs: `2d·d_c + 2d_c·d + 2d_c·d` (KV down + K up + V up) + `2d·d'_c + 2d'_c·d` (Q down + Q up) + `4sd` (attention) + `2d²` (output) = `6d·d_c + 4d·d'_c + 2d² + 4sd`
- Weight: `d·d_c + 2·d_c·d + d·d'_c + d'_c·d + d²` (down_kv + up_k + up_v + down_q + up_q + out)

**SWA:**
- FLOPs: `8d² + 4·min(s,w)·d`
- KV cache 总量 bounded by window_size

### 4.3 FFN 变体公式

**SwiGLU:** FLOPs = `6·d·d_ff`

**MoE:** FLOPs = `6·d·d_expert·top_k + 6·d·d_shared·num_shared_experts` per token
- 第一项: routed experts (top_k 个)
- 第二项: shared experts (如 DeepSeek V3 的 1 个 shared expert, d_shared=18432)
- Router: `2·d·num_experts` (negligible)

**mHC:** overhead ≈ 5-7% of layer time (memory-bandwidth-bound, not compute-bound)
- FLOPs: `O(d · n²)` per sublayer (n=mhc_expansion, 通常 4), 远小于 attention/FFN
- Memory: 扩展 activation 为 `[T, n, d]`，是 roofline bottleneck
- 建模方式: 在 build_layer_ops 中作为额外 op 添加，使用 roofline 模型自动捕获 memory-bound 特性

**MTP (Multi-Token Prediction):**
- 训练时: 每个 depth 额外一次 LM head forward, FLOPs ≈ `2·d·vocab_size·mtp_depth`
- 推理时 (标准): 不使用，FLOPs = 0
- 推理时 (投机解码): `effective_throughput = base × acceptance_len / (1 + draft_overhead)`

### 4.4 Communication 公式

| Collective | Volume (per device) |
|-----------|-------------------|
| AllReduce (ring) | `2 × msg × (N-1)/N` |
| AllGather | `msg × (N-1)/N` |
| ReduceScatter | `msg × (N-1)/N` |
| AllToAll (EP) | `2 × tokens × top_k × hidden × bytes` per MoE layer |
| Ring CP | `2 × (S/CP) × d_kv × bytes × (CP-1)` per layer (2x for K+V separately) |
| P2P (PP) | `batch × seq × hidden × bytes` per micro-batch |

### 4.5 Phase 影响

| Phase | FLOPs multiplier | Memory behavior |
|-------|-----------------|-----------------|
| PREFILL | 1x | activation 不保留 |
| DECODE | 1x (but seq_len=1, kv_len=full) | KV cache 累积 |
| TRAIN_FWD | 1x | activation **保留** (给 backward) |
| TRAIN_BWD_DX | 1x | 读 weight，产生 input gradient |
| TRAIN_BWD_DW | 1x | 读 saved activation，产生 weight gradient |

Training backward total ≈ 2x forward FLOPs.

## 5. L2: Builder + Simulator

### 5.1 Builder (builder.py)

`build_*_step()` 函数将 ModelConfig + ParallelismConfig → `List[SimOp]`。

**并行切分在此体现：** shape 除以 TP/EP 后再传给 ops.py。

**三条主路径：**
- `build_generation_step()` → prefill ops + decode per-token ops
- `build_training_step()` → forward + backward + dp_sync + optimizer, 按 pp_schedule 分 1F1B / ZeroBubble / DualPipe
- `build_layer_ops(layer, parallel, batch, seq_len, phase)` → 单层 op 序列 (共享)

**PP schedule 差异：** 同一套 ops，不同的 depends_on 声明。
- **1F1B:** dx+dw 合并为单一 BWD phase **(MVP, P0)**
- **ZeroBubble:** dw 延迟，不依赖 dx chain **(Next Phase, P1+)**
- **DualPipe:** 4 组件 (attn, a2a_dispatch, mlp, a2a_combine)，通信放到 ep_comm stream **(Next Phase, P1+)**

> **MVP 阶段仅实现 1F1B。** TRAIN_BWD_DX / TRAIN_BWD_DW 分离、ZeroBubble、DualPipe 均为 P1+ 扩展。MVP 使用单一 TRAIN_BWD phase。

### 5.2 Simulator (simulator.py)

多流拓扑模拟 + memory tracking，~200 行。

```python
def simulate(ops: List[SimOp]) -> SimResult:
    # 1. 拓扑排序
    # 2. 多时钟推进 (每个 stream 一个 clock)
    # 3. 同时跟踪 memory (alloc/free by ref_count)
    # 4. 返回 wall_clock + peak_memory
```

**Overlap 处理：** 不同 stream 上的 op 天然并行。barrier 通过 depends_on 表达。simulator 不需要知道"这是什么并行策略"。

## 6. L3: Pipeline Model (pipeline.py)

### 6.1 两阶段 Pipeline

```
Generation (producer) → transfer queue → Training (consumer)
```

RL 的一轮 epoch:
- 总 response 数 = total_prompts × group_size
- 生成时间 = 总 response / gen_throughput
- 训练时间 = 总 response / train_throughput
- **epoch_time = t_first_gen_batch + max(t_gen_remaining, t_train) + t_last_train_step**
- 简化: 当 batch 数足够多时 ≈ `max(gen_time, train_time) + startup_overhead`
- startup_overhead = 一个 gen batch 的时间（训练等待第一批数据）

### 6.2 Long-tail 建模

生成阶段的 wall-clock 由 batch 内最长 response 决定：

```
# Gumbel 近似: E[max of B samples]
effective_len = avg + std × sqrt(2 × ln(batch_size))
t_batch = t_prefill + effective_len × t_per_token
```

### 6.3 Bottleneck Analysis

```python
bottleneck = "GENERATION" if t_gen > t_train else "TRAINING"
slack = max(t_gen, t_train) / min(t_gen, t_train) - 1
```

## 7. Memory Model

### 7.1 设计原则

**Memory 是 op 执行序列的副产品，不独立建模。**

每个 SimOp 已携带 `weight_bytes` 和 `output_bytes`。simulate() 内部同时做 memory tracking：

```python
weight_total = deduplicated_sum(op.weight_bytes for op in ops)
peak_activation = max_of_running_sum(alloc - free, tracked by ref_count)
```

### 7.2 Memory Budget

```
可用 HBM = hbm_capacity × hbm_usable_ratio

训练阶段 peak:
  weights + optimizer_states + peak_activation + gradients

生成阶段 peak:
  weights + kv_cache

RL 额外:
  + reference_model (如果不 offload 到 CPU)

共卡模式:
  peak = max(gen_peak, train_peak, resharding_peak)

分离模式:
  gen_peak 和 train_peak 分别校验
```

### 7.3 框架开销

通过 `hbm_usable_ratio` 配置参数处理（默认 0.85），包含 HCCL buffer、PyTorch allocator、碎片、workspace。可根据实测调整。

### 7.4 Offload / Recomputation 对性能的影响

作为 perf_penalty 系数反馈给 throughput 模型：

| 优化 | Memory 影响 | Perf penalty |
|------|-----------|-------------|
| recompute_attention | 消除 quadratic activation | ~5% (rough default, 待校准) |
| full_recomputation | activation ≈ 0 | ~30% (rough default, 待校准) |
| optimizer_offload | optimizer → CPU | ~15% (PCIe bandwidth dependent) |
| activation_offload | activation → CPU | ~10% (PCIe bandwidth dependent) |
| ref_offload_cpu | ref model → CPU | 0 (不影响训练吞吐) |

注: perf_penalty 为粗略默认值，实际因模型结构而异（recomputation cost 取决于 attention FLOPs 占总 FLOPs 的比例）。后期可通过 benchmark 校准。

## 8. Query Interface (model.py)

### 8.1 核心查询

| 查询 | 输入 | 输出 |
|------|------|------|
| `derive_targets` | model + devices + data + time_budget | gen/train TPS targets |
| `feasibility_check` | model + devices + data | 最短 epoch time |
| `resource_plan` | model + data + time_budget | 最少卡数 |
| `what_if` | base_config + overrides | 前后对比 |
| `sensitivity` | base_config + param + values | 参数扫描 |
| `search_optimal` | model + devices | 最优并行配置 (pruned enumeration, 见 8.2) |

### 8.2 search_optimal 搜索策略

**Pruned enumeration with heuristic constraints:**
- TP ∈ {1, 2, 4, 8} (≤ devices_per_node)
- PP ∈ {1, 2, 4, 8, 16} (≤ num_layers, 需整除)
- EP ∈ {1, num_experts 的因子} (仅 MoE)
- DP = total_devices / (TP × PP × EP)
- ZeRO ∈ {0, 1, 2, 3}
- pp_schedule: 1f1b (PP≤2), interleaved/zero_bubble (PP>2), dualpipe (MoE+EP)

**Pruning rules:**
- Memory check 先于 throughput 计算 (infeasible 直接跳过)
- TP > 1 才启用 SP
- EP > 1 仅对 MoE 模型
- 搜索空间典型 ~100-500 组合，每组 <1ms 计算

### 8.3 TargetReport

```python
@dataclass
class TargetReport:
    epoch_time_hours: float
    within_budget: bool
    bottleneck: str
    gen_tps_target: float
    train_tps_target: float
    gen_time_hours: float
    train_time_hours: float
    memory: MemoryProfile
    parallel_config: ParallelismConfig
```

## 9. CLI (cli.py)

```bash
rl-perf targets --model configs/models/llama3_1_8b.yaml \
                --hardware 910C --devices 64 \
                --prompts 100000 --group-size 8 --time-budget 24h

rl-perf check   --model ... --hardware ... --devices ...
rl-perf plan    --model ... --hardware ... --time-budget ...
rl-perf what-if --base results/last.json --group-size 16
rl-perf sweep   --model ... --param group_size --values 4,8,16,32
rl-perf search  --model ... --hardware ... --devices ...
```

## 10. Demo Models

5 个内置模型配置覆盖关键架构维度：

| 模型 | 类型 | Attention | FFN | 特殊 |
|------|------|-----------|-----|------|
| Llama 3.1 8B | Dense 小 | GQA | SwiGLU | baseline |
| Qwen2.5 72B | Dense 大 | GQA | SwiGLU | 需要 PP+TP |
| Mistral 7B | Dense 小 | GQA+SWA | SwiGLU | Sliding Window |
| Qwen3 235B-A22B | MoE 大 | GQA | MoE (128E, top-8) | 主流 MoE |
| DeepSeek V3 671B | MoE 超大 | MLA | MoE (256E, top-8+1shared) | mHC, MTP |

## 11. Overlap Modeling — Phased Approach

### Phase 1 (MVP): 区域级解析

按通信和计算的依赖关系分类为 exposed (串行) 和 hidden (并行)。硬编码在 builder 的 depends_on + stream 声明中。

### Phase 2: Pipeline chunking + 干扰系数

SP/DP 的 chunk 流水建模，hardware-specific interference factor。

### Phase 2.5: 多流拓扑模拟

Template DAG + 多时钟推进。~200 行，无外部依赖。自动处理所有 overlap。

### Phase 3 (长期): DAG + DES

复用 msmodeling feat/op_dag_des 分支设计，salabim-based，支持动态调度。

## 12. Project Structure

```
RL_modeling/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── models/          # 5 个 demo 模型
│   └── hardware/        # 910C, CM384
├── src/rl_perf/
│   ├── __init__.py
│   ├── cli.py           # ~100 行, CLI 入口
│   ├── model.py         # ~200 行, 查询接口
│   ├── config.py        # ~150 行, 所有 dataclass
│   ├── ops.py           # ~400 行, 算子 cost (roofline+memory)
│   ├── builder.py       # ~300 行, config → op 序列
│   ├── simulator.py     # ~200 行, 多流模拟 + memory tracking
│   ├── pipeline.py      # ~150 行, 两阶段 pipeline
│   └── report.py        # ~100 行, 输出格式化
├── tests/               # 每个核心模块一个 test file
├── notebooks/           # 3 个 demo notebook
└── benchmarks/          # 校准数据 (后期)
```

**总计 ~1600 行核心代码。**

## 13. Development Priority

| 优先级 | 内容 | 产出 |
|--------|------|------|
| P0 | config + ops + builder (FWD + BWD combined) + roofline | 单层 FLOPs/time/memory |
| P0 | simulator (单流) + pipeline + basic CLI + 1F1B PP | 端到端 epoch time 估算 |
| P1 | dx/dw 分离 (5 phase) + ZeroBubble/DualPipe + multi-stream simulator | 高级 PP schedule |
| P1 | 全部 attention 变体 (MLA, SWA) + MoE | DSV3 671B demo |
| P2 | CLI 完善 + what-if + sweep + search | 完整查询接口 |
| P2 | Overlap Phase 2 (chunking + interference) | 精度提升 |
| P3 | Benchmark 校准 + notebooks | 工程可用 |
| P3 | Overlap Phase 2.5 (多流拓扑模拟) | 自动 overlap |

## 14. References

### Papers
- DeepSeek-V3 Technical Report (arXiv:2412.19437)
- HybridFlow/veRL (EuroSys 2025, arXiv:2409.19256)
- OpenRLHF (arXiv:2405.11143)
- RLHFuse (arXiv:2409.13221)
- LUMOS (MLSys 2025, arXiv:2504.09307)
- SimAI (NSDI 2025)
- Zero Bubble PP (arXiv:2401.10241)
- DualPipe (github.com/deepseek-ai/DualPipe)
- mHC (arXiv:2512.24880)
- RollPacker (arXiv:2509.21009)

## 15. Validation Strategy

### 15.1 精度目标: <30% 误差

**度量:** 预测的 wall-clock time vs 实测 wall-clock time (per training step, per generation batch)

### 15.2 验证数据源

| 来源 | 精度 | 可用性 |
|------|------|--------|
| msmodeling TensorCast 推理结果 | 已校准 | 现有 |
| deepseek_v4_modeling roofline 输出 | 解析 | 现有 |
| aiconfigurator 推理 benchmark 数据 | 实测 | 现有 (NVIDIA GPU) |
| 公开论文报告的 training throughput | 参考 | DeepSeek V3, Llama 3, Qwen3 |
| 内部 RL 训练 pilot 实测 | ground truth | 待收集 |

### 15.3 验证流程

1. **P0 阶段:** 对比 deepseek_v4_modeling 的 roofline 输出 (单层 FLOPs/time 一致性)
2. **P1 阶段:** 对比公开论文 training throughput (DeepSeek V3 报告的 MFU, token/s)
3. **P2 阶段:** 对比内部 pilot 训练实测数据

## 16. DualPipe Builder Detail (Next Phase, P1+)

DualPipe 的核心是**双向 pipeline + 4 组件分解**。MVP 不实现，此处为后续扩展设计。

### 16.1 4 组件分解 (per layer)

每层 forward 拆为:
1. `attn` (compute stream) — attention 计算
2. `a2a_dispatch` (ep_comm stream) — EP token dispatch
3. `mlp` (compute stream) — expert/FFN 计算
4. `a2a_combine` (ep_comm stream) — EP 结果聚合

Backward 类似, 但 dx/dw 进一步分离。

### 16.2 双向调度

微批次从 pipeline 两端同时注入:
- "前向" 微批次: stage 0 → stage P-1
- "后向" 微批次: stage P-1 → stage 0

Builder 通过 depends_on 表达两个方向的依赖:
```
forward_micro_batch_1: stage_0_attn → stage_0_mlp → stage_1_attn → ...
backward_micro_batch_1: stage_P-1_attn → stage_P-1_mlp → stage_P-2_attn → ...
```

Simulator 自动发现两个方向可以并行执行（不同 stage 在不同时刻处理不同方向的微批次），bubble ratio 降为 `(PP/2 - 1) × T_chunk / total`。

### Existing Projects (in ~/Projects/)
- msmodeling: TensorCast roofline + memory tracker, ServingCast DES
- aiconfigurator: Operation-level perf DB + config search
- deepseek_v4_modeling: 30+ op cost functions, roofline engine
- SOLAR: SOL methodology
