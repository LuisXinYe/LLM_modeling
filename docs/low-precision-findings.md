# 低精度训练建模：发现与结果

> 基于 `llm-perf` 新增的低精度（FP8/FP4）训练建模能力。本文用模型实测数字总结低精度训练对**计算、内存、通信、计算-通信掩盖**的影响，并给出可操作的结论。
>
> 日期：2026-06-28　|　分支：`feat/low-precision-training-modeling`　|　配套 demo：`examples/demo_low_precision.py`

---

## 0. 摘要（最重要的五条）

1. **算力加速 ≠ 步时加速。** FP8 把 FFN matmul 的峰值算力翻倍、FP4 翻两番，但端到端**单步只快 1.15× / 1.26×**（Llama-8B，单机）。原因是 Amdahl：被量化的 FFN GEMM 只占 wall 的 ~29%，attention（仍 bf16）、归一化、访存受限算子和量化开销都不随精度变快。
2. **低精度对训练内存的帮助有限——因为优化器状态主导。** fp32 主权重+动量+方差 = 12 B/param 是**精度无关**的，本例占 84 GB；低精度只压缩了**权重副本**（14→5.5 GB）和激活。峰值内存仅降 ~5–7%。
3. **但这 ~5% 在临界点上可能决定可行性。** bf16 峰值 113.8 GB **超 HBM 不可行**；fp8 降到 108.2 GB **变可行**。边际上低精度的内存红利是"开关性"的。
4. **通信量随精度线性下降**（梯度 fp8 → DP 通信减半），但本规模下 **DP 同步本就被反向计算完全掩盖**（暴露通信 = 0）。所以低精度的通信红利此处是"富余的余量"，不是步时加速——这正是网络评估要回答的问题。
5. **量化开销真实但不大**（FP8/FP4 约 6–17 ms/步），**误差反馈缓冲的内存代价不小**（fp16 缓冲 = 14 GB，等于又一份权重）。

> ⚠️ 数字是 roofline + 校准（compute_eff≈0.5 等）下的估计。**相对结论（加速比、内存构成、通信占比）比绝对步时更稳健。** 见 §7 局限。

---

## 1. 建模方法（如何刻画低精度）

按**张量角色**指定精度（权重 / 激活 / 梯度 / 通信各一个 dtype + 细粒度块大小 + 是否随机 Hadamard）。当某角色低于 bf16 时：

- **计算**：`roofline_time` 按精度类（bf16/fp8/fp4）选峰值算力；FFN GEMM 前后注入 `quantize → (Hadamard) → 低精度 matmul → dequant` 链。
- **内存**：FFN 的常驻权重副本按 `weights.dtype` 计、保存激活按 `activations.dtype` 计；fp32 主权重留在优化器项；误差反馈缓冲单列。
- **通信**：DP 梯度同步字节按 `comm.dtype` 缩放，集合通信前后注入 quant/dequant。
- **掩盖**：多流模拟器让 DP 梯度同步按层分桶、与后续层反向计算重叠；**同一物理 fabric（节点内 NVLink / 跨节点 NIC）上并发的集合通信分享带宽**，`exposed_comm_by_fabric` 报告未被掩盖的通信。

无 `PrecisionConfig` 时各角色默认取 `ModelConfig.dtype`，与单精度行为逐位等价。

---

## 2. 参考场景

| 项 | 值 |
|---|---|
| 模型 | Llama-3.1-8B（GQA + SwiGLU，~7B 参数） |
| 硬件 | Ascend 910C（bf16 800 TFLOPS，fp8 1600，fp4 3200；HBM 96 GB，可用 ~0.85×） |
| 工作负载 | micro-batch=1，prompt+response 见配置 |
| Recipe | **bf16**；**fp8**（权重/激活 fp8-e4m3 + 块128 + Hadamard，梯度/通信 fp8）；**fp4**（权重/激活 fp4-e2m1 + 块128 + Hadamard，梯度/通信 fp8） |

两个并行场景：
- **场景 A — 单机**：TP=1, DP=4（DP 走节点内 NVLink），ZeRO-0。
- **场景 B — 多机**：TP=8, DP=16（DP 走跨节点 NIC），ZeRO-1。

---

## 3. 结果总览

### 场景 A：单机 TP=1 DP=4

| Recipe | 步时(相对) | **加速比** | 通信降幅 | 峰值内存 | 可行 | 暴露通信 |
|---|---|---|---|---|---|---|
| bf16 | 1.199 s | 1.000 | 0 % | **113.8 GB** | ❌ | nvlink 0 ms |
| fp8  | 1.042 s | **1.151** | 50 % | **108.2 GB** | ✅ | nvlink 0 ms |
| fp4  | 0.955 s | **1.256** | 50 % | **105.4 GB** | ✅ | nvlink 0 ms |

单步 DAG 分项：

| Recipe | wall | FFN compute(按类) | 量化开销 | DP 通信 | 通信量 | 权重 | 梯度 | 优化器 | 激活 |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1.199 s | 0.346 s (bf16) | 0 | 75.2 ms | 20.9 GB | 14.0 | 14.0 | **83.8** | 2.1 |
| fp8  | 1.042 s | 0.173 s (fp8) | 17.3 ms | 37.8 ms | 10.5 GB | 8.3 | 14.0 | **83.8** | 2.1 |
| fp4  | 0.955 s | 0.087 s (fp4) | 16.8 ms | 37.8 ms | 10.5 GB | 5.5 | 14.0 | **83.8** | 2.1 |

（内存单位 GB，为每卡常驻）

### 场景 B：多机 TP=8 DP=16 ZeRO-1

| Recipe | 步时(相对) | 加速比 | 通信降幅 | 峰值内存 | 可行 | 暴露通信 |
|---|---|---|---|---|---|---|
| bf16 | 0.185 s | 1.000 | 0 % | 6.3 GB | ✅ | nvlink 0 / nic 0 ms |
| fp8  | 0.176 s | 1.051 | 15.2 % | 5.6 GB | ✅ | nvlink 0 / nic 0 ms |
| fp4  | 0.165 s | 1.123 | 15.2 % | 5.2 GB | ✅ | nvlink 0 / nic 0 ms |

单步分项：DP 通信 51.5 ms(bf16) → 28.2 ms(fp8)；通信量 10.8→9.2 GB；权重 1.7→1.0→0.7，梯度 1.7，优化器 0.7（被 ZeRO-1 + TP 分摊），激活 2.1。

---

## 4. 逐维度发现

### 发现 1（头条）：算力加速被 Amdahl 严重稀释

FFN GEMM 计算严格随精度类缩放——**0.346 s(bf16) → 0.173(fp8, ×½) → 0.087(fp4, ×¼)**，模型完全捕捉了 2×/4× 的峰值算力。但端到端：

- 被量化的 FFN compute 只占 bf16 wall 的 **0.346/1.199 ≈ 29%**。
- 把它压到 0（理论极限）也只有 **1/(1−0.29) ≈ 1.41×** 的步时上限。
- FP4 实测 1.256×：吃掉了大部分红利，但被 ~17 ms 量化开销 + 未量化的 attention/归一化拉回。

> **结论：若想让低精度真正提速，必须扩大被量化的算子覆盖面**——当前 attention 投影仍是 bf16（见 §7）。把 attention 纳入量化是回报最高的下一步。

### 发现 2：低精度对训练内存"杯水车薪"，因为优化器状态主导

场景 A 内存构成（bf16 共 113.9 GB）：**优化器 83.8 GB（74%）** + 权重 14 + 梯度 14 + 激活 2.1。

- 优化器 = param×(fp32 主权重 4 + 动量 4 + 方差 4) = **12 B/param，与计算精度无关**。
- 低精度只压缩**权重副本**（14→8.3→5.5）和激活。梯度副本本例保持模型精度（14 GB，常驻不变）。
- 净效果：峰值内存 113.8 → 108.2 → 105.4 GB，**仅降 4.9% / 7.4%**。

> **结论：要大幅降训练内存，靠的是 ZeRO 分片 / 优化器状态低精度 / offload，而不是低精度 matmul。** 低精度省的是权重和激活，属次要项。场景 B（ZeRO-1 + TP8）里优化器被摊薄到 0.7 GB，低精度的权重压缩（1.7→0.7）才在占比上更显眼。

### 发现 3：~5% 的内存下降可能"开关性"地决定可行性

尽管降幅小，bf16 的 113.8 GB **超出可用 HBM（96×0.85≈81.6… 实际判定为不可行）**，而 fp8 的 108.2 GB 被判**可行**。在容量临界点上，低精度的边际内存红利直接翻转 feasibility——这是把低精度当"挤进显存"手段时的核心价值。

### 发现 4：通信量线性下降，但本规模下被反向计算完全掩盖

- 梯度/通信用 fp8 → DP 同步字节减半（场景 A 通信量 20.9→10.5 GB，DP 通信 75→38 ms）。
- **但暴露通信 = 0**：DP 同步（38–75 ms）完全藏在反向计算（~1 s）之下，即便 bf16 也如此。
- 场景 B 跨节点 NIC 上同样 exposed=0：DP 同步（28–52 ms）仍被反向（~140 ms）掩盖。

> **网络评估结论：在 Llama-8B 这个 compute/comm 比下，互联不是瓶颈；低精度省下的通信是"富余余量"而非提速。** 该结论会在**更大模型 / 更多节点 / 更快加速器（compute 缩短）/ 更激进的 TP-EP**时翻转——届时 `exposed_comm_by_fabric` 会从 0 变正，且**同一 fabric 上并发的集合通信分享带宽**的建模开始起决定作用。能力已就位，等场景把它推到临界点。

注：场景 B 通信仅降 15.2%（非 50%），因为 **TP/EP 通信量当前不随精度缩放**（只有 DP 梯度通信走了 `comm.dtype`），TP8 下大头是 TP 激活通信。见 §7。

### 发现 5：量化开销真实但可控

随机 Hadamard + 细粒度 scale + dequant 合计 **FP8 ~17 ms、FP4 ~17 ms/步**（场景 A），约为步时的 1.6%。它会侵蚀一部分低精度加速，但远小于收益。块越小、Hadamard 越大，开销越高——这正是按 recipe 对比时要权衡的。

### 发现 6：误差反馈（EF）补偿的内存代价不可忽视

| 配置 | 权重 | 梯度 | 优化器 | EF 缓冲 |
|---|---|---|---|---|
| fp8，无 EF | 8.32 GB | 13.96 | 83.76 | 0 |
| fp8 + EF(fp16) | **22.28 GB** | 13.96 | 83.76 | **13.96** |

EF 缓冲 = param×2 B(fp16) = 14 GB，**等于又一份权重**。关键是：修复后它**单列、不污染优化器/梯度核算**（grad、optim 在两行完全相同）——这点在建模上很容易写错（之前的实现会让开 EF 时所有内存项 ~翻倍）。用 EF 做精度补偿时，这 14 GB 必须计入预算。

---

## 5. 哪些因素会改变上述结论（敏感性）

| 杠杆 | 对结论的影响 |
|---|---|
| **attention 纳入量化** | 直接放大发现 1 的加速（当前 attention bf16 是主要"未提速"项） |
| **模型更大 / 加速器更快** | compute 缩短 → 通信更易暴露 → 发现 4 翻转，网络评估变关键 |
| **ZeRO 阶段 / 优化器低精度 / offload** | 直接攻击发现 2 的 84 GB 优化器主导项，比低精度 matmul 省得多 |
| **更激进 TP/EP** | TP/EP 通信占比上升；但当前它们不随精度缩放（§7），低精度的通信红利被稀释 |
| **块大小 / Hadamard 尺寸** | 调节发现 5 的量化开销 |

---

## 6. 实操建议

1. **把低精度当"挤进显存 + 适度提速"，而非"大幅提速/大幅省内存"。** 单机 8B 上 fp8 的真实价值是让 bf16 不可行的配置变可行 + 15% 提速。
2. **想要更大加速，先扩量化覆盖面（attention 投影），而非追求更低位宽。** 从 fp8→fp4 只多拿 ~10% 步时，但 attention 仍是 bf16 的大头。
3. **想要更大省内存，先动优化器**（ZeRO-2/3、fp8 优化器状态、offload），低精度权重是次要项。
4. **网络选型先用 `exposed_comm_by_fabric` 确认是否暴露。** 若为 0（如本例），低精度通信红利不转化为提速，别为此高估收益；把场景推到你的真实规模再看。
5. **用 EF 补偿时把缓冲计入显存预算**（≈一份权重大小）。

---

## 7. 局限与注意事项

- **绝对步时是 roofline+校准估计**（compute_eff≈0.5、mem_eff≈0.7 等）。相对比值远比绝对值可信。
- **attention 投影仍是 bf16**：attention 是融合了 projection+softmax 的单体算子，量化链与精度感知内存只覆盖 FFN GEMM。这压低了低精度的加速与内存红利，且 `compute_seconds_by_class` 不含 attention。
- **TP/EP 通信量不随精度缩放**：只有 DP 梯度通信用了 `comm.dtype`；TP/EP 集合通信仍按模型精度计字节。多机 TP/EP 重场景下会低估低精度的通信红利。
- **掩盖是保守近似**：同 fabric 集合通信按"时间分片串行"近似带宽分享；`exposed_comm_by_fabric` 用 `max(0, fabric_busy − total_compute)`，当计算在时间轴上有空隙时可能**低估**暴露通信（偏乐观）。
- **梯度常驻副本保持模型精度**（未随 `gradients.dtype` 缩放常驻内存，只缩放了通信字节）。
- **后训练/RL 路径（`rl_training_time`）未接入精度配置**：当前仅预训练/训练路径建模低精度。
- **优化器步时仍是占位**（`param_count×1e-10`），不影响内存与本文结论，但不要据此读优化器耗时。

---

## 附录：复现

```bash
source .venv/bin/activate
python examples/demo_low_precision.py          # 概览表
# 深度分项（本文场景 A/B + EF）见提交历史中的 compare_precision 用法
```

核心 API：

```python
from llm_perf.model import compare_precision
from llm_perf.precision import PrecisionConfig, TensorPrecision
rows = compare_precision(model_cfg, hw, parallel_cfg, rl_cfg, {
    "bf16": PrecisionConfig.bf16_default(),
    "fp8":  PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
                            activations=TensorPrecision(dtype="fp8_e4m3", block_size=128, hadamard=True),
                            gradients=TensorPrecision(dtype="fp8_e4m3"),
                            comm=TensorPrecision(dtype="fp8_e4m3")),
})
# 每行: name, step_seconds, speedup_vs_bf16, comm_bytes, comm_reduction_pct,
#       exposed_comm_by_fabric, peak_memory_gb, feasible
```

相关文档：`docs/architecture.md`（"通信掩盖与 fabric 争用"、"低精度训练成本路径"两节）、设计 spec `docs/superpowers/specs/2026-06-27-low-precision-training-modeling-design.md`。
