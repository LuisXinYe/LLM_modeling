# 建模 FP4 混精预训练方案（Zhou et al. 2025）

> 在 `llm-perf` 中建模论文 *Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models*（Zhou et al., 上海 AI Lab / USTC，arXiv:2502.11458）的混精方法。
>
> 日期：2026-06-29　|　配套：`examples/demo_fp4_paper.py`、`llm_perf.cost_analysis`、`PrecisionConfig.fp4_paper()`

---

## 0. 论文方法（建模对象）

按**模块 × 方向**分配精度（不是全局单一精度）：

| 计算 | 精度 | 理由 |
|---|---|---|
| FFN linear 前向 GEMM | **FP4** | FFN ≈ 57% 算力，主要收益 |
| Attention QKV+O 投影（前向） | **FP8** | "保护"注意力（FP4 下 attention score 趋均匀） |
| MHA 核心（QK^T/softmax/score·V） | **FP16** | FlashAttention，不量化 |
| Linear 反向（wgrad+dgrad） | **FP8** | 反向更敏感（梯度~0.02 在 FP4 下下溢） |
| 激活函数、Norm、主权重 | FP16 / FP32 | 不量化 |

外加细粒度量化（per-token/channel → per-block）与 2 阶段目标精度调度（本次未建模，YAGNI）。

## 1. 在框架里怎么建模

- **配置**：`PrecisionConfig` 新增 `attn_linear` / `ffn_linear`（各含 `fwd`/`bwd` 精度）；`linear_fwd/bwd("attn"|"ffn")` 解析器在未设时回退全局角色（向后兼容）；`PrecisionConfig.fp4_paper()` 一键构造论文 recipe。
- **Attention 拆分**：`op_attention_split` 把注意力拆成**投影 GEMM**（精度感知、走量化链）+ **核心**（FP16）；builder 对 GQA/MHA 发 `attention_proj` + `attention_core` 两个算子（FLOPs 和严格等于原单算子）。MLA/SWA/DSA 回退单算子。
- **方向感知**：`TRAIN_BWD` 用 bwd 精度、前向用 fwd 精度选 compute_class；量化链按模块精度发射 quantize/Hadamard/dequant（fp4_paper 真正计入开销）。
- **分析锚**：`cost_analysis.theoretical_compute_cost` 纯函数，按论文口径（matmul-only、MHA 核心 FP16 留分母）算"占 FP16 的%"，加速倍数参数化。

## 2. 验证结果

### 2.1 Fig-1a 前向 FLOP 拆分 —— 精确复现 ✅

`theoretical_compute_cost`（LLaMA-7B-4K，MHA）：

| 组件 | 模型 | 论文 Fig 1a |
|---|---|---|
| FFN | **57.3%** | 57% |
| Attention linear | **28.4%** | 28.7% |
| MHA core | **14.2%** | 14.3% |

逐项 ±1pp 内吻合——FLOP 口径忠实。

### 2.2 Table-2 理论成本% —— 论文口径不自洽（如实标注）

| Recipe (Attn/FFN/Bwd) | 声明 1/2/4× | 论文隐含 1/1.4/2× | 论文 Table 2 |
|---|---|---|---|
| FP4 / FP4 / FP4 | **35.0%** | **56.7%** | 57.1% |
| FP8 / FP4 / FP8 | **50.8%** | **70.2%** | 66.1% |
| FP16 (base) | 100% | 100% | 100% |

**关键发现**：用论文**正文声明**的 FP16=1×/FP8=2×/FP4=4×，全 FP4 的 FLOP-诚实答案是 **~35%**，而论文 Table 2 写 **57.1%**——系统差 ~21pp。反推可知论文的"computation cost"口径隐含 **FP4 ≈ 2×**（而非声明的 4×），即该表与正文不自洽、且未说明。该成本% 与前/反向倍数无关（分子分母同时缩放），故非反向计数误差。

我们的处理：**不硬凑**。把加速倍数做成参数，默认 1/2/4× 给诚实的 ~35%，用论文隐含的 1/1.4/2× 才得 ~57%；残差与"FP4≈2×"结论写进测试与 demo。

### 2.3 把论文 recipe 套到 Llama-3.1-8B（完整 roofline what-if）

`compare_precision` + `fp4_paper()`（单机 TP=1 DP=4）：

| Recipe | 加速比 | 峰值内存 | 可行 |
|---|---|---|---|
| bf16 | 1.000 | 113.8 GB | ❌ |
| **fp4_paper** | **1.247** | **104.0 GB** | ✅ |

- **加速 1.247×**（已计入 attn-proj/FFN 前后的量化开销；未计开销时为 1.255×）。和上一份低精度发现一致：被量化的 GEMM 只占 wall 一部分，端到端加速被 Amdahl 稀释，远小于 FP4 的 4× 峰值。
- **内存 113.8→104.0 GB**，且 **feasibility 从不可行翻转为可行**——低精度的边际内存红利在容量临界点上是开关性的（优化器 fp32 状态仍主导，故降幅有限）。

## 3. 已知边界与简化

- **Table-2 绝对%在声明 4× 下不可复现**（见 2.2）——这是论文口径问题，已如实标注，非建模缺陷。
- **Attention 拆分仅 GQA/MHA**；MLA/SWA/DSA 保持单算子，不应用模块精度。
- **量化链开销**：fp4_paper 现在会发射 quantize/dequant 并计入时间；块大小在反向仍按前向激活精度键取（二阶小量，接受）。
- **2 阶段精度调度、per-block vs per-token 开销区分**未建模（YAGNI；现有 block_size 已能泛化）。
- 绝对步时是 roofline+校准估计；相对比值（加速比、FLOP 拆分、内存构成）比绝对值可信。

## 4. 复现

```bash
source .venv/bin/activate
python examples/demo_fp4_paper.py
```

```python
from llm_perf.precision import PrecisionConfig
from llm_perf.cost_analysis import theoretical_compute_cost
from llm_perf.model import compare_precision

# 论文 recipe 一键构造
pc = PrecisionConfig.fp4_paper()   # attn-proj FP8, FFN-fwd FP4, linear-bwd FP8

# 理论成本%（参数化加速倍数）
theoretical_compute_cost(model_cfg, pc, seq_len=4096)            # 默认 1/2/4×
theoretical_compute_cost(model_cfg, pc, speed_map={"fp16":1,"fp8":1.4,"fp4":2})

# 套到任意模型/硬件做 what-if
compare_precision(model_cfg, hw, parallel_cfg, rl_cfg,
                  {"bf16": PrecisionConfig.bf16_default(), "fp4_paper": pc})
```

相关：设计 spec `docs/superpowers/specs/2026-06-29-fp4-paper-mixed-precision-modeling-design.md`；低精度总体发现 `docs/low-precision-findings.md`。
