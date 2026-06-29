# 大稀疏比 MoE 网络建模

llm-perf 如何建模**大稀疏比**(专家数多达 1k+、激活的 `top_k` 很小)对互联的影响,
以及如何用它做网络 / 集群选型。

> 稀疏比 = 总专家数 ÷ 激活专家数。细粒度 MoE 把这个比值推得很高(DeepSeek-V3:
> 256 路由专家、top-8 → ~32;新设计追求 50–100+)。此时算力红线被网络盖住:
> dispatch / combine 的 all-to-all 成为主导,且随着专家变细、EP 跨更多节点而增长。

## 这个能力新增了什么

| 层 | 能力 |
|---|---|
| **config** | `ParallelismConfig.moe_node_limit`(0 = 不限)、`moe_imbalance_factor`(≥ 1.0) |
| **ops** | node-limited 分层 all-to-all(拆分机内 / 跨机字节);不均衡感知的专家计算 |
| **builder** | node-limited 路由激活时,发出两个带 fabric 标签的 SimOp(`nic` + `nvlink`) |
| **model** | `sweep_sparse_ratio(grid)` → 逐点 per-fabric 暴露通信扫描 |
| **report** | `format_sparse_sweep(rows)` 表格;`examples/demo_sparse_ratio.py` |

`moe_node_limit = 0` 且 `moe_imbalance_factor = 1.0` 时,字节级复现特性引入前的行为。

## 建模方法

### 1. node-limited 分层路由

一个 token 的 `top_k` 个专家被约束在最多 `M = moe_node_limit` 个不同节点上
(DeepSeek-V3 式)。token 的载荷**每到一个不同的目标节点才跨网络发一次**;落在
同一个已到达节点上的专家,由机内(NVLink/HCCS)复制服务。

```
distinct_dest_nodes = min(top_k, M, nodes_in_ep)        # nodes_in_ep = ceil(ep / devices_per_node)

inter_bytes (NIC)      = tokens · distinct_dest_nodes        · hidden · dtype · imbalance
intra_bytes (NVLink)   = tokens · (top_k − distinct_dest_nodes) · hidden · dtype · imbalance
```

这是相对扁平 all-to-all(`tokens · top_k · hidden`)的关键修正:扁平模型在一个
token 的多个专家落在同一节点时,**高估**了跨机流量。收紧 `M` 把 NIC 字节换成
NVLink 字节——这正是 node-limited 路由提供的杠杆。

对选型偏保守:每个不同目标节点都计为远端(不扣除源所在的本地节点),从而给出
跨机需求的上界。在 `imbalance = 1.0` 时,`inter + intra == 扁平 top_k 流量`
(机内/机外只是字节在两个 fabric 之间挪动,总量守恒)。

### 2. 负载不均衡

`moe_imbalance_factor = f`(最热 EP rank 的流量/计算 ÷ 均值)。all-to-all 是同步
集合通信,耗时由最慢的参与者决定,所以:

- 跨机与机内的 dispatch/combine 字节各 **× f**;
- `op_moe_ffn` 的路由专家 FLOPs **× f**(权重与激活内存不变——本模型里热 rank
  是多算,而非多存)。

### 不在范围内(YAGNI)

细粒度小 GEMM 的效率衰减、概率分布式(p99)不均衡、router 的 softmax-over-1k 开销。
attention 的 token 稀疏由 `op_dsa_attention`(`index_topk`)单独建模。

## 用法

### Python

```python
from llm_perf.config import load_model_config, load_hardware_config, \
    ParallelismConfig, WorkloadConfig
from llm_perf.model import sweep_sparse_ratio
from llm_perf.report import format_sparse_sweep

model = load_model_config("configs/models/deepseekv3_671b.yaml")
hw = load_hardware_config("configs/hardware/ascend_910c.yaml")
base = ParallelismConfig(tp=1, ep=64, dp=1)
rl = WorkloadConfig(group_size=8)

grid = [
    {"moe_node_limit": 0},                                 # 扁平 all-to-all
    {"moe_node_limit": 4},
    {"moe_node_limit": 2},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.3},    # 热专家偏斜
]
rows = sweep_sparse_ratio(model, hw, base, rl, grid)
print(format_sparse_sweep(rows))
```

每个网格项可覆盖 `num_experts`、`top_k`、`ep`、`moe_node_limit`、
`moe_imbalance_factor` 中的任意几个。每行结果包含:`point`、`step_seconds`、
`exposed_comm_by_fabric`、`cross_node_gb`(即 `nic` 暴露指标)、
`peak_memory_gb`、`feasible`。

### Demo

```bash
python examples/demo_sparse_ratio.py
```

## 怎么读结果

`exposed_comm_by_fabric["nic"]` 是首要信号——**没有**被计算盖住的跨机 all-to-all
时间。它决定互联的规格。

- `exposed [nic] = 0` → 该配置下网络**不是**瓶颈,计算完全盖住了 all-to-all,
  不需要更快的 fabric。
- `exposed [nic] > 0` → 跨机通信落在关键路径上;此时收紧 `moe_node_limit`、
  加带宽、或换 EP 布局才有收益。

### Demo 的发现

在 DeepSeek-V3 671B 上的两个场景:

| 场景 | `exposed [nic]` @ `node_limit=0` | @ `node_limit=4` |
|------|----------------------------------|------------------|
| 默认 fabric,ep=64(算力受限) | 0.0 ms | 0.0 ms |
| 降低跨机带宽(10 GB/s,通信受限) | 4428.5 ms | 0.0 ms |

**直接结论:** 满带宽下 671B 模型算力高度受限,MoE 的 all-to-all 被完全 overlap——
无论怎么路由,网络都不是约束。把跨机带宽降下来(或把专家做得更细 / EP 拉得更宽
直到通信暴露),node-limited 路由就成了真正的杠杆:把一个 token 限制在 4 个节点内,
每步抹掉 4.4 s 的暴露跨机时间。

## 结论与选型建议

### 1. 大稀疏比把"算力问题"变成"网络问题",但有前提

稀疏比本身不直接决定网络压力,**是否暴露**才决定。同一个跨机字节量,在算力充裕时
被 overlap(`exposed [nic] = 0`),在算力紧张时落到关键路径上。两件事把系统推向
"通信受限"这一侧:

- **专家做细**:`expert_intermediate_size` 变小 → 每个专家 GEMM 变小 → 单步计算时间
  下降,而 dispatch/combine 字节(∝ `tokens · 路由份额 · hidden`)几乎不变,于是
  comm/compute 比值上升;
- **EP 拉宽 / 跨更多节点**:`nodes_in_ep` 增大 → 扁平 all-to-all 的跨机份额上升,
  同时每卡专家数(`num_experts / ep`)下降、本地计算变少。

**选型含义**:不要孤立地看"专家有多少",要看目标配置落在哪个 regime。先用
`sweep_sparse_ratio` 跑一遍,`exposed [nic]` 是 0 还是正数,直接决定要不要为互联花钱。

### 2. node-limited 路由是跨机带宽的主旋钮,但不是免费的

收紧 `M` 把跨机字节按 `min(top_k, M, nodes_in_ep)` 压下来,代价是把这些字节挪到
机内 NVLink/HCCS(`intra_bytes` 上升)。所以它的收益取决于两个落差:

- **机内 vs 跨机带宽落差越大,收益越大**:NVLink/HCCS 远快于 RoCE/IB 时,把流量从
  NIC 挪到 NVLink 几乎"免费";
- **`top_k` vs `M` 的落差**:只有当 `top_k > M` 时才真正削减跨机份额。`M ≥ top_k`
  等于没限制。

实践上 `M` 的合理区间通常是 2–4(对应 DeepSeek 的 node-limited 设定),既显著压跨机
流量,又不至于把 token 的专家组合挤到一两个节点里伤害模型质量(质量影响不在本模型
范围内,需另行评估)。

### 3. 不均衡是放大器,直接吃掉 overlap 余量

`moe_imbalance_factor` 同时放大 EP 通信字节和热 rank 的专家计算,而且作用在关键路径
(同步集合通信由最慢者决定)。一个本来被勉强盖住的 all-to-all,可能因为
`imbalance = 1.3` 就被顶出 overlap 余量、变成暴露通信。**选型时务必带一个 > 1 的
不均衡因子做敏感性**——按理想均衡算出来的带宽需求是乐观下界。

### 4. 决策树:拿到 `exposed [nic]` 之后怎么办

1. `exposed [nic] = 0`(含带不均衡因子复核后仍为 0)→ 网络非瓶颈,**不要**为更快
   fabric 付费;余量可以拿去换更宽 EP / 更细专家 / 更大 batch。
2. `exposed [nic] > 0` → 依次评估,优先级从低成本到高成本:
   a. 收紧 `moe_node_limit`(零硬件成本,先试);
   b. 调整 EP / 并行布局,让 all-to-all 更多落在机内;
   c. 仍暴露,才考虑加跨机带宽 / 换更高规格互联。

### 5. 这个模型的边界(别过度解读结论)

- 跨机字节是**上界**(不扣源本地节点),给出的是保守的带宽需求;
- 不建模细粒度小 GEMM 的效率衰减,所以在极细专家下,真实"算力受限"程度可能比
  模型乐观——即真实更容易进入通信受限 regime;
- 不建模 router 开销、p99 尾部不均衡、专家质量。`exposed [nic]` 适合做**相对比较和
  趋势判断**(谁更省跨机、`M` 调到多少够),绝对值需用 benchmark 校准
  (见 `docs/calibration-guide.md`)。

## 参考

- 设计 spec:`docs/superpowers/specs/2026-06-28-large-sparse-ratio-moe-modeling-design.md`
- 实现计划:`docs/superpowers/plans/2026-06-28-large-sparse-ratio-moe-modeling.md`
- 架构文档(EP dispatch 一节):`docs/architecture.md`
