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
