"""Demo: how node-limited routing + load imbalance drive cross-node demand.

Run: python examples/demo_sparse_ratio.py

This demo shows the value of `moe_node_limit` via two scenarios:

1. Compute-bound (network NOT the bottleneck): DeepSeek-V3 671B on a
   high-bandwidth cluster (Ascend 910C, 800 Gbps InfiniBand default).
   exposed [nic] is ~0 everywhere because attention + optimizer + shared-expert
   compute fully consumes the compute stream, hiding all MoE all-to-all traffic.
   moe_node_limit still reduces cross-node bytes, but the bytes-level benefit is
   invisible in wall-clock time at this BW/compute ratio.
   See tests/test_model.py (test_sweep_sparse_ratio_*) for the byte-level check.

2. Comm-bound (network IS the bottleneck): same 671B model on a
   bandwidth-constrained cluster (80 Gbps inter-node, e.g. bonded 40GbE or
   heavily over-subscribed fabric). Cross-node all-to-all now exceeds total
   compute time when routing is unconstrained (moe_node_limit=0), so the NIC
   is on the critical path. Tightening moe_node_limit to 4 (each token visits
   at most 4 destination nodes) drops exposed [nic] from >4 s to 0.
"""

from llm_perf.config import (
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.model import sweep_sparse_ratio
from llm_perf.report import format_sparse_sweep

model = load_model_config("configs/models/deepseekv3_671b.yaml")
hw = load_hardware_config("configs/hardware/ascend_910c.yaml")

# WorkloadConfig for a typical RL training micro-step.
# Valid fields: group_size, avg_prompt_len, max_promt_len, avg_response_len,
# max_response_len, std_response_len, train_micro_batch_size,
# gradient_accumulation_steps, train_batch_size, gen_batch_size, etc.
rl = WorkloadConfig(group_size=8)

# ---------------------------------------------------------------------------
# Scenario 1 — Compute-bound (910C default: 800 Gbps InfiniBand, ep=64)
# ---------------------------------------------------------------------------
# With 800 Gbps inter-node BW the 671B model is firmly compute-bound:
#   attention_dsa  ~4800 ms  (no TP, full seq on one rank)
#   optimizer_step ~4200 ms  (proportional to per-rank param count)
#   ffn_moe        ~2800 ms  (4 routed + 1 shared expert per rank at ep=64)
# Total compute ~11 800 ms >> NIC all-to-all ~1600 ms.
# exposed [nic] = max(0, nic_busy - compute) = 0 for every grid point.
# moe_node_limit still cuts cross-node *bytes* (visible in unit tests),
# but the time saving is already hidden inside the compute slack.
print("=" * 70)
print("Scenario 1 — Compute-bound (910C, 800 Gbps IB, ep=64)")
print("  Exposed [nic] is ~0: compute fully hides all MoE all-to-all.")
print("  moe_node_limit reduces bytes but NOT step time at this BW/TFLOPS ratio.")
print("=" * 70)

base_s1 = ParallelismConfig(tp=1, ep=64, dp=1)
grid_s1 = [
    {"moe_node_limit": 0, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 4, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.3},
]
rows_s1 = sweep_sparse_ratio(model, hw, base_s1, rl, grid_s1)
print(format_sparse_sweep(rows_s1))

# ---------------------------------------------------------------------------
# Scenario 2 — Comm-bound (same 671B, same compute, but 80 Gbps inter-node)
# ---------------------------------------------------------------------------
# Same model and compute hardware; only the inter-node fabric is slower
# (80 Gbps ≈ 10 GB/s, e.g. bonded 40GbE or a heavily over-subscribed ToR).
# Now NIC all-to-all at ep=64 / top_k=8 / no node limit exceeds compute time
# by ~4 s per step → exposed [nic] is non-zero and on the critical path.
#
# Tightening moe_node_limit from 0 → 4 halves cross-node traffic
# (each token visits at most 4 of 8 EP nodes instead of all 8), which drops
# exposed [nic] from ~4 400 ms down to 0.  Further tightening to 2 reduces
# bytes even more (2 destination nodes per token).  This is the decision the
# knob was designed to make: cap cross-node fan-out before the fabric saturates.
print()
print("=" * 70)
print("Scenario 2 — Comm-bound (same 671B, 80 Gbps inter-node / 10 GB/s)")
print("  At this bandwidth, unconstrained MoE routing (node_limit=0) leaves")
print("  >4 s of exposed [nic] on the critical path.  Tightening node_limit")
print("  to 4 (or 2) eliminates that exposure entirely.")
print("=" * 70)

# Simulate a bandwidth-constrained inter-node fabric (10 GB/s = 80 Gbps).
hw_bw_limited = hw.model_copy(update={"inter_node_bw_gb_s": 10})
base_s2 = ParallelismConfig(tp=1, ep=64, dp=1)
grid_s2 = [
    {"moe_node_limit": 0, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 4, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.0},
    {"moe_node_limit": 2, "moe_imbalance_factor": 1.3},
]
rows_s2 = sweep_sparse_ratio(model, hw_bw_limited, base_s2, rl, grid_s2)
print(format_sparse_sweep(rows_s2))
