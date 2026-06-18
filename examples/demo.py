#!/usr/bin/env python3
"""rl-perf Python API demo: Pangu-74B-VL RL training performance modeling.
"""

from pathlib import Path
from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from rl_perf.model import RLPerformanceModel
from rl_perf.report import format_table

ROOT = Path(__file__).resolve().parent.parent

# Load Pangu-74B-VL model and Ascend 910B hardware
model = load_model_config(ROOT / "configs/models/pangu_74b_vl.yaml")
hw = load_hardware_config(ROOT / "configs/hardware/ascend_910b.yaml")
perf = RLPerformanceModel(model, hw)

print(f"Model: {model.name}")
print(f"  hidden_size: {model.hidden_size}")
print(f"  num_layers: {model.num_layers}")
print(f"  actual layers: {len(model.get_layers())}")
dense = sum(1 for l in model.get_layers() if l.ffn == "SwiGLU")
moe = sum(1 for l in model.get_layers() if l.ffn == "MoE")
print(f"  Dense layers: {dense}, MoE layers: {moe}")
if model.vision_encoder:
    ve = model.vision_encoder
    from rl_perf.builder import _estimate_vision_encoder_params
    ve_params = _estimate_vision_encoder_params(ve)
    print(f"  Vision Encoder: {ve.num_layers} layers, hidden={ve.hidden_size}, "
          f"image_seq_len={ve.image_seq_len()}, ~{ve_params/1e9:.2f}B params")
print(f"Hardware: {hw.name}")
print(f"  peak_tflops_bf16: {hw.peak_tflops_bf16}")
print(f"  hbm_capacity_gb: {hw.hbm_capacity_gb}")
print()

# RL Configuration from config.yaml
# - train_batch_size: 36
# - max_prompt_length: 4096, max_response_length: 24576
# - rollout.n: 16 (group_size)
# - algorithm: GRPO
# - reference_model: True, ref_offload_cpu: True
# - speculative decoding with MTP (mtp_num_total_tokens: 2)
rl_cfg = RLConfig(
    group_size=16,       # rollout.n = 16
    avg_prompt_len=2048,
    max_promt_len=4096,
    avg_response_len=8192,
    std_response_len=2048,
    max_response_len=24576,
    algorithm="grpo",
    reference_model=True,
    ref_offload_cpu=True,
    use_speculative_decoding=True,
    mtp_acceptance_len=1,
    train_micro_batch_size=1,   # ppo_micro_batch_size_per_gpu: 1
    train_batch_size=36,
    mini_batch_size=36,
    gradient_accumulation_steps=1,
    gen_batch_size=18,          # max_num_seqs: 18
)

# Parallelism from config.yaml
# Actor (Training): TP=4, PP=8, EP=2, CP=4, SP=True, DP=2
#   - expert_model_parallel_size: 2 (inner EP)
#   - Note: 51 layers / PP=8 = [7,7,7,6,6,6,6,6]
# Rollout (Inference): TP=4, EP=8, DP=2
# Ref: TP=1, EP=1, PP=1

# Generation (Rollout): TP=4, EP=8, DP=2 -> 4*8*2 = 64 devices
gen_p = ParallelismConfig(tp=4, pp=1, ep=8, dp=4)

# Training (Actor): TP=4, PP=8, EP=2, DP=2, CP=4, SP=True
# Total devices = TP * PP * CP * DP = 4 * 8 * 4 * 1 = 128
# Offload: param+grad+optimizer all offload to CPU
train_p = ParallelismConfig(
    tp=4, pp=8, ep=2, dp=1, cp=4, sp=True,
    recompute_attention=True,
    full_recomputation=True,    # recompute_granularity: full
    optimizer_offload=True,
)

ref_p = ParallelismConfig(tp=1, pp=1, ep=1, cp=1, dp=128)

# --- 1. Single prediction ---
print("=" * 60)
print("Running performance prediction...")
print("=" * 60)
report = perf.derive_targets(
    total_devices=128,
    rl_cfg=rl_cfg,
    gen_parallel=gen_p,
    train_parallel=train_p,
    ref_parallel = ref_p
)
print(format_table(report))
