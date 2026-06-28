"""Tests for llm_perf.ops — operator cost model and roofline analysis."""

from __future__ import annotations

import pytest

from llm_perf.config import CalibrationConfig, HardwareConfig, Phase
from llm_perf.ops import (
    OpCost,
    comm_time,
    op_allreduce,
    op_alltoall,
    op_gqa_attention,
    op_linear,
    op_mla_attention,
    op_moe_ffn,
    op_mtp_head,
    op_ring_cp,
    op_rmsnorm,
    op_swa_attention,
    op_swiglu_ffn,
    roofline_time,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hw() -> HardwareConfig:
    """Simulated NPU-like hardware for testing."""
    return HardwareConfig(
        name="test_hw",
        peak_tflops_bf16=312.0,  # A100-like
        hbm_capacity_gb=80.0,
        hbm_bandwidth_tb_s=2.0,  # 2 TB/s
        intra_node_bw_gb_s=600.0,
        inter_node_bw_gb_s=100.0,
        inter_node_latency_us=5.0,
        devices_per_node=8,
        calibration=CalibrationConfig(
            compute_eff_large_gemm=0.50,
            compute_eff_small_op=0.20,
            memory_efficiency=0.70,
            comm_efficiency=0.70,
        ),
    )


# ---------------------------------------------------------------------------
# 1. OpCost dataclass
# ---------------------------------------------------------------------------


def test_opcost_dataclass():
    cost = OpCost()
    assert cost.flops == 0
    assert cost.mem_rw == 0
    assert cost.weight_bytes == 0
    assert cost.output_bytes == 0
    assert cost.comm_bytes == 0

    cost2 = OpCost(
        flops=1e12, mem_rw=1e9, weight_bytes=1e8, output_bytes=1e7, comm_bytes=0
    )
    assert cost2.flops == 1e12
    assert cost2.mem_rw == 1e9


# ---------------------------------------------------------------------------
# 2 & 3. roofline_time
# ---------------------------------------------------------------------------


def test_roofline_compute_bound(hw: HardwareConfig):
    """Large GEMM with many FLOPs and small memory: compute-bound."""
    # 312 TFLOPS * 0.5 eff = 156 TFLOPS effective
    # 1e15 FLOPs / 156e12 ≈ 6.4 ms
    cost = OpCost(flops=1e15, mem_rw=1e6)  # tiny memory
    t = roofline_time(cost, hw, is_large_gemm=True)
    compute_t = cost.flops / (
        hw.peak_tflops_bf16 * 1e12 * hw.calibration.compute_eff_large_gemm
    )
    memory_t = cost.mem_rw / (
        hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency
    )
    assert t == pytest.approx(compute_t)
    assert compute_t > memory_t


def test_roofline_memory_bound(hw: HardwareConfig):
    """Small op with large memory footprint: memory-bound."""
    # 2 TB/s * 0.7 eff = 1.4 TB/s
    # 1e13 bytes / 1.4e12 ≈ 7.1 ms
    cost = OpCost(flops=1e6, mem_rw=1e13)  # tiny FLOPs
    t = roofline_time(cost, hw, is_large_gemm=False)
    compute_t = cost.flops / (
        hw.peak_tflops_bf16 * 1e12 * hw.calibration.compute_eff_small_op
    )
    memory_t = cost.mem_rw / (
        hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency
    )
    assert t == pytest.approx(memory_t)
    assert memory_t > compute_t


# ---------------------------------------------------------------------------
# 4 & 5. op_linear FLOPs
# ---------------------------------------------------------------------------


def test_op_linear_forward():
    """FLOPs = 2 * in_features * out_features * batch_tokens for forward."""
    in_f, out_f, batch = 4096, 4096, 128
    cost = op_linear(in_f, out_f, batch, Phase.PREFILL)
    expected = 2 * in_f * out_f * batch
    assert cost.flops == expected


def test_op_linear_backward():
    """Backward FLOPs = 2x forward (dx + dw)."""
    in_f, out_f, batch = 4096, 4096, 128
    fwd = op_linear(in_f, out_f, batch, Phase.PREFILL)
    bwd = op_linear(in_f, out_f, batch, Phase.TRAIN_BWD)
    assert bwd.flops == pytest.approx(2 * fwd.flops)


# ---------------------------------------------------------------------------
# 6 & 7. op_linear output_bytes
# ---------------------------------------------------------------------------


def test_op_linear_output_bytes_train_fwd():
    """During TRAIN_FWD, activation is kept for backward pass."""
    cost = op_linear(4096, 4096, 64, Phase.TRAIN_FWD)
    assert cost.output_bytes > 0
    assert (
        cost.output_bytes == 64 * 4096 * 2
    )  # batch_tokens * out_features * dtype_bytes


def test_op_linear_output_bytes_prefill():
    """During inference (PREFILL), activation is not kept."""
    cost = op_linear(4096, 4096, 64, Phase.PREFILL)
    assert cost.output_bytes == 0


# ---------------------------------------------------------------------------
# 8. op_gqa_attention
# ---------------------------------------------------------------------------


def test_op_gqa_attention_prefill():
    """GQA prefill FLOPs should be positive and include projection + attention."""
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    hidden_size = num_heads * head_dim  # 4096
    batch, seq_len = 2, 512

    cost = op_gqa_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        batch=batch,
        seq_len=seq_len,
        phase=Phase.PREFILL,
    )
    assert cost.flops > 0
    assert cost.weight_bytes > 0
    # GQA should have fewer weight_bytes than MHA (fewer KV heads)
    from llm_perf.ops import op_mha_attention

    mha_cost = op_mha_attention(
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        batch=batch,
        seq_len=seq_len,
        phase=Phase.PREFILL,
    )
    assert cost.weight_bytes < mha_cost.weight_bytes


# ---------------------------------------------------------------------------
# 9. op_swiglu_ffn
# ---------------------------------------------------------------------------


def test_op_swiglu():
    """FLOPs = 6 * hidden * intermediate * batch_tokens."""
    hidden, intermediate, batch_tokens = 4096, 11008, 256
    cost = op_swiglu_ffn(hidden, intermediate, batch_tokens, Phase.PREFILL)
    expected = 6 * hidden * intermediate * batch_tokens
    assert cost.flops == expected


# ---------------------------------------------------------------------------
# 10. op_moe_ffn
# ---------------------------------------------------------------------------


def test_op_moe():
    """MoE FLOPs includes routed + shared experts + router."""
    hidden = 2048
    expert_int = 1024
    n_experts = 8
    n_shared = 2
    shared_int = 2048
    top_k = 2
    batch_tokens = 128

    cost = op_moe_ffn(
        hidden_size=hidden,
        expert_intermediate_size=expert_int,
        num_experts=n_experts,
        num_shared_experts=n_shared,
        shared_intermediate_size=shared_int,
        top_k=top_k,
        batch_tokens=batch_tokens,
        phase=Phase.PREFILL,
    )

    routed_flops = 6 * hidden * expert_int * top_k * batch_tokens
    shared_flops = 6 * hidden * shared_int * n_shared * batch_tokens
    router_flops = 2 * hidden * n_experts * batch_tokens
    expected = routed_flops + shared_flops + router_flops

    assert cost.flops == expected
    assert cost.flops > routed_flops  # shared + router adds to total


# ---------------------------------------------------------------------------
# 11. op_allreduce
# ---------------------------------------------------------------------------


def test_op_allreduce():
    """AllReduce comm_bytes = 2 * msg * (N-1) / N."""
    msg = 1e9  # 1 GB
    N = 8
    cost = op_allreduce(msg, N)
    expected = 2 * msg * (N - 1) / N
    assert cost.comm_bytes == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 12. op_alltoall
# ---------------------------------------------------------------------------


def test_op_alltoall():
    """AllToAll comm_bytes = 2 * tokens * top_k * hidden * dtype_bytes."""
    tokens, hidden, top_k, ep_size = 1024, 2048, 2, 8
    cost = op_alltoall(tokens, hidden, top_k, ep_size, dtype_bytes=2)
    expected = 2 * tokens * top_k * hidden * 2
    assert cost.comm_bytes == expected


# ---------------------------------------------------------------------------
# 13. op_mla_attention
# ---------------------------------------------------------------------------


def test_op_mla_attention():
    """MLA FLOPs > 0 for all phases."""
    kwargs = dict(
        hidden_size=5120,
        num_heads=128,
        head_dim=128,
        kv_compression_dim=512,
        query_compression_dim=1536,
        rope_dim=64,
        batch=2,
        seq_len=512,
    )
    for phase in [Phase.PREFILL, Phase.DECODE, Phase.TRAIN_FWD, Phase.TRAIN_BWD]:
        kv_len = 1024 if phase == Phase.DECODE else None
        cost = op_mla_attention(**kwargs, phase=phase, kv_len=kv_len)
        assert cost.flops > 0, f"MLA FLOPs should be > 0 for phase={phase}"
        assert cost.weight_bytes > 0

    # Training BWD should have 2x forward FLOPs
    fwd = op_mla_attention(**kwargs, phase=Phase.TRAIN_FWD)
    bwd = op_mla_attention(**kwargs, phase=Phase.TRAIN_BWD)
    assert bwd.flops == pytest.approx(2 * fwd.flops)


# ---------------------------------------------------------------------------
# 14. op_swa_attention
# ---------------------------------------------------------------------------


def test_op_swa_attention():
    """SWA FLOPs < GQA FLOPs for long sequences (window caps attention length)."""
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    hidden_size = num_heads * head_dim  # 4096
    batch, seq_len = 2, 2048
    window_size = 256

    swa_cost = op_swa_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        batch=batch,
        seq_len=seq_len,
        phase=Phase.PREFILL,
        window_size=window_size,
    )
    gqa_cost = op_gqa_attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        batch=batch,
        seq_len=seq_len,
        phase=Phase.PREFILL,
    )
    # SWA attention FLOPs should be less because window < seq_len
    assert swa_cost.flops < gqa_cost.flops
    # Projection FLOPs should be the same
    proj_flops = (
        (4 + 4 * num_kv_heads / num_heads) * hidden_size * hidden_size * batch * seq_len
    )
    assert swa_cost.flops >= proj_flops  # at least projections


# ---------------------------------------------------------------------------
# 15. comm_time
# ---------------------------------------------------------------------------


def test_comm_time(hw: HardwareConfig):
    """Verify comm_time returns a reasonable positive value."""
    cost = op_allreduce(1e9, 8)  # 1 GB allreduce
    t_intra = comm_time(cost, hw, group_size=8, is_intra_node=True, algorithm="ring")
    t_inter = comm_time(cost, hw, group_size=8, is_intra_node=False, algorithm="ring")

    assert t_intra > 0
    assert t_inter > 0
    # Inter-node should be slower (lower BW + latency)
    assert t_inter > t_intra

    # Verify intra-node ring formula: bandwidth term + 2*(N-1) latency steps
    expected_intra = cost.comm_bytes / (
        hw.intra_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    ) + 2 * (8 - 1) * (hw.intra_node_latency_us * 1e-6)
    assert t_intra == pytest.approx(expected_intra)

    # Zero comm_bytes => zero time
    zero_cost = OpCost(comm_bytes=0)
    assert comm_time(zero_cost, hw, group_size=8) == 0


def test_comm_time_ring_allreduce_latency(hw: HardwareConfig):
    """Ring AllReduce latency scales with 2*(N-1) steps."""
    cost = OpCost(comm_bytes=1e9)
    t8 = comm_time(cost, hw, group_size=8, is_intra_node=False, algorithm="ring")
    t16 = comm_time(cost, hw, group_size=16, is_intra_node=False, algorithm="ring")
    # 16 nodes has more latency steps than 8 nodes
    assert t16 > t8


def test_comm_time_ring_vs_tree(hw: HardwareConfig):
    """Tree AllReduce has fewer latency steps than ring for large groups."""
    cost = OpCost(comm_bytes=1e6)  # Small message: latency-dominated
    t_ring = comm_time(cost, hw, group_size=64, is_intra_node=False, algorithm="ring")
    t_tree = comm_time(cost, hw, group_size=64, is_intra_node=False, algorithm="tree")
    # Tree: 2*log2(64)=12 steps vs Ring: 2*63=126 steps
    assert t_tree < t_ring


def test_comm_time_intra_node_includes_latency(hw: HardwareConfig):
    """Intra-node ring comm = bandwidth term + 2*(N-1) latency steps."""
    cost = OpCost(comm_bytes=1e9)
    t = comm_time(cost, hw, group_size=8, is_intra_node=True, algorithm="ring")
    expected = 1e9 / (
        hw.intra_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    ) + 2 * (8 - 1) * (hw.intra_node_latency_us * 1e-6)
    assert t == pytest.approx(expected)


def test_comm_time_alltoall(hw: HardwareConfig):
    """AllToAll has minimal latency overhead."""
    cost = OpCost(comm_bytes=1e9)
    t = comm_time(cost, hw, group_size=16, is_intra_node=False, algorithm="alltoall")
    assert t > 0


def test_comm_time_group_size_1(hw: HardwareConfig):
    """Group size 1 = no communication."""
    cost = OpCost(comm_bytes=1e9)
    t = comm_time(cost, hw, group_size=1, is_intra_node=False, algorithm="ring")
    assert t == 0.0


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------


def test_op_rmsnorm_memory_bound(hw: HardwareConfig):
    """RMSNorm should be memory-bound (tiny FLOPs vs large mem_rw)."""
    cost = op_rmsnorm(4096, 1024, Phase.PREFILL)
    assert cost.flops == 5 * 4096 * 1024
    assert cost.mem_rw == 2 * 4096 * 1024 * 2
    t = roofline_time(cost, hw, is_large_gemm=False)
    memory_t = cost.mem_rw / (
        hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency
    )
    assert t == pytest.approx(memory_t)


def test_op_linear_train_bwd_output_bytes():
    """Backward pass does not produce new activations to keep."""
    cost = op_linear(4096, 4096, 64, Phase.TRAIN_BWD)
    assert cost.output_bytes == 0


def test_gqa_flops_scale_with_tp():
    """FLOPs should scale proportionally when heads are TP-partitioned."""
    # Full model: 32 heads, 8 kv_heads
    full = op_gqa_attention(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=4096,
        batch=1,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    # TP=8: 4 heads, 1 kv_head
    tp8 = op_gqa_attention(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        hidden_size=4096,
        batch=1,
        seq_len=512,
        phase=Phase.TRAIN_FWD,
    )
    # Projection FLOPs should be ~1/8 of full
    # (not exact 1/8 because attention FLOPs don't change with TP)
    assert tp8.flops < full.flops
    assert tp8.flops < full.flops / 4  # at least 4x reduction


# ---------------------------------------------------------------------------
# op_mtp_head
# ---------------------------------------------------------------------------


def test_op_mtp_head_train_fwd():
    hidden, vocab, depth, batch = 7168, 129280, 1, 512
    cost = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_FWD)
    expected_flops = 2 * hidden * vocab * depth * batch
    assert cost.flops == expected_flops
    assert cost.weight_bytes == hidden * vocab * 2 * depth
    assert cost.output_bytes == batch * vocab * 2


def test_op_mtp_head_train_bwd():
    hidden, vocab, depth, batch = 7168, 129280, 1, 512
    fwd = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_FWD)
    bwd = op_mtp_head(hidden, vocab, depth, batch, Phase.TRAIN_BWD)
    assert bwd.flops == pytest.approx(2 * fwd.flops)
    assert bwd.output_bytes == 0


def test_op_mtp_head_inference_not_used():
    cost = op_mtp_head(4096, 32000, 1, 64, Phase.PREFILL)
    assert cost.output_bytes == 0
    cost_d = op_mtp_head(4096, 32000, 1, 64, Phase.DECODE)
    assert cost_d.output_bytes == 0


def test_op_mtp_head_depth_scaling():
    base = op_mtp_head(4096, 32000, 1, 128, Phase.TRAIN_FWD)
    doubled = op_mtp_head(4096, 32000, 2, 128, Phase.TRAIN_FWD)
    assert doubled.flops == pytest.approx(2 * base.flops)
    assert doubled.weight_bytes == pytest.approx(2 * base.weight_bytes)


# ---------------------------------------------------------------------------
# op_ring_cp
# ---------------------------------------------------------------------------


def test_op_ring_cp():
    seq_len, cp_size, kv_dim = 4096, 4, 1024
    cost = op_ring_cp(seq_len, cp_size, kv_dim, dtype_bytes=2)
    expected = 2 * (seq_len / cp_size) * kv_dim * 2 * (cp_size - 1)
    assert cost.comm_bytes == pytest.approx(expected)


def test_op_ring_cp_single_rank():
    cost = op_ring_cp(4096, 1, 1024, dtype_bytes=2)
    assert cost.comm_bytes == 0


def test_op_ring_cp_comm_time(hw: HardwareConfig):
    cost = op_ring_cp(4096, 4, 1024, dtype_bytes=2)
    t = comm_time(cost, hw, group_size=4, is_intra_node=False, algorithm="ring_half")
    assert t > 0


# ---------------------------------------------------------------------------
# GQA FLOPs formula verification
# ---------------------------------------------------------------------------


def test_gqa_attention_flops_formula():
    """Verify GQA FLOPs match spec: proj=(4+4G/H)d^2, attn=4*d_qo*s per token."""
    H, G, d_h = 32, 8, 128
    d = H * d_h  # 4096
    batch, seq = 1, 512
    cost = op_gqa_attention(H, G, d_h, d, batch, seq, Phase.PREFILL)

    d_qo = H * d_h
    d_kv = G * d_h
    batch_tokens = batch * seq
    expected_proj = (2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d) * batch_tokens
    expected_attn = 4 * d_qo * seq * batch_tokens  # uses d_qo, not d
    expected = expected_proj + expected_attn
    assert cost.flops == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Decode reads the KV cache (memory traffic scales with context + batch)
# ---------------------------------------------------------------------------


def test_decode_attention_reads_kv_cache():
    """Decode attention mem_rw must include KV cache reads that grow with kv_len.

    Regression guard: without this, decode is dominated by fixed weight loads
    and becomes insensitive to batch / context length (and thus to gen DP).
    """
    H, G, d_h, d, batch = 32, 8, 128, 4096, 16
    short = op_gqa_attention(H, G, d_h, d, batch, 1, Phase.DECODE, kv_len=128)
    long = op_gqa_attention(H, G, d_h, d, batch, 1, Phase.DECODE, kv_len=8192)
    assert long.mem_rw > short.mem_rw
    # Extra traffic = batch * Δkv_len * 2(K,V) * d_kv * dtype_bytes
    d_kv = G * d_h
    assert long.mem_rw - short.mem_rw == pytest.approx(
        batch * (8192 - 128) * 2 * d_kv * 2
    )


def test_prefill_attention_no_kv_cache_read():
    """Prefill does not add a separate KV-cache read term (self-attention)."""
    H, G, d_h, d, batch = 32, 8, 128, 4096, 2
    a = op_gqa_attention(H, G, d_h, d, batch, 512, Phase.PREFILL)
    b = op_gqa_attention(H, G, d_h, d, batch, 512, Phase.PREFILL, kv_len=8192)
    assert a.mem_rw == pytest.approx(b.mem_rw)


def test_roofline_uses_compute_class_peak():
    hw = HardwareConfig(
        name="t", peak_tflops_bf16=100.0,
        peak_tflops={"bf16": 100.0, "fp8": 200.0, "fp4": 400.0},
        hbm_capacity_gb=80, hbm_bandwidth_tb_s=3.0,
    )
    cost = OpCost(flops=1e12, mem_rw=0)
    t_bf16 = roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16")
    t_fp8 = roofline_time(cost, hw, is_large_gemm=True, compute_class="fp8")
    t_fp4 = roofline_time(cost, hw, is_large_gemm=True, compute_class="fp4")
    assert t_fp8 == pytest.approx(t_bf16 / 2, rel=1e-6)
    assert t_fp4 == pytest.approx(t_bf16 / 4, rel=1e-6)


def test_roofline_default_compute_class_is_bf16():
    hw = HardwareConfig(
        name="t", peak_tflops_bf16=100.0,
        peak_tflops={"bf16": 100.0, "fp8": 200.0, "fp4": 400.0},
        hbm_capacity_gb=80, hbm_bandwidth_tb_s=3.0,
    )
    cost = OpCost(flops=1e12, mem_rw=0)
    assert roofline_time(cost, hw, is_large_gemm=True) == pytest.approx(
        roofline_time(cost, hw, is_large_gemm=True, compute_class="bf16")
    )


def test_tflops_for_missing_class_falls_back_to_bf16():
    hw = HardwareConfig(
        name="t", peak_tflops_bf16=100.0,
        hbm_capacity_gb=80, hbm_bandwidth_tb_s=3.0,
    )  # no peak_tflops given
    assert hw.tflops_for("fp8") == 100.0  # conservative fallback
    assert hw.tflops_for("bf16") == 100.0


def test_dsa_decode_models_decoupled_rope():
    """DSA decode must model the MLA-style decoupled RoPE: rope projection in
    weight + FLOPs, and the rope key (∝ rope_dim) in the KV-cache read."""
    from llm_perf.ops import op_dsa_attention

    H, Nq, r = 7168, 128, 64
    base = dict(
        hidden_size=H, num_heads=Nq, head_dim=128, q_lora_rank=1536,
        o_lora_rank=512, o_groups=8, compress_ratio=4, compress_c_kv=512,
        compress_coeff=1.0, index_n_heads=64, index_head_dim=128, index_topk=512,
        window_size=4096, batch=16, seq_len=1, phase=Phase.DECODE, kv_len=4096,
    )
    a = op_dsa_attention(**base, rope_dim=0)
    b = op_dsa_attention(**base, rope_dim=r)
    # rope projection weight: Q rope (H*Nq*r) + K rope (H*r)
    w_delta = (H * Nq * r + H * r) * 2
    assert b.weight_bytes - a.weight_bytes == pytest.approx(w_delta)
    # mem_rw delta = weight delta + KV rope-key read (batch*L*rope_dim*dtype)
    kv_delta = 16 * 4096 * r * 2
    assert b.mem_rw - a.mem_rw == pytest.approx(w_delta + kv_delta)
    assert b.flops > a.flops  # rope projection FLOPs added


# ---------------------------------------------------------------------------
# Low-precision overhead ops: quantize, dequant, Hadamard, compensation-add
# ---------------------------------------------------------------------------


def test_quantize_is_memory_bound_and_adds_scale_bytes():
    from llm_perf.ops import op_quantize

    # bf16 (2B) -> fp8 (1B), 1024 elems, block 128 -> 8 scales*4B
    cost = op_quantize(1024, "bf16", "fp8_e4m3", block_size=128, scale_bytes=4)
    # read 1024*2 + write 1024*1 + scale writes 8*4
    assert cost.mem_rw == pytest.approx(1024 * 2 + 1024 * 1 + 32)
    assert cost.flops > 0  # small reduction work, but tiny
    # memory-bound: compute time must not dominate at typical hw
    assert cost.mem_rw > cost.flops


def test_dequant_mirrors_quantize_direction():
    from llm_perf.ops import op_dequant

    cost = op_dequant(1024, "fp8_e4m3", "bf16", block_size=128, scale_bytes=4)
    # read 1024*1 + write 1024*2 + read scales 8*4
    assert cost.mem_rw == pytest.approx(1024 * 1 + 1024 * 2 + 32)


def test_hadamard_fwht_flops_scale_with_log_block():
    from llm_perf.ops import op_hadamard

    # FWHT: tokens*d*log2(block) MACs *2 flops
    cost = op_hadamard(tokens=100, d=256, block=128)
    assert cost.flops == pytest.approx(2 * 100 * 256 * 7)  # log2(128)=7
    assert cost.mem_rw > 0


def test_compensation_add_is_elementwise_memory():
    from llm_perf.ops import op_compensation_add

    cost = op_compensation_add(2048, "fp16")
    # read EF buffer + read grad + write EF buffer = 3 * numel * 2B
    assert cost.mem_rw == pytest.approx(3 * 2048 * 2)
    assert cost.weight_bytes == 0


# ---------------------------------------------------------------------------
# Hierarchical all-to-all and imbalance-aware MoE FFN
# ---------------------------------------------------------------------------


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
    from llm_perf.config import Phase
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
