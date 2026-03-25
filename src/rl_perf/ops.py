"""Operator cost model using roofline analysis.

Each op function returns an OpCost describing FLOPs, memory, and communication.
Use roofline_time() to convert OpCost → seconds given a HardwareConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rl_perf.config import HardwareConfig, Phase


@dataclass
class OpCost:
    flops: float = 0  # FLOPs count
    mem_rw: float = 0  # HBM read+write bytes (for roofline)
    weight_bytes: float = 0  # Persistent weight memory
    output_bytes: float = 0  # Transient activation/gradient memory
    comm_bytes: float = 0  # Communication bytes (for comm ops)


def roofline_time(
    cost: OpCost, hw: HardwareConfig, is_large_gemm: bool = True
) -> float:
    """Roofline model: time = max(compute_time, memory_time).

    Uses two-tier calibration: large GEMM vs small ops.
    """
    eff = (
        hw.calibration.compute_eff_large_gemm
        if is_large_gemm
        else hw.calibration.compute_eff_small_op
    )
    compute_time = (
        cost.flops / (hw.peak_tflops_bf16 * 1e12 * eff) if cost.flops > 0 else 0
    )
    memory_time = (
        cost.mem_rw / (hw.hbm_bandwidth_tb_s * 1e12 * hw.calibration.memory_efficiency)
        if cost.mem_rw > 0
        else 0
    )
    return max(compute_time, memory_time)


def comm_time(
    cost: OpCost,
    hw: HardwareConfig,
    group_size: int = 1,
    is_intra_node: bool = True,
    algorithm: str = "ring",
) -> float:
    """Communication time with algorithm-aware latency modeling.

    Ring AllReduce: 2*(N-1) steps, each msg/N, latency per step
    Ring AllGather/ReduceScatter: (N-1) steps
    Tree AllReduce: 2*ceil(log2(N)) steps (lower latency, same bandwidth)
    AllToAll: (N-1) P2P exchanges in parallel; modeled as 1 latency step
    P2P: single transfer

    Ref: Thakur et al. "Optimization of Collective Communication Operations in MPICH"
    """
    if cost.comm_bytes <= 0:
        return 0.0

    bw = hw.intra_node_bw_gb_s if is_intra_node else hw.inter_node_bw_gb_s
    bw_bytes = bw * 1e9 * hw.calibration.comm_efficiency
    lat = 0.0 if is_intra_node else hw.inter_node_latency_us * 1e-6

    N = group_size
    if N <= 1:
        return 0.0

    if algorithm == "ring":
        # Ring AllReduce: 2*(N-1) steps
        num_steps = 2 * (N - 1)
        return cost.comm_bytes / bw_bytes + num_steps * lat
    elif algorithm == "ring_half":
        # AllGather or ReduceScatter: (N-1) steps
        num_steps = N - 1
        return cost.comm_bytes / bw_bytes + num_steps * lat
    elif algorithm == "tree":
        # Tree (double binary tree): 2*ceil(log2(N)) steps

        num_steps = 2 * math.ceil(math.log2(max(N, 2)))
        return cost.comm_bytes / bw_bytes + num_steps * lat
    elif algorithm == "alltoall":
        # AllToAll: conceptually (N-1) parallel exchanges; 1 latency step (optimistic)
        return cost.comm_bytes / bw_bytes + lat
    elif algorithm == "p2p":
        return cost.comm_bytes / bw_bytes + lat
    else:
        return cost.comm_bytes / bw_bytes + lat


# ---------------------------------------------------------------------------
# Compute operator cost functions
# ---------------------------------------------------------------------------


def op_linear(
    in_features: int,
    out_features: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """Linear projection. FLOPs = 2*M*N*K per token. Ref: Megatron-LM paper."""
    fwd_flops = 2 * in_features * out_features * batch_tokens
    if phase == Phase.TRAIN_BWD:
        # dx (2*M*N*K) + dw (2*M*N*K) => 2x forward
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    weight_b = in_features * out_features * dtype_bytes
    # mem_rw: read weight + read input + write output
    mem_rw = (
        weight_b
        + batch_tokens * in_features * dtype_bytes
        + batch_tokens * out_features * dtype_bytes
    )
    # Activation kept only during forward of training (for backward pass)
    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * out_features * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_gqa_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
) -> OpCost:
    """GQA attention. FLOPs = 2d·d_qo + 4d·d_kv + 4sd per token. Ref: GQA paper, Llama 2."""
    H = num_heads
    G = num_kv_heads
    d = hidden_size
    batch_tokens = batch * seq_len

    # TP-partitioned output dimensions derived from head counts
    d_qo = H * head_dim  # Q/O output dim (TP-partitioned via num_heads)
    d_kv = G * head_dim  # KV output dim (TP-partitioned via num_kv_heads)

    # Effective KV length for attention computation
    if phase in (Phase.DECODE,):
        L = kv_len if kv_len is not None else seq_len
        # For decode, query seq_len=1 but batch_tokens=batch*seq_len (seq_len=1 typically)
    else:
        L = seq_len  # prefill/train: attend to self

    # Projection FLOPs: Q(2*d*d_qo) + K(2*d*d_kv) + V(2*d*d_kv) + O(2*d_qo*d)
    proj_flops = (
        2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d
    ) * batch_tokens

    # Attention FLOPs: QK^T(2*d_qo*L) + Softmax*V(2*d_qo*L) = 4*d_qo*L per query token
    # Uses d_qo (TP-partitioned) not d, since attention is computed per-head
    if phase == Phase.DECODE:
        attn_flops = 4 * d_qo * L * batch
    else:
        attn_flops = 4 * d_qo * L * batch_tokens

    fwd_flops = proj_flops + attn_flops

    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    # Weight bytes: Q(d*d_qo) + K(d*d_kv) + V(d*d_kv) + O(d_qo*d)
    weight_b = (d * d_qo + d * d_kv + d * d_kv + d_qo * d) * dtype_bytes

    # mem_rw: read weights + read input + write output (output dim is d_qo)
    mem_rw = (
        weight_b + batch_tokens * d * dtype_bytes + batch_tokens * d_qo * dtype_bytes
    )

    # Output activation kept for backward
    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * d * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_mha_attention(
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
) -> OpCost:
    """MHA attention. Calls GQA with num_kv_heads=num_heads."""
    return op_gqa_attention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        batch=batch,
        seq_len=seq_len,
        phase=phase,
        kv_len=kv_len,
        dtype_bytes=dtype_bytes,
    )


def op_swa_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    window_size: int,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
) -> OpCost:
    """SWA. Same as GQA but attention FLOPs capped by window_size. Ref: Mistral 7B."""
    H = num_heads
    G = num_kv_heads
    d = hidden_size
    batch_tokens = batch * seq_len

    # TP-partitioned output dimensions derived from head counts
    d_qo = H * head_dim  # Q/O output dim (TP-partitioned via num_heads)
    d_kv = G * head_dim  # KV output dim (TP-partitioned via num_kv_heads)

    # Effective KV length capped by window size
    if phase == Phase.DECODE:
        L = min(kv_len if kv_len is not None else seq_len, window_size)
    else:
        L = min(seq_len, window_size)

    # Projection FLOPs: Q(2*d*d_qo) + K(2*d*d_kv) + V(2*d*d_kv) + O(2*d_qo*d)
    proj_flops = (
        2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d
    ) * batch_tokens

    if phase == Phase.DECODE:
        attn_flops = 4 * d_qo * L * batch
    else:
        attn_flops = 4 * d_qo * L * batch_tokens

    fwd_flops = proj_flops + attn_flops

    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    # Weight bytes: Q(d*d_qo) + K(d*d_kv) + V(d*d_kv) + O(d_qo*d)
    weight_b = (d * d_qo + d * d_kv + d * d_kv + d_qo * d) * dtype_bytes
    # mem_rw: read weights + read input + write output (output dim is d_qo)
    mem_rw = (
        weight_b + batch_tokens * d * dtype_bytes + batch_tokens * d_qo * dtype_bytes
    )

    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * d * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_mla_attention(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    kv_compression_dim: int,
    query_compression_dim: int,
    rope_dim: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
) -> OpCost:
    """MLA. Training: 6d·d_c + 4d·d'_c + 2d² + 4sd. Inference: absorbed. Ref: DeepSeek-V2."""
    d = hidden_size
    d_c = kv_compression_dim  # KV compression dim (c_KV in paper)
    d_c_q = query_compression_dim  # query compression dim (c_Q)
    r = rope_dim
    batch_tokens = batch * seq_len

    if phase == Phase.DECODE:
        L = kv_len if kv_len is not None else seq_len
        # decode: 1 query token per batch item
        query_tokens = batch
    else:
        L = seq_len
        query_tokens = batch_tokens

    if phase in (Phase.TRAIN_FWD, Phase.TRAIN_BWD):
        # Training: no weight absorption
        # Q down-proj: d -> d_c_q: 2*d*d_c_q
        # Q up-proj: d_c_q -> d: 2*d_c_q*d
        # KV down-proj: d -> d_c: 2*d*d_c
        # K up-proj: d_c -> d: 2*d_c*d
        # V up-proj: d_c -> d: 2*d_c*d
        # O proj: d -> d: 2*d*d
        # Attention: 4*s*d
        proj_flops = (
            2 * d * d_c_q  # Q down
            + 2 * d_c_q * d  # Q up
            + 2 * d * d_c  # KV down
            + 2 * d_c * d  # K up
            + 2 * d_c * d  # V up
            + 2 * d * d  # O proj
        ) * query_tokens
        attn_flops = 4 * d * L * query_tokens
        fwd_flops = proj_flops + attn_flops

        # Simplifying: 6*d*d_c + 4*d*d_c_q + 2*d^2 + 4*s*d per token
        # (matches docstring)

        if phase == Phase.TRAIN_BWD:
            flops = 2 * fwd_flops
        else:
            flops = fwd_flops

    else:
        # Inference (PREFILL / DECODE): with weight absorption
        # Absorbed: Q absorbs up-proj, K/V absorbed into single latent lookup
        # FLOPs: 6*d*d_c + 4*L*d_c + 2*d*num_heads*rope_dim (per query token)
        proj_flops = (6 * d * d_c + 2 * d * num_heads * r) * query_tokens
        attn_flops = 4 * L * d_c * query_tokens
        flops = proj_flops + attn_flops

    # Weight bytes: all projection matrices
    # Q down (d * d_c_q) + Q up (d_c_q * d) + KV down (d * d_c) + K up (d_c * d) + V up (d_c * d) + O (d * d)
    weight_b = (
        d * d_c_q + d_c_q * d + d * d_c + d_c * d + d_c * d + d * d
    ) * dtype_bytes

    mem_rw = weight_b + query_tokens * d * dtype_bytes + query_tokens * d * dtype_bytes

    if phase == Phase.TRAIN_FWD:
        output_b = query_tokens * d * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_swiglu_ffn(
    hidden_size: int,
    intermediate_size: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """SwiGLU FFN. FLOPs = 6·d·d_ff. Ref: Shazeer 2020."""
    # gate proj + up proj + down proj = 3 linear layers
    # gate: hidden -> intermediate (2*hidden*intermediate*batch)
    # up:   hidden -> intermediate (2*hidden*intermediate*batch)
    # down: intermediate -> hidden (2*intermediate*hidden*batch)
    # Total fwd = 6*hidden*intermediate*batch
    fwd_flops = 6 * hidden_size * intermediate_size * batch_tokens

    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    weight_b = 3 * hidden_size * intermediate_size * dtype_bytes

    mem_rw = (
        weight_b
        + batch_tokens * hidden_size * dtype_bytes  # read input
        + batch_tokens * hidden_size * dtype_bytes  # write output
    )

    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * hidden_size * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_moe_ffn(
    hidden_size: int,
    expert_intermediate_size: int,
    num_experts: int,
    num_shared_experts: int,
    shared_intermediate_size: int,
    top_k: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """MoE FFN. FLOPs = 6·d·d_e·top_k + 6·d·d_s·n_shared. Ref: DeepSeek-V3."""
    routed_flops = 6 * hidden_size * expert_intermediate_size * top_k * batch_tokens
    shared_flops = (
        6 * hidden_size * shared_intermediate_size * num_shared_experts * batch_tokens
        if num_shared_experts > 0
        else 0
    )
    router_flops = 2 * hidden_size * num_experts * batch_tokens

    fwd_flops = routed_flops + shared_flops + router_flops

    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    # Routed expert weights (all experts loaded to device in MoE)
    routed_weight_b = (
        num_experts * 3 * hidden_size * expert_intermediate_size * dtype_bytes
    )
    shared_weight_b = (
        num_shared_experts * 3 * hidden_size * shared_intermediate_size * dtype_bytes
        if num_shared_experts > 0
        else 0
    )
    router_weight_b = num_experts * hidden_size * dtype_bytes
    weight_b = routed_weight_b + shared_weight_b + router_weight_b

    mem_rw = (
        weight_b
        + batch_tokens * hidden_size * dtype_bytes  # read input
        + batch_tokens * hidden_size * dtype_bytes  # write output
    )

    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * hidden_size * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_rmsnorm(
    hidden_size: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """RMSNorm. Memory-bound element-wise op."""
    # ~5 ops per element: square, sum, rsqrt, scale, multiply
    flops = 5 * hidden_size * batch_tokens
    # Read input + write output
    mem_rw = 2 * hidden_size * batch_tokens * dtype_bytes
    weight_b = hidden_size * dtype_bytes  # gamma

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=0,
    )


def op_mhc_residual(
    hidden_size: int,
    expansion: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """mHC residual. Memory-bandwidth-bound. Ref: arXiv:2512.24880."""
    # Small matmul on [expansion, expansion] matrix
    flops = 2 * hidden_size * expansion * expansion * batch_tokens
    # Read expanded + write + intermediates: memory-bound
    mem_rw = batch_tokens * expansion * hidden_size * dtype_bytes * 3
    # H_res + H_pre + H_post per sublayer
    weight_b = 3 * expansion * expansion * dtype_bytes

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=0,
    )


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
    output_b = (
        batch_tokens * vocab_size * dtype_bytes if phase == Phase.TRAIN_FWD else 0
    )
    return OpCost(
        flops=flops, mem_rw=mem_rw, weight_bytes=weight_b, output_bytes=output_b
    )


# ---------------------------------------------------------------------------
# Communication operator cost functions
# ---------------------------------------------------------------------------


def op_allreduce(msg_bytes: float, group_size: int) -> OpCost:
    """AllReduce. Ring-allreduce: 2 * msg * (N-1)/N bytes transferred."""
    comm_b = 2 * msg_bytes * (group_size - 1) / group_size
    return OpCost(comm_bytes=comm_b)


def op_allgather(msg_bytes: float, group_size: int) -> OpCost:
    """AllGather. Each device sends msg_bytes * (N-1)/N."""
    comm_b = msg_bytes * (group_size - 1) / group_size
    return OpCost(comm_bytes=comm_b)


def op_reducescatter(msg_bytes: float, group_size: int) -> OpCost:
    """ReduceScatter. Each device sends msg_bytes * (N-1)/N."""
    comm_b = msg_bytes * (group_size - 1) / group_size
    return OpCost(comm_bytes=comm_b)


def op_alltoall(
    tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    dtype_bytes: int = 2,
) -> OpCost:
    """AllToAll for MoE expert dispatch + combine. comm_bytes = 2*tokens*top_k*hidden*bytes."""
    comm_b = 2 * tokens * top_k * hidden_size * dtype_bytes
    return OpCost(comm_bytes=comm_b)


def op_p2p(msg_bytes: float) -> OpCost:
    """Point-to-point communication."""
    return OpCost(comm_bytes=msg_bytes)


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
