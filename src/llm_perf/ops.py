"""Operator cost model using roofline analysis.

Each op function returns an OpCost describing FLOPs, memory, and communication.
Use roofline_time() to convert OpCost → seconds given a HardwareConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from llm_perf.config import HardwareConfig, Phase
from llm_perf.precision import dtype_bytes as _dtype_bytes


@dataclass
class OpCost:
    flops: float = 0  # FLOPs count
    mem_rw: float = 0  # HBM read+write bytes (for roofline)
    weight_bytes: float = 0  # Persistent weight memory
    output_bytes: float = 0  # Transient activation/gradient memory
    comm_bytes: float = 0  # Communication bytes (for comm ops)


def roofline_time(
    cost: OpCost, hw: HardwareConfig, is_large_gemm: bool = True,
    compute_class: str = "bf16",
) -> float:
    """Roofline model: time = max(compute_time, memory_time).

    Uses two-tier calibration (large GEMM vs small ops) and per-precision peak
    TFLOPS selected by compute_class ("bf16"|"fp8"|"fp4").
    """
    eff = (
        hw.calibration.compute_eff_large_gemm
        if is_large_gemm
        else hw.calibration.compute_eff_small_op
    )
    peak = hw.tflops_for(compute_class)
    compute_time = (
        cost.flops / (peak * 1e12 * eff) if cost.flops > 0 else 0
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
    lat = hw.intra_node_latency_us * 1e-6 if is_intra_node else hw.inter_node_latency_us * 1e-6

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

    # KV cache read: during decode each query reads all L past K,V pairs from the
    # cache (K + V, each d_kv per token). This dominates decode memory traffic for
    # long contexts and makes decode time scale with batch and context length.
    if phase == Phase.DECODE:
        mem_rw += batch * L * 2 * d_kv * dtype_bytes

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

    # KV cache read during decode (bounded by the sliding window via L above).
    if phase == Phase.DECODE:
        mem_rw += batch * L * 2 * d_kv * dtype_bytes

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
    # + decoupled RoPE projections: Q rope (d * num_heads * r) + K rope (d * r, shared)
    weight_b = (
        d * d_c_q + d_c_q * d + d * d_c + d_c * d + d_c * d + d * d
        + d * num_heads * r + d * r
    ) * dtype_bytes

    mem_rw = weight_b + query_tokens * d * dtype_bytes + query_tokens * d * dtype_bytes

    # KV cache read during decode: MLA caches a single compressed latent (d_c)
    # plus the RoPE key (r) per token — read once to reconstruct K and V.
    if phase == Phase.DECODE:
        mem_rw += batch * L * (d_c + r) * dtype_bytes

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


def op_dsa_attention(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    o_groups: int,
    compress_ratio: int,
    compress_c_kv: int,
    compress_coeff: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    window_size: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
    rope_dim: int = 0,
) -> OpCost:
    """DSA (DeepSeek Sparse Attention) operator cost model.

    Comprises three sub-ops: KV Compression, Lightning Index (C4A only),
    and Compressed Attention, plus projection FLOPs.

    Args:
        hidden_size: Model hidden dimension (d).
        num_heads: Number of query heads (TP-partitioned).
        head_dim: Dimension per attention head.
        q_lora_rank: Q LoRA rank (qlr).
        o_lora_rank: O LoRA rank (olr).
        o_groups: O projection block-diagonal groups.
        compress_ratio: KV compression ratio: 4 (C4A), 128 (C128A), 0 (non-DSA).
        compress_c_kv: Compressed KV dimension.
        compress_coeff: Compression cost coefficient: 1.0 (C4A), 0.5 (C128A).
        index_n_heads: Lightning Index head count.
        index_head_dim: Lightning Index head dimension.
        index_topk: Top-K compressed KV entries for attention.
        window_size: Sliding window size for SWA component.
        batch: Batch size.
        seq_len: Sequence length.
        phase: Execution phase.
        kv_len: KV cache length for decode.
        dtype_bytes: Bytes per element.
    """
    H = hidden_size
    Nq = num_heads
    d_h = head_dim
    qlr = q_lora_rank
    olr = o_lora_rank
    g = o_groups
    c_kv = compress_c_kv
    coeff = compress_coeff
    idx_nh = index_n_heads
    idx_hd = index_head_dim
    topk = index_topk
    W = window_size
    r = rope_dim
    use_index = compress_ratio == 4  # Lightning Index only for C4A

    if phase == Phase.DECODE:
        L = kv_len if kv_len is not None else seq_len
        query_tokens = batch
    else:
        L = seq_len
        query_tokens = batch * seq_len

    S = seq_len  # for prefill/train, this is the full sequence
    S_comp = S // compress_ratio if compress_ratio > 0 else S

    # ---- 1. Projection FLOPs (all phases) ----
    # Q down: d -> qlr
    proj_flops = 2 * H * qlr * query_tokens
    # Q up: qlr -> Nq*d_h
    proj_flops += 2 * qlr * (Nq * d_h) * query_tokens
    # KV proj (MQA): d -> d_h (single KV head)
    kv_dim = d_h  # MQA
    proj_flops += 2 * H * kv_dim * query_tokens
    # O proj_a: Nq*d_h -> olr
    proj_flops += 2 * (Nq * d_h) * olr * query_tokens
    # O proj_b: Ng*olr -> d
    Ng = max(1, Nq // g)  # heads per group
    proj_flops += 2 * (Ng * olr) * H * query_tokens
    # Decoupled RoPE projection (MLA-style): Q rope (num_heads * r) + K rope (r)
    proj_flops += 2 * H * Nq * r * query_tokens

    # ---- 2. KV Compression FLOPs ----
    comp_flops = 0.0
    if compress_ratio > 0 and c_kv > 0 and coeff > 0:
        if phase in (Phase.PREFILL, Phase.TRAIN_FWD, Phase.TRAIN_BWD):
            # Projection + group compression
            comp_flops = coeff * (8 * batch * S * H * c_kv + (14 * g - 1) * c_kv * batch * S_comp)
        elif phase == Phase.DECODE:
            # Check alignment: decode aligned if kv_len is multiple of compress_ratio
            aligned = (kv_len is not None and kv_len % compress_ratio == 0)
            if aligned:
                comp_flops = coeff * (8 * batch * H * c_kv + (14 * g - 1) * batch * c_kv)
            else:
                # Unaligned: only projection, no group compression
                comp_flops = coeff * 8 * batch * H * c_kv

    # ---- 3. Lightning Index FLOPs (C4A only) ----
    index_flops = 0.0
    if use_index and idx_nh > 0 and idx_hd > 0:
        # IQ projection: qlr -> idx_nh*idx_hd
        index_flops += 2 * qlr * (idx_nh * idx_hd) * query_tokens

        # Index KV compression (same structure as KV compression, with idx_hd, coeff=1.0)
        idx_coeff = 1.0
        if phase in (Phase.PREFILL, Phase.TRAIN_FWD, Phase.TRAIN_BWD):
            index_flops += idx_coeff * (8 * batch * S * H * idx_hd + (14 * g - 1) * idx_hd * batch * S_comp)
            # Index score computation for prefill
            index_flops += (
                batch * idx_nh * S * S_comp * idx_hd * 2  # QK^T
                + batch * idx_nh * S * S_comp              # softmax
                + batch * idx_nh * S * S_comp * 3          # top-k selection
                + batch * S * S_comp * math.log(max(S, 2))  # argsort overhead
            )
        elif phase == Phase.DECODE:
            aligned = (kv_len is not None and kv_len % compress_ratio == 0)
            if aligned:
                S_comp_decode = (kv_len or seq_len) // compress_ratio
                index_flops += idx_coeff * (8 * batch * H * idx_hd + (14 * g - 1) * batch * idx_hd)
                # Index score for decode
                index_flops += (
                    batch * idx_nh * S_comp_decode * idx_hd * 2  # QK^T
                    + batch * idx_nh * S_comp_decode              # softmax
                    + batch * idx_nh * S_comp_decode * 3          # top-k
                )
            else:
                index_flops += idx_coeff * 8 * batch * H * idx_hd

    # ---- 4. SWA Attention FLOPs ----
    swa_flops = 0.0
    if W > 0:
        # SWA uses full KV dim (d_h for MQA)
        if phase == Phase.DECODE:
            swa_L = min(L, W)
            swa_flops = 2 * batch * Nq * 1 * swa_L * kv_dim * 2 * 1.3
        else:
            swa_L = min(S, W)
            swa_flops = 2 * batch * Nq * S * swa_L * kv_dim * 2 * 1.3

    # ---- 5. Compressed Attention FLOPs ----
    # Only when compress_ratio > 1 (C4A or C128A); ratio=1 means SWA-only
    ca_flops = 0.0
    if compress_ratio > 1 and c_kv > 0:
        n_attend = topk if use_index else S_comp
        if phase == Phase.DECODE:
            ca_flops = 2 * batch * Nq * 1 * n_attend * c_kv * 2 * 1.3
        else:
            ca_flops = 2 * batch * Nq * S * n_attend * c_kv * 2 * 1.3

    fwd_flops = proj_flops + comp_flops + index_flops + swa_flops + ca_flops

    if phase == Phase.TRAIN_BWD:
        flops = 2 * fwd_flops
    else:
        flops = fwd_flops

    # ---- Weight bytes ----
    weight_b = (
        H * qlr                           # Q down
        + qlr * (Nq * d_h)                # Q up
        + H * kv_dim                      # KV (MQA)
        + (Nq * d_h) * olr               # O proj_a
        + (Ng * olr) * H                  # O proj_b
        + H * Nq * r                      # Q decoupled RoPE proj
        + H * r                           # K decoupled RoPE proj (shared)
    ) * dtype_bytes

    # Index IQ weight (C4A only)
    if use_index and idx_nh > 0 and idx_hd > 0:
        weight_b += qlr * idx_nh * idx_hd * dtype_bytes

    mem_rw = weight_b + query_tokens * H * dtype_bytes + query_tokens * H * dtype_bytes

    # KV cache read during decode: SWA window (MQA K+V of dim d_h, bounded by W)
    # + compressed KV latent (c_kv) + Lightning Index entries (C4A), the
    # compressed parts read at L/compress_ratio length.
    if phase == Phase.DECODE:
        kv_read = batch * min(L, W) * 2 * kv_dim * dtype_bytes if W > 0 else 0.0
        # Decoupled RoPE key (uncompressed, full context length, single shared head)
        kv_read += batch * L * r * dtype_bytes
        if compress_ratio > 1 and c_kv > 0:
            comp_len = L // compress_ratio
            kv_read += batch * comp_len * c_kv * dtype_bytes
            if use_index and idx_hd > 0:
                kv_read += batch * comp_len * idx_hd * dtype_bytes
        mem_rw += kv_read

    if phase == Phase.TRAIN_FWD:
        output_b = query_tokens * H * dtype_bytes
    else:
        output_b = 0

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=output_b,
    )


def op_dsa_index_score_allreduce(
    batch: int,
    seq_len: int,
    compress_ratio: int,
    index_n_heads: int,
    index_head_dim: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
) -> OpCost:
    """Communication cost for Lightning Index score AllReduce (DSA C4A).

    Only produces comm_bytes; compute is handled by op_dsa_attention.
    """
    g = compress_ratio
    if g <= 1 or index_n_heads <= 0:
        return OpCost()

    if phase in (Phase.PREFILL, Phase.TRAIN_FWD, Phase.TRAIN_BWD):
        comm_b = batch * seq_len * (seq_len // g) * dtype_bytes
    elif phase == Phase.DECODE:
        L = kv_len if kv_len is not None else seq_len
        comm_b = batch * (L // g) * dtype_bytes
    else:
        comm_b = 0

    return OpCost(comm_bytes=comm_b)


def op_swiglu_ffn(
    hidden_size: int,
    intermediate_size: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
    weight_dtype_bytes: float | None = None,
    act_dtype_bytes: float | None = None,
) -> OpCost:
    """SwiGLU FFN. FLOPs = 6·d·d_ff. Ref: Shazeer 2020.

    For low-precision training the resident weight copy (weight_bytes) and the
    saved activation (output_bytes) may be smaller than the compute dtype:
    weight_dtype_bytes / act_dtype_bytes override those terms while mem_rw (the
    roofline traffic) keeps the compute dtype. Both default to dtype_bytes.
    """
    w_bytes = dtype_bytes if weight_dtype_bytes is None else weight_dtype_bytes
    a_bytes = dtype_bytes if act_dtype_bytes is None else act_dtype_bytes
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

    weight_b = 3 * hidden_size * intermediate_size * w_bytes

    mem_rw = (
        3 * hidden_size * intermediate_size * dtype_bytes
        + batch_tokens * hidden_size * dtype_bytes  # read input
        + batch_tokens * hidden_size * dtype_bytes  # write output
    )

    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * hidden_size * a_bytes
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
    weight_dtype_bytes: float | None = None,
    act_dtype_bytes: float | None = None,
) -> OpCost:
    """MoE FFN. FLOPs = 6·d·d_e·top_k + 6·d·d_s·n_shared. Ref: DeepSeek-V3.

    weight_dtype_bytes / act_dtype_bytes size the resident weight copy and the
    saved activation for low-precision training; mem_rw (roofline traffic) keeps
    the compute dtype. Both default to dtype_bytes.
    """
    w_bytes = dtype_bytes if weight_dtype_bytes is None else weight_dtype_bytes
    a_bytes = dtype_bytes if act_dtype_bytes is None else act_dtype_bytes
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
        num_experts * 3 * hidden_size * expert_intermediate_size * w_bytes
    )
    shared_weight_b = (
        num_shared_experts * 3 * hidden_size * shared_intermediate_size * w_bytes
        if num_shared_experts > 0
        else 0
    )
    router_weight_b = num_experts * hidden_size * w_bytes
    weight_b = routed_weight_b + shared_weight_b + router_weight_b

    # mem_rw keeps the compute dtype (roofline traffic), independent of the
    # resident weight precision above.
    mem_rw_weight = (
        num_experts * 3 * hidden_size * expert_intermediate_size * dtype_bytes
        + (num_shared_experts * 3 * hidden_size * shared_intermediate_size * dtype_bytes
           if num_shared_experts > 0 else 0)
        + num_experts * hidden_size * dtype_bytes
    )
    mem_rw = (
        mem_rw_weight
        + batch_tokens * hidden_size * dtype_bytes  # read input
        + batch_tokens * hidden_size * dtype_bytes  # write output
    )

    if phase == Phase.TRAIN_FWD:
        output_b = batch_tokens * hidden_size * a_bytes
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


def op_mhc_pre_fused(
    hidden_size: int,
    expansion: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """Fused mHC pre + Sinkhorn: single kernel for residual mixing + aggregation.

    Fuses: Sinkhorn normalization of H_res, residual mixing H_res @ x_l,
    and input aggregation H_pre · x_l into one kernel.

    With fusion, intermediates stay in on-chip SRAM/registers.
    Only true inputs and outputs traverse HBM.

    Ref: arXiv:2512.24880, DeepSeek V4 perf_model.
    """
    n = expansion
    D = hidden_size
    T = batch_tokens

    # FLOPs: identical to unfused pre + sinkhorn combined
    # pre:      2*T*(n²+2n)*n*D + 5*T*n + 2*T*n²  (H_res matmul + H_pre dot)
    # sinkhorn: T*n² + 40*T*n*(2n-1)               (20 iters × row/col norm)
    cube_flops = 2 * T * (n**2 + 2 * n) * n * D + 5 * T * n + 2 * T * n**2
    vec_ops = T * n**2 + 40 * T * n * (2 * n - 1) + 2 * T * n
    flops = cube_flops + vec_ops

    # Fused HBM traffic: ONLY input read + output writes
    # BF16 for inference (H_res is doubly stochastic, spectral norm ≤ 1)
    bpe = dtype_bytes  # 2 for BF16
    mem_rw = (
        bpe * T * n * D           # read x_l [T, n, D]
        + bpe * T * n * D         # write residual [T, n, D]
        + bpe * T * D             # write sub_input [T, D]
    )

    # Weights: H_res_logits [n,n] + H_pre_logits [n] (FP32, negligible)
    weight_b = 4 * (n * n + n)

    if phase == Phase.TRAIN_BWD:
        flops = 2 * flops

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=0,
    )


def op_mhc_post_fused(
    hidden_size: int,
    expansion: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """Fused mHC post: distribute sub-layer output back to n streams + residual add.

    Mathematical operation:
      x_{l+1}[t, i, :] = residual[t, i, :] + H_post[i] × sub_output[t, :]

    Ref: arXiv:2512.24880, DeepSeek V4 perf_model.
    """
    n = expansion
    D = hidden_size
    T = batch_tokens

    # FLOPs: same as unfused
    cube_flops = 2 * T * n**2 * D + 3 * T * n * D
    vec_ops = 0
    flops = cube_flops + vec_ops

    # Fused HBM traffic
    bpe = dtype_bytes
    mem_rw = (
        bpe * T * n * D           # read residual [T, n, D]
        + bpe * T * D             # read sub_output [T, D]
        + bpe * T * n * D         # write x_{l+1} [T, n, D]
    )

    weight_b = 4 * n  # H_post_logits [n] (FP32, negligible)

    if phase == Phase.TRAIN_BWD:
        flops = 2 * flops

    return OpCost(
        flops=flops,
        mem_rw=mem_rw,
        weight_bytes=weight_b,
        output_bytes=0,
    )


def op_mhc_post_pre_fused(
    hidden_size: int,
    expansion: int,
    batch_tokens: int,
    phase: Phase,
    dtype_bytes: int = 2,
) -> OpCost:
    """Deep-fused mHC: post(sublayer_k) + sinkhorn(k+1) + pre(sublayer_k+1).

    Fuses across the sublayer boundary (e.g. post_attn → pre_moe),
    eliminating the intermediate x_{l+0.5} from HBM entirely.

    This is the most important fusion: it saves bpe × T × n × D × 2 bytes
    (read+write of x_{l+0.5}) compared to separate fused post + fused pre.

    Ref: arXiv:2512.24880, DeepSeek V4 perf_model.
    """
    n = expansion
    D = hidden_size
    T = batch_tokens

    # FLOPs: sum of post + sinkhorn + pre
    cube_flops = (
        (2 * T * n**2 * D + 3 * T * n * D)                    # post
        + (2 * T * (n**2 + 2 * n) * n * D + 5 * T * n + 2 * T * n**2)  # pre
    )
    vec_ops = T * n**2 + 40 * T * n * (2 * n - 1) + 2 * T * n  # sinkhorn + pre
    flops = cube_flops + vec_ops

    # Deep-fused HBM traffic: x_{l+0.5} never written to HBM
    bpe = dtype_bytes
    mem_rw = (
        bpe * T * n * D           # read residual_k [T, n, D]
        + bpe * T * D             # read sub_output_k [T, D]
        + bpe * T * n * D         # write residual_{k+1} [T, n, D]
        + bpe * T * D             # write sub_input_{k+1} [T, D]
    )

    # Weights: H_res + H_pre + H_post for both sublayers (FP32, negligible)
    weight_b = 4 * (n * n + 2 * n)

    if phase == Phase.TRAIN_BWD:
        flops = 2 * flops

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


def op_alltoall_dispatch(
    tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    dtype_bytes: int = 2,
) -> OpCost:
    """AllToAll dispatch: send tokens to their target expert EP ranks.

    Each token is routed to top_k experts; the dispatch AllToAll redistributes
    tokens so that each EP rank receives the tokens destined for its local experts.
    comm_bytes = tokens * top_k * hidden * dtype_bytes.
    """
    comm_b = tokens * top_k * hidden_size * dtype_bytes
    return OpCost(comm_bytes=comm_b)


def op_alltoall_combine(
    tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    dtype_bytes: int = 2,
) -> OpCost:
    """AllToAll combine: gather expert outputs back to original EP ranks.

    After local expert computation, the combine AllToAll sends each token's
    expert output back to the EP rank that originally owned the token.
    comm_bytes = tokens * top_k * hidden * dtype_bytes.
    """
    comm_b = tokens * top_k * hidden_size * dtype_bytes
    return OpCost(comm_bytes=comm_b)


def op_alltoall(
    tokens: int,
    hidden_size: int,
    top_k: int,
    ep_size: int,
    dtype_bytes: int = 2,
) -> OpCost:
    """AllToAll for MoE expert dispatch + combine (combined cost).

    DEPRECATED: Use op_alltoall_dispatch + op_alltoall_combine instead.
    Kept for backward compatibility. comm_bytes = 2*tokens*top_k*hidden*bytes.
    """
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


def op_quantize(
    numel: float, in_dtype: str, out_dtype: str,
    block_size: int = 0, scale_bytes: int = 4,
) -> OpCost:
    """Quantize: per-block scale reduction + cast. Memory-bound element-wise.

    Reads numel*in_bytes, writes numel*out_bytes, plus scale-metadata writes.
    FLOPs ~ a few per element (max-abs reduce + divide). Ref: fine-grained
    block quantization (DeepSeek-V3); cost dominated by HBM traffic.
    """
    in_b = _dtype_bytes(in_dtype)
    out_b = _dtype_bytes(out_dtype)
    if block_size <= 0:
        scale_b = scale_bytes
    else:
        scale_b = (-(-int(numel) // block_size)) * scale_bytes
    mem_rw = numel * in_b + numel * out_b + scale_b
    flops = 3 * numel  # abs, compare/reduce, divide
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_dequant(
    numel: float, in_dtype: str, out_dtype: str,
    block_size: int = 0, scale_bytes: int = 4,
) -> OpCost:
    """Dequantize: upcast + apply scale. Memory-bound mirror of op_quantize."""
    in_b = _dtype_bytes(in_dtype)
    out_b = _dtype_bytes(out_dtype)
    if block_size <= 0:
        scale_b = scale_bytes
    else:
        scale_b = (-(-int(numel) // block_size)) * scale_bytes
    mem_rw = numel * in_b + numel * out_b + scale_b
    flops = numel  # one multiply per element
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_hadamard(tokens: float, d: int, block: int) -> OpCost:
    """Stochastic Hadamard via Fast Walsh-Hadamard Transform (FWHT).

    FWHT over a block of size B costs B*log2(B) MACs; applied to tokens*d/B
    blocks => tokens*d*log2(B) MACs. Ref: QuaRot / stochastic Hadamard rotation
    for outlier-free low-precision training.
    """
    b = block if block and block > 1 else 128
    macs = tokens * d * math.log2(b)
    flops = 2 * macs
    in_b = 2  # operates on bf16/fp16 activations before quant
    mem_rw = 2 * tokens * d * in_b  # read + write the rotated tensor
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_compensation_add(numel: float, dtype: str) -> OpCost:
    """Error-feedback residual add: ef_buf += (x - dequant(quant(x))).

    Memory-bound: read EF buffer + read residual + write EF buffer.
    Ref: error feedback / residual accumulation in low-precision training.
    """
    b = _dtype_bytes(dtype)
    mem_rw = 3 * numel * b
    flops = numel
    return OpCost(flops=flops, mem_rw=mem_rw, weight_bytes=0, output_bytes=0)


def op_attention_split(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: int | None = None,
    dtype_bytes: int = 2,
    proj_weight_dtype_bytes: float | None = None,
    proj_act_dtype_bytes: float | None = None,
) -> tuple["OpCost", "OpCost"]:
    """Split GQA/MHA attention into (projection GEMM, core attention) costs.

    Projection = QKV + O linears (quantizable; carries all weights + activation
    traffic). Core = QK^T + softmax*V (kept FP16 per Zhou et al. 2025). FLOPs of
    the two parts sum to op_gqa_attention's total. proj_weight/act_dtype_bytes
    size the resident weight copy / saved activation for low precision (default
    dtype_bytes). Ref: GQA paper; Zhou et al. arXiv:2502.11458.
    """
    H, G, d = num_heads, num_kv_heads, hidden_size
    batch_tokens = batch * seq_len
    d_qo = H * head_dim
    d_kv = G * head_dim
    L = (
        (kv_len if kv_len is not None else seq_len)
        if phase == Phase.DECODE
        else seq_len
    )
    wb = dtype_bytes if proj_weight_dtype_bytes is None else proj_weight_dtype_bytes
    ab = dtype_bytes if proj_act_dtype_bytes is None else proj_act_dtype_bytes

    proj_flops = (2 * d * d_qo + 2 * d * d_kv + 2 * d * d_kv + 2 * d_qo * d) * batch_tokens
    attn_flops = (
        (4 * d_qo * L * batch)
        if phase == Phase.DECODE
        else (4 * d_qo * L * batch_tokens)
    )
    bwd = 2 if phase == Phase.TRAIN_BWD else 1

    weight_b = (d * d_qo + d * d_kv + d * d_kv + d_qo * d) * wb
    proj_mem = (
        weight_b + batch_tokens * d * dtype_bytes + batch_tokens * d_qo * dtype_bytes
    )
    proj_out = batch_tokens * d * ab if phase == Phase.TRAIN_FWD else 0
    proj = OpCost(
        flops=bwd * proj_flops,
        mem_rw=proj_mem,
        weight_bytes=weight_b,
        output_bytes=proj_out,
    )

    core_mem = 2 * batch_tokens * d_qo * dtype_bytes  # read scores / write context (approx)
    if phase == Phase.DECODE:
        core_mem += batch * L * 2 * d_kv * dtype_bytes
    core = OpCost(flops=bwd * attn_flops, mem_rw=core_mem, weight_bytes=0, output_bytes=0)
    return proj, core
