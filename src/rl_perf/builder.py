"""Builder: converts ModelConfig + ParallelismConfig into SimOp sequences.

Parallelism shapes are divided HERE — ops.py knows nothing about TP/EP.
Each op produces a SimOp with name, stream, duration, depends_on,
weight_bytes, and output_bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rl_perf import ops
from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    RLConfig,
)


@dataclass
class SimOp:
    name: str
    stream: str  # "compute", "tp_comm", "ep_comm", "dp_comm"
    duration: float  # seconds
    depends_on: List[int] = field(default_factory=list)
    weight_bytes: float = 0
    output_bytes: float = 0
    consumers: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_intra_node(group_size: int, hw: HardwareConfig) -> bool:
    return group_size <= hw.devices_per_node


_COLLECTIVE_OPS = {
    "allreduce": (ops.op_allreduce, "ring"),
    "allgather": (ops.op_allgather, "ring_half"),
    "reducescatter": (ops.op_reducescatter, "ring_half"),
}


def _build_tp_comm(
    name: str,
    collective: str,
    batch: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int,
    tp: int,
    hw: HardwareConfig,
    dep_idx: int,
) -> SimOp:
    """Build a TP collective comm SimOp (allreduce, allgather, or reducescatter)."""
    op_fn, algorithm = _COLLECTIVE_OPS[collective]
    msg = batch * seq_len * hidden_size * dtype_bytes
    cost = op_fn(msg, group_size=tp)
    duration = ops.comm_time(
        cost, hw, group_size=tp, is_intra_node=_is_intra_node(tp, hw), algorithm=algorithm
    )
    return SimOp(
        name=name,
        stream="tp_comm",
        duration=duration,
        depends_on=[dep_idx],
        weight_bytes=0,
        output_bytes=0,
    )


# ---------------------------------------------------------------------------
# build_layer_ops
# ---------------------------------------------------------------------------


def build_layer_ops(
    layer_cfg: LayerConfig,
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    hw: HardwareConfig,
    batch: int,
    seq_len: int,
    phase: Phase,
    kv_len: Optional[int] = None,
    index_offset: int = 0,
) -> List[SimOp]:
    """Build SimOps for one transformer layer.

    Parameters
    ----------
    index_offset:
        Global index of the first op returned (used so depends_on indices are
        globally consistent when ops from multiple layers are concatenated).
    """
    tp = parallel_cfg.tp
    ep = parallel_cfg.ep
    cp = parallel_cfg.cp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes
    batch_tokens = batch * seq_len
    attn_seq_len = seq_len // cp if cp > 1 else seq_len

    result: List[SimOp] = []

    def _idx(local: int) -> int:
        """Convert local (within-layer) index to global index."""
        return index_offset + local

    # ---- 0. Pre-attention RMSNorm ----------------------------------------
    norm1_cost = ops.op_rmsnorm(d, batch_tokens, phase, dtype_bytes)
    norm1 = SimOp(
        name="rmsnorm_pre_attn",
        stream="compute",
        duration=ops.roofline_time(norm1_cost, hw, is_large_gemm=False),
        depends_on=[],  # first op in layer; caller chains layers together
        weight_bytes=norm1_cost.weight_bytes,
        output_bytes=norm1_cost.output_bytes,
    )
    result.append(norm1)  # local idx 0

    # ---- CP Ring comm (before attention) -------------------------------------
    if cp > 1:
        attn_type = layer_cfg.attention
        if attn_type == "MLA":
            kv_dim = layer_cfg.kv_compression_dim + layer_cfg.rope_dim
        elif attn_type in ("GQA", "MHA", "SWA"):
            tp_kv_heads = max(1, layer_cfg.num_kv_heads // tp)
            kv_dim = tp_kv_heads * layer_cfg.head_dim
        else:
            kv_dim = layer_cfg.num_kv_heads * layer_cfg.head_dim

        cp_cost = ops.op_ring_cp(seq_len, cp, kv_dim, dtype_bytes)
        cp_ring = SimOp(
            name="cp_ring_kv",
            stream="cp_comm",
            duration=ops.comm_time(
                cp_cost, hw, group_size=cp,
                is_intra_node=_is_intra_node(cp, hw), algorithm="ring_half"
            ),
            depends_on=[_idx(len(result) - 1)],  # depends on norm1
            weight_bytes=0,
            output_bytes=0,
        )
        result.append(cp_ring)

    # ---- 1. Attention -------------------------------------------------------
    attn_type = layer_cfg.attention

    if attn_type in ("GQA", "MHA"):
        tp_num_heads = max(1, layer_cfg.num_heads // tp)
        tp_kv_heads = max(1, layer_cfg.num_kv_heads // tp)
        attn_cost = ops.op_gqa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=attn_seq_len,
            phase=phase,
            kv_len=kv_len if kv_len is not None else None,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "SWA":
        tp_num_heads = max(1, layer_cfg.num_heads // tp)
        tp_kv_heads = max(1, layer_cfg.num_kv_heads // tp)
        attn_cost = ops.op_swa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=attn_seq_len,
            phase=phase,
            window_size=layer_cfg.window_size,
            kv_len=kv_len if kv_len is not None else None,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "MLA":
        tp_num_heads = max(1, layer_cfg.num_heads // tp)
        attn_cost = ops.op_mla_attention(
            hidden_size=d,
            num_heads=tp_num_heads,
            head_dim=layer_cfg.head_dim,
            kv_compression_dim=layer_cfg.kv_compression_dim,
            query_compression_dim=layer_cfg.query_compression_dim,
            rope_dim=layer_cfg.rope_dim,
            batch=batch,
            seq_len=attn_seq_len,
            phase=phase,
            kv_len=kv_len if kv_len is not None else None,
            dtype_bytes=dtype_bytes,
        )
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    # SP: insert AllGather before attention if sp=True and tp>1
    if parallel_cfg.sp and tp > 1:
        ag_attn = _build_tp_comm(
            name="tp_allgather_attn", collective="allgather",
            batch=batch,
            seq_len=seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm1
        )
        result.append(ag_attn)

    attn_dep_idx = _idx(len(result) - 1)  # depends on allgather (SP) or norm1 (no SP)
    attn_op = SimOp(
        name=f"attention_{attn_type.lower()}",
        stream="compute",
        duration=ops.roofline_time(attn_cost, hw, is_large_gemm=True),
        depends_on=[attn_dep_idx],
        weight_bytes=attn_cost.weight_bytes,
        output_bytes=attn_cost.output_bytes,
    )
    result.append(attn_op)

    # ---- 2. TP comm (attention) ---------------------------------------------
    if tp > 1:
        if parallel_cfg.sp:
            tp_attn_comm = _build_tp_comm(
                name="tp_reducescatter_attn", collective="reducescatter",
                batch=batch,
                seq_len=seq_len,
                hidden_size=d,
                dtype_bytes=dtype_bytes,
                tp=tp,
                hw=hw,
                dep_idx=_idx(len(result) - 1),  # depends on attention
            )
        else:
            tp_attn_comm = _build_tp_comm(
                name="tp_allreduce_attn", collective="allreduce",
                batch=batch,
                seq_len=seq_len,
                hidden_size=d,
                dtype_bytes=dtype_bytes,
                tp=tp,
                hw=hw,
                dep_idx=_idx(len(result) - 1),  # depends on attention
            )
        result.append(tp_attn_comm)
        last_compute_idx = _idx(len(result) - 1)
    else:
        last_compute_idx = _idx(len(result) - 1)

    # ---- 3. Pre-FFN RMSNorm -------------------------------------------------
    norm2_cost = ops.op_rmsnorm(d, batch_tokens, phase, dtype_bytes)
    norm2 = SimOp(
        name="rmsnorm_pre_ffn",
        stream="compute",
        duration=ops.roofline_time(norm2_cost, hw, is_large_gemm=False),
        depends_on=[last_compute_idx],
        weight_bytes=norm2_cost.weight_bytes,
        output_bytes=norm2_cost.output_bytes,
    )
    result.append(norm2)  # local idx = len(result)-1
    last_compute_local = len(result) - 1

    # ---- 4. FFN -------------------------------------------------------------
    # SP: insert AllGather before FFN if sp=True and tp>1
    if parallel_cfg.sp and tp > 1:
        ag_ffn = _build_tp_comm(
            name="tp_allgather_ffn", collective="allgather",
            batch=batch,
            seq_len=seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm2
        )
        result.append(ag_ffn)
        last_compute_local = len(result) - 1

    ffn_type = layer_cfg.ffn

    if ffn_type == "SwiGLU":
        tp_intermediate = max(1, layer_cfg.intermediate_size // tp)
        ffn_cost = ops.op_swiglu_ffn(
            hidden_size=d,
            intermediate_size=tp_intermediate,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        ffn_op = SimOp(
            name="ffn_swiglu",
            stream="compute",
            duration=ops.roofline_time(ffn_cost, hw, is_large_gemm=True),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
        )
        result.append(ffn_op)
        last_compute_local = len(result) - 1

    elif ffn_type == "MoE":
        ep_num_experts = max(1, layer_cfg.num_experts // ep)
        ffn_cost = ops.op_moe_ffn(
            hidden_size=d,
            expert_intermediate_size=layer_cfg.expert_intermediate_size,
            num_experts=ep_num_experts,
            num_shared_experts=layer_cfg.num_shared_experts,
            shared_intermediate_size=layer_cfg.shared_intermediate_size,
            top_k=layer_cfg.top_k,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        ffn_op = SimOp(
            name="ffn_moe",
            stream="compute",
            duration=ops.roofline_time(ffn_cost, hw, is_large_gemm=True),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
        )
        result.append(ffn_op)
        last_compute_local = len(result) - 1

        # EP AllToAll (dispatch + combine) — emitted on ep_comm stream
        if ep > 1:
            a2a_cost = ops.op_alltoall(
                tokens=batch_tokens,
                hidden_size=d,
                top_k=layer_cfg.top_k,
                ep_size=ep,
                dtype_bytes=dtype_bytes,
            )
            ep_comm = SimOp(
                name="ep_alltoall",
                stream="ep_comm",
                duration=ops.comm_time(
                    a2a_cost, hw, group_size=ep, is_intra_node=_is_intra_node(ep, hw), algorithm="alltoall"
                ),
                depends_on=[_idx(last_compute_local)],
                weight_bytes=0,
                output_bytes=0,
            )
            result.append(ep_comm)
            last_compute_local = len(result) - 1
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")

    # ---- 5. TP comm (FFN) ----------------------------------------------------
    if tp > 1:
        if parallel_cfg.sp:
            tp_ffn_comm = _build_tp_comm(
                name="tp_reducescatter_ffn", collective="reducescatter",
                batch=batch,
                seq_len=seq_len,
                hidden_size=d,
                dtype_bytes=dtype_bytes,
                tp=tp,
                hw=hw,
                dep_idx=_idx(last_compute_local),
            )
        else:
            tp_ffn_comm = _build_tp_comm(
                name="tp_allreduce_ffn", collective="allreduce",
                batch=batch,
                seq_len=seq_len,
                hidden_size=d,
                dtype_bytes=dtype_bytes,
                tp=tp,
                hw=hw,
                dep_idx=_idx(last_compute_local),
            )
        result.append(tp_ffn_comm)
        last_compute_local = len(result) - 1

    # ---- 6. mHC residual (optional) -----------------------------------------
    if layer_cfg.residual == "mHC":
        mhc_cost = ops.op_mhc_residual(
            hidden_size=d,
            expansion=layer_cfg.mhc_expansion,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        mhc_op = SimOp(
            name="mhc_residual",
            stream="compute",
            duration=ops.roofline_time(mhc_cost, hw, is_large_gemm=False),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=mhc_cost.weight_bytes,
            output_bytes=mhc_cost.output_bytes,
        )
        result.append(mhc_op)

    # Zero out weight_bytes for BWD ops to avoid double-counting in simulator
    if phase == Phase.TRAIN_BWD:
        for op in result:
            op.weight_bytes = 0

    return result


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def _estimate_param_count(model_cfg: ModelConfig, parallel_cfg: ParallelismConfig) -> int:
    """Quick estimate of model parameters per TP shard for gradient sync sizing."""
    tp = parallel_cfg.tp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes

    param_bytes = 0
    for layer_cfg in model_cfg.get_layers():
        # Attention
        attn_type = layer_cfg.attention
        if attn_type in ("GQA", "MHA", "SWA"):
            tp_num_heads = max(1, layer_cfg.num_heads // tp)
            tp_kv_heads = max(1, layer_cfg.num_kv_heads // tp)
            # Q: d*num_heads*head_dim; K,V: d*kv_heads*head_dim
            # O: RowParallel, input dim = tp_num_heads * head_dim
            q_params = d * tp_num_heads * layer_cfg.head_dim
            kv_params = 2 * d * tp_kv_heads * layer_cfg.head_dim
            o_params = tp_num_heads * layer_cfg.head_dim * d
            param_bytes += (q_params + kv_params + o_params) * dtype_bytes
        elif attn_type == "MLA":
            w_b = (
                d * layer_cfg.query_compression_dim
                + layer_cfg.query_compression_dim * d
                + d * layer_cfg.kv_compression_dim
                + layer_cfg.kv_compression_dim * d
                + layer_cfg.kv_compression_dim * d
                + d * d
            ) * dtype_bytes
            param_bytes += w_b

        # FFN
        ffn_type = layer_cfg.ffn
        if ffn_type == "SwiGLU":
            tp_intermediate = max(1, layer_cfg.intermediate_size // tp)
            param_bytes += 3 * d * tp_intermediate * dtype_bytes
        elif ffn_type == "MoE":
            ep = parallel_cfg.ep
            ep_num_experts = max(1, layer_cfg.num_experts // ep)
            param_bytes += (
                ep_num_experts * 3 * d * layer_cfg.expert_intermediate_size * dtype_bytes
            )
            # Shared experts (not EP-partitioned)
            if layer_cfg.num_shared_experts > 0:
                shared_int = layer_cfg.shared_intermediate_size or layer_cfg.intermediate_size
                param_bytes += (
                    layer_cfg.num_shared_experts * 3 * d * shared_int * dtype_bytes
                )

        # RMSNorm (small, 2 per layer)
        param_bytes += 2 * d * dtype_bytes

    return int(param_bytes / dtype_bytes)  # return as element count


# ---------------------------------------------------------------------------
# build_training_step
# ---------------------------------------------------------------------------


def build_training_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: RLConfig,
) -> List[SimOp]:
    """Build SimOps for one full training micro-step (forward + backward + DP sync + optim).

    MVP: PP stage 0 only (builds all layers / pp).
    """
    pp = parallel_cfg.pp
    dp = parallel_cfg.dp
    dtype_bytes = model_cfg.dtype_bytes

    seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
    batch = rl_cfg.train_micro_batch_size

    all_layers = model_cfg.get_layers()
    num_layers_total = len(all_layers)
    # PP stage 0: take the first chunk of layers
    stage_layers_count = max(1, num_layers_total // pp)
    stage_layers = all_layers[:stage_layers_count]

    all_ops: List[SimOp] = []

    # Dependency chain: track the global index of the last op in the main sequence
    prev_dep: Optional[int] = None

    # ------ Forward pass -------------------------------------------------------
    for layer_cfg in stage_layers:
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_FWD,
            index_offset=offset,
        )
        # Chain: first op of this layer depends on last op of previous layer
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ MTP head (forward) ------------------------------------------------
    mtp_depth = 0
    if model_cfg.auxiliary:
        mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)

    if mtp_depth > 0:
        mtp_fwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * seq_len,
            phase=Phase.TRAIN_FWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_fwd = SimOp(
            name="mtp_head_fwd",
            stream="compute",
            duration=ops.roofline_time(mtp_fwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=mtp_fwd_cost.weight_bytes,
            output_bytes=mtp_fwd_cost.output_bytes,
        )
        all_ops.append(mtp_fwd)
        prev_dep = len(all_ops) - 1

    # ------ MTP head (backward) — before layer backward (backprop output→input)
    if mtp_depth > 0:
        mtp_bwd_cost = ops.op_mtp_head(
            hidden_size=model_cfg.hidden_size,
            vocab_size=model_cfg.vocab_size,
            mtp_depth=mtp_depth,
            batch_tokens=batch * seq_len,
            phase=Phase.TRAIN_BWD,
            dtype_bytes=dtype_bytes,
        )
        mtp_bwd = SimOp(
            name="mtp_head_bwd",
            stream="compute",
            duration=ops.roofline_time(mtp_bwd_cost, hw, is_large_gemm=True),
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=0,
            output_bytes=0,
        )
        all_ops.append(mtp_bwd)
        prev_dep = len(all_ops) - 1

    # ------ Backward pass (reversed layers) ------------------------------------
    for layer_cfg in reversed(stage_layers):
        offset = len(all_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=seq_len,
            phase=Phase.TRAIN_BWD,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ DP gradient sync ---------------------------------------------------
    if dp > 1:
        param_count = _estimate_param_count(model_cfg, parallel_cfg)
        grad_bytes = param_count * dtype_bytes  # gradient same dtype as weights

        if parallel_cfg.zero_stage < 3:
            # AllReduce (ring)
            dp_cost = ops.op_allreduce(grad_bytes, group_size=dp)
        else:
            # ZeRO-3: ReduceScatter
            dp_cost = ops.op_reducescatter(grad_bytes, group_size=dp)

        dp_algorithm = "ring_half" if parallel_cfg.zero_stage >= 3 else "ring"
        dp_duration = ops.comm_time(
            dp_cost, hw, group_size=dp, is_intra_node=_is_intra_node(dp, hw), algorithm=dp_algorithm
        )
        dp_sync = SimOp(
            name="dp_grad_sync",
            stream="dp_comm",
            duration=dp_duration,
            depends_on=[prev_dep] if prev_dep is not None else [],
            weight_bytes=0,
            output_bytes=0,
        )
        all_ops.append(dp_sync)
        prev_dep = len(all_ops) - 1

    # ------ Optimizer step (placeholder) ---------------------------------------
    param_count = _estimate_param_count(model_cfg, parallel_cfg)
    optim_duration = param_count * 1e-10  # rough placeholder
    optim_op = SimOp(
        name="optimizer_step",
        stream="compute",
        duration=optim_duration,
        depends_on=[prev_dep] if prev_dep is not None else [],
        weight_bytes=0,
        output_bytes=0,
    )
    all_ops.append(optim_op)

    return all_ops


# ---------------------------------------------------------------------------
# build_generation_step
# ---------------------------------------------------------------------------


def build_generation_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: RLConfig,
) -> Tuple[List[SimOp], List[SimOp]]:
    """Build SimOps for generation (prefill + decode-per-token).

    Returns (prefill_ops, decode_per_token_ops).
    """
    pp = parallel_cfg.pp
    all_layers = model_cfg.get_layers()
    num_layers_total = len(all_layers)
    stage_layers_count = max(1, num_layers_total // pp)
    stage_layers = all_layers[:stage_layers_count]

    batch = rl_cfg.gen_batch_size
    prompt_len = rl_cfg.avg_prompt_len
    kv_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len

    # ------ Prefill ------------------------------------------------------------
    prefill_ops: List[SimOp] = []
    prev_dep: Optional[int] = None

    for layer_cfg in stage_layers:
        offset = len(prefill_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=prompt_len,
            phase=Phase.PREFILL,
            kv_len=None,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        prefill_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(prefill_ops) - 1

    # ------ Decode (per token) -------------------------------------------------
    decode_ops: List[SimOp] = []
    prev_dep = None

    for layer_cfg in stage_layers:
        offset = len(decode_ops)
        layer_ops = build_layer_ops(
            layer_cfg=layer_cfg,
            model_cfg=model_cfg,
            parallel_cfg=parallel_cfg,
            hw=hw,
            batch=batch,
            seq_len=1,
            phase=Phase.DECODE,
            kv_len=kv_len,
            index_offset=offset,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        decode_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(decode_ops) - 1

    return prefill_ops, decode_ops
