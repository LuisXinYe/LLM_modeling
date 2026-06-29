"""Builder: converts ModelConfig + ParallelismConfig into SimOp sequences.

Parallelism shapes are divided HERE — ops.py knows nothing about TP/EP.
Each op produces a SimOp with name, stream, duration, depends_on,
weight_bytes, and output_bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from llm_perf import ops
from llm_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    WorkloadConfig,
)
from llm_perf.precision import (
    PrecisionConfig,
    compute_class as _compute_class,
    dtype_bytes as _prec_dtype_bytes,
)


def _split_stages(
    all_layers: List[LayerConfig], pp: int
) -> List[List[LayerConfig]]:
    """Split layers into pp pipeline stages, allowing uneven distribution.

    When num_layers is not divisible by pp, the first (num_layers % pp) stages
    get one extra layer — matching Megatron-LM's behavior.

    E.g. 51 layers / pp=8 → [7, 7, 7, 6, 6, 6, 6, 6]
    """
    n = len(all_layers)
    base = n // pp
    remainder = n % pp
    stages = []
    offset = 0
    for i in range(pp):
        count = base + (1 if i < remainder else 0)
        stages.append(all_layers[offset : offset + count])
        offset += count
    return stages


def _validate_parallelism(
    all_layers: List[LayerConfig], parallel_cfg: ParallelismConfig
):
    """Validate parallelism config against model dimensions."""
    tp = parallel_cfg.tp
    pp = parallel_cfg.pp

    if pp > len(all_layers):
        raise ValueError(
            f"pp={pp} exceeds num_layers={len(all_layers)}. "
            f"pp must be <= number of model layers."
        )

    for i, layer in enumerate(all_layers):
        if layer.num_heads % tp != 0:
            raise ValueError(
                f"Layer {i}: num_heads={layer.num_heads} not divisible by tp={tp}. "
                f"Choose tp that evenly divides num_heads."
            )
        if layer.attention in ("GQA", "MHA", "SWA") and layer.num_kv_heads % tp != 0:
            raise ValueError(
                f"Layer {i}: num_kv_heads={layer.num_kv_heads} not divisible by tp={tp}. "
                f"Choose tp that evenly divides num_kv_heads."
            )
        if layer.ffn == "SwiGLU" and layer.intermediate_size % tp != 0:
            raise ValueError(
                f"Layer {i}: intermediate_size={layer.intermediate_size} not divisible by tp={tp}. "
                f"Choose tp that evenly divides intermediate_size."
            )
        if layer.attention == "DSA" and layer.compress_ratio == 4 and layer.index_n_heads % tp != 0:
            raise ValueError(
                f"Layer {i}: index_n_heads={layer.index_n_heads} not divisible by tp={tp}. "
                f"Choose tp that evenly divides index_n_heads for DSA C4A layers."
            )


@dataclass
class SimOp:
    name: str
    stream: str  # "compute", "tp_comm", "ep_comm", "dp_comm"
    duration: float  # seconds
    depends_on: List[int] = field(default_factory=list)
    weight_bytes: float = 0
    output_bytes: float = 0
    comm_bytes: float = 0  # communication volume in bytes (for comm ops)
    consumers: Optional[List[int]] = None
    param_count: float = 0  # architecture param count (set once, on the optimizer op)
    fabric: Optional[str] = None  # "nvlink" (intra-node) | "nic" (inter-node) | None for compute
    op_class: Optional[str] = None  # compute pipe class for GEMM ops: "bf16" | "fp8" | "fp4"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_intra_node(group_size: int, hw: HardwareConfig) -> bool:
    return group_size <= hw.devices_per_node


def _fabric(group_size: int, hw: HardwareConfig) -> str:
    """Physical fabric a collective traverses: intra-node NVLink/HCCS vs inter-node NIC."""
    return "nvlink" if _is_intra_node(group_size, hw) else "nic"


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
    """Build a TP collective comm SimOp (allreduce, allgather, or reducescatter).

    Note: seq_len should already be CP-local (i.e. seq_len // cp) when called
    from build_layer_ops, which handles CP sharding internally.
    """
    op_fn, algorithm = _COLLECTIVE_OPS[collective]
    msg = batch * seq_len * hidden_size * dtype_bytes
    cost = op_fn(msg, group_size=tp)
    duration = ops.comm_time(
        cost,
        hw,
        group_size=tp,
        is_intra_node=_is_intra_node(tp, hw),
        algorithm=algorithm,
    )
    return SimOp(
        name=name,
        stream="tp_comm",
        duration=duration,
        depends_on=[dep_idx],
        weight_bytes=0,
        output_bytes=0,
        comm_bytes=cost.comm_bytes,
        fabric=_fabric(tp, hw),
    )


# ---------------------------------------------------------------------------
# _inject_quant_chain
# ---------------------------------------------------------------------------


def _inject_quant_chain(
    result: List[SimOp],
    gemm_op: SimOp,
    numel: float,
    tokens: float,
    d: int,
    pc: PrecisionConfig,
    hw: HardwareConfig,
    index_offset: int,
    dep_idx: Optional[int],
    component_name: str = "",
) -> int:
    """Prepend quantize(+hadamard) and append dequant around gemm_op.

    Returns the global index of the dequant op (the new chain tail). When the
    activation compute_class is bf16 (no low precision), appends gemm_op as-is
    and returns its index.

    When component_name is in pc.high_precision_layers, the component uses the
    high-precision dtype path (bf16 branch) even if activations are low-precision.
    """
    act = pc.activations
    # High-precision layer override: skip quant chain for this component
    if component_name and component_name in pc.high_precision_layers:
        gemm_op.depends_on = [dep_idx] if dep_idx is not None else []
        result.append(gemm_op)
        return index_offset + len(result) - 1
    if _compute_class(act.dtype) == "bf16":
        gemm_op.depends_on = [dep_idx] if dep_idx is not None else []
        result.append(gemm_op)
        return index_offset + len(result) - 1

    prev = dep_idx
    q = SimOp(
        name="quantize_act",
        stream="compute",
        duration=ops.roofline_time(
            ops.op_quantize(numel, "bf16", act.dtype, act.block_size, act.scale_bytes),
            hw,
            is_large_gemm=False,
        ),
        depends_on=[prev] if prev is not None else [],
    )
    result.append(q)
    prev = index_offset + len(result) - 1

    if act.hadamard:
        h = SimOp(
            name="hadamard_act",
            stream="compute",
            duration=ops.roofline_time(
                ops.op_hadamard(tokens, d, act.hadamard_block),
                hw,
                is_large_gemm=False,
            ),
            depends_on=[prev],
        )
        result.append(h)
        prev = index_offset + len(result) - 1

    gemm_op.depends_on = [prev]
    result.append(gemm_op)
    prev = index_offset + len(result) - 1

    dq = SimOp(
        name="dequant_out",
        stream="compute",
        duration=ops.roofline_time(
            ops.op_dequant(numel, act.dtype, "bf16", act.block_size, act.scale_bytes),
            hw,
            is_large_gemm=False,
        ),
        depends_on=[prev],
    )
    result.append(dq)
    return index_offset + len(result) - 1


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
    precision_cfg: Optional[PrecisionConfig] = None,
) -> List[SimOp]:
    """Build SimOps for one transformer layer.

    Parameters
    ----------
    index_offset:
        Global index of the first op returned (used so depends_on indices are
        globally consistent when ops from multiple layers are concatenated).
    precision_cfg:
        Precision configuration. None resolves to PrecisionConfig.bf16_default(),
        which reproduces legacy single-dtype behavior.
    """
    pc = precision_cfg or PrecisionConfig.from_model_dtype(model_cfg.dtype)
    tp = parallel_cfg.tp
    ep = parallel_cfg.ep
    cp = parallel_cfg.cp
    sp = parallel_cfg.sp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes
    # CP shards the sequence across ranks; all per-rank ops use the local length
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    # SP further shards the sequence across TP ranks (only when sp=True and tp>1)
    sp_seq_len = local_seq_len // tp if (sp and tp > 1) else local_seq_len

    # batch_tokens for ops that run on SP-sharded data (rmsnorm, residual, etc.)
    sp_batch_tokens = batch * sp_seq_len
    # batch_tokens for ops that run on full CP-local data (attention, FFN)
    batch_tokens = batch * local_seq_len

    result: List[SimOp] = []

    def _idx(local: int) -> int:
        """Convert local (within-layer) index to global index."""
        return index_offset + local

    # ---- 0. Pre-attention RMSNorm ----------------------------------------
    # RMSNorm runs on SP-sharded data (seq_len/cp/tp when SP on, seq_len/cp otherwise)
    norm1_cost = ops.op_rmsnorm(d, sp_batch_tokens, phase, dtype_bytes)
    norm1 = SimOp(
        name="rmsnorm_pre_attn",
        stream="compute",
        duration=ops.roofline_time(norm1_cost, hw, is_large_gemm=False),
        depends_on=[],  # first op in layer; caller chains layers together
        weight_bytes=norm1_cost.weight_bytes,
        output_bytes=norm1_cost.output_bytes,
    )
    result.append(norm1)  # local idx 0

    # ---- SP AllGather (before CP ring / attention) ----------------------------
    # SP AllGather restores sequence from sp_seq_len to local_seq_len (CP-local).
    # Must happen BEFORE CP ring, because CP ring operates on full CP-local sequence.
    # Comm volume is sized by the GATHERED (full) sequence, so this AllGather
    # matches the ReduceScatter below — together they equal one AllReduce.
    if sp and tp > 1:
        ag_attn = _build_tp_comm(
            name="tp_allgather_attn",
            collective="allgather",
            batch=batch,
            seq_len=local_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),  # depends on norm1
        )
        result.append(ag_attn)

    # ---- CP Ring comm (before attention) -------------------------------------
    # CP Ring operates on the full CP-local sequence (after SP AllGather if SP is on).
    if cp > 1:
        attn_type = layer_cfg.attention
        if attn_type == "MLA":
            kv_dim = layer_cfg.kv_compression_dim + layer_cfg.rope_dim
        elif attn_type == "DSA":
            kv_dim = layer_cfg.compress_c_kv
        elif attn_type in ("GQA", "MHA", "SWA"):
            tp_kv_heads = layer_cfg.num_kv_heads // tp
            kv_dim = tp_kv_heads * layer_cfg.head_dim
        else:
            kv_dim = layer_cfg.num_kv_heads * layer_cfg.head_dim

        cp_cost = ops.op_ring_cp(seq_len, cp, kv_dim, dtype_bytes)
        cp_ring = SimOp(
            name="cp_ring_kv",
            stream="cp_comm",
            duration=ops.comm_time(
                cp_cost,
                hw,
                group_size=cp,
                is_intra_node=_is_intra_node(cp, hw),
                algorithm="ring_half",
            ),
            depends_on=[_idx(len(result) - 1)],  # depends on SP AG or norm1
            weight_bytes=0,
            output_bytes=0,
            comm_bytes=cp_cost.comm_bytes,
            fabric=_fabric(cp, hw),
        )
        result.append(cp_ring)

    # ---- 1. Attention -------------------------------------------------------
    attn_type = layer_cfg.attention

    if attn_type in ("GQA", "MHA"):
        tp_num_heads = layer_cfg.num_heads // tp
        tp_kv_heads = layer_cfg.num_kv_heads // tp
        attn_cost = ops.op_gqa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "SWA":
        tp_num_heads = layer_cfg.num_heads // tp
        tp_kv_heads = layer_cfg.num_kv_heads // tp
        attn_cost = ops.op_swa_attention(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            window_size=layer_cfg.window_size,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "MLA":
        tp_num_heads = layer_cfg.num_heads // tp
        attn_cost = ops.op_mla_attention(
            hidden_size=d,
            num_heads=tp_num_heads,
            head_dim=layer_cfg.head_dim,
            kv_compression_dim=layer_cfg.kv_compression_dim,
            query_compression_dim=layer_cfg.query_compression_dim,
            rope_dim=layer_cfg.rope_dim,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
    elif attn_type == "DSA":
        tp_num_heads = layer_cfg.num_heads // tp
        tp_index_heads = layer_cfg.index_n_heads // tp if layer_cfg.index_n_heads > 0 else 0
        attn_cost = ops.op_dsa_attention(
            hidden_size=d,
            num_heads=tp_num_heads,
            head_dim=layer_cfg.head_dim,
            q_lora_rank=layer_cfg.q_lora_rank,
            o_lora_rank=layer_cfg.o_lora_rank,
            o_groups=layer_cfg.o_groups,
            compress_ratio=layer_cfg.compress_ratio,
            compress_c_kv=layer_cfg.compress_c_kv,
            compress_coeff=layer_cfg.compress_coeff,
            index_n_heads=tp_index_heads,
            index_head_dim=layer_cfg.index_head_dim,
            index_topk=layer_cfg.index_topk,
            window_size=layer_cfg.window_size,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
            rope_dim=layer_cfg.rope_dim,
        )
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    attn_dep_idx = _idx(len(result) - 1)  # depends on CP ring / SP AG / norm1
    attn_is_large_gemm = attn_type != "DSA"  # DSA uses small_gemm for mixed cube+vec ops

    if attn_type in ("GQA", "MHA") and pc.attn_linear is not None:
        # Split path: precision-aware proj (QKV + O linears) + FP16 core (scores/softmax)
        attn_prec = (
            pc.linear_bwd("attn") if phase == Phase.TRAIN_BWD else pc.linear_fwd("attn")
        )
        proj_cost, core_cost = ops.op_attention_split(
            num_heads=tp_num_heads,
            num_kv_heads=tp_kv_heads,
            head_dim=layer_cfg.head_dim,
            hidden_size=d,
            batch=batch,
            seq_len=local_seq_len,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
            proj_weight_dtype_bytes=_prec_dtype_bytes(attn_prec.dtype),
            proj_act_dtype_bytes=_prec_dtype_bytes(attn_prec.dtype),
        )
        proj_op = SimOp(
            name="attention_proj",
            stream="compute",
            duration=ops.roofline_time(
                proj_cost, hw, is_large_gemm=True,
                compute_class=_compute_class(attn_prec.dtype),
            ),
            depends_on=[attn_dep_idx],  # overridden by _inject_quant_chain
            weight_bytes=proj_cost.weight_bytes,
            output_bytes=proj_cost.output_bytes,
            op_class=_compute_class(attn_prec.dtype),
        )
        proj_tail = _inject_quant_chain(
            result=result,
            gemm_op=proj_op,
            numel=batch_tokens * d,
            tokens=batch_tokens,
            d=d,
            pc=pc,
            hw=hw,
            index_offset=index_offset,
            dep_idx=attn_dep_idx,
            component_name="attn",
        )
        core_op = SimOp(
            name="attention_core",
            stream="compute",
            duration=ops.roofline_time(core_cost, hw, is_large_gemm=True),  # FP16 core
            depends_on=[proj_tail],
            weight_bytes=0,
            output_bytes=core_cost.output_bytes,
        )
        result.append(core_op)
    else:
        attn_op = SimOp(
            name=f"attention_{attn_type.lower()}",
            stream="compute",
            duration=ops.roofline_time(attn_cost, hw, is_large_gemm=attn_is_large_gemm),
            depends_on=[attn_dep_idx],
            weight_bytes=attn_cost.weight_bytes,
            output_bytes=attn_cost.output_bytes,
        )
        result.append(attn_op)

    # ---- 1b. Lightning Index AllReduce (DSA C4A only, when TP > 1) ----------
    if attn_type == "DSA" and layer_cfg.compress_ratio == 4 and tp > 1:
        index_allreduce_cost = ops.op_dsa_index_score_allreduce(
            batch=batch,
            seq_len=local_seq_len,
            compress_ratio=layer_cfg.compress_ratio,
            index_n_heads=layer_cfg.index_n_heads,
            index_head_dim=layer_cfg.index_head_dim,
            phase=phase,
            kv_len=kv_len,
            dtype_bytes=dtype_bytes,
        )
        if index_allreduce_cost.comm_bytes > 0:
            index_allreduce_duration = ops.comm_time(
                index_allreduce_cost,
                hw,
                group_size=tp,
                is_intra_node=_is_intra_node(tp, hw),
                algorithm="ring",
            )
            index_allreduce_op = SimOp(
                name="tp_allreduce_index_score",
                stream="tp_comm",
                duration=index_allreduce_duration,
                depends_on=[_idx(len(result) - 1)],
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=index_allreduce_cost.comm_bytes,
                fabric=_fabric(tp, hw),
            )
            result.append(index_allreduce_op)

    # ---- 2. TP comm (attention) ---------------------------------------------
    if tp > 1:
        # SP: ReduceScatter (returns to SP-sharded state)
        # No SP: AllReduce
        collective = "reducescatter" if sp else "allreduce"
        tp_attn_comm = _build_tp_comm(
            name=f"tp_{collective}_attn",
            collective=collective,
            batch=batch,
            seq_len=local_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(len(result) - 1),
        )
        result.append(tp_attn_comm)
    last_compute_idx = _idx(len(result) - 1)

    # ---- 3. mHC post-attn + pre-MoE (deep-fused, optional) ------------------
    # Placed after attention TP comm, before pre-FFN RMSNorm.
    # This deep-fused kernel eliminates x_{l+0.5} from HBM.
    if layer_cfg.residual == "mHC":
        mhc_mid_cost = ops.op_mhc_post_pre_fused(
            hidden_size=d,
            expansion=layer_cfg.mhc_expansion,
            batch_tokens=sp_batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        mhc_mid_op = SimOp(
            name="mhc_post_attn_pre_moe",
            stream="compute",
            duration=ops.roofline_time(mhc_mid_cost, hw, is_large_gemm=False),
            depends_on=[last_compute_idx],
            weight_bytes=mhc_mid_cost.weight_bytes,
            output_bytes=mhc_mid_cost.output_bytes,
        )
        result.append(mhc_mid_op)
        last_compute_idx = _idx(len(result) - 1)

    # ---- 4. Pre-FFN RMSNorm -------------------------------------------------
    # After TP ReduceScatter (SP), data is back to SP-sharded state.
    # So RMSNorm runs on SP-sharded data.
    norm2_cost = ops.op_rmsnorm(d, sp_batch_tokens, phase, dtype_bytes)
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
    # SP: insert AllGather before FFN to restore from sp_seq_len to local_seq_len.
    # Comm volume is sized by the GATHERED (full) sequence (matches ReduceScatter).
    if sp and tp > 1:
        ag_ffn = _build_tp_comm(
            name="tp_allgather_ffn",
            collective="allgather",
            batch=batch,
            seq_len=local_seq_len,
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
        tp_intermediate = layer_cfg.intermediate_size // tp
        ffn_prec = pc.linear_bwd("ffn") if phase == Phase.TRAIN_BWD else pc.linear_fwd("ffn")
        ffn_cc = _compute_class(ffn_prec.dtype)
        if pc.ffn_linear is not None:
            ffn_w_bytes = ffn_a_bytes = _prec_dtype_bytes(ffn_prec.dtype)
        else:
            ffn_w_bytes = _prec_dtype_bytes(pc.weights.dtype)
            ffn_a_bytes = _prec_dtype_bytes(pc.activations.dtype)
        ffn_cost = ops.op_swiglu_ffn(
            hidden_size=d,
            intermediate_size=tp_intermediate,
            batch_tokens=batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
            weight_dtype_bytes=ffn_w_bytes,
            act_dtype_bytes=ffn_a_bytes,
        )
        ffn_op = SimOp(
            name="ffn_swiglu",
            stream="compute",
            duration=ops.roofline_time(
                ffn_cost, hw, is_large_gemm=True,
                compute_class=ffn_cc,
            ),
            depends_on=[],  # set by _inject_quant_chain
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
            op_class=ffn_cc,
        )
        tail_idx = _inject_quant_chain(
            result=result,
            gemm_op=ffn_op,
            numel=batch_tokens * d,
            tokens=batch_tokens,
            d=d,
            pc=pc,
            hw=hw,
            index_offset=index_offset,
            dep_idx=_idx(last_compute_local),
            component_name="ffn",
        )
        last_compute_local = tail_idx - index_offset

    elif ffn_type == "MoE":
        ep_num_experts = max(1, layer_cfg.num_experts // ep)

        # EP AllToAll dispatch: send tokens to target expert EP ranks
        if ep > 1:
            a2a_dispatch_cost = ops.op_alltoall_dispatch(
                tokens=batch_tokens,
                hidden_size=d,
                top_k=layer_cfg.top_k,
                ep_size=ep,
                dtype_bytes=dtype_bytes,
            )
            ep_dispatch = SimOp(
                name="ep_alltoall_dispatch",
                stream="ep_comm",
                duration=ops.comm_time(
                    a2a_dispatch_cost,
                    hw,
                    group_size=ep,
                    is_intra_node=_is_intra_node(ep, hw),
                    algorithm="alltoall",
                ),
                depends_on=[_idx(last_compute_local)],
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=a2a_dispatch_cost.comm_bytes,
                fabric=_fabric(ep, hw),
            )
            result.append(ep_dispatch)
            last_compute_local = len(result) - 1

        ffn_prec = pc.linear_bwd("ffn") if phase == Phase.TRAIN_BWD else pc.linear_fwd("ffn")
        ffn_cc = _compute_class(ffn_prec.dtype)
        if pc.ffn_linear is not None:
            ffn_w_bytes = ffn_a_bytes = _prec_dtype_bytes(ffn_prec.dtype)
        else:
            ffn_w_bytes = _prec_dtype_bytes(pc.weights.dtype)
            ffn_a_bytes = _prec_dtype_bytes(pc.activations.dtype)
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
            weight_dtype_bytes=ffn_w_bytes,
            act_dtype_bytes=ffn_a_bytes,
        )
        ffn_op = SimOp(
            name="ffn_moe",
            stream="compute",
            duration=ops.roofline_time(
                ffn_cost, hw, is_large_gemm=True,
                compute_class=ffn_cc,
            ),
            depends_on=[],  # set by _inject_quant_chain
            weight_bytes=ffn_cost.weight_bytes,
            output_bytes=ffn_cost.output_bytes,
            op_class=ffn_cc,
        )
        tail_idx = _inject_quant_chain(
            result=result,
            gemm_op=ffn_op,
            numel=batch_tokens * d,
            tokens=batch_tokens,
            d=d,
            pc=pc,
            hw=hw,
            index_offset=index_offset,
            dep_idx=_idx(last_compute_local),
            component_name="ffn",
        )
        last_compute_local = tail_idx - index_offset

        # EP AllToAll combine: gather expert outputs back to original EP ranks
        if ep > 1:
            a2a_combine_cost = ops.op_alltoall_combine(
                tokens=batch_tokens,
                hidden_size=d,
                top_k=layer_cfg.top_k,
                ep_size=ep,
                dtype_bytes=dtype_bytes,
            )
            ep_combine = SimOp(
                name="ep_alltoall_combine",
                stream="ep_comm",
                duration=ops.comm_time(
                    a2a_combine_cost,
                    hw,
                    group_size=ep,
                    is_intra_node=_is_intra_node(ep, hw),
                    algorithm="alltoall",
                ),
                depends_on=[_idx(last_compute_local)],
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=a2a_combine_cost.comm_bytes,
                fabric=_fabric(ep, hw),
            )
            result.append(ep_combine)
            last_compute_local = len(result) - 1
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")

    # ---- 5. TP comm (FFN) ----------------------------------------------------
    if tp > 1:
        collective = "reducescatter" if sp else "allreduce"
        tp_ffn_comm = _build_tp_comm(
            name=f"tp_{collective}_ffn",
            collective=collective,
            batch=batch,
            seq_len=local_seq_len,
            hidden_size=d,
            dtype_bytes=dtype_bytes,
            tp=tp,
            hw=hw,
            dep_idx=_idx(last_compute_local),
        )
        result.append(tp_ffn_comm)
        last_compute_local = len(result) - 1

    # ---- 6. mHC pre-attn + post-moe (optional) -------------------------------
    # mHC (manifold Hyper Connection) replaces standard residual connections.
    # With fused kernels, each layer has 3 mHC ops:
    #   a) mhc_pre_attn: BEFORE norm1 (inserted at start of layer)
    #   b) mhc_post_attn_pre_moe: after attn TP comm, before norm2 (already inserted above)
    #   c) mhc_post_moe: after FFN TP ReduceScatter (inserted here)
    # Ref: arXiv:2512.24880
    if layer_cfg.residual == "mHC":
        # 6a. mHC pre-attention: placed BEFORE norm1
        # In the V4 reference, mhc_pre comes before RMSNorm.
        # We insert it at the beginning by shifting norm1's dependency.
        mhc_pre_cost = ops.op_mhc_pre_fused(
            hidden_size=d,
            expansion=layer_cfg.mhc_expansion,
            batch_tokens=sp_batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        mhc_pre_op = SimOp(
            name="mhc_pre_attn",
            stream="compute",
            duration=ops.roofline_time(mhc_pre_cost, hw, is_large_gemm=False),
            depends_on=[],  # first op; caller chains layers together
            weight_bytes=mhc_pre_cost.weight_bytes,
            output_bytes=mhc_pre_cost.output_bytes,
        )
        # Insert mhc_pre before norm1, and make norm1 depend on it
        result.insert(0, mhc_pre_op)
        result[1].depends_on = [_idx(0)]  # norm1 now depends on mhc_pre

        # 6c. mHC post-MoE: after FFN TP ReduceScatter
        mhc_post_cost = ops.op_mhc_post_fused(
            hidden_size=d,
            expansion=layer_cfg.mhc_expansion,
            batch_tokens=sp_batch_tokens,
            phase=phase,
            dtype_bytes=dtype_bytes,
        )
        mhc_post_op = SimOp(
            name="mhc_post_moe",
            stream="compute",
            duration=ops.roofline_time(mhc_post_cost, hw, is_large_gemm=False),
            depends_on=[_idx(last_compute_local)],
            weight_bytes=mhc_post_cost.weight_bytes,
            output_bytes=mhc_post_cost.output_bytes,
        )
        result.append(mhc_post_op)

    # Zero out weight_bytes for BWD ops to avoid double-counting in simulator
    if phase == Phase.TRAIN_BWD:
        for op in result:
            op.weight_bytes = 0

    return result


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def _estimate_param_count(
    model_cfg: ModelConfig,
    parallel_cfg: ParallelismConfig,
    all_layers: List[LayerConfig],
) -> int:
    """Quick estimate of model parameters per TP shard for gradient sync sizing."""
    tp = parallel_cfg.tp
    d = model_cfg.hidden_size
    dtype_bytes = model_cfg.dtype_bytes

    param_bytes = 0
    for layer_cfg in all_layers:
        # Attention
        attn_type = layer_cfg.attention
        if attn_type in ("GQA", "MHA", "SWA"):
            tp_num_heads = layer_cfg.num_heads // tp
            tp_kv_heads = layer_cfg.num_kv_heads // tp
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
        elif attn_type == "DSA":
            tp_num_heads = layer_cfg.num_heads // tp
            q_params = d * layer_cfg.q_lora_rank + layer_cfg.q_lora_rank * tp_num_heads * layer_cfg.head_dim
            kv_params = d * layer_cfg.head_dim  # MQA
            o_params = tp_num_heads * layer_cfg.head_dim * layer_cfg.o_lora_rank + (layer_cfg.o_groups * layer_cfg.o_lora_rank * d) // tp
            param_bytes += (q_params + kv_params + o_params) * dtype_bytes
            if layer_cfg.compress_ratio == 4:
                tp_index_heads = layer_cfg.index_n_heads // tp
                param_bytes += layer_cfg.q_lora_rank * tp_index_heads * layer_cfg.index_head_dim * dtype_bytes

        # FFN
        ffn_type = layer_cfg.ffn
        if ffn_type == "SwiGLU":
            tp_intermediate = layer_cfg.intermediate_size // tp
            param_bytes += 3 * d * tp_intermediate * dtype_bytes
        elif ffn_type == "MoE":
            ep = parallel_cfg.ep
            ep_num_experts = max(1, layer_cfg.num_experts // ep)
            param_bytes += (
                ep_num_experts
                * 3
                * d
                * layer_cfg.expert_intermediate_size
                * dtype_bytes
            )
            # Shared experts (not EP-partitioned)
            if layer_cfg.num_shared_experts > 0:
                shared_int = (
                    layer_cfg.shared_intermediate_size or layer_cfg.intermediate_size
                )
                param_bytes += (
                    layer_cfg.num_shared_experts * 3 * d * shared_int * dtype_bytes
                )

        # RMSNorm (small, 2 per layer)
        param_bytes += 2 * d * dtype_bytes

        # mHC weights (FP32): 4 matrices of [n,n] per layer
        # H_res_attn, H_res_moe, H_pre (×2), H_post (×2) → 4 * n² FP32 params
        if layer_cfg.residual == "mHC":
            n = layer_cfg.mhc_expansion
            param_bytes += 4 * 3 * n * n * dtype_bytes

    return int(param_bytes / dtype_bytes)  # return as element count


# ---------------------------------------------------------------------------
# build_forward_pass
# ---------------------------------------------------------------------------


def build_forward_pass(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    include_mtp: bool = True,
    name_prefix: str = "",
    stage_layers: Optional[List[LayerConfig]] = None,
) -> List[SimOp]:
    """Build SimOps for a forward-pass-only (no backward, no DP sync, no optimizer).

    Used for reference model forward.

    Parameters
    ----------
    include_mtp: Whether to include MTP head forward ops.
    name_prefix: Prefix for op names to disambiguate (e.g. "ref_").
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp
    dtype_bytes = model_cfg.dtype_bytes

    seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
    batch = rl_cfg.train_micro_batch_size
    cp = parallel_cfg.cp
    local_seq_len = seq_len // cp if cp > 1 else seq_len

    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

    all_ops: List[SimOp] = []
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
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

    # ------ MTP head (forward only) -------------------------------------------
    if include_mtp:
        mtp_depth = 0
        if model_cfg.auxiliary:
            mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)

        if mtp_depth > 0:
            mtp_fwd_cost = ops.op_mtp_head(
                hidden_size=model_cfg.hidden_size,
                vocab_size=model_cfg.vocab_size,
                mtp_depth=mtp_depth,
                batch_tokens=batch * local_seq_len,
                phase=Phase.TRAIN_FWD,
                dtype_bytes=dtype_bytes,
            )
            mtp_fwd = SimOp(
                name=f"{name_prefix}mtp_head_fwd",
                stream="compute",
                duration=ops.roofline_time(mtp_fwd_cost, hw, is_large_gemm=True),
                depends_on=[prev_dep] if prev_dep is not None else [],
                weight_bytes=mtp_fwd_cost.weight_bytes,
                output_bytes=mtp_fwd_cost.output_bytes,
            )
            all_ops.append(mtp_fwd)
            prev_dep = len(all_ops) - 1

    # Apply name prefix
    if name_prefix:
        for op in all_ops:
            op.name = f"{name_prefix}{op.name}"

    return all_ops


# ---------------------------------------------------------------------------
# build_training_step
# ---------------------------------------------------------------------------


def build_training_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    stage_layers: Optional[List[LayerConfig]] = None,
    precision_cfg: Optional[PrecisionConfig] = None,
) -> List[SimOp]:
    """Build SimOps for one full training micro-step (forward + backward + DP sync + optim).

    Parameters
    ----------
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    precision_cfg:
        Precision configuration. None resolves to PrecisionConfig.bf16_default(),
        which reproduces legacy single-dtype behavior.
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp
    dp = parallel_cfg.dp
    dtype_bytes = model_cfg.dtype_bytes
    pc = precision_cfg or PrecisionConfig.from_model_dtype(model_cfg.dtype)

    seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
    batch = rl_cfg.train_micro_batch_size
    cp = parallel_cfg.cp
    local_seq_len = seq_len // cp if cp > 1 else seq_len

    # PP stage layers: use provided or default to stage 0
    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

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
            precision_cfg=precision_cfg,
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
            batch_tokens=batch * local_seq_len,
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
            batch_tokens=batch * local_seq_len,
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
    # DP gradient sync is bucketed per layer and issued as soon as that
    # layer's backward finishes, on the separate "dp_comm" stream — it then
    # overlaps with the backward compute of earlier (still-pending) layers,
    # instead of waiting for the whole backward pass to complete. This
    # mirrors real systems (PyTorch DDP / Megatron gradient bucketing).
    dp_sync_indices: List[int] = []
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
            precision_cfg=precision_cfg,
        )
        if prev_dep is not None and layer_ops:
            layer_ops[0].depends_on = [prev_dep]
        all_ops.extend(layer_ops)
        if layer_ops:
            prev_dep = len(all_ops) - 1

        if dp > 1 and layer_ops:
            layer_param_count = _estimate_param_count(model_cfg, parallel_cfg, [layer_cfg])
            grad_dtype = pc.gradients.dtype
            comm_dtype = pc.comm.dtype
            # Collective message size (potentially reduced to comm dtype)
            comm_bytes_local = layer_param_count * _prec_dtype_bytes(comm_dtype)

            if parallel_cfg.zero_stage < 3:
                # AllReduce (ring)
                dp_cost = ops.op_allreduce(comm_bytes_local, group_size=dp)
            else:
                # ZeRO-3: ReduceScatter
                dp_cost = ops.op_reducescatter(comm_bytes_local, group_size=dp)

            dp_algorithm = "ring_half" if parallel_cfg.zero_stage >= 3 else "ring"
            dp_duration = ops.comm_time(
                dp_cost,
                hw,
                group_size=dp,
                is_intra_node=_is_intra_node(dp, hw),
                algorithm=dp_algorithm,
            )

            # Low-precision comm: inject quantize_grad before collective
            dep_for_dp_sync = prev_dep  # default: last backward op of this layer
            if _compute_class(comm_dtype) != "bf16":
                quant_cost = ops.op_quantize(
                    layer_param_count,
                    grad_dtype,
                    comm_dtype,
                    pc.comm.block_size,
                    pc.comm.scale_bytes,
                )
                quant_op = SimOp(
                    name="quantize_grad",
                    stream="compute",
                    duration=ops.roofline_time(quant_cost, hw),
                    depends_on=[prev_dep] if prev_dep is not None else [],
                    weight_bytes=0,
                    output_bytes=0,
                )
                all_ops.append(quant_op)
                dep_for_dp_sync = len(all_ops) - 1

            dp_sync = SimOp(
                name="dp_grad_sync",
                stream="dp_comm",
                duration=dp_duration,
                depends_on=[dep_for_dp_sync],  # quantize_grad or last bwd op
                weight_bytes=0,
                output_bytes=0,
                comm_bytes=dp_cost.comm_bytes,
                fabric=_fabric(dp, hw),
            )
            all_ops.append(dp_sync)
            dp_sync_idx = len(all_ops) - 1

            # Low-precision comm: inject dequant_grad after collective
            if _compute_class(comm_dtype) != "bf16":
                dequant_cost = ops.op_dequant(
                    layer_param_count,
                    comm_dtype,
                    grad_dtype,
                    pc.comm.block_size,
                    pc.comm.scale_bytes,
                )
                dequant_op = SimOp(
                    name="dequant_grad",
                    stream="compute",
                    duration=ops.roofline_time(dequant_cost, hw),
                    depends_on=[dp_sync_idx],
                    weight_bytes=0,
                    output_bytes=0,
                )
                all_ops.append(dequant_op)
                dp_sync_indices.append(len(all_ops) - 1)  # optimizer waits for dequant
            else:
                dp_sync_indices.append(dp_sync_idx)

    # ------ Optimizer step (placeholder) ---------------------------------------
    # Must wait for the last backward op AND the last (dp_comm-serialized)
    # gradient sync bucket — whichever finishes later.
    param_count = _estimate_param_count(model_cfg, parallel_cfg, stage_layers)
    optim_depends = [prev_dep] if prev_dep is not None else []
    if dp_sync_indices:
        optim_depends.append(dp_sync_indices[-1])
    optim_duration = param_count * 1e-10  # rough placeholder
    optim_op = SimOp(
        name="optimizer_step",
        stream="compute",
        duration=optim_duration,
        depends_on=optim_depends,
        weight_bytes=0,
        output_bytes=0,
        param_count=param_count,
    )
    all_ops.append(optim_op)

    # ------ Error-feedback buffers + compensation add ---------------------------
    # When error_feedback=True and weights are low-precision (not bf16), each
    # param has a persistent EF buffer in ef_dtype to accumulate quantization
    # error across steps. Also inject one compensation_add op per step.
    if pc.error_feedback and _compute_class(pc.weights.dtype) != "bf16":
        ef_buf = SimOp(
            name="ef_buffer",
            stream="compute",
            duration=0.0,
            weight_bytes=param_count * _prec_dtype_bytes(pc.ef_dtype),
            output_bytes=0,
        )
        all_ops.append(ef_buf)

        comp_cost = ops.op_compensation_add(param_count, pc.ef_dtype)
        comp_op = SimOp(
            name="compensation_add",
            stream="compute",
            duration=ops.roofline_time(comp_cost, hw, is_large_gemm=False),
            weight_bytes=0,
            output_bytes=0,
        )
        all_ops.append(comp_op)

    return all_ops


# ---------------------------------------------------------------------------
# build_generation_step
# ---------------------------------------------------------------------------


def build_generation_step(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    stage_layers: Optional[List[LayerConfig]] = None,
) -> Tuple[List[SimOp], List[SimOp]]:
    """Build SimOps for generation (prefill + decode-per-token).

    Returns (prefill_ops, decode_per_token_ops).

    Parameters
    ----------
    stage_layers: Optional per-stage layer list. If None, uses _split_stages()[0].
    """
    all_layers = model_cfg.get_layers()
    _validate_parallelism(all_layers, parallel_cfg)
    pp = parallel_cfg.pp

    if stage_layers is None:
        stage_layers = _split_stages(all_layers, pp)[0]

    batch = rl_cfg.gen_batch_size / parallel_cfg.dp
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