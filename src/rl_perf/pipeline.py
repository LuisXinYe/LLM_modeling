import math

from rl_perf import ops
from rl_perf.builder import (
    build_training_step,
    build_generation_step,
    build_forward_pass,
    build_vision_encoder_step,
    build_vision_encoder_step_fwd,
    _split_stages,
)
from rl_perf.config import Phase
from rl_perf.report import TrainBreakdown
from rl_perf.simulator import simulate


def _cpu_offload_transfer_time(weight_bytes: float, hw) -> float:
    """Time to transfer weights from CPU to GPU over PCIe/HCCS.

    When model weights are offloaded to CPU memory, they must be transferred
    to GPU before each forward pass. The transfer time is:
        t = weight_bytes / (cpu_gpu_bw * 1e9)

    Args:
        weight_bytes: Total weight bytes to transfer (per device, after TP sharding).
        hw: HardwareConfig with cpu_gpu_bw_gb_s field.

    Returns:
        Transfer time in seconds.
    """
    bw_bytes = hw.cpu_gpu_bw_gb_s * 1e9
    return weight_bytes / bw_bytes if bw_bytes > 0 else 0.0


def _compute_pp_p2p_time(model_cfg, hw, parallel_cfg, batch_size, seq_len):
    """PP bubble P2P time for one pipeline step (fwd + bwd).

    In 1F1B, the pipeline bubble at the start of each step contains (pp-1)
    idle micro-batches. Each idle micro-batch requires 2 P2P transfers per
    stage boundary (1 forward activation + 1 backward gradient).
    Total bubble P2Ps = (pp-1) * 2.

    In steady state, P2P transfers overlap with the next micro-batch's
    compute, so only the bubble P2Ps are on the critical path.
    This function returns the total bubble P2P time for the entire step,
    NOT per micro-batch — callers should NOT multiply by micro-batch count.

    Each P2P incurs both bandwidth cost (size / bw) and a latency cost,
    consistent with ops.comm_time's P2P modeling.

    Note: CP shards the sequence across CP ranks, so the activation tensor
    between PP stages is CP-local: local_seq_len = seq_len // cp.
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        return 0.0
    tp = parallel_cfg.tp
    cp = parallel_cfg.cp
    # CP shards the sequence; each PP P2P sends only the local shard
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    # Activation tensor between stages (TP-sharded, each rank sends its shard)
    activation_bytes = batch_size * local_seq_len * model_cfg.hidden_size * model_cfg.dtype_bytes
    activation_bytes_per_tp = activation_bytes // tp
    bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    lat = hw.inter_node_latency_us * 1e-6
    num_p2p = (pp - 1) * 2
    t_p2p = activation_bytes_per_tp / bw_bytes + lat
    return num_p2p * t_p2p


def _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, batch_size, seq_len):
    """PP bubble P2P time for one pipeline step (forward only, no backward).

    Same as _compute_pp_p2p_time but for forward-only passes (e.g. reference,
    reward, old_logprob). Only (pp-1) forward P2Ps are on the critical path.
    This is the total bubble P2P time for the step — callers should NOT
    multiply by micro-batch count.

    Each P2P incurs both bandwidth cost and latency, consistent with
    ops.comm_time's P2P modeling.

    Note: CP shards the sequence across CP ranks, so the activation tensor
    between PP stages is CP-local: local_seq_len = seq_len // cp.
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        return 0.0
    tp = parallel_cfg.tp
    cp = parallel_cfg.cp
    # CP shards the sequence; each PP P2P sends only the local shard
    local_seq_len = seq_len // cp if cp > 1 else seq_len
    activation_bytes = batch_size * local_seq_len * model_cfg.hidden_size * model_cfg.dtype_bytes
    activation_bytes_per_tp = activation_bytes // tp
    bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
    lat = hw.inter_node_latency_us * 1e-6
    num_p2p = pp - 1  # forward only
    t_p2p = activation_bytes_per_tp / bw_bytes + lat
    return num_p2p * t_p2p


def _pp_bubble_time(t_non_bubble, parallel_cfg, num_micro_batches):
    """PP bubble idle time for a pipeline execution.

    In 1F1B, the pipeline warmup/cooldown leaves (pp-1) micro-batches
    of idle time on some stages. Each micro-batch takes
    t_non_bubble / M time (compute + recompute + p2p), so the bubble
    is (pp-1) * (t_non_bubble / M) = t_non_bubble * (pp-1) / M.

    We express this as:
      bubble_ratio = (pp-1) / (M + pp-1)
      t_bubble = t_non_bubble * bubble_ratio
    
    This avoids the circular dependency: t_bubble depends on
    t_non_bubble (which excludes the bubble itself), and
    t_step = t_non_bubble + t_bubble.

    Args:
      t_non_bubble: Total non-bubble time (compute + recompute + p2p).
      parallel_cfg: ParallelismConfig with pp field.
      num_micro_batches: Number of micro-batches in this pipeline execution.

    Returns:
      Bubble idle time in seconds.
     """
    if parallel_cfg.pp <= 1 or num_micro_batches <= 0:
            return 0.0
    M = num_micro_batches
    bubble_ratio = (parallel_cfg.pp - 1) / (M + parallel_cfg.pp - 1)
    return t_non_bubble * bubble_ratio


def _optimizer_offload_time(model_cfg, hw, weight_bytes):
    """CPU offload transfer time for optimizer states.

    ZeRO-Offload: gradients (GPU→CPU, bf16) + updated master weights (CPU→GPU, fp32).
    Total = param_count * (2 + 4) = param_count * 6 bytes.
    weight_bytes is per-device (after TP sharding).
    """
    param_count = weight_bytes / model_cfg.dtype_bytes
    grad_bytes = param_count * 2    # bf16 gradients GPU→CPU
    master_bytes = param_count * 4  # fp32 master weights CPU→GPU
    total_bytes = grad_bytes + master_bytes
    return total_bytes / (hw.cpu_gpu_bw_gb_s * 1e9)


def effective_response_len(
    avg: int, std: int = None, batch_size: int = 1, max_len: int = None
) -> float:
    """Gumbel approximation for expected max of batch_size samples.
    E[max of B samples] ≈ avg + std * sqrt(2 * ln(B))
    Falls back to max_len if std not provided."""
    if std is not None and std > 0 and batch_size > 1:
        return avg + std * math.sqrt(2 * math.log(batch_size))
    if max_len is not None:
        return max_len
    return avg


def _simulate_slowest_stage(build_fn, model_cfg, hw, parallel_cfg, rl_cfg, **kwargs):
    """Simulate all PP stages and return (slowest_time, stage0_sim_result).

    For mixed-layer models (e.g. 4 SwiGLU + 47 MoE), different PP stages
    have different layer compositions and thus different compute times.
    The pipeline time is determined by the slowest stage.

    For multimodal models, the Vision Encoder (ViT) runs on PP stage 0
    before the LLM layers. Its compute time is added to stage 0's LLM
    time to determine the effective stage 0 time, since the pipeline
    bottleneck is the slowest stage.

    Returns (max_wall_clock_time, stage0_sim_result) where stage0_sim_result
    provides weight_bytes and peak_activation_bytes for memory analysis.

    Args:
        build_fn: Builder function (build_training_step, build_forward_pass, etc.)
        model_cfg: ModelConfig.
        hw: HardwareConfig.
        parallel_cfg: ParallelismConfig.
        rl_cfg: RLConfig.
        **kwargs: Additional keyword arguments passed to build_fn.

    Returns:
        Tuple of (slowest_wall_clock_time, stage0_sim_result).
    """
    pp = parallel_cfg.pp
    if pp <= 1:
        ops_list = build_fn(model_cfg, hw, parallel_cfg, rl_cfg, **kwargs)
        sim = simulate(ops_list)
        fwd_only = build_fn is build_forward_pass
        t_vit = _vit_per_microbatch(model_cfg, hw, parallel_cfg, rl_cfg, forward_only=fwd_only)
        return sim.wall_clock_time + t_vit, sim

    all_layers = model_cfg.get_layers()
    stages = _split_stages(all_layers, pp)

    # ViT runs on stage 0; compute its per-micro-batch time once
    fwd_only = build_fn is build_forward_pass
    t_vit = _vit_per_microbatch(model_cfg, hw, parallel_cfg, rl_cfg, forward_only=fwd_only)

    slowest_time = 0.0
    stage0_sim = None

    for i, stage_layers in enumerate(stages):
        ops_list = build_fn(
            model_cfg, hw, parallel_cfg, rl_cfg,
            stage_layers=stage_layers, **kwargs,
        )
        sim = simulate(ops_list)
        stage_time = sim.wall_clock_time
        if i == 0:
            stage0_sim = sim
            # ViT runs on stage 0, so its time adds to stage 0
            stage_time += t_vit
        if stage_time > slowest_time:
            slowest_time = stage_time

    return slowest_time, stage0_sim


def _vit_per_microbatch(model_cfg, hw, parallel_cfg, rl_cfg, forward_only=False):
    """Compute ViT per-micro-batch time. Returns 0 if no vision encoder.

    Args:
        forward_only: If True, compute ViT forward-only time (for reference,
            old_logprob, reward sub-steps). If False, compute ViT fwd+bwd.
    """
    if not model_cfg.vision_encoder:
        return 0.0
    if forward_only:
        vit_ops = build_vision_encoder_step_fwd(
            model_cfg, hw, parallel_cfg, rl_cfg.train_micro_batch_size
        )
    else:
        vit_ops = build_vision_encoder_step(
            model_cfg, hw, parallel_cfg, rl_cfg.train_micro_batch_size
        )
    return simulate(vit_ops).wall_clock_time


def generation_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total generation time in seconds. Returns (total_time, sim_result, t_per_batch).

    GRPO group-aware generation:
      group_size=16 means each prompt is sampled 16 times.
      A gen_batch of 64 samples covers 64/16=4 distinct prompts.
      Prefill is paid once per prompt (4 times), not per response (64 times).
      Decode is paid per response (64 times × eff_len tokens).

    PPO single-sample generation:
      group_size=1, each response is from a distinct prompt.
      Prefill and decode are both per response.

    For multi-stage PP, the slowest stage determines the pipeline time.
    We simulate all stages and use the maximum prefill/decode times.
    """

    # Vision encoder forward (multimodal models)
    t_vit_fwd = 0.0
    if model_cfg.vision_encoder:
        vit_ops = build_vision_encoder_step_fwd(model_cfg, hw, parallel_cfg, rl_cfg.gen_batch_size)
        t_vit_fwd = simulate(vit_ops).wall_clock_time

    prefill_ops, decode_ops = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)
    prefill_sim = simulate(prefill_ops)
    t_prefill = prefill_sim.wall_clock_time + t_vit_fwd
    t_decode_per_token = simulate(decode_ops).wall_clock_time

    eff_len = effective_response_len(
        avg=rl_cfg.avg_response_len,
        std=rl_cfg.std_response_len,
        batch_size=rl_cfg.gen_batch_size,
        max_len=rl_cfg.max_response_len,
    )

    t_step = t_prefill + eff_len * t_decode_per_token

    # Speculative decoding throughput multiplier (spec §4.3)
    if rl_cfg.use_speculative_decoding:
        mtp_depth = (model_cfg.auxiliary or {}).get("mtp_depth", 0)
        if mtp_depth > 0:
            acceptance_len = rl_cfg.mtp_acceptance_len or mtp_depth
            draft_cost = ops.op_mtp_head(
                model_cfg.hidden_size,
                model_cfg.vocab_size,
                mtp_depth,
                batch_tokens=rl_cfg.gen_batch_size,
                phase=Phase.DECODE,
                dtype_bytes=model_cfg.dtype_bytes,
            )
            draft_overhead = (
                ops.roofline_time(draft_cost, hw) / t_decode_per_token
                if t_decode_per_token > 0
                else 0
            )
            throughput_multiplier = acceptance_len / (1 + draft_overhead)
            if throughput_multiplier > 0:
                t_step = (
                    t_prefill
                    + (eff_len / throughput_multiplier)
                    * t_decode_per_token
                )

    return prefill_sim, t_step


def training_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total training time in seconds. Returns (total_time, sim_result, step_breakdown).

    Note: advantage computation (GRPO group normalization, PPO GAE) is O(B×S)
    element-wise ops on CPU/GPU, negligible compared to O(B×S×D²×L) forward
    passes. Not modeled explicitly.

    Each step of the training phase:
      1. reward:        reward model forward (if reward_model)
      2. old_log_prob:  policy forward to compute old log probs
      3. update_actor:  policy fwd+bwd update

    Each step uses the latest model weights. No ppo_epochs inner loop
    (each mini-batch is used exactly once, then discarded).

    Note: advantage computation (GRPO group norm, PPO GAE) is O(B×S)
    element-wise, negligible vs forward passes — not modeled explicitly.
    """

    t_vit = 0.0
    t_pp_p2p_old = 0.0
    t_pp_p2p_reward = 0.0
    # --- 1. Vision encoder forward+backward (multimodal models) ---
    if model_cfg.vision_encoder:
        t_vit = _vit_per_microbatch(model_cfg, hw, parallel_cfg, rl_cfg)

    # --- 2. Reward model forward (if enabled) ---
    t_reward_fwd = 0.0
    if rl_cfg.reward_model:
        t_reward_fwd, _ = _simulate_slowest_stage(
            build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="reward_"
        )
        t_pp_p2p_reward = _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, rl_cfg.train_micro_batch_size, rl_cfg.avg_prompt_len + rl_cfg.avg_response_len)

    # --- 3. old_log_prob forward (GRPO & PPO) ---
    t_old_fwd, _ = _simulate_slowest_stage(
        build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="old_"
    )
    t_pp_p2p_old = _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, rl_cfg.train_micro_batch_size, rl_cfg.avg_prompt_len + rl_cfg.avg_response_len)


    # --- 3. Policy update compute: fwd + bwd ---
    t_policy_update, train_sim = _simulate_slowest_stage(
        build_training_step, model_cfg, hw, parallel_cfg, rl_cfg
    )

    t_pp_p2p_policy = _compute_pp_p2p_time(model_cfg, hw, parallel_cfg, rl_cfg.train_micro_batch_size, rl_cfg.avg_prompt_len + rl_cfg.avg_response_len)
    t_optim_offload_policy = 0.0
    if parallel_cfg.optimizer_offload:
        t_optim_offload_policy = _optimizer_offload_time(model_cfg, hw, train_sim.weight_bytes)

    # --- Per-step total ---
    t_reward_fwd *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_old_fwd *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_policy_update *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_optim_offload_policy *= rl_cfg.train_batch_size / parallel_cfg.dp
    t_vit *= rl_cfg.train_batch_size / parallel_cfg.dp

    t_pp_p2p_total = t_pp_p2p_old + t_pp_p2p_reward + t_pp_p2p_policy

    # Compute total (before bubble and recompute overheads)
    t_compute = (
        t_reward_fwd
        + t_old_fwd
        + t_policy_update
        + t_optim_offload_policy
    )

    # Recomputation overhead
    recompute_penalty = 1.0
    if parallel_cfg.full_recomputation:
        recompute_penalty *= 1.33
    elif parallel_cfg.recompute_attention:
        recompute_penalty *= 1.05
    if parallel_cfg.activation_offload:
        recompute_penalty *= 1.10
    t_recompute = t_compute * (recompute_penalty - 1.0)

    # PP bubble overhead: the per-micro-batch time that determines bubble
    # duration must include compute + recompute + p2p (everything except
    # the bubble itself).  Using only t_compute would underestimate the
    # bubble because each micro-batch also spends time on recomputation
    # and p2p transfers while the pipeline is in steady state.
    t_non_bubble = t_compute + t_recompute + t_pp_p2p_total
    M_train = (rl_cfg.train_batch_size
                / rl_cfg.gradient_accumulation_steps
                / rl_cfg.train_micro_batch_size
                / parallel_cfg.dp)
    t_bubble = _pp_bubble_time(t_non_bubble, parallel_cfg, M_train)
    
    t_step = t_non_bubble + t_bubble

    step_bd = TrainBreakdown(
        vit=t_vit,
        reward_fwd=t_reward_fwd,
        old_logprob_fwd=t_old_fwd,
        policy_update=t_policy_update,
        pp_p2p=t_pp_p2p_total,
        pp_bubble=t_bubble,
        recompute=t_recompute,
        optim_offload=t_optim_offload_policy,
        total=t_step,
    )

    return t_step, train_sim, step_bd


def ref_time(model_cfg, hw, parallel_cfg, rl_cfg):
    """Total reference time in seconds. Returns (total_time, sim_result).

    ViT time is included in _simulate_slowest_stage (added to stage 0),
    since the reference forward also runs ViT on PP stage 0.
    """

    # --- 1. Reference model forward (if enabled) ---
    t_ref_fwd = 0.0
    ref_sim = None
    t_pp_p2p_ref = 0.0
    t_ref_offload = 0.0
    if rl_cfg.reference_model:
        t_ref_fwd, ref_sim = _simulate_slowest_stage(
            build_forward_pass, model_cfg, hw, parallel_cfg, rl_cfg, name_prefix="ref_"
        )
        t_pp_p2p_ref = _compute_pp_p2p_time_fwd(model_cfg, hw, parallel_cfg, rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp, rl_cfg.avg_prompt_len + rl_cfg.avg_response_len)
        # CPU offload: add weight transfer time (CPU → GPU before forward)
        if rl_cfg.ref_offload_cpu:
            t_ref_offload = _cpu_offload_transfer_time(ref_sim.weight_bytes, hw)

    t_ref_fwd *= rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp

    # PP bubble: ref forward runs N micro-batches through the pipeline,
    # incurring one bubble per pipeline execution.
    M_ref = rl_cfg.train_batch_size * rl_cfg.group_size / parallel_cfg.dp
    t_ref_non_bubble = t_ref_fwd + t_pp_p2p_ref + t_ref_offload
    t_ref_bubble = _pp_bubble_time(t_ref_non_bubble, parallel_cfg, M_ref)

    t_ref = t_ref_non_bubble + t_ref_bubble

    return t_ref, ref_sim


def step_time(
    t_gen: float,
    t_train: float,
    t_ref: float,
    startup_overhead: float = 0,
    colocated: bool = False,
    t_reshard: float = 0,
) -> float:
    """Compute wall-clock time.

    Args:
        t_gen: Generation phase time.
        t_train: Training phase time.
        t_ref: Reference phase time.
        startup_overhead: Startup overhead (non-colocated only).
        colocated: If True, phases run sequentially on same devices.
        t_reshard: Resharding time between phases (colocated only).
    """
    if colocated:
        return t_gen + t_ref + t_train + t_reshard
    return max(t_gen, t_train, t_ref) + startup_overhead


def _reshard_time(
    model_cfg,
    hw,
    src_parallel,
    dst_parallel,
) -> float:
    """Estimate time to reshard model weights between two parallelism configs.

    When the parallelism strategy changes between phases (e.g. gen→train,
    train→ref), model weights must be redistributed across devices.
    This involves collective communication (AllGather + ReduceScatter or
    AllToAll) proportional to the total model weight volume.

    Model:
      - The full model weights must be reassembled from the source layout
        and redistributed into the destination layout.
      - Each device holds weight_bytes / (tp_src * pp_src * ep_src) in the
        source layout and needs weight_bytes / (tp_dst * pp_dst * ep_dst)
        in the destination layout.
      - Resharding is modeled as an AllToAll over the device group that
        spans both layouts, with total communication volume ≈ total_weight.
      - For PP changes, each device needs to load/unload different layer
        sets, but the total data movement is still bounded by total_weight.

    Args:
        model_cfg: ModelConfig.
        hw: HardwareConfig.
        src_parallel: Source ParallelismConfig.
        dst_parallel: Destination ParallelismConfig.

    Returns:
        Estimated resharding time in seconds.
    """
    # If parallelism is identical, no resharding needed
    if (src_parallel.tp == dst_parallel.tp
            and src_parallel.pp == dst_parallel.pp
            and src_parallel.ep == dst_parallel.ep
            and src_parallel.dp == dst_parallel.dp):
        return 0.0

    # Estimate total model weight bytes (before any sharding)
    all_layers = model_cfg.get_layers()
    total_weight_bytes = 0.0
    dtype_bytes = model_cfg.dtype_bytes
    d = model_cfg.hidden_size

    for layer_cfg in all_layers:
        # Attention weights
        if layer_cfg.attention in ("GQA", "MHA", "SWA"):
            q_params = d * layer_cfg.num_heads * layer_cfg.head_dim
            kv_params = 2 * d * layer_cfg.num_kv_heads * layer_cfg.head_dim
            o_params = layer_cfg.num_heads * layer_cfg.head_dim * d
            total_weight_bytes += (q_params + kv_params + o_params) * dtype_bytes
        elif layer_cfg.attention == "MLA":
            total_weight_bytes += (
                d * layer_cfg.query_compression_dim
                + layer_cfg.query_compression_dim * d
                + d * layer_cfg.kv_compression_dim
                + layer_cfg.kv_compression_dim * d
                + layer_cfg.kv_compression_dim * d
                + d * d
            ) * dtype_bytes

        # FFN weights
        if layer_cfg.ffn == "SwiGLU":
            total_weight_bytes += 3 * d * layer_cfg.intermediate_size * dtype_bytes
        elif layer_cfg.ffn == "MoE":
            expert_int = layer_cfg.expert_intermediate_size or layer_cfg.intermediate_size
            total_weight_bytes += (
                layer_cfg.num_experts
                * 3
                * d
                * expert_int
                * dtype_bytes
            )
            if layer_cfg.num_shared_experts > 0:
                shared_int = (
                    layer_cfg.shared_intermediate_size or layer_cfg.intermediate_size
                )
                total_weight_bytes += (
                    layer_cfg.num_shared_experts
                    * 3 * d * shared_int * dtype_bytes
                )

        # RMSNorm
        total_weight_bytes += 2 * d * dtype_bytes

    # Add embedding + LM head
    total_weight_bytes += 2 * model_cfg.vocab_size * d * dtype_bytes

    # Add MTP head weight if present
    if model_cfg.auxiliary:
        mtp_depth = model_cfg.auxiliary.get("mtp_depth", 0)
        if mtp_depth > 0:
            total_weight_bytes += mtp_depth * d * model_cfg.vocab_size * dtype_bytes

    # Communication volume: each device sends/receives its shard.
    # Total data movement ≈ total_weight_bytes (each byte moves once).
    # Use the number of devices involved in the resharding group.
    # In colocated mode, the resharding group is the union of src and dst
    # device groups. Since they share the same physical devices, we use
    # the max of the two.
    n_devices = max(src_parallel.total_devices, dst_parallel.total_devices)

    # Determine if communication is intra-node or inter-node
    is_intra = n_devices <= hw.devices_per_node

    if is_intra:
        bw_bytes = hw.intra_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
        lat = hw.intra_node_latency_us * 1e-6
    else:
        bw_bytes = hw.inter_node_bw_gb_s * 1e9 * hw.calibration.comm_efficiency
        lat = hw.inter_node_latency_us * 1e-6

    # Model as AllToAll: each device sends total_weight/N to every other device.
    # AllToAll bandwidth cost: per-device volume / bandwidth.
    # Per-device communication volume ≈ total_weight * (N-1)/N ≈ total_weight.
    # Add latency for the AllToAll operation.
    comm_bytes = total_weight_bytes * (n_devices - 1) / n_devices
    t_reshard = comm_bytes / bw_bytes + lat

    return t_reshard
