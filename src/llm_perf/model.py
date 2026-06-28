from llm_perf.config import (
    HardwareConfig,
    ModelConfig,
    ParallelismConfig,
    WorkloadConfig,
)
from llm_perf.post_training import (
    step_time,
    generation_time,
    pretraining_time,
    training_time,
    ref_time,
    _reshard_time,
)
from llm_perf.builder import _split_stages
from llm_perf.inference import effective_response_len, prefill_decode_times
from llm_perf.report import MemoryProfile, TargetReport


class LLMPerformanceModel:
    """Top-level facade for LLM performance estimation (inference / training / post-training).

    Combines pipeline timing (generation + training), memory profiling,
    and feasibility checking into a single interface.
    """

    def __init__(self, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.model = model_cfg
        self.hw = hw_cfg

    def derive_targets(
        self,
        total_devices,
        rl_cfg,
        gen_parallel,
        train_parallel,
        ref_parallel,
    ):
        """Derive throughput targets and feasibility for one RL step.

        Args:
            total_devices: Total number of accelerator devices available.
            rl_cfg: WorkloadConfig describing the workload.
            gen_parallel: ParallelismConfig for the generation phase.
            train_parallel: ParallelismConfig for the training phase.
            ref_parallel: ParallelismConfig for the reference phase.

        Returns:
            TargetReport with step time, TPS targets, memory profile,
            and feasibility verdict.
        """
        # Validate device layout consistency
        for label, par in [
            ("gen_parallel", gen_parallel),
            ("ref_parallel", ref_parallel),
            ("train_parallel", train_parallel),
        ]:
            if par.total_devices > total_devices:
                raise ValueError(
                    f"{label} requires {par.total_devices} devices "
                    f"but only {total_devices} available."
                )

        # Compute generation and training times
        gen_sim, t_gen = generation_time(self.model, self.hw, gen_parallel, rl_cfg)
        t_train, train_sim, step_bd = training_time(
            self.model, self.hw, train_parallel, rl_cfg
        )
        t_ref, ref_sim = ref_time(self.model, self.hw, ref_parallel, rl_cfg)

        # Resharding time: phases run serially on the shared device pool
        # (gen → ref → train); when the parallelism layout changes between
        # phases, model weights must be redistributed across devices.
        t_reshard_gen_ref = _reshard_time(
            self.model, self.hw, gen_parallel, ref_parallel
        )
        t_reshard_ref_train = _reshard_time(
            self.model, self.hw, ref_parallel, train_parallel
        )
        t_reshard = t_reshard_gen_ref + t_reshard_ref_train

        t_step = step_time(t_gen, t_train, t_ref, t_reshard=t_reshard)

        # Compute TPS targets (single-rank perspective)
        # Each rank processes local_seq_len = seq_len / cp tokens due to CP.
        avg_tokens = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
        train_local_tokens = (
            avg_tokens / train_parallel.cp if train_parallel.cp > 1 else avg_tokens
        )
        ref_local_tokens = (
            avg_tokens / ref_parallel.cp if ref_parallel.cp > 1 else avg_tokens
        )

        gen_tps = (
            rl_cfg.gen_batch_size * rl_cfg.avg_response_len / t_gen if t_gen > 0 else 0
        )
        train_tps = (
            rl_cfg.train_batch_size * train_local_tokens / t_train if t_train > 0 else 0
        )
        ref_tps = (
            rl_cfg.train_batch_size
            * rl_cfg.group_size
            / ref_parallel.dp
            * ref_local_tokens
            / t_ref
            if t_ref > 0
            else 0
        )
        gen_sps = rl_cfg.gen_batch_size / t_gen if t_gen > 0 else 0
        train_sps = rl_cfg.train_batch_size / t_train if t_train > 0 else 0
        ref_sps = (
            rl_cfg.train_batch_size * rl_cfg.gen_batch_size / ref_parallel.dp / t_ref
            if t_ref > 0
            else 0
        )

        # Memory profile
        memory = self._compute_memory_profile(
            train_sim,
            gen_sim,
            ref_sim,
            train_parallel,
            gen_parallel,
            ref_parallel,
            rl_cfg,
        )

        feasible = memory.train_feasible and memory.gen_feasible and memory.ref_feasible

        return TargetReport(
            step_time_seconds=t_step,
            gen_tps_target=gen_tps,
            train_tps_target=train_tps,
            ref_tps_target=ref_tps,
            gen_samples_per_sec=gen_sps,
            train_samples_per_sec=train_sps,
            ref_samples_per_sec=ref_sps,
            gen_time_seconds=t_gen,
            train_time_seconds=t_train,
            ref_time_seconds=t_ref,
            reshard_gen_ref_seconds=t_reshard_gen_ref,
            reshard_ref_train_seconds=t_reshard_ref_train,
            train_breakdown=step_bd,
            memory=memory,
            gen_parallel=gen_parallel,
            train_parallel=train_parallel,
            ref_parallel=ref_parallel,
            feasible=feasible,
            exposed_comm_by_fabric=dict(train_sim.exposed_comm_by_fabric),
            quant_overhead_seconds=train_sim.quant_overhead_seconds,
            compute_seconds_by_class=dict(train_sim.compute_seconds_by_class),
        )

    def feasibility_check(
        self,
        total_devices,
        rl_cfg,
        gen_parallel,
        train_parallel,
        ref_parallel=None,
    ):
        """Convenience alias for derive_targets; returns the same TargetReport."""
        if ref_parallel is None:
            ref_parallel = train_parallel
        return self.derive_targets(
            total_devices,
            rl_cfg,
            gen_parallel,
            train_parallel,
            ref_parallel,
        )

    def _mtp_weight_gb(self) -> float:
        """Extra weight (GB) for an MTP / speculative-decoding head, if present."""
        if not self.model.auxiliary:
            return 0.0
        mtp_depth = self.model.auxiliary.get("mtp_depth", 0)
        if mtp_depth <= 0:
            return 0.0
        return (
            mtp_depth
            * self.model.hidden_size
            * self.model.vocab_size
            * self.model.dtype_bytes
            / 1e9
        )

    def _train_state_gb(self, train_sim, train_parallel):
        """Resident per-device (weight, gradient, optimizer) memory in GB.

        Mixed-precision Adam holds, per parameter:
          - weights in model dtype (e.g. bf16 = 2 B),
          - a gradient buffer in model dtype (2 B),
          - optimizer state = 12 B (fp32 master + momentum + variance).

        ZeRO shards these across the DP group:
          stage 1 → optimizer, stage 2 → +gradients, stage 3 → +parameters.
        CPU offload removes a component from resident HBM entirely.
        """
        p = train_parallel
        # param_count comes from the architecture (set by the builder on the
        # optimizer op), NOT inferred from weight_bytes — which is now both
        # precision-scaled and EF-excluded. Fall back to the old inference for
        # SimResults that predate the field (param_count == 0).
        param_count = train_sim.param_count or (
            train_sim.weight_bytes / self.model.dtype_bytes
        )
        # Resident low-precision weight copy (sized by weights.dtype in builder).
        weight_bytes = float(train_sim.weight_bytes)
        # Error-feedback buffer: resident weight-like state, reported separately
        # so it never feeds param_count / optimizer sizing.
        ef_bytes = float(getattr(train_sim, "ef_buffer_bytes", 0.0))
        grad_bytes = param_count * self.model.dtype_bytes  # gradients in model dtype
        optim_bytes = param_count * 12  # fp32 master + momentum + variance

        if p.zero_stage >= 1:
            optim_bytes /= p.dp
        if p.zero_stage >= 2:
            grad_bytes /= p.dp
        if p.zero_stage >= 3:
            weight_bytes /= p.dp
            ef_bytes /= p.dp

        if p.param_offload:
            weight_bytes = 0.0
        if p.grad_offload:
            grad_bytes = 0.0
        if p.optimizer_offload:
            optim_bytes = 0.0

        # Fold EF in after param-count-derived terms are fixed.
        weight_bytes += ef_bytes
        return weight_bytes / 1e9, grad_bytes / 1e9, optim_bytes / 1e9

    def _train_activation_gb(self, train_parallel, rl_cfg) -> float:
        """Per-device activation memory (GB) retained from forward to backward.

        The simulator's peak_activation only captures the instantaneous working
        set, because backward ops do not depend on their forward activations —
        so forward activations are freed during the forward pass instead of
        being held until backward. For training (fwd+bwd) the real footprint is
        the full retained forward stack: every layer's intermediate activations
        live until that layer's backward.

        We estimate it per layer as the sum of that layer's forward compute-op
        output_bytes (already TP/CP/SP-sharded and attention/FFN-variant aware),
        then apply:
          - full_recomputation: keep only the layer input (b·s·h), recompute
            the rest in backward;
          - recompute_attention: drop the attention block's activations;
          - PP 1F1B: stage 0 holds up to `pp` in-flight micro-batches;
          - activation_offload: stream all but ~one layer to CPU.

        (Reference/generation phases are forward-only, so their instantaneous
        simulator peak is already correct — this adjustment is training-only.)
        """
        from llm_perf.builder import build_layer_ops
        from llm_perf.config import Phase

        p = train_parallel
        stage_layers = _split_stages(self.model.get_layers(), p.pp)[0]
        batch = rl_cfg.train_micro_batch_size
        seq_len = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
        local_seq = seq_len // p.cp if p.cp > 1 else seq_len
        d = self.model.hidden_size

        total_bytes = 0.0
        for layer in stage_layers:
            if p.full_recomputation:
                # Only the layer input checkpoint is kept; rest recomputed.
                keep = batch * local_seq * d * self.model.dtype_bytes
            else:
                fwd_ops = build_layer_ops(
                    layer_cfg=layer,
                    model_cfg=self.model,
                    parallel_cfg=p,
                    hw=self.hw,
                    batch=batch,
                    seq_len=seq_len,
                    phase=Phase.TRAIN_FWD,
                )
                keep = sum(op.output_bytes for op in fwd_ops if op.stream == "compute")
                if p.recompute_attention:
                    keep -= sum(
                        op.output_bytes
                        for op in fwd_ops
                        if op.stream == "compute" and "attention" in op.name
                    )
            total_bytes += keep

        # PP activation scaling by schedule (GPipe and 1F1B have the SAME bubble
        # time but very different activation memory):
        #   - GPipe: all M micro-batches are forwarded before any backward, so
        #     stage 0 holds all M micro-batches' activations → peak ∝ M.
        #   - 1F1B / zero-bubble / interleaved: forward and backward are
        #     interleaved, so stage 0 only holds the warmup depth ≈ min(pp, M).
        if p.pp > 1:
            micro_batches = max(
                1,
                rl_cfg.train_batch_size
                / max(1, rl_cfg.gradient_accumulation_steps)
                / max(1, rl_cfg.train_micro_batch_size)
                / max(1, p.dp),
            )
            if p.pp_schedule == "gpipe":
                factor = micro_batches
            else:  # 1f1b, zero_bubble, interleaved
                factor = min(p.pp, micro_batches)
            total_bytes *= factor

        # Activation offload streams all but roughly one layer to CPU.
        if p.activation_offload and stage_layers:
            total_bytes /= len(stage_layers)

        return total_bytes / 1e9

    def _kv_cache_gb(self, gen_parallel, rl_cfg) -> float:
        """Per-device KV cache size (GB) under the generation parallelism.

        Iterates the layers of the first PP stage. Each device holds
        gen_batch_size / dp sequences of length avg_prompt + max_response.
        MLA stores a compressed latent; SWA/DSA bound the cache by the
        attention window.
        """
        all_layers = self.model.get_layers()
        stage_layers = _split_stages(all_layers, gen_parallel.pp)[0]
        kv_total = 0
        gen_batch_per_device = max(
            1, -(-rl_cfg.gen_batch_size // gen_parallel.dp)
        )  # ceil division
        max_kv_seq = rl_cfg.avg_prompt_len + rl_cfg.max_response_len
        for layer in stage_layers:
            if layer.attention == "MLA":
                kv_per_token = (
                    layer.kv_compression_dim + layer.rope_dim
                ) * self.model.dtype_bytes
            elif layer.attention == "DSA":
                # SWA KV cache (all DSA layers have SWA, bounded by window_size)
                swa_kv_heads_per_device = (
                    layer.num_kv_heads
                )  # MQA: replicated, not TP-split
                swa_kv_per_token = (
                    2
                    * swa_kv_heads_per_device
                    * layer.head_dim
                    * self.model.dtype_bytes
                )
                kv_total += (
                    swa_kv_per_token
                    * gen_batch_per_device
                    * min(max_kv_seq, layer.window_size)
                )
                # Decoupled RoPE key (MLA-style): uncompressed, kept for the full
                # context length, single shared head.
                if layer.rope_dim > 0:
                    rope_kv_per_token = layer.rope_dim * self.model.dtype_bytes
                    kv_total += rope_kv_per_token * gen_batch_per_device * max_kv_seq
                # Compressed KV cache (only for ratio > 1)
                if layer.compress_ratio > 1:
                    comp_seq = max_kv_seq // layer.compress_ratio
                    comp_kv_per_token = layer.compress_c_kv * self.model.dtype_bytes
                    kv_total += comp_kv_per_token * gen_batch_per_device * comp_seq
                    # Index KV cache (C4A only)
                    if layer.compress_ratio == 4 and layer.index_head_dim > 0:
                        idx_kv_per_token = layer.index_head_dim * self.model.dtype_bytes
                        kv_total += idx_kv_per_token * gen_batch_per_device * comp_seq
                continue
            elif layer.attention == "SWA" and layer.window_size > 0:
                kv_heads_per_device = layer.num_kv_heads // gen_parallel.tp
                kv_per_token = (
                    2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
                )
                # SWA KV cache bounded by window_size
                kv_total += (
                    kv_per_token
                    * gen_batch_per_device
                    * min(max_kv_seq, layer.window_size)
                )
                continue
            else:
                kv_heads_per_device = layer.num_kv_heads // gen_parallel.tp
                kv_per_token = (
                    2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
                )
            kv_total += kv_per_token * gen_batch_per_device * max_kv_seq
        return kv_total / 1e9

    def derive_inference(self, total_devices, rl_cfg, gen_parallel):
        """Inference-only modeling: prefill + decode generation, no training.

        Returns a dict with generation throughput, prefill/decode timing,
        and the generation memory footprint (weights + KV cache).
        """
        if gen_parallel.total_devices > total_devices:
            raise ValueError(
                f"gen_parallel requires {gen_parallel.total_devices} devices "
                f"but only {total_devices} available."
            )

        gen_sim, t_gen = generation_time(self.model, self.hw, gen_parallel, rl_cfg)

        # Prefill / decode split for visualization (slowest-stage prefill time;
        # see inference.prefill_decode_times()).
        t_prefill, _, _ = prefill_decode_times(
            self.model, self.hw, gen_parallel, rl_cfg
        )
        t_decode = max(t_gen - t_prefill, 0.0)
        eff_len = effective_response_len(
            avg=rl_cfg.avg_response_len,
            std=rl_cfg.std_response_len,
            batch_size=rl_cfg.gen_batch_size,
            max_len=rl_cfg.max_response_len,
        )

        gen_weight_gb = gen_sim.weight_bytes / 1e9 + self._mtp_weight_gb()
        kv_cache_gb = self._kv_cache_gb(gen_parallel, rl_cfg)
        total_gen_gb = gen_weight_gb + kv_cache_gb
        usable = self.hw.usable_hbm_gb

        gen_tps = (
            rl_cfg.gen_batch_size * rl_cfg.avg_response_len / t_gen if t_gen > 0 else 0
        )
        gen_sps = rl_cfg.gen_batch_size / t_gen if t_gen > 0 else 0
        # Decode token latency (per output token, across the batch).
        decode_ms_per_token = (t_decode / eff_len * 1000) if eff_len > 0 else 0

        return {
            "gen_time_seconds": t_gen,
            "prefill_seconds": t_prefill,
            "decode_seconds": t_decode,
            "eff_response_len": eff_len,
            "decode_ms_per_token": decode_ms_per_token,
            "gen_tps_target": gen_tps,
            "gen_samples_per_sec": gen_sps,
            "gen_weight_gb": gen_weight_gb,
            "kv_cache_gb": kv_cache_gb,
            "total_gen_gb": total_gen_gb,
            "usable_hbm_gb": usable,
            "gen_feasible": total_gen_gb < usable,
        }

    def derive_pretraining(
        self, total_devices, rl_cfg, train_parallel, precision_cfg=None
    ):
        """Pretraining-only modeling: one fwd+bwd+optimizer step, no RL.

        Returns a dict with step time, throughput, the training-step
        breakdown, and the training memory footprint.

        Args:
            precision_cfg: Optional PrecisionConfig. When high_precision_period > 0,
                the step time is a blended estimate across low/high-precision steps.
        """
        if train_parallel.total_devices > total_devices:
            raise ValueError(
                f"train_parallel requires {train_parallel.total_devices} devices "
                f"but only {total_devices} available."
            )

        t_step, train_sim, bd = pretraining_time(
            self.model, self.hw, train_parallel, rl_cfg, precision_cfg=precision_cfg
        )

        weight_gb, grad_gb, optimizer_gb = self._train_state_gb(
            train_sim, train_parallel
        )
        activation_peak_gb = self._train_activation_gb(train_parallel, rl_cfg)
        total_train_gb = weight_gb + grad_gb + optimizer_gb + activation_peak_gb
        usable = self.hw.usable_hbm_gb

        avg_tokens = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
        train_local_tokens = (
            avg_tokens / train_parallel.cp if train_parallel.cp > 1 else avg_tokens
        )
        train_tps = (
            rl_cfg.train_batch_size * train_local_tokens / t_step if t_step > 0 else 0
        )
        train_sps = rl_cfg.train_batch_size / t_step if t_step > 0 else 0

        return {
            "step_time_seconds": t_step,
            "train_tps_target": train_tps,
            "train_samples_per_sec": train_sps,
            "breakdown": {
                "policy_update": bd.policy_update,
                "recompute": bd.recompute,
                "pp_bubble": bd.pp_bubble,
                "optim_offload": bd.optim_offload,
                "total": bd.total,
            },
            "weight_gb": weight_gb,
            "grad_gb": grad_gb,
            "optimizer_gb": optimizer_gb,
            "activation_peak_gb": activation_peak_gb,
            "total_train_gb": total_train_gb,
            "usable_hbm_gb": usable,
            "train_feasible": total_train_gb < usable,
        }

    def _compute_memory_profile(
        self,
        train_sim,
        gen_sim,
        ref_sim,
        train_parallel,
        gen_parallel,
        ref_parallel,
        rl_cfg,
    ):
        """Compute per-device memory breakdown for training, generation, and reference.

        Combines SimResult-derived values (weights, activations) with analytical
        estimates for optimizer states, KV cache, and reference model.

        Args:
            train_sim: SimResult from the training phase simulation.
            gen_sim: SimResult from the generation phase simulation.
            ref_sim: SimResult from the reference phase simulation.
            train_parallel: ParallelismConfig for training.
            gen_parallel: ParallelismConfig for generation.
            ref_parallel: ParallelismConfig for reference.
            rl_cfg: WorkloadConfig workload specification.

        Returns:
            MemoryProfile with per-component memory in GB and feasibility flags.
        """

        # From SimResult (ephemeral memory)
        gen_weight_gb = gen_sim.weight_bytes / 1e9
        # Full (unsharded-by-ZeRO) per-device weight, used for reward model.
        train_weight_full_gb = train_sim.weight_bytes / 1e9

        # Resident weight / gradient / optimizer (ZeRO + offload aware).
        train_weight_gb, grad_gb, optimizer_gb = self._train_state_gb(
            train_sim, train_parallel
        )
        # Retained forward activation stack (recompute / offload / PP aware).
        activation_peak_gb = self._train_activation_gb(train_parallel, rl_cfg)

        # KV cache for generation (per-device), see _kv_cache_gb.
        kv_cache_gb = self._kv_cache_gb(gen_parallel, rl_cfg)

        # Reference model
        ref_weight_gb = ref_sim.weight_bytes / 1e9 if ref_sim else 0
        ref_offload = rl_cfg.ref_offload_cpu or ref_parallel.param_offload
        ref_gb = ref_weight_gb if (rl_cfg.reference_model and not ref_offload) else 0
        ref_activation_peak_gb = ref_sim.peak_activation_bytes / 1e9 if ref_sim else 0

        # Reward model (same architecture as policy, forward-only, no optimizer).
        # Uses the full per-device weight (not ZeRO-sharded).
        reward_model_gb = train_weight_full_gb if rl_cfg.reward_model else 0

        # Totals
        total_train = (
            train_weight_gb
            + grad_gb
            + optimizer_gb
            + activation_peak_gb
            + ref_gb
            + reward_model_gb
        )

        # Generation weight: gen_sim may not include MTP head weights
        # (build_generation_step doesn't build MTP ops). Add them if present.
        gen_weight_total = gen_weight_gb + self._mtp_weight_gb()

        # KV cache only exists during the gen sub-step within each step, then freed.
        # Training peak memory does NOT coexist with full KV cache.
        # We still report total_gen for the gen sub-step feasibility check.
        total_gen = gen_weight_total + kv_cache_gb

        # Reference phase: ref model weights + ref forward activation peak.
        # Ref phase is forward-only, no optimizer or KV cache.
        # When ref_offload_cpu=True, weights are on CPU and stream to GPU
        # layer-by-layer during forward. Peak GPU memory is only the
        # activation peak (which includes per-layer working set), not the
        # full model weight. When not offloaded, all weights are resident.
        if ref_offload:
            total_ref = ref_activation_peak_gb
        else:
            total_ref = ref_weight_gb + ref_activation_peak_gb
        usable = self.hw.usable_hbm_gb

        return MemoryProfile(
            weight_gb=train_weight_gb,
            grad_gb=grad_gb,
            gen_weight_gb=gen_weight_total,
            ref_weight_gb=ref_weight_gb if ref_sim else 0,
            optimizer_gb=optimizer_gb,
            activation_peak_gb=activation_peak_gb,
            kv_cache_gb=kv_cache_gb,
            ref_model_gb=ref_gb,
            ref_activation_peak_gb=ref_activation_peak_gb,
            reward_model_gb=reward_model_gb,
            total_train_gb=total_train,
            total_gen_gb=total_gen,
            total_ref_gb=total_ref,
            usable_hbm_gb=usable,
            train_feasible=total_train < usable,
            gen_feasible=total_gen < usable,
            ref_feasible=total_ref < usable,
        )

    def what_if(
        self,
        base_config,
        overrides,
        total_devices,
        gen_parallel,
        train_parallel,
        ref_parallel=None,
    ):
        """Run a what-if scenario by merging overrides into a base WorkloadConfig.

        Args:
            base_config: Dict of base WorkloadConfig field values.
            overrides: Dict of fields to override (e.g. {"group_size": 16}).
            total_devices: Total number of accelerator devices.
            gen_parallel: ParallelismConfig for generation.
            train_parallel: ParallelismConfig for training.
            ref_parallel: ParallelismConfig for reference. Defaults to train_parallel.

        Returns:
            TargetReport for the modified configuration.
        """
        if ref_parallel is None:
            ref_parallel = train_parallel
        rl_cfg = WorkloadConfig(**{**base_config, **overrides})
        return self.derive_targets(
            total_devices, rl_cfg, gen_parallel, train_parallel, ref_parallel
        )

    def sensitivity(
        self,
        rl_cfg,
        param_name,
        values,
        total_devices,
        gen_parallel,
        train_parallel,
        ref_parallel=None,
    ):
        """Sweep a single WorkloadConfig parameter across multiple values.

        Args:
            rl_cfg: Base WorkloadConfig instance.
            param_name: Name of the WorkloadConfig field to sweep (e.g. "group_size").
            values: Iterable of values to try for the parameter.
            total_devices: Total number of accelerator devices.
            gen_parallel: ParallelismConfig for generation.
            train_parallel: ParallelismConfig for training.
            ref_parallel: ParallelismConfig for reference. Defaults to train_parallel.

        Returns:
            List[TargetReport], one per value in the sweep.

        Raises:
            ValueError: If param_name is not a valid WorkloadConfig field.
        """
        if ref_parallel is None:
            ref_parallel = train_parallel
        if param_name not in WorkloadConfig.model_fields:
            raise ValueError(f"Unknown WorkloadConfig field: {param_name}")
        results = []
        for v in values:
            cfg = rl_cfg.model_copy(update={param_name: v})
            results.append(
                self.derive_targets(
                    total_devices, cfg, gen_parallel, train_parallel, ref_parallel
                )
            )
        return results


def compare_precision(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    recipes: dict,
) -> list:
    """Compare multiple precision recipes on the same model/hardware/parallel config.

    For each named recipe, runs the training-step simulation and returns a list of
    per-recipe dicts with performance and communication metrics.

    Args:
        model_cfg: ModelConfig.
        hw: HardwareConfig (must have peak_tflops for fp8/fp4 if those are used).
        parallel_cfg: ParallelismConfig for the training phase.
        rl_cfg: WorkloadConfig providing training workload parameters.
        recipes: Dict mapping recipe name → PrecisionConfig.

    Returns:
        List of dicts, one per recipe, each with:
            name, step_seconds, speedup_vs_bf16, comm_bytes, comm_reduction_pct,
            exposed_comm_by_fabric, peak_memory_gb, feasible.

    The baseline for speedup and comm_reduction_pct is the recipe named "bf16"
    if present, otherwise the first recipe in the dict.
    """
    # Determine baseline recipe key
    baseline_key = "bf16" if "bf16" in recipes else next(iter(recipes))

    # Shared performance model instance for memory accounting (reuses the exact
    # ZeRO-sharding / offload / activation-recompute path from derive_pretraining).
    perf_model = LLMPerformanceModel(model_cfg, hw)

    # Run simulation for each recipe
    computed = {}
    for name, precision_cfg in recipes.items():
        t_step, train_sim, _bd = pretraining_time(
            model_cfg, hw, parallel_cfg, rl_cfg, precision_cfg=precision_cfg
        )
        # Exact per-device memory: ZeRO-stage-aware weight/grad/optimizer sharding
        # + retained activation stack (mirrors derive_pretraining lines 401-405).
        weight_gb, grad_gb, optimizer_gb = perf_model._train_state_gb(
            train_sim, parallel_cfg
        )
        activation_peak_gb = perf_model._train_activation_gb(parallel_cfg, rl_cfg)
        peak_memory_gb = weight_gb + grad_gb + optimizer_gb + activation_peak_gb
        computed[name] = {
            "t_step": t_step,
            "train_sim": train_sim,
            "peak_memory_gb": peak_memory_gb,
        }

    t_baseline = computed[baseline_key]["t_step"]
    comm_baseline = computed[baseline_key]["train_sim"].total_comm_bytes

    rows = []
    for name in recipes:  # preserve insertion order
        data = computed[name]
        t_step = data["t_step"]
        train_sim = data["train_sim"]
        peak_memory_gb = data["peak_memory_gb"]
        comm_bytes = train_sim.total_comm_bytes

        speedup = t_baseline / t_step if t_step > 0 else 1.0
        comm_reduction_pct = (
            (1.0 - comm_bytes / comm_baseline) * 100.0 if comm_baseline > 0 else 0.0
        )

        rows.append(
            {
                "name": name,
                "step_seconds": t_step,
                "speedup_vs_bf16": speedup,
                "comm_bytes": comm_bytes,
                "comm_reduction_pct": comm_reduction_pct,
                "exposed_comm_by_fabric": dict(train_sim.exposed_comm_by_fabric),
                "peak_memory_gb": peak_memory_gb,
                "feasible": peak_memory_gb <= hw.usable_hbm_gb,
            }
        )

    return rows


def sweep_sparse_ratio(
    model_cfg: ModelConfig,
    hw: HardwareConfig,
    base_parallel_cfg: ParallelismConfig,
    rl_cfg: WorkloadConfig,
    grid: list,
) -> list:
    """Sweep large-sparse-ratio MoE knobs and report per-fabric network demand.

    Each `grid` entry is a dict overriding any of: num_experts, top_k, ep,
    moe_node_limit, moe_imbalance_factor. For every point the training step is
    simulated and the per-fabric exposed communication is reported — the primary
    signal for cross-node interconnect sizing.

    Returns one dict per point with: point, step_seconds, exposed_comm_by_fabric,
    cross_node_gb, peak_memory_gb, feasible.
    """
    rows = []
    for point in grid:
        # Apply parallel-config overrides (ep, node_limit, imbalance).
        par_overrides = {
            k: point[k]
            for k in ("ep", "moe_node_limit", "moe_imbalance_factor")
            if k in point
        }
        par = base_parallel_cfg.model_copy(update=par_overrides)

        # Apply per-layer MoE overrides (num_experts, top_k) to a model copy.
        layer_overrides = {k: point[k] for k in ("num_experts", "top_k") if k in point}
        if layer_overrides:
            new_layers = [
                lc.model_copy(update=layer_overrides) for lc in model_cfg.layers
            ]
            mc = model_cfg.model_copy(update={"layers": new_layers})
        else:
            mc = model_cfg

        # Build perf_model from the per-point model copy so activation-memory
        # accounting reads the correct layer config (e.g. num_experts/top_k).
        perf_model = LLMPerformanceModel(mc, hw)

        t_step, train_sim, _bd = pretraining_time(mc, hw, par, rl_cfg)
        weight_gb, grad_gb, optimizer_gb = perf_model._train_state_gb(train_sim, par)
        activation_peak_gb = perf_model._train_activation_gb(par, rl_cfg)
        peak_memory_gb = weight_gb + grad_gb + optimizer_gb + activation_peak_gb

        exposed = dict(train_sim.exposed_comm_by_fabric)
        cross_node_gb = train_sim.exposed_comm_by_fabric.get("nic", 0.0)

        rows.append(
            {
                "point": point,
                "step_seconds": t_step,
                "exposed_comm_by_fabric": exposed,
                "cross_node_gb": cross_node_gb,
                "peak_memory_gb": peak_memory_gb,
                "feasible": peak_memory_gb <= hw.usable_hbm_gb,
            }
        )
    return rows
