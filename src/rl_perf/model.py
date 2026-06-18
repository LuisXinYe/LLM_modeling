from rl_perf.config import HardwareConfig, ModelConfig, RLConfig
from rl_perf.pipeline import (
    step_time,
    generation_time,
    training_time,
    ref_time,
    _reshard_time,
)
from rl_perf.builder import _estimate_vision_encoder_params, _split_stages
from rl_perf.report import MemoryProfile, TargetReport


class RLPerformanceModel:
    """Top-level facade for RL training performance estimation.

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
            rl_cfg: RLConfig describing the workload.
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
        gen_sim, t_gen = generation_time(
            self.model, self.hw, gen_parallel, rl_cfg
        )
        t_train, train_sim, step_bd = training_time(
            self.model, self.hw, train_parallel, rl_cfg
        )
        t_ref, ref_sim = ref_time(
            self.model, self.hw, ref_parallel, rl_cfg
        )

        # Startup overhead: full gen batch time (prefill + decode)
        startup = t_gen

        # Resharding time: when parallelism changes between phases,
        # model weights must be redistributed across devices.
        # Colocated execution order: gen → ref → train
        t_reshard_gen_ref = 0.0
        t_reshard_ref_train = 0.0
        if rl_cfg.colocated:
            t_reshard_gen_ref = _reshard_time(
                self.model, self.hw, gen_parallel, ref_parallel
            )
            t_reshard_ref_train = _reshard_time(
                self.model, self.hw, ref_parallel, train_parallel
            )
        t_reshard = t_reshard_gen_ref + t_reshard_ref_train

        t_step = step_time(
            t_gen,
            t_train,
            t_ref,
            startup,
            colocated=rl_cfg.colocated,
            t_reshard=t_reshard,
        )

        # Compute TPS targets (single-rank perspective)
        # Each rank processes local_seq_len = seq_len / cp tokens due to CP.
        avg_tokens = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
        train_local_tokens = avg_tokens / train_parallel.cp if train_parallel.cp > 1 else avg_tokens
        ref_local_tokens = avg_tokens / ref_parallel.cp if ref_parallel.cp > 1 else avg_tokens

        gen_tps = rl_cfg.gen_batch_size * rl_cfg.avg_response_len / t_gen if t_gen > 0 else 0
        train_tps = rl_cfg.train_batch_size * train_local_tokens / t_train if t_train > 0 else 0
        ref_tps = rl_cfg.train_batch_size * rl_cfg.group_size / ref_parallel.dp * ref_local_tokens / t_ref if t_ref > 0 else 0
        gen_sps = rl_cfg.gen_batch_size / t_gen if t_gen > 0 else 0
        train_sps = rl_cfg.train_batch_size / t_train if t_train > 0 else 0
        ref_sps = rl_cfg.train_batch_size * rl_cfg.gen_batch_size / ref_parallel.dp / t_ref if t_ref > 0 else 0

        # Memory profile
        memory = self._compute_memory_profile(
            train_sim, gen_sim, ref_sim, train_parallel, gen_parallel, ref_parallel, rl_cfg
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

    def _compute_memory_profile(
        self, train_sim, gen_sim, ref_sim, train_parallel, gen_parallel, ref_parallel, rl_cfg
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
            rl_cfg: RLConfig workload specification.

        Returns:
            MemoryProfile with per-component memory in GB and feasibility flags.
        """

        # From SimResult (ephemeral memory)
        train_weight_gb = train_sim.weight_bytes / 1e9
        gen_weight_gb = gen_sim.weight_bytes / 1e9
        activation_peak_gb = train_sim.peak_activation_bytes / 1e9

        # Optimizer: 12 bytes per param (Adam fp32 master + momentum + variance)
        # When optimizer_offload=True, optimizer states reside on CPU and are
        # transferred in small chunks during the step — not resident on GPU.
        param_count = train_sim.weight_bytes / self.model.dtype_bytes
        optim_bytes = param_count * 12
        if train_parallel.zero_stage >= 1:
            optim_bytes /= train_parallel.dp
        optimizer_gb = 0.0 if train_parallel.optimizer_offload else optim_bytes / 1e9

        # KV cache for generation — iterate all layers per PP stage
        all_layers = self.model.get_layers()
        stage_layers = _split_stages(all_layers, gen_parallel.pp)[0]
        kv_total = 0
        max_kv_seq = rl_cfg.avg_prompt_len + rl_cfg.max_response_len
        for layer in stage_layers:
            if layer.attention == "MLA":
                kv_per_token = (
                    layer.kv_compression_dim + layer.rope_dim
                ) * self.model.dtype_bytes
            elif layer.attention == "SWA" and layer.window_size > 0:
                kv_heads_per_device = layer.num_kv_heads // gen_parallel.tp
                kv_per_token = (
                    2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
                )
                # SWA KV cache bounded by window_size
                kv_total += (
                    kv_per_token
                    * rl_cfg.gen_batch_size
                    * min(max_kv_seq, layer.window_size)
                )
                continue
            else:
                kv_heads_per_device = layer.num_kv_heads // gen_parallel.tp
                kv_per_token = (
                    2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
                )
            kv_total += kv_per_token * rl_cfg.gen_batch_size * max_kv_seq
        kv_cache_gb = kv_total / 1e9

        # Reference model
        ref_weight_gb = ref_sim.weight_bytes / 1e9 if ref_sim else 0
        ref_offload = rl_cfg.ref_offload_cpu or ref_parallel.param_offload
        ref_gb = (
            ref_weight_gb
            if (rl_cfg.reference_model and not ref_offload)
            else 0
        )
        ref_activation_peak_gb = (
            ref_sim.peak_activation_bytes / 1e9 if ref_sim else 0
        )

        # Reward model (same architecture as policy, forward-only, no optimizer)
        reward_model_gb = train_weight_gb if rl_cfg.reward_model else 0

        # Totals
        total_train = (
            train_weight_gb
            + optimizer_gb
            + activation_peak_gb
            + ref_gb
            + reward_model_gb
        )

        # Vision encoder (ViT) memory for multimodal models
        ve_weight_gb = 0.0
        ve_optimizer_gb = 0.0
        ve_weight_gen_gb = 0.0
        ve_weight_ref_gb = 0.0
        if self.model.vision_encoder:
            ve = self.model.vision_encoder
            ve_param_count = _estimate_vision_encoder_params(ve)
            ve_weight_total_gb = ve_param_count * self.model.dtype_bytes / 1e9
            # Training: ViT weights TP-sharded on same devices as LLM
            ve_weight_gb = ve_weight_total_gb / train_parallel.tp
            # Generation: ViT weights TP-sharded on gen devices
            ve_weight_gen_gb = ve_weight_total_gb / gen_parallel.tp
            # Reference: ViT weights TP-sharded on ref devices
            ve_weight_ref_gb = ve_weight_total_gb / ref_parallel.tp
            # Optimizer states for ViT (Adam: 12 bytes per param)
            ve_optim_bytes = ve_param_count * 12
            if train_parallel.zero_stage >= 1:
                ve_optim_bytes /= train_parallel.dp
            ve_optimizer_gb = 0.0 if train_parallel.optimizer_offload else ve_optim_bytes / 1e9
            total_train += ve_weight_gb + ve_optimizer_gb

        # KV cache only exists during the gen sub-step within each step, then freed.
        # Training peak memory does NOT coexist with full KV cache.
        # We still report total_gen for the gen sub-step feasibility check.
        # Generation also needs ViT weights on GPU (multimodal models).
        total_gen = gen_weight_gb + kv_cache_gb + ve_weight_gen_gb

        # Reference phase: ref model weights + ref forward activation peak.
        # Ref phase is forward-only, no optimizer or KV cache.
        # When ref_offload_cpu=True, weights are on CPU and stream to GPU
        # layer-by-layer during forward. Peak GPU memory is only the
        # activation peak (which includes per-layer working set), not the
        # full model weight. When not offloaded, all weights are resident.
        # Reference also needs ViT weights on GPU (multimodal models).
        if ref_offload:
            total_ref = ref_activation_peak_gb + ve_weight_ref_gb
        else:
            total_ref = ref_weight_gb + ref_activation_peak_gb + ve_weight_ref_gb
        usable = self.hw.usable_hbm_gb

        return MemoryProfile(
            weight_gb=train_weight_gb,
            gen_weight_gb=gen_weight_gb,
            ref_weight_gb=ref_weight_gb if ref_sim else 0,
            optimizer_gb=optimizer_gb,
            activation_peak_gb=activation_peak_gb,
            kv_cache_gb=kv_cache_gb,
            ref_model_gb=ref_gb,
            ref_activation_peak_gb=ref_activation_peak_gb,
            reward_model_gb=reward_model_gb,
            ve_weight_gb=ve_weight_gb,
            ve_optimizer_gb=ve_optimizer_gb,
            ve_weight_gen_gb=ve_weight_gen_gb,
            ve_weight_ref_gb=ve_weight_ref_gb,
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
        """Run a what-if scenario by merging overrides into a base RLConfig.

        Args:
            base_config: Dict of base RLConfig field values.
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
        rl_cfg = RLConfig(**{**base_config, **overrides})
        return self.derive_targets(
            total_devices, rl_cfg, gen_parallel, train_parallel, ref_parallel
        )

    def sensitivity(
        self, rl_cfg, param_name, values, total_devices, gen_parallel, train_parallel,
        ref_parallel=None,
    ):
        """Sweep a single RLConfig parameter across multiple values.

        Args:
            rl_cfg: Base RLConfig instance.
            param_name: Name of the RLConfig field to sweep (e.g. "group_size").
            values: Iterable of values to try for the parameter.
            total_devices: Total number of accelerator devices.
            gen_parallel: ParallelismConfig for generation.
            train_parallel: ParallelismConfig for training.
            ref_parallel: ParallelismConfig for reference. Defaults to train_parallel.

        Returns:
            List[TargetReport], one per value in the sweep.

        Raises:
            ValueError: If param_name is not a valid RLConfig field.
        """
        if ref_parallel is None:
            ref_parallel = train_parallel
        if param_name not in RLConfig.model_fields:
            raise ValueError(f"Unknown RLConfig field: {param_name}")
        results = []
        for v in values:
            cfg = rl_cfg.model_copy(update={param_name: v})
            results.append(
                self.derive_targets(total_devices, cfg, gen_parallel, train_parallel, ref_parallel)
            )
        return results
