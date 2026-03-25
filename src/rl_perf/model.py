from rl_perf.config import HardwareConfig, ModelConfig, RLConfig
from rl_perf.pipeline import bottleneck_analysis, epoch_time, generation_time, training_time
from rl_perf.report import MemoryProfile, TargetReport


class RLPerformanceModel:
    def __init__(self, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.model = model_cfg
        self.hw = hw_cfg

    def derive_targets(self, total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None):
        # Compute generation and training times
        t_gen, gen_sim, t_per_batch = generation_time(self.model, self.hw, gen_parallel, rl_cfg)
        t_train, train_sim = training_time(self.model, self.hw, train_parallel, rl_cfg)

        # Startup overhead: full gen batch time (prefill + decode)
        startup = t_per_batch

        t_epoch = epoch_time(t_gen, t_train, startup, colocated=rl_cfg.colocated)
        bottleneck, slack = bottleneck_analysis(t_gen, t_train)

        # Compute TPS targets
        total_responses = rl_cfg.total_responses
        avg_tokens = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len

        gen_tps = total_responses * rl_cfg.avg_response_len / t_gen if t_gen > 0 else 0
        train_tps = total_responses * avg_tokens / t_train if t_train > 0 else 0
        gen_sps = total_responses / t_gen if t_gen > 0 else 0
        train_sps = total_responses / t_train if t_train > 0 else 0

        # Memory profile
        memory = self._compute_memory_profile(train_sim, gen_sim, train_parallel, gen_parallel, rl_cfg)

        within_budget = time_budget_hours is None or (t_epoch / 3600) <= time_budget_hours
        feasible = within_budget and memory.train_feasible and memory.gen_feasible

        return TargetReport(
            epoch_time_hours=t_epoch / 3600,
            within_budget=within_budget,
            bottleneck=bottleneck,
            bottleneck_slack=slack,
            gen_tps_target=gen_tps,
            train_tps_target=train_tps,
            gen_samples_per_sec=gen_sps,
            train_samples_per_sec=train_sps,
            gen_time_hours=t_gen / 3600,
            train_time_hours=t_train / 3600,
            memory=memory,
            gen_parallel=gen_parallel,
            train_parallel=train_parallel,
            feasible=feasible,
        )

    def feasibility_check(self, total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None):
        return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=time_budget_hours)

    def _compute_memory_profile(self, train_sim, gen_sim, train_parallel, gen_parallel, rl_cfg):
        """Memory profile: weight/activation from SimResult, KV/optimizer/ref analytical."""

        # From SimResult (ephemeral memory)
        train_weight_gb = train_sim.weight_bytes / 1e9
        gen_weight_gb = gen_sim.weight_bytes / 1e9
        activation_peak_gb = train_sim.peak_activation_bytes / 1e9

        # Optimizer: 12 bytes per param (Adam fp32 master + momentum + variance)
        param_count = train_sim.weight_bytes / self.model.dtype_bytes
        optim_bytes = param_count * 12
        if train_parallel.zero_stage >= 1:
            optim_bytes /= train_parallel.dp
        optimizer_gb = optim_bytes / 1e9

        # KV cache for generation — iterate all layers per PP stage
        all_layers = self.model.get_layers()
        layers_per_stage = max(1, len(all_layers) // gen_parallel.pp)
        stage_layers = all_layers[:layers_per_stage]
        kv_total = 0
        max_kv_seq = rl_cfg.avg_prompt_len + rl_cfg.max_response_len
        for layer in stage_layers:
            if layer.attention == "MLA":
                kv_per_token = (layer.kv_compression_dim + layer.rope_dim) * self.model.dtype_bytes
            elif layer.attention == "SWA" and layer.window_size > 0:
                kv_heads_per_device = max(1, layer.num_kv_heads // gen_parallel.tp)
                kv_per_token = 2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
                # SWA KV cache bounded by window_size
                kv_total += kv_per_token * rl_cfg.gen_batch_size * min(max_kv_seq, layer.window_size)
                continue
            else:
                kv_heads_per_device = max(1, layer.num_kv_heads // gen_parallel.tp)
                kv_per_token = 2 * kv_heads_per_device * layer.head_dim * self.model.dtype_bytes
            kv_total += kv_per_token * rl_cfg.gen_batch_size * max_kv_seq
        kv_cache_gb = kv_total / 1e9

        # Reference model
        ref_gb = train_weight_gb if (rl_cfg.reference_model and not rl_cfg.ref_offload_cpu) else 0

        # Totals
        total_train = train_weight_gb + optimizer_gb + activation_peak_gb + ref_gb
        total_gen = gen_weight_gb + kv_cache_gb
        usable = self.hw.usable_hbm_gb

        return MemoryProfile(
            weight_gb=train_weight_gb,
            optimizer_gb=optimizer_gb,
            activation_peak_gb=activation_peak_gb,
            kv_cache_gb=kv_cache_gb,
            ref_model_gb=ref_gb,
            total_train_gb=total_train,
            total_gen_gb=total_gen,
            usable_hbm_gb=usable,
            train_feasible=total_train < usable,
            gen_feasible=total_gen < usable,
        )

    def what_if(self, base_config, overrides,
                total_devices, gen_parallel, train_parallel, time_budget_hours=None):
        """base_config + overrides → TargetReport for comparison."""
        rl_cfg = RLConfig(**{**base_config, **overrides})
        return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours)

    def sensitivity(self, rl_cfg, param_name, values,
                    total_devices, gen_parallel, train_parallel):
        """Sweep one parameter across values."""
        if param_name not in RLConfig.model_fields:
            raise ValueError(f"Unknown RLConfig field: {param_name}")
        results = []
        for v in values:
            cfg = rl_cfg.model_copy(update={param_name: v})
            results.append(self.derive_targets(total_devices, cfg, gen_parallel, train_parallel))
        return results
