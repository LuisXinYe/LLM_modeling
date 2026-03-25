import math
from rl_perf.config import ModelConfig, HardwareConfig, ParallelismConfig, RLConfig
from rl_perf.pipeline import generation_time, training_time, epoch_time, bottleneck_analysis
from rl_perf.builder import build_training_step, build_generation_step
from rl_perf.simulator import simulate
from rl_perf.report import TargetReport, MemoryProfile


class RLPerformanceModel:
    def __init__(self, model_cfg: ModelConfig, hw_cfg: HardwareConfig):
        self.model = model_cfg
        self.hw = hw_cfg

    def derive_targets(self, total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None):
        # Compute generation and training times
        t_gen = generation_time(self.model, self.hw, gen_parallel, rl_cfg)
        t_train = training_time(self.model, self.hw, train_parallel, rl_cfg)

        # Startup overhead: one gen batch time
        prefill_ops, decode_ops = build_generation_step(self.model, self.hw, gen_parallel, rl_cfg)
        t_prefill = simulate(prefill_ops).wall_clock_time
        startup = t_prefill  # simplified

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
        memory = self._compute_memory(train_parallel, gen_parallel, rl_cfg)

        within_budget = time_budget_hours is None or (t_epoch / 3600) <= time_budget_hours

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
        )

    def feasibility_check(self, total_devices, rl_cfg, gen_parallel, train_parallel):
        return self.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=None)

    def _compute_memory(self, train_parallel, gen_parallel, rl_cfg):
        """Estimate memory profile per device."""
        model = self.model
        hw = self.hw

        # Weight memory per device (simplified: total params * bytes / TP / PP)
        layer = model.get_layers()[0]
        d = model.hidden_size
        dtype_b = model.dtype_bytes

        # Per-layer weight estimate
        if layer.ffn == "MoE":
            # Attention + MoE weights
            attn_w = 4 * d * d * dtype_b  # Q,K,V,O (approximate for GQA)
            router_w = d * layer.num_experts * dtype_b
            expert_w = (layer.num_experts * 3 * d * layer.expert_intermediate_size * dtype_b) / train_parallel.ep
            shared_w = layer.num_shared_experts * 3 * d * (layer.shared_intermediate_size or layer.intermediate_size) * dtype_b
            per_layer_w = (attn_w + router_w + expert_w + shared_w) / train_parallel.tp
        else:
            attn_w = 4 * d * d * dtype_b
            ffn_w = 3 * d * layer.intermediate_size * dtype_b
            per_layer_w = (attn_w + ffn_w) / train_parallel.tp

        layers_per_stage = model.num_layers / train_parallel.pp
        total_weight = per_layer_w * layers_per_stage

        # Embedding + LM head
        embed_w = model.vocab_size * d * dtype_b / train_parallel.tp
        total_weight += embed_w

        weight_gb = total_weight / 1e9

        # Optimizer: 12 bytes per param (Adam fp32 master + momentum + variance)
        param_count = total_weight / dtype_b
        optim_bytes = param_count * 12
        if train_parallel.zero_stage >= 1:
            optim_bytes /= train_parallel.dp
        optimizer_gb = optim_bytes / 1e9

        # Activation memory per layer: Megatron-LM formula sbh * (34 + 5*a*s/h)
        # The "34" factor accounts for: LayerNorm(4) + QKV(6) + attn_output(2) +
        # FFN_gate_up(8) + FFN_down(2) + dropout_masks + intermediates.
        # Note: approximate; MoE models may differ. Ref: Megatron SC'21 paper.
        s = rl_cfg.avg_prompt_len + rl_cfg.avg_response_len
        b = rl_cfg.train_micro_batch_size
        act_per_layer = s * b * d * 34 * dtype_b / train_parallel.tp
        if train_parallel.recompute_attention:
            act_per_layer *= 0.5  # ~50% reduction
        if train_parallel.full_recomputation:
            act_per_layer *= 0.1  # ~90% reduction
        # PP buffers
        pp_buffers = train_parallel.pp
        activation_gb = act_per_layer * layers_per_stage * pp_buffers / 1e9

        # KV cache for generation
        if layer.attention == "MLA":
            kv_per_token = (layer.kv_compression_dim + layer.rope_dim) * dtype_b
        else:
            kv_heads_per_device = max(1, layer.num_kv_heads // gen_parallel.tp)
            kv_per_token = 2 * kv_heads_per_device * layer.head_dim * dtype_b
        kv_total = kv_per_token * model.num_layers * rl_cfg.gen_batch_size * (rl_cfg.avg_prompt_len + rl_cfg.max_response_len)
        kv_cache_gb = kv_total / 1e9

        # Reference model
        ref_gb = weight_gb if (rl_cfg.reference_model and not rl_cfg.ref_offload_cpu) else 0

        # Totals
        total_train = weight_gb + optimizer_gb + activation_gb + ref_gb
        total_gen = weight_gb + kv_cache_gb
        usable = hw.usable_hbm_gb

        return MemoryProfile(
            weight_gb=weight_gb,
            optimizer_gb=optimizer_gb,
            activation_peak_gb=activation_gb,
            kv_cache_gb=kv_cache_gb,
            ref_model_gb=ref_gb,
            total_train_gb=total_train,
            total_gen_gb=total_gen,
            usable_hbm_gb=usable,
            train_feasible=total_train < usable,
            gen_feasible=total_gen < usable,
        )
