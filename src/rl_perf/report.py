import json
from dataclasses import asdict, dataclass, field


@dataclass
class TrainBreakdown:
    """Train-phase time breakdown for one training step (one parameter update).

    All times are in seconds. In online mode, generation is part of each step;
    in offline mode, generation is a separate phase and gen_step is 0.
    """

    vit: float = 0.0       # Vision encoder forward+backward
    reward_fwd: float = 0.0    # Reward model forward
    old_logprob_fwd: float = 0.0  # Old policy log-prob forward
    policy_update: float = 0.0 # Policy fwd+bwd update
    pp_p2p: float = 0.0        # PP inter-stage P2P communication
    pp_bubble: float = 0.0     # PP bubble idle time (stages waiting for other stages)
    recompute: float = 0.0     # Recomputation / activation offload overhead
    optim_offload: float = 0.0 # Optimizer CPU offload transfer
    total: float = 0.0         # Sum of all sub-steps

    @property
    def training_only(self) -> float:
        """Step time excluding generation (pure training sub-steps)."""
        return self.total - self.gen_step


@dataclass
class MemoryProfile:
    """Per-device memory breakdown for training, generation, and reference phases.

    All sizes are in GB. Feasibility flags indicate whether total fits in HBM.
    """

    weight_gb: float  # Model weight memory in GB (per device, after TP/PP sharding)
    gen_weight_gb: float  # Generation phase weight memory in GB (per device, gen parallelism)
    ref_weight_gb: float  # Reference phase weight memory in GB (per device, ref parallelism)
    optimizer_gb: float  # Optimizer state memory in GB (Adam master weights + moments)
    activation_peak_gb: float  # Peak activation memory in GB during training
    kv_cache_gb: float  # KV cache memory in GB during generation
    ref_model_gb: float  # Reference model weight memory in GB (0 if offloaded/absent)
    ref_activation_peak_gb: float  # Reference forward peak activation memory in GB
    reward_model_gb: float  # Reward model weight memory in GB (0 if no reward model)
    ve_weight_gb: float  # Vision encoder weight memory in GB (0 if no ViT)
    ve_optimizer_gb: float  # Vision encoder optimizer memory in GB (0 if no ViT)
    ve_weight_gen_gb: float  # ViT weight in GB per device during generation phase
    ve_weight_ref_gb: float  # ViT weight in GB per device during reference phase
    total_train_gb: float  # Total training memory: weights + optimizer + activations + ref + reward + ve
    total_gen_gb: float  # Total generation memory: weights + KV cache
    total_ref_gb: float  # Total reference memory: ref weights + ref activation peak
    usable_hbm_gb: float  # Usable HBM capacity in GB (after framework overhead)
    train_feasible: bool  # True if total_train_gb < usable_hbm_gb
    gen_feasible: bool  # True if total_gen_gb < usable_hbm_gb
    ref_feasible: bool  # True if total_ref_gb < usable_hbm_gb


@dataclass
class TargetReport:
    """Complete performance and feasibility report for one RL step."""

    step_time_seconds: float  # Single-step wall-clock time in seconds
    gen_tps_target: float  # Required generation throughput in tokens/s
    train_tps_target: float  # Required training throughput in tokens/s
    ref_tps_target: float  # Required reference throughput in tokens/s
    gen_samples_per_sec: float  # Generation throughput in samples/s
    train_samples_per_sec: float  # Training throughput in samples/s
    ref_samples_per_sec: float  # Reference throughput in samples/s
    gen_time_seconds: float  # Generation phase wall-clock time in seconds
    train_time_seconds: float  # Training phase wall-clock time in seconds
    ref_time_seconds: float  # Reference phase wall-clock time in seconds
    reshard_gen_ref_seconds: float = 0.0  # Resharding time gen→ref phase transition
    reshard_ref_train_seconds: float = 0.0  # Resharding time ref→train phase transition
    train_breakdown: TrainBreakdown = field(default_factory=TrainBreakdown)  # Train-phase time breakdown
    memory: MemoryProfile = None  # Per-device memory breakdown
    gen_parallel: object = None  # ParallelismConfig used for generation
    train_parallel: object = None  # ParallelismConfig used for training
    ref_parallel: object = None  # ParallelismConfig used for reference
    feasible: bool = True  # True if no OOM


def format_table(report: TargetReport) -> str:
    """Format a TargetReport as a human-readable text table.

    Args:
        report: The TargetReport to format.

    Returns:
        Multi-line string with step time, TPS targets, and memory summary.
    """
    reasons = []
    if report.memory and (not report.memory.train_feasible or not report.memory.gen_feasible or not report.memory.ref_feasible):
        reasons.append("OOM")
    status_str = "FEASIBLE" if not reasons else f"NOT FEASIBLE: {' + '.join(reasons)}"
    lines = [
        "=" * 60,
        "          RL Training Performance Report",
        "=" * 60,
        f" Step time:         {report.step_time_seconds:.1f} s  [{status_str}]",
        "-" * 60,
        " Generation:",
        f"   TPS target:      {report.gen_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.gen_samples_per_sec:.2f}",
        f"   Time:            {report.gen_time_seconds:.1f} s",
    ]
    if report.reshard_gen_ref_seconds > 0:
        lines.append("-" * 60)
        lines.append(" Reshard (gen→ref):")
        lines.append(f"   Time:            {report.reshard_gen_ref_seconds:.1f} s")
    lines.append("-" * 60)
    lines += [
        " Reference:",
        f"   TPS target:      {report.ref_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.ref_samples_per_sec:.2f}",
        f"   Time:            {report.ref_time_seconds:.1f} s",
    ]
    if report.reshard_ref_train_seconds > 0:
        lines.append("-" * 60)
        lines.append(" Reshard (ref→train):")
        lines.append(f"   Time:            {report.reshard_ref_train_seconds:.1f} s")
    lines.append("-" * 60)
    lines += [
        " Training:",
        f"   TPS target:      {report.train_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.train_samples_per_sec:.2f}",
    ]
    sb = report.train_breakdown
    if sb and sb.total > 0:
        lines.append(f"   Time:            {sb.total:.1f} s")
        sub_steps = [
            ("vit", "vit", sb.vit),
            ("reward_fwd", "reward_fwd", sb.reward_fwd),
            ("old_logprob_fwd", "old_logp_fwd", sb.old_logprob_fwd),
            ("policy_update", "policy_update", sb.policy_update),
            ("pp_p2p", "pp_p2p", sb.pp_p2p),
            ("pp_bubble", "pp_bubble", sb.pp_bubble),
            ("recompute", "recompute", sb.recompute),
            ("optim_offload", "optim_offload", sb.optim_offload),
        ]
        for _, label, t in sub_steps:
            if t > 0:
                pct = t / sb.total * 100
                lines.append(f"     {label:16s} {t:8.1f} s  ({pct:5.1f}%)")
    lines.append("-" * 60)
    if report.memory:
        mem = report.memory
        lines.append(" Memory:")
        lines.append(f"   Train: {mem.total_train_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.train_feasible else 'OOM'}]")
        lines.append(f"   Gen:   {mem.total_gen_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.gen_feasible else 'OOM'}]")
        if mem.total_ref_gb > 0:
            lines.append(f"   Ref:   {mem.total_ref_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.ref_feasible else 'OOM'}]")
        if mem.reward_model_gb > 0:
            lines.append(f"   Reward model:    {mem.reward_model_gb:.1f} GB")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_json(report: TargetReport) -> str:
    """Serialize a TargetReport to a JSON string.

    Args:
        report: The TargetReport to serialize.

    Returns:
        Pretty-printed JSON string (indent=2).
    """
    return json.dumps(asdict(report), indent=2, default=str)
