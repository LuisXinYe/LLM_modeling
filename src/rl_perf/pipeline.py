import math
from rl_perf.config import ModelConfig, HardwareConfig, ParallelismConfig, RLConfig
from rl_perf.builder import build_training_step, build_generation_step
from rl_perf.simulator import simulate


def effective_response_len(avg: int, std: int = None, batch_size: int = 1, max_len: int = None) -> float:
    """Gumbel approximation for expected max of batch_size samples.
    E[max of B samples] ≈ avg + std * sqrt(2 * ln(B))
    Falls back to max_len if std not provided."""
    if std is not None and std > 0 and batch_size > 1:
        return avg + std * math.sqrt(2 * math.log(batch_size))
    if max_len is not None:
        return max_len
    return avg


def generation_time(model_cfg, hw, parallel_cfg, rl_cfg) -> float:
    """Total generation time in seconds for all responses."""
    prefill_ops, decode_ops = build_generation_step(model_cfg, hw, parallel_cfg, rl_cfg)

    t_prefill = simulate(prefill_ops).wall_clock_time
    t_decode_per_token = simulate(decode_ops).wall_clock_time

    eff_len = effective_response_len(
        avg=rl_cfg.avg_response_len,
        std=rl_cfg.std_response_len,
        batch_size=rl_cfg.gen_batch_size,
        max_len=rl_cfg.max_response_len,
    )

    t_per_batch = t_prefill + eff_len * t_decode_per_token

    total_responses = rl_cfg.total_responses
    gen_dp = parallel_cfg.dp  # number of independent generation instances
    batches = math.ceil(total_responses / (rl_cfg.gen_batch_size * gen_dp))

    return batches * t_per_batch


def training_time(model_cfg, hw, parallel_cfg, rl_cfg) -> float:
    """Total training time in seconds for all responses."""
    train_ops = build_training_step(model_cfg, hw, parallel_cfg, rl_cfg)
    t_step = simulate(train_ops).wall_clock_time

    # PP bubble ratio: (pp-1) / (M + pp-1) where M = gradient_accumulation_steps
    pp = parallel_cfg.pp
    if pp > 1:
        M = rl_cfg.gradient_accumulation_steps
        bubble_ratio = (pp - 1) / (M + pp - 1)
        t_step *= (1 + bubble_ratio)

    # Apply perf penalties for recomputation/offload
    penalty = 1.0
    if parallel_cfg.full_recomputation:
        penalty *= 1.30
    elif parallel_cfg.recompute_attention:
        penalty *= 1.05
    if parallel_cfg.optimizer_offload:
        penalty *= 1.15
    if parallel_cfg.activation_offload:
        penalty *= 1.10
    t_step *= penalty

    total_responses = rl_cfg.total_responses
    effective_batch = rl_cfg.train_micro_batch_size * rl_cfg.gradient_accumulation_steps * parallel_cfg.dp
    num_steps = math.ceil(total_responses / effective_batch)

    return num_steps * t_step


def epoch_time(t_gen: float, t_train: float, startup_overhead: float = 0, colocated: bool = False) -> float:
    """Two-stage pipeline epoch time.
    Colocated: same GPUs, gen+train serial.
    Separated: different GPU pools, gen/train parallel."""
    if colocated:
        return t_gen + t_train
    return max(t_gen, t_train) + startup_overhead


def bottleneck_analysis(t_gen: float, t_train: float):
    """Returns (bottleneck_name, slack_ratio)."""
    if abs(t_gen - t_train) < 1e-9:
        return "BALANCED", 0.0
    if t_gen > t_train:
        return "GENERATION", t_gen / t_train - 1
    return "TRAINING", t_train / t_gen - 1
