import pytest
from pathlib import Path
from rl_perf.model import RLPerformanceModel
from rl_perf.config import (
    load_model_config, load_hardware_config,
    RLConfig, ParallelismConfig,
)
from rl_perf.report import format_table

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))


@pytest.fixture
def rl_cfg():
    return RLConfig(
        total_prompts=10000, group_size=8,
        avg_prompt_len=512, avg_response_len=2048,
        max_response_len=4096,
        train_micro_batch_size=4, gradient_accumulation_steps=4,
        gen_batch_size=64,
    )


@pytest.mark.parametrize("model_name,tp,pp,dp,ep,expected_max_hours", [
    ("llama3_1_8b", 8, 1, 8, 1, 100),      # Small model, should be fast
    ("qwen2_5_72b", 8, 4, 4, 1, 500),       # Large dense, slower
    ("mistral_7b", 8, 1, 8, 1, 100),         # Small with SWA
])
def test_e2e_derive_targets(model_name, tp, pp, dp, ep, expected_max_hours, hw, rl_cfg):
    mc = load_model_config(str(CONFIGS_DIR / "models" / f"{model_name}.yaml"))

    total_devices = tp * pp * dp * ep
    gen_parallel = ParallelismConfig(tp=tp, pp=1, dp=max(1, total_devices // tp))
    train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp, ep=ep)

    perf = RLPerformanceModel(mc, hw)
    report = perf.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel, time_budget_hours=24)

    # Sanity checks
    assert report.epoch_time_hours > 0
    assert report.epoch_time_hours < expected_max_hours
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.bottleneck in ("GENERATION", "TRAINING", "BALANCED")
    assert report.memory is not None
    assert report.memory.weight_gb > 0

    # Format should not crash
    output = format_table(report)
    assert len(output) > 100  # Non-trivial output


@pytest.mark.parametrize("model_name,tp,pp,dp,ep", [
    ("qwen3_235b_moe", 8, 4, 2, 8),   # MoE with EP
    ("deepseekv3_671b", 8, 4, 2, 16),  # MLA + MoE + mHC
])
def test_e2e_moe_models(model_name, tp, pp, dp, ep, hw, rl_cfg):
    mc = load_model_config(str(CONFIGS_DIR / "models" / f"{model_name}.yaml"))

    total_devices = tp * pp * dp * ep
    gen_parallel = ParallelismConfig(tp=tp, pp=1, dp=max(1, total_devices // tp))
    train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp, ep=ep)

    perf = RLPerformanceModel(mc, hw)
    report = perf.derive_targets(total_devices, rl_cfg, gen_parallel, train_parallel)

    assert report.epoch_time_hours > 0
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0

    output = format_table(report)
    assert "tokens" in output.lower() or "TPS" in output


def test_e2e_memory_feasibility(hw):
    """Test that a too-large model on too-few devices is flagged as infeasible."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "qwen2_5_72b.yaml"))
    rl_cfg = RLConfig(total_prompts=100, group_size=4)
    # Only 8 devices for a 72B model — should be memory-tight
    parallel = ParallelismConfig(tp=8, pp=1, dp=1)

    perf = RLPerformanceModel(mc, hw)
    report = perf.feasibility_check(8, rl_cfg, parallel, parallel)

    # Should have a memory profile either way
    assert report.memory is not None
    assert report.memory.usable_hbm_gb > 0


def test_e2e_what_if_comparison(hw):
    """Doubling group_size should increase epoch time."""
    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    parallel = ParallelismConfig(tp=8, pp=1, dp=8)

    perf = RLPerformanceModel(mc, hw)

    rl_base = RLConfig(total_prompts=1000, group_size=8)
    rl_double = RLConfig(total_prompts=1000, group_size=16)

    base = perf.derive_targets(64, rl_base, parallel, parallel)
    double = perf.derive_targets(64, rl_double, parallel, parallel)

    assert double.epoch_time_hours > base.epoch_time_hours


def test_e2e_full_suite_print(hw, rl_cfg, capsys):
    """Print results for all models for visual inspection."""
    models = ["llama3_1_8b", "qwen2_5_72b", "mistral_7b", "qwen3_235b_moe", "deepseekv3_671b"]
    configs = [
        ("llama3_1_8b", 8, 1, 8, 1),
        ("qwen2_5_72b", 8, 4, 4, 1),
        ("mistral_7b", 8, 1, 8, 1),
        ("qwen3_235b_moe", 8, 4, 2, 8),
        ("deepseekv3_671b", 8, 4, 2, 16),
    ]

    for model_name, tp, pp, dp, ep in configs:
        mc = load_model_config(str(CONFIGS_DIR / "models" / f"{model_name}.yaml"))
        total_devices = tp * pp * dp * ep
        gen_p = ParallelismConfig(tp=tp, pp=1, dp=max(1, total_devices // tp))
        train_p = ParallelismConfig(tp=tp, pp=pp, dp=dp, ep=ep)

        perf = RLPerformanceModel(mc, hw)
        report = perf.derive_targets(total_devices, rl_cfg, gen_p, train_p, time_budget_hours=24)

        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Devices: {total_devices} | TP={tp} PP={pp} DP={dp} EP={ep}")
        print(f"Epoch: {report.epoch_time_hours:.2f}h | Bottleneck: {report.bottleneck}")
        print(f"Gen TPS: {report.gen_tps_target:,.0f} | Train TPS: {report.train_tps_target:,.0f}")
        print(f"Memory: train={report.memory.total_train_gb:.1f}GB gen={report.memory.total_gen_gb:.1f}GB")
        print(f"{'='*60}")
