"""Tests for model.py (LLMPerformanceModel) and report.py (format_table)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_perf.config import (
    HardwareConfig,
    ModelConfig,
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.model import LLMPerformanceModel
from llm_perf.precision import PrecisionConfig, TensorPrecision
from llm_perf.report import TargetReport, format_json, format_table

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@pytest.fixture
def model_cfg() -> ModelConfig:
    return load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))


@pytest.fixture
def hw() -> HardwareConfig:
    return load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))


@pytest.fixture
def parallel_cfg() -> ParallelismConfig:
    return ParallelismConfig(tp=8, pp=1, dp=8, ep=1)


@pytest.fixture
def rl_cfg() -> WorkloadConfig:
    return WorkloadConfig(
        total_prompts=64,
        group_size=4,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gradient_accumulation_steps=1,
        gen_batch_size=16,
    )


@pytest.fixture
def perf_model(model_cfg, hw) -> LLMPerformanceModel:
    return LLMPerformanceModel(model_cfg, hw)


# ---------------------------------------------------------------------------
# Test 1: derive_targets — basic sanity
# ---------------------------------------------------------------------------


def test_derive_targets(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.derive_targets(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
        ref_parallel=parallel_cfg,
    )
    assert isinstance(report, TargetReport)
    assert report.step_time_seconds > 0
    assert report.gen_tps_target > 0
    assert report.train_tps_target > 0
    assert report.ref_tps_target > 0


# ---------------------------------------------------------------------------
# Test 2: feasibility_check — ref_parallel defaults to train_parallel
# ---------------------------------------------------------------------------


def test_feasibility_check(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    assert isinstance(report, TargetReport)
    assert report.step_time_seconds > 0
    assert isinstance(report.feasible, bool)


# ---------------------------------------------------------------------------
# Test 3: what-if group_size — larger group should take longer training time
# ---------------------------------------------------------------------------


def test_what_if_group_size(perf_model, parallel_cfg):
    rl_small = WorkloadConfig(
        total_prompts=64,
        group_size=8,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    rl_large = WorkloadConfig(
        total_prompts=64,
        group_size=16,
        avg_prompt_len=256,
        avg_response_len=512,
        max_response_len=1024,
        train_micro_batch_size=2,
        gen_batch_size=16,
    )
    report_small = perf_model.derive_targets(64, rl_small, parallel_cfg, parallel_cfg, parallel_cfg)
    report_large = perf_model.derive_targets(64, rl_large, parallel_cfg, parallel_cfg, parallel_cfg)

    # group_size=16 → 2x responses per prompt → longer step time overall
    assert report_large.step_time_seconds > report_small.step_time_seconds


# ---------------------------------------------------------------------------
# Test 4: memory profile — all numeric fields positive, booleans exist
# ---------------------------------------------------------------------------


def test_memory_profile(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    mem = report.memory
    assert mem.weight_gb > 0
    assert mem.optimizer_gb > 0
    assert mem.activation_peak_gb > 0
    assert mem.kv_cache_gb > 0
    assert mem.total_train_gb > 0
    assert mem.total_gen_gb > 0
    assert mem.usable_hbm_gb > 0
    assert isinstance(mem.train_feasible, bool)
    assert isinstance(mem.gen_feasible, bool)


# ---------------------------------------------------------------------------
# Test 5: format_table returns string with key words
# ---------------------------------------------------------------------------


def test_format_table(perf_model, rl_cfg, parallel_cfg):
    report = perf_model.feasibility_check(
        total_devices=64,
        rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg,
        train_parallel=parallel_cfg,
    )
    table = format_table(report)
    assert isinstance(table, str)
    assert "Step time" in table
    assert "Generation" in table
    assert "Training" in table
    assert "Memory" in table
    # Should contain either OK or OOM
    assert "OK" in table or "OOM" in table
    # Should contain FEASIBLE status
    assert "FEASIBLE" in table


# ---------------------------------------------------------------------------
# Test 6: format_json — round-trips through JSON, key fields present
# ---------------------------------------------------------------------------


def test_format_json(perf_model, rl_cfg):
    import json as json_mod

    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p, train_p)

    result = format_json(report)
    parsed = json_mod.loads(result)
    assert "step_time_seconds" in parsed
    assert "memory" in parsed
    assert parsed["memory"]["weight_gb"] > 0


# ---------------------------------------------------------------------------
# Test 7: memory from SimResult
# ---------------------------------------------------------------------------


def test_memory_from_sim_result(perf_model, rl_cfg):
    """Memory profile should use SimResult weight_bytes."""
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    report = perf_model.derive_targets(64, rl_cfg, gen_p, train_p, train_p)
    assert report.memory.weight_gb > 0
    assert report.memory.activation_peak_gb > 0


# ---------------------------------------------------------------------------
# Test 8: what_if
# ---------------------------------------------------------------------------


def test_what_if(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    base = perf_model.derive_targets(64, rl_cfg, gen_p, train_p, train_p)

    result = perf_model.what_if(
        base_config=rl_cfg.model_dump(),
        overrides={"group_size": 16},
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert result.step_time_seconds > base.step_time_seconds


# ---------------------------------------------------------------------------
# Test 9: sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    results = perf_model.sensitivity(
        rl_cfg=rl_cfg, param_name="group_size", values=[4, 8, 16],
        total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
    )
    assert len(results) == 3
    assert results[2].step_time_seconds > results[0].step_time_seconds


# ---------------------------------------------------------------------------
# Test 10: sensitivity invalid param
# ---------------------------------------------------------------------------


def test_sensitivity_invalid_param(perf_model, rl_cfg):
    gen_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    train_p = ParallelismConfig(tp=8, pp=1, dp=8, ep=1)
    with pytest.raises(ValueError, match="Unknown WorkloadConfig field"):
        perf_model.sensitivity(
            rl_cfg=rl_cfg, param_name="nonexistent_field", values=[1, 2],
            total_devices=64, gen_parallel=gen_p, train_parallel=train_p,
        )


# ---------------------------------------------------------------------------
# Test 11: weight_bytes no double count
# ---------------------------------------------------------------------------


def test_weight_bytes_no_double_count():
    """Verify builder zeroes weight_bytes on BWD ops so SimResult doesn't double-count."""
    from llm_perf.builder import build_training_step
    from llm_perf.simulator import simulate

    mc = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    rl = WorkloadConfig(total_prompts=100, group_size=4, train_micro_batch_size=2, gen_batch_size=8)
    parallel = ParallelismConfig(tp=1, pp=1, dp=1, ep=1)

    ops_list = build_training_step(mc, hw, parallel, rl)
    fwd_weight = sum(op.weight_bytes for op in ops_list)
    # Since BWD ops have weight_bytes=0, this sum equals fwd-only weight
    sim = simulate(ops_list)
    assert sim.weight_bytes == pytest.approx(fwd_weight)


# ---------------------------------------------------------------------------
# Test 12: feasible field — tracks memory feasibility
# ---------------------------------------------------------------------------


def test_feasible_field_tracks_memory(perf_model, rl_cfg, parallel_cfg):
    """feasible should be True when all phases fit in memory."""
    report = perf_model.derive_targets(
        total_devices=64, rl_cfg=rl_cfg,
        gen_parallel=parallel_cfg, train_parallel=parallel_cfg,
        ref_parallel=parallel_cfg,
    )
    assert hasattr(report, 'feasible')
    assert isinstance(report.feasible, bool)
    expected = (
        report.memory.train_feasible
        and report.memory.gen_feasible
        and report.memory.ref_feasible
    )
    assert report.feasible is expected


# ---------------------------------------------------------------------------
# Memory modeling: gradient buffer, ZeRO sharding, recompute/offload
# ---------------------------------------------------------------------------


def test_pretraining_memory_includes_gradient(perf_model, rl_cfg, parallel_cfg):
    """Mixed-precision training holds a gradient buffer (same dtype as weights)."""
    r = perf_model.derive_pretraining(64, rl_cfg, parallel_cfg)
    assert r["grad_gb"] > 0
    assert r["grad_gb"] == pytest.approx(r["weight_gb"], rel=0.01)
    assert r["total_train_gb"] == pytest.approx(
        r["weight_gb"] + r["grad_gb"] + r["optimizer_gb"] + r["activation_peak_gb"],
        rel=1e-3,
    )


def test_pretraining_zero3_shards_weight_grad_optimizer(perf_model, rl_cfg):
    base = perf_model.derive_pretraining(
        64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8, zero_stage=0)
    )
    z3 = perf_model.derive_pretraining(
        64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8, zero_stage=3)
    )
    for key in ("weight_gb", "grad_gb", "optimizer_gb"):
        assert z3[key] == pytest.approx(base[key] / 8, rel=0.02), key


def test_pretraining_recompute_and_offload_reduce_activation(perf_model, rl_cfg):
    base = perf_model.derive_pretraining(64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8))
    full = perf_model.derive_pretraining(
        64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8, full_recomputation=True)
    )
    off = perf_model.derive_pretraining(
        64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8, activation_offload=True)
    )
    # Retained activation stack is well above the old instantaneous-only peak.
    assert base["activation_peak_gb"] > 0.5
    assert full["activation_peak_gb"] < base["activation_peak_gb"]
    assert off["activation_peak_gb"] < full["activation_peak_gb"]


def test_pretraining_grad_offload_zeroes_gradient(perf_model, rl_cfg):
    off = perf_model.derive_pretraining(
        64, rl_cfg, ParallelismConfig(tp=8, pp=1, dp=8, grad_offload=True)
    )
    assert off["grad_gb"] == 0


def test_gpipe_holds_more_activation_than_1f1b(perf_model):
    """GPipe keeps all M micro-batches' activations; 1F1B only the warmup depth.

    With M=32 (train_batch 64 / mbs 1 / dp 2) and pp=4, GPipe holds 32 vs
    1F1B's min(pp, M)=4 → GPipe activation ≈ 8x. zero_bubble ≈ 1F1B.
    """
    wl = WorkloadConfig(
        group_size=4, train_batch_size=64,
        train_micro_batch_size=1, gradient_accumulation_steps=1,
    )

    def act(sched):
        return perf_model.derive_pretraining(
            64, wl, ParallelismConfig(tp=4, pp=4, dp=2, pp_schedule=sched)
        )["activation_peak_gb"]

    a_1f1b, a_gpipe, a_zb = act("1f1b"), act("gpipe"), act("zero_bubble")
    assert a_gpipe == pytest.approx(a_1f1b * 8, rel=0.01)
    assert a_zb == pytest.approx(a_1f1b)


# ---------------------------------------------------------------------------
# Test: periodic high-precision step blending
# ---------------------------------------------------------------------------


def test_periodic_high_precision_blends_step_time(model_cfg, hw):
    """With high_precision_period=N=4, blended step time == 0.75*t_low + 0.25*t_high.

    Uses the existing fixture model+hw. Builds three precision configs:
      - fp8_low: fp8 weights, no periodic override (high_precision_period=0)
      - bf16_high: all-bf16 (high_precision_dtype default)
      - fp8_blend: fp8 weights with high_precision_period=4

    Asserts the blended step time matches the linear combination exactly.
    """
    perf_model = LLMPerformanceModel(model_cfg, hw)

    rl = WorkloadConfig(
        total_prompts=8,
        group_size=2,
        avg_prompt_len=256,
        avg_response_len=256,
        train_micro_batch_size=1,
        gen_batch_size=4,
    )
    train_parallel = ParallelismConfig(tp=1, pp=1, dp=1)

    # Low-precision config: fp8 weights, no period override
    fp8_low = PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3"))
    # High-precision config: all-bf16 (the high_precision_dtype default)
    bf16_high = PrecisionConfig()  # all bf16 = bf16_default
    # Blend config: fp8 with high_precision_period=4
    fp8_blend = PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3"),
        high_precision_period=4,
        high_precision_dtype="bf16",
    )

    t_low = perf_model.derive_pretraining(1, rl, train_parallel, precision_cfg=fp8_low)[
        "step_time_seconds"
    ]
    t_high = perf_model.derive_pretraining(1, rl, train_parallel, precision_cfg=bf16_high)[
        "step_time_seconds"
    ]
    t_blend = perf_model.derive_pretraining(1, rl, train_parallel, precision_cfg=fp8_blend)[
        "step_time_seconds"
    ]

    # N=4 → weight = (1 - 1/4) = 0.75 low + 0.25 high
    assert t_blend == pytest.approx(0.75 * t_low + 0.25 * t_high, rel=1e-3)


# ---------------------------------------------------------------------------
# Test: compare_precision
# ---------------------------------------------------------------------------


def test_compare_precision_orders_speedup_and_memory():
    from llm_perf.model import compare_precision

    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    recipes = {
        "bf16": PrecisionConfig.bf16_default(),
        "fp8": PrecisionConfig(
            weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
            activations=TensorPrecision(dtype="fp8_e4m3", block_size=128),
            comm=TensorPrecision(dtype="fp8_e4m3"),
        ),
    }
    rows = compare_precision(model, hw, pc, rl, recipes)
    by_name = {r["name"]: r for r in rows}
    assert by_name["bf16"]["speedup_vs_bf16"] == pytest.approx(1.0)
    assert by_name["fp8"]["speedup_vs_bf16"] >= 1.0
    assert by_name["fp8"]["comm_reduction_pct"] > 0
    assert (
        "nic" in by_name["fp8"]["exposed_comm_by_fabric"]
        or "nvlink" in by_name["fp8"]["exposed_comm_by_fabric"]
    )


def test_compare_precision_memory_is_monotonic_non_increasing():
    """Final-review #2: low-precision weights/activations must reduce peak memory.

    bf16 >= fp8 >= fp4 on peak_memory_gb (weight + activation terms shrink; the
    fp32-master optimizer term is constant, so the ordering is non-increasing).
    """
    from llm_perf.model import compare_precision

    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    recipes = {
        "bf16": PrecisionConfig.bf16_default(),
        "fp8": PrecisionConfig(
            weights=TensorPrecision(dtype="fp8_e4m3", block_size=128),
            activations=TensorPrecision(dtype="fp8_e4m3", block_size=128),
        ),
        "fp4": PrecisionConfig(
            weights=TensorPrecision(dtype="fp4_e2m1", block_size=128),
            activations=TensorPrecision(dtype="fp4_e2m1", block_size=128),
        ),
    }
    by_name = {r["name"]: r for r in compare_precision(model, hw, pc, rl, recipes)}
    m_bf16 = by_name["bf16"]["peak_memory_gb"]
    m_fp8 = by_name["fp8"]["peak_memory_gb"]
    m_fp4 = by_name["fp4"]["peak_memory_gb"]
    assert m_bf16 >= m_fp8 >= m_fp4
    # The weight copy genuinely shrinks, so fp8 must be strictly below bf16.
    assert m_fp8 < m_bf16


def test_error_feedback_does_not_inflate_optimizer_or_grad_memory(model_cfg, hw):
    """Final-review #1: EF buffer must be a separate resident term, not folded
    into the param_count-bearing weight_bytes (which previously ~doubled all
    memory when error_feedback was enabled)."""
    from llm_perf.builder import build_training_step
    from llm_perf.simulator import simulate

    pc = ParallelismConfig(tp=1, dp=1, zero_stage=0)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    fp8_no_ef = PrecisionConfig(weights=TensorPrecision(dtype="fp8_e4m3"))
    fp8_ef = PrecisionConfig(
        weights=TensorPrecision(dtype="fp8_e4m3"), error_feedback=True, ef_dtype="fp16"
    )

    sim_no = simulate(build_training_step(model_cfg, hw, pc, rl, precision_cfg=fp8_no_ef))
    sim_ef = simulate(build_training_step(model_cfg, hw, pc, rl, precision_cfg=fp8_ef))

    pm = LLMPerformanceModel(model_cfg, hw)
    w0, g0, o0 = pm._train_state_gb(sim_no, pc)
    w1, g1, o1 = pm._train_state_gb(sim_ef, pc)

    assert sim_ef.ef_buffer_bytes > 0
    assert g1 == pytest.approx(g0)  # gradients invariant to EF
    assert o1 == pytest.approx(o0)  # optimizer invariant to EF
    # weight term rises by EXACTLY the EF buffer (not ~2x everything)
    assert (w1 - w0) * 1e9 == pytest.approx(sim_ef.ef_buffer_bytes, rel=1e-6)


def test_fp4_paper_recipe_through_compare_precision():
    from llm_perf.model import compare_precision
    from llm_perf.precision import PrecisionConfig
    model = load_model_config(str(CONFIGS_DIR / "models" / "llama3_1_8b.yaml"))
    hw = load_hardware_config(str(CONFIGS_DIR / "hardware" / "ascend_910c.yaml"))
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    rows = compare_precision(model, hw, pc, rl, {
        "bf16": PrecisionConfig.bf16_default(),
        "fp4_paper": PrecisionConfig.fp4_paper(),
    })
    by = {r["name"]: r for r in rows}
    assert by["fp4_paper"]["speedup_vs_bf16"] > 1.0
    assert by["fp4_paper"]["speedup_vs_bf16"] <= 4.0


def test_default_precision_follows_model_dtype(model_cfg, hw):
    """Final-review #3: with no PrecisionConfig, roles default to ModelConfig.dtype
    (not hardcoded bf16). A fp8 model's default DP grad comm uses fp8 bytes."""
    from llm_perf.builder import build_training_step

    # from_model_dtype unit behavior
    pc_fp8 = PrecisionConfig.from_model_dtype("fp8")
    assert pc_fp8.weights.dtype == "fp8"
    assert pc_fp8.comm.dtype == "fp8"
    assert pc_fp8.master_dtype == "fp32"

    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    fp8_model = model_cfg.model_copy(update={"dtype": "fp8"})

    # default (precision_cfg=None) → comm follows model dtype (fp8 = 1 B)
    ops_default = build_training_step(fp8_model, hw, pc, rl)
    # forced bf16 comm (2 B) for comparison
    ops_bf16 = build_training_step(
        fp8_model, hw, pc, rl, precision_cfg=PrecisionConfig.bf16_default()
    )
    dp_default = sum(o.comm_bytes for o in ops_default if o.name == "dp_grad_sync")
    dp_bf16 = sum(o.comm_bytes for o in ops_bf16 if o.name == "dp_grad_sync")
    assert dp_default == pytest.approx(dp_bf16 / 2, rel=1e-6)
