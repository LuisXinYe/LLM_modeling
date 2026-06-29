"""Demo: model Zhou et al. 2025 FP4 mixed-precision pretraining scheme."""
from llm_perf.config import load_model_config, load_hardware_config, ParallelismConfig, WorkloadConfig
from llm_perf.precision import PrecisionConfig, ModuleLinearPrecision, TensorPrecision
from llm_perf.cost_analysis import theoretical_compute_cost
from llm_perf.model import compare_precision


def main():
    model = load_model_config("configs/models/llama3_1_8b.yaml")
    hw = load_hardware_config("configs/hardware/ascend_910c.yaml")

    print("== Fig 1a forward FLOP split (LLaMA-7B-4K) ==")
    fs = theoretical_compute_cost(model, PrecisionConfig.bf16_default(), seq_len=4096)["forward_split"]
    for k, v in fs.items():
        print(f"  {k:12s} {v*100:5.1f}%")

    print("\n== Table-2 recipes — theoretical compute cost % ==")
    def rec(a, f, b):
        return PrecisionConfig(
            attn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=a), bwd=TensorPrecision(dtype=b)),
            ffn_linear=ModuleLinearPrecision(fwd=TensorPrecision(dtype=f), bwd=TensorPrecision(dtype=b)))
    rows = {
        "FP4/FP4/FP4": rec("fp4_e2m1", "fp4_e2m1", "fp4_e2m1"),
        "FP8/FP4/FP8": rec("fp8_e4m3", "fp4_e2m1", "fp8_e4m3"),
        "FP16 (base)": PrecisionConfig.bf16_default(),
    }
    for name, pc in rows.items():
        stated = theoretical_compute_cost(model, pc, seq_len=4096)["cost_pct"]
        implied = theoretical_compute_cost(model, pc, seq_len=4096,
                                           speed_map={"fp16": 1.0, "fp8": 1.4, "fp4": 2.0})["cost_pct"]
        print(f"  {name:12s} stated(1/2/4x)={stated:5.1f}%  paper-implied={implied:5.1f}%")
    print("  NOTE: paper Table 2 all-FP4 = 57.1%; FLOP-honest 1/2/4x gives ~36% (implies FP4~2x).")

    print("\n== Apply fp4_paper() recipe to Llama-3.1-8B (full roofline) ==")
    pc = ParallelismConfig(tp=1, dp=4)
    rl = WorkloadConfig(total_prompts=8, group_size=2, train_micro_batch_size=1)
    for r in compare_precision(model, hw, pc, rl,
                               {"bf16": PrecisionConfig.bf16_default(), "fp4_paper": PrecisionConfig.fp4_paper()}):
        print(f"  {r['name']:10s} speedup={r['speedup_vs_bf16']:.3f} peakMem={r['peak_memory_gb']:.1f}GB feasible={r['feasible']}")


if __name__ == "__main__":
    main()
