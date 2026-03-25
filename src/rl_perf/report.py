import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class MemoryProfile:
    weight_gb: float
    optimizer_gb: float
    activation_peak_gb: float
    kv_cache_gb: float
    ref_model_gb: float
    total_train_gb: float
    total_gen_gb: float
    usable_hbm_gb: float
    train_feasible: bool
    gen_feasible: bool


@dataclass
class TargetReport:
    epoch_time_hours: float
    within_budget: bool
    bottleneck: str
    bottleneck_slack: float
    gen_tps_target: float
    train_tps_target: float
    gen_samples_per_sec: float
    train_samples_per_sec: float
    gen_time_hours: float
    train_time_hours: float
    memory: MemoryProfile
    gen_parallel: object = None
    train_parallel: object = None
    feasible: bool = True


def format_table(report: TargetReport) -> str:
    """Format as rich-compatible text table."""
    reasons = []
    if not report.within_budget:
        reasons.append("OVER TIME")
    if not report.memory.train_feasible or not report.memory.gen_feasible:
        reasons.append("OOM")
    status_str = "FEASIBLE" if not reasons else f"NOT FEASIBLE: {' + '.join(reasons)}"
    mem = report.memory
    lines = [
        "=" * 60,
        "          RL Training Performance Report",
        "=" * 60,
        f" Epoch time:        {report.epoch_time_hours:.2f} hours  [{status_str}]",
        f" Bottleneck:        {report.bottleneck} (slack: {report.bottleneck_slack:.1%})",
        "-" * 60,
        " Generation:",
        f"   TPS target:      {report.gen_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.gen_samples_per_sec:.2f}",
        f"   Time:            {report.gen_time_hours:.2f} hours",
        "-" * 60,
        " Training:",
        f"   TPS target:      {report.train_tps_target:,.0f} tokens/s",
        f"   Samples/s:       {report.train_samples_per_sec:.2f}",
        f"   Time:            {report.train_time_hours:.2f} hours",
        "-" * 60,
        " Memory:",
        f"   Train: {mem.total_train_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.train_feasible else 'OOM'}]",
        f"   Gen:   {mem.total_gen_gb:.1f}/{mem.usable_hbm_gb:.1f} GB  [{'OK' if mem.gen_feasible else 'OOM'}]",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_json(report: TargetReport) -> str:
    """JSON serialization of TargetReport."""
    return json.dumps(asdict(report), indent=2, default=str)
