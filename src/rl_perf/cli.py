import typer
from pathlib import Path
from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from rl_perf.model import RLPerformanceModel
from rl_perf.report import format_table

app = typer.Typer(help="RL Training Performance Modeling Tool")

# Hardware shortname mapping
HW_SHORTCUTS = {
    "910C": "configs/hardware/ascend_910c.yaml",
    "CM384": "configs/hardware/cloudmatrix_384.yaml",
}


def resolve_hardware(hw: str) -> str:
    """Resolve hardware shortname or path."""
    if hw in HW_SHORTCUTS:
        return HW_SHORTCUTS[hw]
    return hw


@app.command()
def targets(
    model: str = typer.Option(..., "--model", "-m", help="Model config YAML path"),
    hardware: str = typer.Option(..., "--hardware", "-hw", help="Hardware config YAML or shortname (910C, CM384)"),
    devices: int = typer.Option(..., "--devices", "-d", help="Total device count"),
    prompts: int = typer.Option(..., "--prompts", "-p", help="Total prompt count"),
    group_size: int = typer.Option(8, "--group-size", "-g"),
    time_budget: float = typer.Option(None, "--time-budget", "-t", help="Time budget in hours"),
    avg_prompt_len: int = typer.Option(512, "--avg-prompt-len"),
    avg_response_len: int = typer.Option(2048, "--avg-response-len"),
    max_response_len: int = typer.Option(4096, "--max-response-len"),
    gen_batch: int = typer.Option(64, "--gen-batch"),
    train_batch: int = typer.Option(4, "--train-batch"),
    grad_acc: int = typer.Option(1, "--grad-acc"),
    tp: int = typer.Option(8, "--tp"),
    pp: int = typer.Option(1, "--pp"),
):
    """Derive TPS targets given model, hardware, and RL config."""
    mc = load_model_config(model)
    hw = load_hardware_config(resolve_hardware(hardware))

    rl_cfg = RLConfig(
        total_prompts=prompts, group_size=group_size,
        avg_prompt_len=avg_prompt_len, avg_response_len=avg_response_len,
        max_response_len=max_response_len,
        train_micro_batch_size=train_batch, gradient_accumulation_steps=grad_acc,
        gen_batch_size=gen_batch,
    )

    # Auto-derive parallelism
    dp = max(1, devices // (tp * pp))
    gen_parallel = ParallelismConfig(tp=tp, pp=1, dp=max(1, devices // tp))
    train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp)

    perf = RLPerformanceModel(mc, hw)
    report = perf.derive_targets(devices, rl_cfg, gen_parallel, train_parallel, time_budget)

    typer.echo(format_table(report))


@app.command()
def check(
    model: str = typer.Option(..., "--model", "-m"),
    hardware: str = typer.Option(..., "--hardware", "-hw"),
    devices: int = typer.Option(..., "--devices", "-d"),
    prompts: int = typer.Option(10000, "--prompts", "-p"),
    group_size: int = typer.Option(8, "--group-size", "-g"),
    tp: int = typer.Option(8, "--tp"),
    pp: int = typer.Option(1, "--pp"),
):
    """Quick feasibility check."""
    mc = load_model_config(model)
    hw = load_hardware_config(resolve_hardware(hardware))

    rl_cfg = RLConfig(total_prompts=prompts, group_size=group_size)
    dp = max(1, devices // (tp * pp))
    gen_parallel = ParallelismConfig(tp=tp, dp=max(1, devices // tp))
    train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp)

    perf = RLPerformanceModel(mc, hw)
    report = perf.feasibility_check(devices, rl_cfg, gen_parallel, train_parallel)

    typer.echo(format_table(report))
