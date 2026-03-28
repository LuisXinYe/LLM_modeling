from enum import Enum
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from rl_perf.config import load_model_config, load_hardware_config, RLConfig, ParallelismConfig
from rl_perf.model import RLPerformanceModel
from rl_perf.report import format_table, format_json

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


class OutputFormat(str, Enum):
    table = "table"
    json = "json"

app = typer.Typer(help="RL Training Performance Modeling Tool")

# Hardware shortname mapping
HW_SHORTCUTS = {
    "910C": str(_CONFIGS_DIR / "hardware" / "ascend_910c.yaml"),
    "CM384": str(_CONFIGS_DIR / "hardware" / "cloudmatrix_384.yaml"),
}


def resolve_hardware(hw: str) -> str:
    """Resolve hardware shortname or path."""
    if hw in HW_SHORTCUTS:
        return HW_SHORTCUTS[hw]
    return hw


def _format_output(report, fmt: OutputFormat) -> str:
    """Format report as table or JSON."""
    if fmt == OutputFormat.json:
        return format_json(report)
    return format_table(report)


def _run_safely(func):
    """Run func, catching common errors and printing friendly messages."""
    try:
        return func()
    except FileNotFoundError as e:
        typer.echo(f"Error: file not found: {e}", err=True)
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        typer.echo(f"Error: invalid YAML: {e}", err=True)
        raise typer.Exit(1)
    except ValidationError as e:
        typer.echo(f"Error: invalid config: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


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
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", "-f", help="Output format"),
):
    """Derive TPS targets given model, hardware, and RL config."""
    def _run():
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

        typer.echo(_format_output(report, fmt))

    _run_safely(_run)


@app.command()
def check(
    model: str = typer.Option(..., "--model", "-m", help="Model config YAML path"),
    hardware: str = typer.Option(..., "--hardware", "-hw", help="Hardware config YAML or shortname (910C, CM384)"),
    devices: int = typer.Option(..., "--devices", "-d"),
    prompts: int = typer.Option(10000, "--prompts", "-p"),
    group_size: int = typer.Option(8, "--group-size", "-g"),
    tp: int = typer.Option(8, "--tp"),
    pp: int = typer.Option(1, "--pp"),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", "-f", help="Output format"),
):
    """Quick feasibility check."""
    def _run():
        mc = load_model_config(model)
        hw = load_hardware_config(resolve_hardware(hardware))

        rl_cfg = RLConfig(total_prompts=prompts, group_size=group_size)
        dp = max(1, devices // (tp * pp))
        gen_parallel = ParallelismConfig(tp=tp, dp=max(1, devices // tp))
        train_parallel = ParallelismConfig(tp=tp, pp=pp, dp=dp)

        perf = RLPerformanceModel(mc, hw)
        report = perf.feasibility_check(devices, rl_cfg, gen_parallel, train_parallel)

        typer.echo(_format_output(report, fmt))

    _run_safely(_run)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (0.0.0.0 for LAN)"),
    port: int = typer.Option(7860, "--port", help="Port number"),
    share: bool = typer.Option(False, "--share", help="Create public Gradio link"),
):
    """Launch the web GUI."""
    from rl_perf.ui.app import launch
    launch(host=host, port=port, share=share)
