from typer.testing import CliRunner
from rl_perf.cli import app

runner = CliRunner()


def test_targets_help():
    result = runner.invoke(app, ["targets", "--help"])
    assert result.exit_code == 0
    assert "model" in result.output.lower()


def test_targets_basic():
    result = runner.invoke(app, [
        "targets",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "configs/hardware/ascend_910c.yaml",
        "--devices", "64", "--prompts", "1000", "--group-size", "8",
        "--time-budget", "24",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "TPS" in result.output or "tokens" in result.output.lower()


def test_check_basic():
    result = runner.invoke(app, [
        "check",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "configs/hardware/ascend_910c.yaml",
        "--devices", "64", "--prompts", "1000",
    ])
    assert result.exit_code == 0, f"CLI failed: {result.output}"
