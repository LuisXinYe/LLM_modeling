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


def test_targets_missing_model_file():
    """Missing model file should give friendly error, not traceback."""
    result = runner.invoke(app, [
        "targets",
        "--model", "nonexistent.yaml",
        "--hardware", "910C",
        "--devices", "64", "--prompts", "1000",
    ])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_targets_json_format():
    """--format json should output valid JSON."""
    import json as json_mod
    result = runner.invoke(app, [
        "targets",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "910C",
        "--devices", "64", "--prompts", "1000",
        "--format", "json",
    ])
    assert result.exit_code == 0
    parsed = json_mod.loads(result.output)
    assert "epoch_time_hours" in parsed


def test_check_json_format():
    import json as json_mod
    result = runner.invoke(app, [
        "check",
        "--model", "configs/models/llama3_1_8b.yaml",
        "--hardware", "910C",
        "--devices", "64",
        "--format", "json",
    ])
    assert result.exit_code == 0
    parsed = json_mod.loads(result.output)
    assert "epoch_time_hours" in parsed


def test_check_hardware_help():
    """check --help should mention shortnames."""
    result = runner.invoke(app, ["check", "--help"])
    assert "910C" in result.output
