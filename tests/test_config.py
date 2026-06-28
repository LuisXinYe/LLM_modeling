"""Tests for llm_perf.config module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    Phase,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_layer_config_defaults():
    layer = LayerConfig()
    assert layer.attention == "GQA"
    assert layer.num_heads == 64
    assert layer.num_kv_heads == 8
    assert layer.head_dim == 128
    assert layer.ffn == "SwiGLU"
    assert layer.num_experts == 1


def test_model_config_expands_layers():
    layer = LayerConfig(num_heads=16, num_kv_heads=4)
    model = ModelConfig(
        name="test-model",
        hidden_size=2048,
        vocab_size=32000,
        num_layers=4,
        default_layer=layer,
    )
    layers = model.get_layers()
    assert len(layers) == 4
    assert all(layer.num_heads == 16 for layer in layers)


def test_model_config_raises_without_layers():
    model = ModelConfig(
        name="test-model",
        hidden_size=2048,
        vocab_size=32000,
        num_layers=4,
    )
    with pytest.raises(ValueError, match="Must provide default_layer or layers"):
        model.get_layers()


def test_model_config_dtype_bytes():
    model = ModelConfig(
        name="test", hidden_size=512, vocab_size=1000, num_layers=2, dtype="bf16"
    )
    assert model.dtype_bytes == 2

    model_fp8 = ModelConfig(
        name="test", hidden_size=512, vocab_size=1000, num_layers=2, dtype="fp8"
    )
    assert model_fp8.dtype_bytes == 1


def test_hardware_config_usable_hbm():
    hw = HardwareConfig(
        name="Test HW",
        peak_tflops_bf16=400.0,
        hbm_capacity_gb=128.0,
        hbm_bandwidth_tb_s=2.0,
        hbm_usable_ratio=0.85,
    )
    assert hw.usable_hbm_gb == pytest.approx(128 * 0.85)


def test_hardware_config_default_calibration():
    hw = HardwareConfig(
        name="Test HW",
        peak_tflops_bf16=400.0,
        hbm_capacity_gb=80.0,
        hbm_bandwidth_tb_s=2.0,
    )
    assert hw.calibration.compute_eff_large_gemm == pytest.approx(0.50)
    assert hw.calibration.memory_efficiency == pytest.approx(0.70)


def test_parallelism_config_total_devices():
    p = ParallelismConfig(tp=8, pp=4, dp=4, ep=1)
    assert p.total_devices == 128


def test_parallelism_config_defaults():
    p = ParallelismConfig()
    assert p.total_devices == 1
    assert p.zero_stage == 0
    assert p.sp is False


def test_phase_enum():
    assert Phase.PREFILL.value == "prefill"
    assert Phase.DECODE.value == "decode"
    assert Phase.TRAIN_FWD.value == "train_fwd"
    assert Phase.TRAIN_BWD.value == "train_bwd"


def test_rl_config_defaults():
    rl = WorkloadConfig()
    assert rl.group_size == 16
    assert rl.avg_prompt_len == 2048
    assert rl.avg_response_len == 2048
    assert rl.reference_model is True


def test_load_model_config_yaml():
    config_data = {
        "name": "Test-Model",
        "hidden_size": 1024,
        "vocab_size": 50000,
        "num_layers": 8,
        "dtype": "fp16",
        "default_layer": {
            "attention": "MHA",
            "num_heads": 16,
            "num_kv_heads": 16,
            "head_dim": 64,
            "ffn": "SwiGLU",
            "intermediate_size": 4096,
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config_data, tmp)
        tmp_path = tmp.name

    model = load_model_config(tmp_path)
    assert model.name == "Test-Model"
    assert model.hidden_size == 1024
    assert model.num_layers == 8
    assert model.dtype == "fp16"
    assert model.dtype_bytes == 2
    layers = model.get_layers()
    assert len(layers) == 8
    assert layers[0].attention == "MHA"


# ---------------------------------------------------------------------------
# Integration tests using real YAML files
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_load_real_llama_config():
    path = CONFIGS_DIR / "models" / "llama3_1_8b.yaml"
    model = load_model_config(str(path))
    assert model.name == "Llama-3.1-8B"
    assert model.hidden_size == 4096
    assert model.vocab_size == 128256
    assert model.num_layers == 32
    assert model.dtype == "bf16"
    assert model.dtype_bytes == 2
    layers = model.get_layers()
    assert len(layers) == 32
    assert layers[0].attention == "GQA"
    assert layers[0].num_heads == 32
    assert layers[0].num_kv_heads == 8
    assert layers[0].intermediate_size == 14336


def test_load_real_hardware_config():
    path = CONFIGS_DIR / "hardware" / "ascend_910c.yaml"
    hw = load_hardware_config(str(path))
    assert hw.name == "Ascend 910C"
    assert hw.peak_tflops_bf16 == pytest.approx(800.0)
    assert hw.hbm_capacity_gb == pytest.approx(128.0)
    assert hw.hbm_bandwidth_tb_s == pytest.approx(3.2)
    assert hw.usable_hbm_gb == pytest.approx(128 * 0.85)
    assert hw.devices_per_node == 8
    assert hw.calibration.compute_eff_large_gemm == pytest.approx(0.50)
    assert hw.calibration.comm_efficiency == pytest.approx(0.70)


@pytest.mark.parametrize("name", [
    "llama3_1_8b", "qwen2_5_72b", "mistral_7b", "qwen3_235b_moe", "deepseekv3_671b",
])
def test_load_all_model_configs(name):
    mc = load_model_config(str(CONFIGS_DIR / "models" / f"{name}.yaml"))
    assert mc.num_layers > 0
    assert len(mc.get_layers()) == mc.num_layers


def test_rlconfig_speculative_decoding_defaults():
    cfg = WorkloadConfig(total_prompts=1000)
    assert cfg.use_speculative_decoding is False
    assert cfg.mtp_acceptance_len is None


def test_rlconfig_speculative_decoding_set():
    cfg = WorkloadConfig(total_prompts=1000, use_speculative_decoding=True, mtp_acceptance_len=3)
    assert cfg.use_speculative_decoding is True
    assert cfg.mtp_acceptance_len == 3


# ---------------------------------------------------------------------------
# ParallelismConfig validation
# ---------------------------------------------------------------------------


def test_parallelism_config_tp_positive():
    with pytest.raises(Exception):
        ParallelismConfig(tp=0)


def test_parallelism_config_pp_positive():
    with pytest.raises(Exception):
        ParallelismConfig(pp=0)


def test_parallelism_config_dp_positive():
    with pytest.raises(Exception):
        ParallelismConfig(dp=0)


def test_parallelism_config_ep_positive():
    with pytest.raises(Exception):
        ParallelismConfig(ep=0)


def test_parallelism_config_valid():
    cfg = ParallelismConfig(tp=4, pp=2, dp=4, ep=1)
    assert cfg.tp == 4


def test_moe_sparse_knobs_default_and_validate():
    from llm_perf.config import ParallelismConfig
    import pytest

    # Defaults reproduce current behavior
    p = ParallelismConfig(ep=8)
    assert p.moe_node_limit == 0
    assert p.moe_imbalance_factor == 1.0

    # Accepts explicit values
    p2 = ParallelismConfig(ep=64, moe_node_limit=4, moe_imbalance_factor=1.3)
    assert p2.moe_node_limit == 4
    assert p2.moe_imbalance_factor == 1.3

    # node_limit must be >= 0
    with pytest.raises(ValueError):
        ParallelismConfig(moe_node_limit=-1)

    # imbalance_factor must be >= 1.0
    with pytest.raises(ValueError):
        ParallelismConfig(moe_imbalance_factor=0.9)
