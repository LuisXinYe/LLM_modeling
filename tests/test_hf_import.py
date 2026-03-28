"""Tests for HuggingFace config.json → ModelConfig converter."""

from __future__ import annotations

import pytest

from rl_perf.ui.hf_import import SUPPORTED_ARCHITECTURES, hf_config_to_model_config


# ---------------------------------------------------------------------------
# Fixtures: minimal HF config.json dicts
# ---------------------------------------------------------------------------

LLAMA_HF = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 128256,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "torch_dtype": "bfloat16",
}

MISTRAL_SLIDING_HF = {
    "architectures": ["MistralForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 32000,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "sliding_window": 4096,
    "torch_dtype": "bfloat16",
}

MIXTRAL_HF = {
    "architectures": ["MixtralForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 32000,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "torch_dtype": "bfloat16",
}

QWEN2_HF = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "vocab_size": 152064,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 29568,
    "torch_dtype": "bfloat16",
}

DEEPSEEK_HF = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "vocab_size": 129280,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "head_dim": 64,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "torch_dtype": "bfloat16",
}


# ---------------------------------------------------------------------------
# SUPPORTED_ARCHITECTURES
# ---------------------------------------------------------------------------


def test_supported_architectures_contains_all_five():
    expected = {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "MistralForCausalLM",
        "DeepseekV3ForCausalLM",
        "MixtralForCausalLM",
    }
    assert expected == set(SUPPORTED_ARCHITECTURES)
    assert len(SUPPORTED_ARCHITECTURES) == 5


# ---------------------------------------------------------------------------
# LlamaForCausalLM → GQA + SwiGLU
# ---------------------------------------------------------------------------


def test_llama_basic_conversion():
    cfg = hf_config_to_model_config(LLAMA_HF, name="Llama-3-8B")
    assert cfg.name == "Llama-3-8B"
    assert cfg.hidden_size == 4096
    assert cfg.num_layers == 32
    assert cfg.vocab_size == 128256
    assert cfg.dtype == "bf16"


def test_llama_layer_gqa_swiglu():
    cfg = hf_config_to_model_config(LLAMA_HF)
    layer = cfg.default_layer
    assert layer is not None
    assert layer.attention == "GQA"
    assert layer.ffn == "SwiGLU"
    assert layer.num_heads == 32
    assert layer.num_kv_heads == 8
    assert layer.head_dim == 128
    assert layer.intermediate_size == 14336


def test_llama_mha_when_kv_equals_heads():
    """When num_kv_heads == num_heads, use MHA."""
    hf = {**LLAMA_HF, "num_key_value_heads": 32}
    cfg = hf_config_to_model_config(hf)
    assert cfg.default_layer.attention == "MHA"


# ---------------------------------------------------------------------------
# MistralForCausalLM → SWA when sliding_window present
# ---------------------------------------------------------------------------


def test_mistral_sliding_window():
    cfg = hf_config_to_model_config(MISTRAL_SLIDING_HF)
    layer = cfg.default_layer
    assert layer.attention == "SWA"
    assert layer.window_size == 4096
    assert layer.ffn == "SwiGLU"


def test_mistral_no_sliding_window_is_gqa():
    hf = {k: v for k, v in MISTRAL_SLIDING_HF.items() if k != "sliding_window"}
    cfg = hf_config_to_model_config(hf)
    assert cfg.default_layer.attention == "GQA"
    assert cfg.default_layer.window_size == 0


# ---------------------------------------------------------------------------
# MixtralForCausalLM → GQA + MoE
# ---------------------------------------------------------------------------


def test_mixtral_moe():
    cfg = hf_config_to_model_config(MIXTRAL_HF)
    layer = cfg.default_layer
    assert layer.attention == "GQA"
    assert layer.ffn == "MoE"
    assert layer.num_experts == 8
    assert layer.top_k == 2
    assert layer.expert_intermediate_size == 14336


# ---------------------------------------------------------------------------
# Qwen2ForCausalLM → GQA + SwiGLU
# ---------------------------------------------------------------------------


def test_qwen2_conversion():
    cfg = hf_config_to_model_config(QWEN2_HF, name="Qwen2-72B")
    assert cfg.hidden_size == 8192
    assert cfg.num_layers == 80
    assert cfg.vocab_size == 152064
    layer = cfg.default_layer
    assert layer.attention == "GQA"
    assert layer.ffn == "SwiGLU"
    assert layer.num_heads == 64
    assert layer.num_kv_heads == 8
    assert layer.intermediate_size == 29568


# ---------------------------------------------------------------------------
# DeepseekV3ForCausalLM → MLA + MoE
# ---------------------------------------------------------------------------


def test_deepseek_mla_moe():
    cfg = hf_config_to_model_config(DEEPSEEK_HF, name="DeepSeek-V3")
    layer = cfg.default_layer
    assert layer.attention == "MLA"
    assert layer.ffn == "MoE"
    assert layer.kv_compression_dim == 512
    assert layer.query_compression_dim == 1536
    assert layer.rope_dim == 64
    assert layer.num_experts == 256
    assert layer.top_k == 8
    assert layer.num_shared_experts == 1


# ---------------------------------------------------------------------------
# Unsupported architecture → ValueError
# ---------------------------------------------------------------------------


def test_unsupported_architecture_raises():
    hf = {**LLAMA_HF, "architectures": ["GPT2LMHeadModel"]}
    with pytest.raises(ValueError, match="Unsupported"):
        hf_config_to_model_config(hf)


def test_missing_architectures_key_raises():
    hf = {k: v for k, v in LLAMA_HF.items() if k != "architectures"}
    with pytest.raises((ValueError, KeyError)):
        hf_config_to_model_config(hf)
