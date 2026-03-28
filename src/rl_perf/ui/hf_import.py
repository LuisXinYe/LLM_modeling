"""HuggingFace config.json → ModelConfig converter.

Supports importing model architectures from HuggingFace Hub and converting
their configuration dictionaries into rl_perf ModelConfig objects.
"""

from __future__ import annotations

from rl_perf.config import LayerConfig, ModelConfig

SUPPORTED_ARCHITECTURES: list[str] = [
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "DeepseekV3ForCausalLM",
    "MixtralForCausalLM",
]

_DTYPE_MAP = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8": "fp8",
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp8": "fp8",
}


def _dtype(hf: dict) -> str:
    raw = hf.get("torch_dtype", "bfloat16")
    return _DTYPE_MAP.get(str(raw), "bf16")


def _head_dim(hf: dict) -> int:
    """Resolve head_dim from HF config; fall back to hidden_size // num_heads."""
    if "head_dim" in hf:
        return int(hf["head_dim"])
    return int(hf["hidden_size"]) // int(hf["num_attention_heads"])


def _convert_llama_qwen2(hf: dict) -> LayerConfig:
    """LlamaForCausalLM / Qwen2ForCausalLM: GQA (or MHA) + SwiGLU."""
    num_heads = int(hf["num_attention_heads"])
    num_kv_heads = int(hf.get("num_key_value_heads", num_heads))
    attention = "MHA" if num_kv_heads == num_heads else "GQA"
    return LayerConfig(
        attention=attention,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_head_dim(hf),
        ffn="SwiGLU",
        intermediate_size=int(hf["intermediate_size"]),
    )


def _convert_mistral(hf: dict) -> LayerConfig:
    """MistralForCausalLM: GQA + SwiGLU; SWA when sliding_window present."""
    num_heads = int(hf["num_attention_heads"])
    num_kv_heads = int(hf.get("num_key_value_heads", num_heads))
    sliding_window = hf.get("sliding_window")
    attention = (
        "SWA" if sliding_window else ("MHA" if num_kv_heads == num_heads else "GQA")
    )
    window_size = int(sliding_window) if sliding_window else 0
    return LayerConfig(
        attention=attention,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_head_dim(hf),
        ffn="SwiGLU",
        intermediate_size=int(hf["intermediate_size"]),
        window_size=window_size,
    )


def _convert_mixtral(hf: dict) -> LayerConfig:
    """MixtralForCausalLM: GQA + MoE."""
    num_heads = int(hf["num_attention_heads"])
    num_kv_heads = int(hf.get("num_key_value_heads", num_heads))
    attention = "MHA" if num_kv_heads == num_heads else "GQA"
    num_experts = int(hf["num_local_experts"])
    top_k = int(hf["num_experts_per_tok"])
    # Mixtral uses intermediate_size for the per-expert hidden dim
    expert_intermediate_size = int(hf["intermediate_size"])
    return LayerConfig(
        attention=attention,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_head_dim(hf),
        ffn="MoE",
        intermediate_size=0,
        num_experts=num_experts,
        top_k=top_k,
        expert_intermediate_size=expert_intermediate_size,
    )


def _convert_deepseek_v3(hf: dict) -> LayerConfig:
    """DeepseekV3ForCausalLM: MLA + MoE."""
    num_heads = int(hf["num_attention_heads"])
    num_kv_heads = int(hf.get("num_key_value_heads", num_heads))
    kv_compression_dim = int(hf.get("kv_lora_rank", 0))
    query_compression_dim = int(hf.get("q_lora_rank", 0))
    rope_dim = int(hf.get("qk_rope_head_dim", 0))
    num_experts = int(hf.get("n_routed_experts", 1))
    top_k = int(hf.get("num_experts_per_tok", 1))
    num_shared_experts = int(hf.get("n_shared_experts", 0))
    # DeepSeek uses moe_intermediate_size for expert hidden dim
    expert_intermediate_size = int(hf.get("moe_intermediate_size", 0))
    # intermediate_size may be for shared experts or dense fallback
    shared_intermediate_size = (
        int(hf.get("intermediate_size", 0)) if num_shared_experts else 0
    )
    return LayerConfig(
        attention="MLA",
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=_head_dim(hf),
        ffn="MoE",
        intermediate_size=0,
        kv_compression_dim=kv_compression_dim,
        query_compression_dim=query_compression_dim,
        rope_dim=rope_dim,
        num_experts=num_experts,
        top_k=top_k,
        num_shared_experts=num_shared_experts,
        expert_intermediate_size=expert_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
    )


_ARCH_CONVERTERS = {
    "LlamaForCausalLM": _convert_llama_qwen2,
    "Qwen2ForCausalLM": _convert_llama_qwen2,
    "MistralForCausalLM": _convert_mistral,
    "MixtralForCausalLM": _convert_mixtral,
    "DeepseekV3ForCausalLM": _convert_deepseek_v3,
}


def hf_config_to_model_config(hf: dict, name: str = "") -> ModelConfig:
    """Convert a HuggingFace config.json dict to a ModelConfig.

    Args:
        hf: Dictionary parsed from HuggingFace config.json.
        name: Optional human-readable model name. Defaults to the architecture
              name if not provided.

    Returns:
        A ModelConfig populated from the HuggingFace configuration.

    Raises:
        ValueError: If the architecture is not in SUPPORTED_ARCHITECTURES.
        KeyError: If required fields are missing from the HF config dict.
    """
    architectures = hf["architectures"]
    arch = architectures[0] if architectures else ""

    if arch not in _ARCH_CONVERTERS:
        raise ValueError(
            f"Unsupported architecture: {arch!r}. Supported: {SUPPORTED_ARCHITECTURES}"
        )

    layer = _ARCH_CONVERTERS[arch](hf)
    model_name = name or arch

    return ModelConfig(
        name=model_name,
        hidden_size=int(hf["hidden_size"]),
        vocab_size=int(hf["vocab_size"]),
        num_layers=int(hf["num_hidden_layers"]),
        dtype=_dtype(hf),
        default_layer=layer,
    )


def fetch_hf_config(model_id: str) -> dict:
    """Download config.json from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B").

    Returns:
        Parsed config.json as a dictionary.

    Raises:
        ImportError: If huggingface_hub is not installed.
        Exception: If the model cannot be fetched from HuggingFace Hub.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for fetching HF configs. "
            "Install it with: pip install huggingface-hub"
        ) from e

    import json

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path) as f:
        return json.load(f)
