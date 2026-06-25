"""FastAPI backend for llm-perf web GUI.

Serves static files (index.html, styles.css, app.js) and provides REST
endpoints for model prediction, search, and configuration loading.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.model import LLMPerformanceModel
from llm_perf.search import pareto_search, sensitivity_sweep
from llm_perf.ui.hf_import import fetch_hf_config, hf_config_to_model_config

_STATIC_DIR = Path(__file__).parent / "static"
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

_MODEL_TEMPLATES = {
    "Llama-3.1-8B": "llama3_1_8b",
    "Qwen2.5-72B": "qwen2_5_72b",
    "Mistral-7B": "mistral_7b",
    "Qwen3-235B-MoE": "qwen3_235b_moe",
    "DeepSeekV3-671B": "deepseekv3_671b",
    "DeepSeek-V4": "deepseekv4",
}

# Alias the model names used in runtime YAMLs (which follow the demo naming)
# to their model stems, so /api/presets can resolve them even when they differ
# from the _MODEL_TEMPLATES keys used by the model-template dropdown.
_MODEL_ALIASES = {
    "DeepSeek-V3-671B": "deepseekv3_671b",
    "Qwen3-235B-A22B": "qwen3_235b_moe",
}

_HW_TEMPLATES = {
    "Ascend 910B": "ascend_910b",
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}

_NETWORK_TEMPLATES = {
    "HCCS+RoCE (910B)": "hccs_roce_910b",
    "HCCS+RoCE (910C)": "hccs_roce_910c",
    "CloudMatrix 384 Fullmesh": "cloudmatrix_384_fullmesh",
}

_RUNTIME_TEMPLATES = {
    "DeepSeekV3-671B (128x 910C)": "deepseekv3_671b_128x_910c",
    "Llama-3.1-8B (8x 910C)": "llama3_1_8b_8x_910c",
    "Mistral-7B (8x 910C)": "mistral_7b_8x_910c",
    "Qwen2.5-72B (32x 910C)": "qwen2_5_72b_32x_910c",
    "Qwen3-235B-MoE (64x 910C)": "qwen3_235b_moe_64x_910c",
}

app = FastAPI(title="llm-perf", docs_url=None, redoc_url=None)


# ── Pydantic models for request/response ──────────────────────────


class LayerInput(BaseModel):
    attention: str = "GQA"
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn: str = "SwiGLU"
    intermediate_size: int = 14336
    residual: str = "standard"
    num_experts: int = 1
    top_k: int = 1
    num_shared_experts: int = 0
    expert_intermediate_size: int = 0
    shared_intermediate_size: int = 0
    kv_compression_dim: int = 0
    query_compression_dim: int = 0
    rope_dim: int = 0
    window_size: int = 0
    mhc_expansion: int = 4
    # DSA (DeepSeek Sparse Attention) params
    compress_ratio: int = 0
    compress_c_kv: int = 0
    compress_coeff: float = 0.0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    q_lora_rank: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0


class ModelInput(BaseModel):
    name: str = "Llama-3.1-8B"
    hidden_size: int = 4096
    vocab_size: int = 128256
    num_layers: int = 32
    dtype: str = "bf16"
    layer: LayerInput = LayerInput()
    auxiliary: dict | None = None
    layers_summary: str | None = None


class ParallelismInput(BaseModel):
    tp: int = 1
    pp: int = 1
    dp: int = 8
    ep: int = 1
    cp: int = 1
    cp_type: str = "ring"
    sp: bool = False
    zero_stage: int = 0
    pp_schedule: str = "1f1b"
    recompute_attention: bool = False
    full_recomputation: bool = False
    optimizer_offload: bool = False
    activation_offload: bool = False
    param_offload: bool = False
    grad_offload: bool = False


class WorkloadInput(BaseModel):
    group_size: int = 8
    avg_prompt_len: int = 512
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: int | None = None
    train_micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 36
    gen_batch_size: int = 64
    reward_model: bool = False
    reference_model: bool = True
    ref_offload_cpu: bool = False
    use_speculative_decoding: bool = False
    mtp_acceptance_len: int | None = None


class PredictRequest(BaseModel):
    scenario: str = "post_training"  # "inference" | "pretraining" | "post_training"
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    network: str | None = None
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    gen_parallelism: ParallelismInput | None = None
    ref_parallelism: ParallelismInput | None = None
    rl: WorkloadInput = WorkloadInput()


class SearchConfig(BaseModel):
    mode: str = "pareto"
    device_counts: list[int] = [8, 16, 32, 64, 128]
    optimization_target: str = "step_time_seconds"
    sweep_param: str = "group_size"
    sweep_values: list[int] = [4, 8, 16, 32]


class SearchRequest(BaseModel):
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    network: str | None = None
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    gen_parallelism: ParallelismInput | None = None
    ref_parallelism: ParallelismInput | None = None
    rl: WorkloadInput = WorkloadInput()
    search: SearchConfig = SearchConfig()


class HFImportRequest(BaseModel):
    model_id: str


# ── Helpers ───────────────────────────────────────────────────────


def _expand_layers_summary(
    summary: str, default_layer: LayerConfig
) -> list[LayerConfig] | None:
    """Expand a layers_summary string into a list of LayerConfig objects.

    Supports two formats:
    - Old: '4x GQA+SwiGLU, 47x GQA+MoE E80'
    - New: '4x GQA+SwiGLU, 47x GQA+MoE E80/K8/SE1/EI1280/SI2560'

    The new format includes full MoE params (E=num_experts, K=top_k,
    SE=num_shared_experts, EI=expert_intermediate_size,
    SI=shared_intermediate_size) for accurate reconstruction.

    Only the FFN type and MoE params are taken from the summary string.
    The attention type is always inherited from default_layer (which reflects
    the user's UI selection), so changing attention in the UI takes effect
    even for mixed-layer models.

    Returns None if the summary cannot be parsed (fallback to default_layer).
    """
    import re

    parts = [p.strip() for p in summary.split(",")]
    layers = []
    for part in parts:
        # Pattern: "{count}x {attention}+{ffn}[ MoE params]"
        # Old MoE: "47x GQA+MoE E80"
        # New MoE: "47x GQA+MoE E80/K8/SE1/EI1280/SI2560"
        m = re.match(
            r"(\d+)x\s+(\w+)\+(\w+)"
            r"(?:\s+E(\d+)"           # E = num_experts
            r"(?:/K(\d+))?"           # K = top_k (optional)
            r"(?:/SE(\d+))?"          # SE = num_shared_experts (optional)
            r"(?:/EI(\d+))?"          # EI = expert_intermediate_size (optional)
            r"(?:/SI(\d+))?"          # SI = shared_intermediate_size (optional)
            r")?",
            part,
        )
        if not m:
            return None
        count = int(m.group(1))
        # attn from summary is ignored — user's UI selection takes priority
        ffn = m.group(3)
        n_experts = int(m.group(4)) if m.group(4) else None
        top_k = int(m.group(5)) if m.group(5) else None
        n_shared = int(m.group(6)) if m.group(6) else None
        ei = int(m.group(7)) if m.group(7) else None
        si = int(m.group(8)) if m.group(8) else None

        for _ in range(count):
            lc = default_layer.model_copy()
            # Keep lc.attention from default_layer (user's selection)
            lc.ffn = ffn
            if ffn == "MoE":
                if n_experts is not None:
                    lc.num_experts = n_experts
                if top_k is not None:
                    lc.top_k = top_k
                if n_shared is not None:
                    lc.num_shared_experts = n_shared
                if ei is not None:
                    lc.expert_intermediate_size = ei
                elif lc.expert_intermediate_size == 0 and lc.intermediate_size > 0:
                    lc.expert_intermediate_size = lc.intermediate_size
                if si is not None:
                    lc.shared_intermediate_size = si
                elif lc.shared_intermediate_size == 0 and lc.num_shared_experts > 0 and lc.intermediate_size > 0:
                    lc.shared_intermediate_size = lc.intermediate_size
            else:
                # Non-MoE layer: zero out MoE params
                lc.num_experts = 1
                lc.top_k = 1
                lc.num_shared_experts = 0
                lc.expert_intermediate_size = 0
                lc.shared_intermediate_size = 0
            layers.append(lc)
    return layers if layers else None


def _build_model_config(m: ModelInput) -> ModelConfig:
    layer = LayerConfig(
        attention=m.layer.attention,
        num_heads=m.layer.num_heads,
        num_kv_heads=m.layer.num_kv_heads,
        head_dim=m.layer.head_dim,
        ffn=m.layer.ffn,
        intermediate_size=m.layer.intermediate_size,
        residual=m.layer.residual,
        num_experts=m.layer.num_experts,
        top_k=m.layer.top_k,
        num_shared_experts=m.layer.num_shared_experts,
        expert_intermediate_size=m.layer.expert_intermediate_size or m.layer.intermediate_size,
        shared_intermediate_size=m.layer.shared_intermediate_size or m.layer.intermediate_size,
        kv_compression_dim=m.layer.kv_compression_dim,
        query_compression_dim=m.layer.query_compression_dim,
        rope_dim=m.layer.rope_dim,
        window_size=m.layer.window_size,
        mhc_expansion=m.layer.mhc_expansion,
        compress_ratio=m.layer.compress_ratio,
        compress_c_kv=m.layer.compress_c_kv,
        compress_coeff=m.layer.compress_coeff,
        index_n_heads=m.layer.index_n_heads,
        index_head_dim=m.layer.index_head_dim,
        index_topk=m.layer.index_topk,
        q_lora_rank=m.layer.q_lora_rank,
        o_lora_rank=m.layer.o_lora_rank,
        o_groups=m.layer.o_groups,
    )

    # Build layers list from layers_summary if available (mixed-layer models)
    layers = _expand_layers_summary(m.layers_summary, layer) if m.layers_summary else None

    return ModelConfig(
        name=m.name,
        hidden_size=m.hidden_size,
        vocab_size=m.vocab_size,
        num_layers=m.num_layers,
        dtype=m.dtype,
        default_layer=layer,
        layers=layers,
        auxiliary=m.auxiliary,
    )


def _load_yaml_raw(path: Path) -> dict:
    """Load a YAML file and return the raw dict."""
    import yaml as _yaml

    with open(path, encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


def _build_hw_config(hw_name: str, network_name: str | None = None) -> HardwareConfig:
    """Build a HardwareConfig by merging device specs from hardware/ and
    interconnect specs from network/.

    If *network_name* is not provided, auto-detect a matching network config
    based on the hardware name (e.g. "Ascend 910C" → "HCCS+RoCE (910C)").
    """
    stem = _HW_TEMPLATES.get(hw_name)
    if not stem:
        raise HTTPException(status_code=400, detail=f"Unknown hardware: {hw_name}")
    hw = load_hardware_config(str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml"))

    # Auto-detect network if not explicitly provided
    if not network_name:
        hw_key = stem.replace("ascend_", "").replace("cloudmatrix_", "")
        for net_name, net_stem in _NETWORK_TEMPLATES.items():
            if hw_key in net_stem:
                network_name = net_name
                break

    if network_name:
        net_stem = _NETWORK_TEMPLATES.get(network_name)
        if net_stem:
            net_data = _load_yaml_raw(_CONFIGS_DIR / "network" / f"{net_stem}.yaml")
            tiers = net_data.get("tiers", [])
            intra = next((t for t in tiers if t.get("name") == "intra_node"), tiers[0] if tiers else {})
            inter = next((t for t in tiers if t.get("name") == "inter_node"), tiers[-1] if len(tiers) > 1 else {})
            hw.intra_node_bw_gb_s = intra.get("bandwidth_gb_s", hw.intra_node_bw_gb_s)
            hw.intra_node_latency_us = intra.get("latency_us", hw.intra_node_latency_us)
            hw.inter_node_bw_gb_s = inter.get("bandwidth_gb_s", hw.inter_node_bw_gb_s)
            hw.inter_node_latency_us = inter.get("latency_us", hw.inter_node_latency_us)
            hw.devices_per_node = net_data.get("devices_per_node", hw.devices_per_node)

    return hw


def _build_parallelism(p: ParallelismInput) -> ParallelismConfig:
    return ParallelismConfig(
        tp=p.tp,
        pp=p.pp,
        dp=p.dp,
        ep=p.ep,
        cp=p.cp,
        cp_type=p.cp_type,
        sp=p.sp,
        zero_stage=p.zero_stage,
        pp_schedule=p.pp_schedule,
        recompute_attention=p.recompute_attention,
        full_recomputation=p.full_recomputation,
        optimizer_offload=p.optimizer_offload,
        activation_offload=p.activation_offload,
        param_offload=p.param_offload,
        grad_offload=p.grad_offload,
    )


def _build_rl_config(r: WorkloadInput) -> WorkloadConfig:
    std = r.std_response_len if r.std_response_len and r.std_response_len > 0 else None
    mtp = r.mtp_acceptance_len if r.use_speculative_decoding else None
    return WorkloadConfig(
        group_size=r.group_size,
        avg_prompt_len=r.avg_prompt_len,
        avg_response_len=r.avg_response_len,
        max_response_len=r.max_response_len,
        std_response_len=std,
        train_micro_batch_size=r.train_micro_batch_size,
        gradient_accumulation_steps=r.gradient_accumulation_steps,
        train_batch_size=r.train_batch_size,
        gen_batch_size=r.gen_batch_size,
        reward_model=r.reward_model,
        reference_model=r.reference_model,
        ref_offload_cpu=r.ref_offload_cpu,
        use_speculative_decoding=r.use_speculative_decoding,
        mtp_acceptance_len=mtp,
    )


def _topology_data(
    par: ParallelismInput, hw: HardwareConfig, num_layers: int
) -> list[dict]:
    """Compute rank mapping for topology visualization."""
    tp, ep, pp, dp = par.tp, par.ep, par.pp, par.dp
    total = tp * ep * pp * dp
    # Compute per-stage layer ranges (supports uneven PP splits, e.g. 51/8)
    base = num_layers // pp if pp > 0 else num_layers
    remainder = num_layers % pp if pp > 0 else 0
    stage_layer_start = []
    stage_layer_end = []
    offset = 0
    for i in range(pp):
        count = base + (1 if i < remainder else 0)
        stage_layer_start.append(offset)
        stage_layer_end.append(offset + count - 1)
        offset += count
    ranks = []
    for g in range(total):
        r = g
        tp_rank = r % tp
        r //= tp
        ep_rank = r % ep
        r //= ep
        pp_stage = r % pp
        r //= pp
        dp_rank = r
        ranks.append(
            {
                "global_rank": g,
                "node": g // hw.devices_per_node,
                "local_gpu": g % hw.devices_per_node,
                "tp_rank": tp_rank,
                "pp_stage": pp_stage,
                "dp_rank": dp_rank,
                "ep_rank": ep_rank,
                "layer_start": stage_layer_start[pp_stage],
                "layer_end": stage_layer_end[pp_stage],
            }
        )
    return ranks


# ── Endpoints ─────────────────────────────────────────────────────


def _layer_to_dict(layer: LayerConfig) -> dict:
    """Convert a LayerConfig to a JSON-serializable dict."""
    return {
        "attention": layer.attention,
        "num_heads": layer.num_heads,
        "num_kv_heads": layer.num_kv_heads,
        "head_dim": layer.head_dim,
        "ffn": layer.ffn,
        "intermediate_size": layer.intermediate_size,
        "residual": layer.residual,
        "num_experts": layer.num_experts,
        "top_k": layer.top_k,
        "num_shared_experts": layer.num_shared_experts,
        "expert_intermediate_size": layer.expert_intermediate_size,
        "shared_intermediate_size": layer.shared_intermediate_size,
        "kv_compression_dim": layer.kv_compression_dim,
        "query_compression_dim": layer.query_compression_dim,
        "rope_dim": layer.rope_dim,
        "window_size": layer.window_size,
        "mhc_expansion": layer.mhc_expansion,
        "compress_ratio": layer.compress_ratio,
        "compress_c_kv": layer.compress_c_kv,
        "compress_coeff": layer.compress_coeff,
        "index_n_heads": layer.index_n_heads,
        "index_head_dim": layer.index_head_dim,
        "index_topk": layer.index_topk,
        "q_lora_rank": layer.q_lora_rank,
        "o_lora_rank": layer.o_lora_rank,
        "o_groups": layer.o_groups,
    }


def _compute_layers_summary(mc: ModelConfig) -> str | None:
    """Compute a human-readable layers summary for mixed-layer models.

    Returns None for uniform-layer models (single default_layer).

    MoE groups include expert params for accurate reconstruction:
      "4x GQA+SwiGLU, 47x GQA+MoE E80/K8/SE1/EI1280/SI2560"
    where E=num_experts, K=top_k, SE=num_shared_experts,
    EI=expert_intermediate_size, SI=shared_intermediate_size.
    """
    if not mc.layers:
        return None
    parts = []
    i = 0
    while i < len(mc.layers):
        layer = mc.layers[i]
        j = i + 1
        while j < len(mc.layers) and _layer_signature(mc.layers[j]) == _layer_signature(layer):
            j += 1
        count = j - i
        if layer.ffn == "MoE":
            ffn_desc = (
                f"MoE E{layer.num_experts}/K{layer.top_k}"
                f"/SE{layer.num_shared_experts}"
                f"/EI{layer.expert_intermediate_size}"
                f"/SI{layer.shared_intermediate_size}"
            )
        else:
            ffn_desc = layer.ffn
        parts.append(f"{count}x {layer.attention}+{ffn_desc}")
        i = j
    return ", ".join(parts)


def _layer_signature(layer: LayerConfig) -> tuple:
    """Return a hashable signature for layer type comparison."""
    return (
        layer.attention,
        layer.ffn,
        layer.num_experts,
        layer.top_k,
        layer.num_shared_experts,
    )


@app.get("/api/presets")
def get_presets():
    """Load all runtime YAMLs from configs/runtime/, resolve references, return as dict."""
    presets = {}
    runtime_dir = _CONFIGS_DIR / "runtime"
    if runtime_dir.exists():
        for yaml_file in sorted(runtime_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue
            data = _load_yaml_raw(yaml_file)
            name = data.get("name", yaml_file.stem)

            # Resolve model reference: string → load from models/
            model_ref = data.get("model")
            if isinstance(model_ref, str):
                model_stem = _MODEL_TEMPLATES.get(model_ref) or _MODEL_ALIASES.get(model_ref)
                if model_stem:
                    model_path = _CONFIGS_DIR / "models" / f"{model_stem}.yaml"
                    if model_path.exists():
                        mc = load_model_config(str(model_path))
                        layer = mc.default_layer or (mc.layers[0] if mc.layers else LayerConfig())
                        model_entry = {
                            "name": mc.name,
                            "hidden_size": mc.hidden_size,
                            "vocab_size": mc.vocab_size,
                            "num_layers": mc.num_layers,
                            "dtype": mc.dtype,
                            "layer": _layer_to_dict(layer),
                        }
                        if mc.auxiliary:
                            model_entry["auxiliary"] = mc.auxiliary
                        layers_summary = _compute_layers_summary(mc)
                        if layers_summary:
                            model_entry["layers_summary"] = layers_summary
                        data["model"] = model_entry

            presets[name] = data
    return {"presets": presets}


@app.get("/api/models")
def get_models():
    templates = {}
    for display_name, stem in _MODEL_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "models" / f"{stem}.yaml"
        if yaml_path.exists():
            mc = load_model_config(str(yaml_path))
            # Use first layer from layers[] if default_layer is None
            if mc.default_layer:
                layer = mc.default_layer
            elif mc.layers:
                layer = mc.layers[0]
            else:
                layer = LayerConfig()
            entry = {
                "name": mc.name,
                "hidden_size": mc.hidden_size,
                "vocab_size": mc.vocab_size,
                "num_layers": mc.num_layers,
                "dtype": mc.dtype,
                "layer": _layer_to_dict(layer),
            }
            if mc.auxiliary:
                entry["auxiliary"] = mc.auxiliary
            layers_summary = _compute_layers_summary(mc)
            if layers_summary:
                entry["layers_summary"] = layers_summary
            templates[display_name] = entry
    return {"templates": templates}


@app.get("/api/hardware")
def get_hardware():
    profiles = {}
    for display_name, stem in _HW_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "hardware" / f"{stem}.yaml"
        if yaml_path.exists():
            hw = load_hardware_config(str(yaml_path))
            entry = {
                "hbm_gb": hw.hbm_capacity_gb,
                "tflops_bf16": hw.peak_tflops_bf16,
            }
            # Populate devices_per_node from the matching network config
            for net_name, net_stem in _NETWORK_TEMPLATES.items():
                net_path = _CONFIGS_DIR / "network" / f"{net_stem}.yaml"
                if net_path.exists():
                    net_data = _load_yaml_raw(net_path)
                    # Match by substring (e.g. "910C" in network name matches "910C" hardware)
                    hw_key = stem.replace("ascend_", "").replace("cloudmatrix_", "")
                    if hw_key in net_stem:
                        entry["devices_per_node"] = net_data.get("devices_per_node", 8)
                        break
            if "devices_per_node" not in entry:
                entry["devices_per_node"] = hw.devices_per_node
            profiles[display_name] = entry
    return {"profiles": profiles}


@app.get("/api/network")
def get_network():
    """Load all network YAMLs from configs/network/, return as dict keyed by name."""
    profiles = {}
    for display_name, stem in _NETWORK_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "network" / f"{stem}.yaml"
        if yaml_path.exists():
            data = _load_yaml_raw(yaml_path)
            tiers = data.get("tiers", [])
            profiles[display_name] = {
                "devices_per_node": data.get("devices_per_node", 8),
                "tiers": tiers,
            }
    return {"profiles": profiles}


@app.get("/api/runtime")
def get_runtime():
    """Load all runtime YAMLs from configs/runtime/, return as dict keyed by name."""
    profiles = {}
    for display_name, stem in _RUNTIME_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "runtime" / f"{stem}.yaml"
        if yaml_path.exists():
            data = _load_yaml_raw(yaml_path)
            profiles[display_name] = data
    return {"profiles": profiles}


def _predict_inference(req: PredictRequest):
    """Inference scenario: prefill + decode generation only."""
    model_cfg = _build_model_config(req.model)
    hw_cfg = _build_hw_config(req.hardware, req.network)
    rl_cfg = _build_rl_config(req.rl)
    # The single parallelism block edited in the UI maps to generation.
    gen_par_in = req.gen_parallelism or req.parallelism
    gen_par = _build_parallelism(gen_par_in)

    perf = LLMPerformanceModel(model_cfg, hw_cfg)
    r = perf.derive_inference(req.total_devices, rl_cfg, gen_par)
    topo = _topology_data(gen_par_in, hw_cfg, model_cfg.num_layers)

    return {
        "scenario": "inference",
        "kpis": {
            "gen_tps_target": round(r["gen_tps_target"], 0),
            "gen_samples_per_sec": round(r["gen_samples_per_sec"], 3),
            "gen_time_seconds": round(r["gen_time_seconds"], 2),
            "prefill_seconds": round(r["prefill_seconds"], 3),
            "decode_seconds": round(r["decode_seconds"], 2),
            "decode_ms_per_token": round(r["decode_ms_per_token"], 3),
            "kv_cache_gb": round(r["kv_cache_gb"], 2),
            "feasible": r["gen_feasible"],
        },
        "memory": {
            "gen_weight_gb": round(r["gen_weight_gb"], 2),
            "kv_cache_gb": round(r["kv_cache_gb"], 2),
            "total_gen_gb": round(r["total_gen_gb"], 2),
            "usable_hbm_gb": round(r["usable_hbm_gb"], 2),
            "gen_feasible": r["gen_feasible"],
        },
        "timeline": {
            "prefill_seconds": round(r["prefill_seconds"], 3),
            "decode_seconds": round(r["decode_seconds"], 2),
            "gen_seconds": round(r["gen_time_seconds"], 2),
        },
        "topology": {
            "ranks": topo,
            "tp": gen_par_in.tp,
            "pp": gen_par_in.pp,
            "dp": gen_par_in.dp,
            "ep": gen_par_in.ep,
        },
    }


def _predict_pretraining(req: PredictRequest):
    """Pretraining scenario: one fwd+bwd+optimizer step, no RL sub-steps."""
    model_cfg = _build_model_config(req.model)
    hw_cfg = _build_hw_config(req.hardware, req.network)
    rl_cfg = _build_rl_config(req.rl)
    train_par_in = req.parallelism
    train_par = _build_parallelism(train_par_in)

    perf = LLMPerformanceModel(model_cfg, hw_cfg)
    r = perf.derive_pretraining(req.total_devices, rl_cfg, train_par)
    topo = _topology_data(train_par_in, hw_cfg, model_cfg.num_layers)

    return {
        "scenario": "pretraining",
        "kpis": {
            "step_time_seconds": round(r["step_time_seconds"], 2),
            "train_tps_target": round(r["train_tps_target"], 0),
            "train_samples_per_sec": round(r["train_samples_per_sec"], 3),
            "total_train_gb": round(r["total_train_gb"], 2),
            "feasible": r["train_feasible"],
        },
        "memory": {
            "weight_gb": round(r["weight_gb"], 2),
            "grad_gb": round(r["grad_gb"], 2),
            "optimizer_gb": round(r["optimizer_gb"], 2),
            "activation_peak_gb": round(r["activation_peak_gb"], 2),
            "total_train_gb": round(r["total_train_gb"], 2),
            "usable_hbm_gb": round(r["usable_hbm_gb"], 2),
            "train_feasible": r["train_feasible"],
        },
        "timeline": {
            "breakdown": {k: round(v, 3) for k, v in r["breakdown"].items()},
        },
        "topology": {
            "ranks": topo,
            "tp": train_par_in.tp,
            "pp": train_par_in.pp,
            "dp": train_par_in.dp,
            "ep": train_par_in.ep,
        },
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        if req.scenario == "inference":
            return _predict_inference(req)
        if req.scenario == "pretraining":
            return _predict_pretraining(req)

        model_cfg = _build_model_config(req.model)
        hw_cfg = _build_hw_config(req.hardware, req.network)
        train_par = _build_parallelism(req.parallelism)
        rl_cfg = _build_rl_config(req.rl)

        # Use explicit gen/ref parallelism if provided, fall back to TP-derived
        if req.gen_parallelism:
            gen_par = _build_parallelism(req.gen_parallelism)
        else:
            gen_ep = req.parallelism.ep
            gen_dp = max(1, req.total_devices // req.parallelism.tp // gen_ep) if gen_ep > 0 else req.total_devices // req.parallelism.tp
            gen_par = ParallelismConfig(tp=req.parallelism.tp, pp=1, ep=gen_ep, dp=gen_dp)

        if req.ref_parallelism:
            ref_par = _build_parallelism(req.ref_parallelism)
        else:
            ref_dp = req.total_devices // req.parallelism.tp if req.parallelism.tp > 0 else 1
            ref_par = ParallelismConfig(tp=req.parallelism.tp, pp=1, dp=ref_dp)

        perf = LLMPerformanceModel(model_cfg, hw_cfg)
        report = perf.derive_targets(req.total_devices, rl_cfg, gen_par, train_par, ref_par)
        mem = report.memory

        gen_t = report.gen_time_seconds
        train_t = report.train_time_seconds
        ref_t = report.ref_time_seconds

        topo = _topology_data(req.parallelism, hw_cfg, model_cfg.num_layers)

        return {
            "scenario": "post_training",
            "kpis": {
                "step_time_seconds": round(report.step_time_seconds, 1),
                "gen_tps_target": round(report.gen_tps_target, 0),
                "train_tps_target": round(report.train_tps_target, 0),
                "ref_tps_target": round(report.ref_tps_target, 0),
                "gen_time_seconds": round(gen_t, 1),
                "train_time_seconds": round(train_t, 1),
                "ref_time_seconds": round(ref_t, 1),
                "reshard_gen_ref_seconds": round(report.reshard_gen_ref_seconds, 2),
                "reshard_ref_train_seconds": round(report.reshard_ref_train_seconds, 2),
                "feasible": report.feasible,
            },
            "memory": {
                "weight_gb": round(mem.weight_gb, 2),
                "grad_gb": round(mem.grad_gb, 2),
                "gen_weight_gb": round(mem.gen_weight_gb, 2),
                "ref_weight_gb": round(mem.ref_weight_gb, 2),
                "optimizer_gb": round(mem.optimizer_gb, 2),
                "activation_peak_gb": round(mem.activation_peak_gb, 2),
                "ref_model_gb": round(mem.ref_model_gb, 2),
                "ref_activation_peak_gb": round(mem.ref_activation_peak_gb, 2),
                "reward_model_gb": round(mem.reward_model_gb, 2),
                "kv_cache_gb": round(mem.kv_cache_gb, 2),
                "total_train_gb": round(mem.total_train_gb, 2),
                "total_gen_gb": round(mem.total_gen_gb, 2),
                "total_ref_gb": round(mem.total_ref_gb, 2),
                "usable_hbm_gb": round(mem.usable_hbm_gb, 2),
                "train_feasible": mem.train_feasible,
                "gen_feasible": mem.gen_feasible,
                "ref_feasible": mem.ref_feasible,
            },
            "timeline": {
                "gen_seconds": round(gen_t, 1),
                "train_seconds": round(train_t, 1),
                "ref_seconds": round(ref_t, 1),
                "reshard_gen_ref_seconds": round(report.reshard_gen_ref_seconds, 2),
                "reshard_ref_train_seconds": round(report.reshard_ref_train_seconds, 2),
                "step_time_seconds": round(report.step_time_seconds, 1),
            },
            "topology": {
                "ranks": topo,
                "tp": req.parallelism.tp,
                "pp": req.parallelism.pp,
                "dp": req.parallelism.dp,
                "ep": req.parallelism.ep,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


@app.post("/api/search")
def search(req: SearchRequest):
    try:
        model_cfg = _build_model_config(req.model)
        hw_cfg = _build_hw_config(req.hardware, req.network)
        rl_cfg = _build_rl_config(req.rl)
        perf = LLMPerformanceModel(model_cfg, hw_cfg)

        if req.search.mode == "pareto":
            sr = pareto_search(perf, hw_cfg, rl_cfg, req.search.device_counts)
            results = []
            for r in sr:
                tp_cfg = r.train_parallel
                results.append(
                    {
                        "devices": r.devices,
                        "parallelism": {
                            "tp": tp_cfg.tp,
                            "pp": tp_cfg.pp,
                            "dp": tp_cfg.dp,
                            "ep": tp_cfg.ep,
                            "cp": tp_cfg.cp,
                            "sp": tp_cfg.sp,
                        },
                        "step_time_seconds": round(r.report.step_time_seconds, 1),
                        "gen_tps": round(r.report.gen_tps_target, 0),
                        "train_tps": round(r.report.train_tps_target, 0),
                        "ref_tps": round(r.report.ref_tps_target, 0),
                        "feasible": r.is_feasible,
                        "is_pareto": r.is_pareto,
                        "is_oom": r.is_oom,
                    }
                )
            return {
                "results": results,
                "status": (f"Pareto search complete. {len(sr)} configs evaluated."),
            }

        else:
            tp_v = req.parallelism.tp
            train_par = _build_parallelism(req.parallelism)

            # Use explicit gen/ref parallelism if provided, fall back to
            # TP-derived defaults (same logic as /api/predict).
            if req.gen_parallelism:
                gen_par = _build_parallelism(req.gen_parallelism)
            else:
                gen_ep = train_par.ep
                gen_dp = max(1, req.total_devices // tp_v // gen_ep) if gen_ep > 0 else req.total_devices // tp_v
                gen_par = ParallelismConfig(
                    tp=tp_v, pp=1, ep=gen_ep, dp=gen_dp
                )

            if req.ref_parallelism:
                ref_par = _build_parallelism(req.ref_parallelism)
            else:
                ref_dp = req.total_devices // tp_v if tp_v > 0 else 1
                ref_par = ParallelismConfig(tp=tp_v, pp=1, dp=ref_dp)

            sweep = sensitivity_sweep(
                perf,
                hw_cfg,
                rl_cfg,
                param_name=req.search.sweep_param,
                values=req.search.sweep_values,
                total_devices=req.total_devices,
                gen_parallel=gen_par,
                train_parallel=train_par,
                ref_parallel=ref_par,
            )
            results = []
            for val, sr in zip(req.search.sweep_values, sweep):
                results.append(
                    {
                        "devices": sr.devices,
                        "parallelism": {
                            "tp": train_par.tp,
                            "pp": train_par.pp,
                            "dp": train_par.dp,
                            "ep": train_par.ep,
                            "cp": train_par.cp,
                            "sp": train_par.sp,
                        },
                        "step_time_seconds": round(sr.report.step_time_seconds, 1),
                        "gen_tps": round(sr.report.gen_tps_target, 0),
                        "train_tps": round(sr.report.train_tps_target, 0),
                        "ref_tps": round(sr.report.ref_tps_target, 0),
                        "feasible": sr.is_feasible,
                        "is_pareto": False,
                        "is_oom": sr.is_oom,
                        "sweep_value": val,
                    }
                )
            return {
                "results": results,
                "status": (
                    f"Sensitivity sweep complete. {len(sweep)} values evaluated."
                ),
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


@app.post("/api/hf-import")
def hf_import(req: HFImportRequest):
    try:
        hf_cfg = fetch_hf_config(req.model_id)
        mc = hf_config_to_model_config(hf_cfg, name=req.model_id)
        layer = mc.default_layer or (mc.layers[0] if mc.layers else LayerConfig())
        result = {
            "name": mc.name,
            "hidden_size": mc.hidden_size,
            "vocab_size": mc.vocab_size,
            "num_layers": mc.num_layers,
            "dtype": mc.dtype,
            "layer": _layer_to_dict(layer),
        }
        if mc.auxiliary:
            result["auxiliary"] = mc.auxiliary
        layers_summary = _compute_layers_summary(mc)
        if layers_summary:
            result["layers_summary"] = layers_summary
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


# ── Static files & SPA fallback ───────────────────────────────────

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


def launch(host: str = "127.0.0.1", port: int = 7860):
    """Launch the web GUI."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)