"""FastAPI backend for rl-perf web GUI.

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

from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
    VisionEncoderConfig,
    load_hardware_config,
    load_model_config,
)
from rl_perf.model import RLPerformanceModel
from rl_perf.search import pareto_search, sensitivity_sweep
from rl_perf.ui.hf_import import fetch_hf_config, hf_config_to_model_config

_STATIC_DIR = Path(__file__).parent / "static"
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

_MODEL_TEMPLATES = {
    "Llama-3.1-8B": "llama3_1_8b",
    "Qwen2.5-72B": "qwen2_5_72b",
    "Mistral-7B": "mistral_7b",
    "Qwen3-235B-MoE": "qwen3_235b_moe",
    "DeepSeekV3-671B": "deepseekv3_671b",
    "Pangu-74B-VL": "pangu_74b_vl",
}

_HW_TEMPLATES = {
    "Ascend 910B": "ascend_910b",
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}

app = FastAPI(title="rl-perf", docs_url=None, redoc_url=None)


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


class VisionEncoderInput(BaseModel):
    hidden_size: int = 1536
    num_layers: int = 26
    num_heads: int = 16
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 4608
    ffn: str = "SwiGLU"
    anyres_max_pixels: int = 1806336
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    mm_vision_token_down_size: int = 2


class ModelInput(BaseModel):
    name: str = "Llama-3.1-8B"
    hidden_size: int = 4096
    vocab_size: int = 128256
    num_layers: int = 32
    dtype: str = "bf16"
    layer: LayerInput = LayerInput()
    vision_encoder: VisionEncoderInput | None = None
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


class RLInput(BaseModel):
    group_size: int = 8
    avg_prompt_len: int = 512
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: int | None = None
    train_micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 36
    gen_batch_size: int = 64
    algorithm: str = "grpo"
    ppo_epochs: int = 1
    reward_model: bool = False
    colocated: bool = True
    online_rollout: bool = True
    reference_model: bool = True
    ref_offload_cpu: bool = False
    use_speculative_decoding: bool = False
    mtp_acceptance_len: int | None = None


class PredictRequest(BaseModel):
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    gen_parallelism: ParallelismInput | None = None
    ref_parallelism: ParallelismInput | None = None
    rl: RLInput = RLInput()


class SearchConfig(BaseModel):
    mode: str = "pareto"
    device_counts: list[int] = [8, 16, 32, 64, 128]
    optimization_target: str = "step_time_seconds"
    sweep_param: str = "group_size"
    sweep_values: list[int] = [4, 8, 16, 32]


class SearchRequest(BaseModel):
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    gen_parallelism: ParallelismInput | None = None
    ref_parallelism: ParallelismInput | None = None
    rl: RLInput = RLInput()
    search: SearchConfig = SearchConfig()


class HFImportRequest(BaseModel):
    model_id: str


# ── Helpers ───────────────────────────────────────────────────────


def _expand_layers_summary(
    summary: str, default_layer: LayerConfig
) -> list[LayerConfig] | None:
    """Expand a layers_summary string like '4x GQA+SwiGLU, 47x GQA+MoE E80' into a
    list of LayerConfig objects.

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
        # Pattern: "{count}x {attention}+{ffn}[ E{num_experts}]"
        m = re.match(r"(\d+)x\s+(\w+)\+(\w+)(?:\s+E(\d+))?", part)
        if not m:
            return None
        count = int(m.group(1))
        # attn from summary is ignored — user's UI selection takes priority
        ffn = m.group(3)
        n_experts = int(m.group(4)) if m.group(4) else None

        for _ in range(count):
            lc = default_layer.model_copy()
            # Keep lc.attention from default_layer (user's selection)
            lc.ffn = ffn
            if ffn == "MoE":
                if n_experts is not None:
                    lc.num_experts = n_experts
                # MoE layers need expert_intermediate_size; if zero, fall back to
                # the dense intermediate_size.
                if lc.expert_intermediate_size == 0 and lc.intermediate_size > 0:
                    lc.expert_intermediate_size = lc.intermediate_size
                if lc.shared_intermediate_size == 0 and lc.num_shared_experts > 0 and lc.intermediate_size > 0:
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
    )
    ve_cfg = None
    if m.vision_encoder:
        ve = m.vision_encoder
        ve_cfg = VisionEncoderConfig(
            hidden_size=ve.hidden_size,
            num_layers=ve.num_layers,
            num_heads=ve.num_heads,
            num_kv_heads=ve.num_kv_heads,
            head_dim=ve.head_dim,
            ffn=ve.ffn,
            intermediate_size=ve.intermediate_size,
            anyres_max_pixels=ve.anyres_max_pixels,
            patch_size=ve.patch_size,
            temporal_patch_size=ve.temporal_patch_size,
            merge_size=ve.merge_size,
            mm_vision_token_down_size=ve.mm_vision_token_down_size,
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
        vision_encoder=ve_cfg,
        auxiliary=m.auxiliary,
    )


def _build_hw_config(hw_name: str) -> HardwareConfig:
    stem = _HW_TEMPLATES.get(hw_name)
    if not stem:
        raise HTTPException(status_code=400, detail=f"Unknown hardware: {hw_name}")
    return load_hardware_config(str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml"))


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


def _build_rl_config(r: RLInput) -> RLConfig:
    std = r.std_response_len if r.std_response_len and r.std_response_len > 0 else None
    mtp = r.mtp_acceptance_len if r.use_speculative_decoding else None
    return RLConfig(
        group_size=r.group_size,
        avg_prompt_len=r.avg_prompt_len,
        avg_response_len=r.avg_response_len,
        max_response_len=r.max_response_len,
        std_response_len=std,
        train_micro_batch_size=r.train_micro_batch_size,
        gradient_accumulation_steps=r.gradient_accumulation_steps,
        train_batch_size=r.train_batch_size,
        gen_batch_size=r.gen_batch_size,
        algorithm=r.algorithm,
        ppo_epochs=r.ppo_epochs,
        reward_model=r.reward_model,
        colocated=r.colocated,
        online_rollout=r.online_rollout,
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
    }


def _ve_to_dict(ve: VisionEncoderConfig) -> dict:
    """Convert a VisionEncoderConfig to a JSON-serializable dict."""
    return {
        "hidden_size": ve.hidden_size,
        "num_layers": ve.num_layers,
        "num_heads": ve.num_heads,
        "num_kv_heads": ve.num_kv_heads,
        "head_dim": ve.head_dim,
        "intermediate_size": ve.intermediate_size,
        "ffn": ve.ffn,
        "anyres_max_pixels": ve.anyres_max_pixels,
        "patch_size": ve.patch_size,
        "temporal_patch_size": ve.temporal_patch_size,
        "merge_size": ve.merge_size,
        "mm_vision_token_down_size": ve.mm_vision_token_down_size,
        "image_seq_len": ve.image_seq_len(),
    }


def _compute_layers_summary(mc: ModelConfig) -> str | None:
    """Compute a human-readable layers summary for mixed-layer models.

    Returns None for uniform-layer models (single default_layer).
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
        ffn_desc = f"{layer.ffn}" + (f" E{layer.num_experts}" if layer.ffn == "MoE" else "")
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
    """Load all preset YAMLs from configs/presets/, return as dict keyed by name."""
    import yaml

    presets = {}
    presets_dir = _CONFIGS_DIR / "presets"
    if presets_dir.exists():
        for yaml_file in sorted(presets_dir.glob("*.yaml")):
            if yaml_file.name.startswith("_"):
                continue
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            presets[data.get("name", yaml_file.stem)] = data
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
            if mc.vision_encoder:
                entry["vision_encoder"] = _ve_to_dict(mc.vision_encoder)
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
            profiles[display_name] = {
                "devices_per_node": hw.devices_per_node,
                "hbm_gb": hw.hbm_capacity_gb,
                "tflops_bf16": hw.peak_tflops_bf16,
            }
    return {"profiles": profiles}


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        model_cfg = _build_model_config(req.model)
        hw_cfg = _build_hw_config(req.hardware)
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

        perf = RLPerformanceModel(model_cfg, hw_cfg)
        report = perf.derive_targets(req.total_devices, rl_cfg, gen_par, train_par, ref_par)
        mem = report.memory

        gen_t = report.gen_time_seconds
        train_t = report.train_time_seconds
        ref_t = report.ref_time_seconds

        topo = _topology_data(req.parallelism, hw_cfg, model_cfg.num_layers)

        return {
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
                "gen_weight_gb": round(mem.gen_weight_gb, 2),
                "ref_weight_gb": round(mem.ref_weight_gb, 2),
                "optimizer_gb": round(mem.optimizer_gb, 2),
                "activation_peak_gb": round(mem.activation_peak_gb, 2),
                "ref_model_gb": round(mem.ref_model_gb, 2),
                "ref_activation_peak_gb": round(mem.ref_activation_peak_gb, 2),
                "reward_model_gb": round(mem.reward_model_gb, 2),
                "kv_cache_gb": round(mem.kv_cache_gb, 2),
                "ve_weight_gb": round(mem.ve_weight_gb, 2),
                "ve_optimizer_gb": round(mem.ve_optimizer_gb, 2),
                "ve_weight_gen_gb": round(mem.ve_weight_gen_gb, 2),
                "ve_weight_ref_gb": round(mem.ve_weight_ref_gb, 2),
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
                "colocated": req.rl.colocated,
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
        hw_cfg = _build_hw_config(req.hardware)
        rl_cfg = _build_rl_config(req.rl)
        perf = RLPerformanceModel(model_cfg, hw_cfg)

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
        if mc.vision_encoder:
            result["vision_encoder"] = _ve_to_dict(mc.vision_encoder)
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
