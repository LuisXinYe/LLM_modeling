from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    TRAIN_FWD = "train_fwd"
    TRAIN_BWD = "train_bwd"  # MVP: combined backward


class LayerConfig(BaseModel):
    attention: str = "GQA"  # MHA, GQA, MLA, SWA, Mamba
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn: str = "SwiGLU"  # SwiGLU, MoE
    intermediate_size: int = 11008
    # MoE params (only when ffn="MoE")
    num_experts: int = 1
    num_shared_experts: int = 0
    top_k: int = 1
    expert_intermediate_size: int = 0
    shared_intermediate_size: int = 0  # shared expert hidden dim
    # MLA params
    kv_compression_dim: int = 0
    query_compression_dim: int = 0
    rope_dim: int = 0
    # SWA params
    window_size: int = 0
    # Residual
    residual: str = "standard"  # standard, mHC
    mhc_expansion: int = 4


class ModelConfig(BaseModel):
    name: str
    hidden_size: int
    vocab_size: int
    num_layers: int
    dtype: str = "bf16"
    default_layer: Optional[LayerConfig] = None
    layers: Optional[List[LayerConfig]] = None
    auxiliary: Optional[dict] = None  # mtp_depth etc.

    def get_layers(self) -> List[LayerConfig]:
        if self.layers:
            return self.layers
        if self.default_layer:
            return [self.default_layer] * self.num_layers
        raise ValueError("Must provide default_layer or layers")

    @property
    def dtype_bytes(self) -> int:
        return {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[self.dtype]


class CalibrationConfig(BaseModel):
    compute_eff_large_gemm: float = 0.50
    compute_eff_small_op: float = 0.20
    memory_efficiency: float = 0.70
    comm_efficiency: float = 0.70


class HardwareConfig(BaseModel):
    name: str
    peak_tflops_bf16: float
    hbm_capacity_gb: float
    hbm_bandwidth_tb_s: float
    hbm_usable_ratio: float = 0.85
    intra_node_bw_gb_s: float = 400
    inter_node_bw_gb_s: float = 100
    inter_node_latency_us: float = 5
    devices_per_node: int = 8
    calibration: CalibrationConfig = CalibrationConfig()

    @property
    def usable_hbm_gb(self) -> float:
        return self.hbm_capacity_gb * self.hbm_usable_ratio


class RLConfig(BaseModel):
    total_prompts: int
    group_size: int = 8
    avg_prompt_len: int = 512
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: Optional[int] = None
    train_micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gen_batch_size: int = 64
    reference_model: bool = True
    ref_offload_cpu: bool = False
    colocated: bool = False

    @property
    def total_responses(self) -> int:
        return self.total_prompts * self.group_size


class ParallelismConfig(BaseModel):
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1
    cp: int = 1
    cp_type: str = "ring"
    sp: bool = False
    zero_stage: int = 0
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 1
    recompute_attention: bool = False
    full_recomputation: bool = False
    optimizer_offload: bool = False
    activation_offload: bool = False

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.dp * self.ep


def load_model_config(path: str) -> ModelConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


def load_hardware_config(path: str) -> HardwareConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return HardwareConfig(**data)
