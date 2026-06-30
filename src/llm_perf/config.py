from __future__ import annotations

from enum import Enum
from typing import List, Optional

import yaml
from pydantic import BaseModel, field_validator


class Phase(str, Enum):
    """Execution phase of an LLM forward/backward pipeline.

    Used to select the appropriate cost model (prefill vs decode vs training).
    """

    PREFILL = "prefill"
    DECODE = "decode"
    TRAIN_FWD = "train_fwd"
    TRAIN_BWD = "train_bwd"  # MVP: combined backward




class LayerConfig(BaseModel):
    """Per-layer architecture configuration.

    Attributes:
        attention: Attention variant — one of "MHA", "GQA", "MLA", "SWA", "DSA".
        num_heads: Number of query attention heads. Must divide evenly by TP degree.
        num_kv_heads: Number of key/value heads (< num_heads for GQA).
        head_dim: Dimension per attention head.
        ffn: Feed-forward variant — "SwiGLU" or "MoE".
        intermediate_size: FFN hidden dimension (used when ffn="SwiGLU").
        num_experts: Number of MoE experts (1 = dense FFN).
        num_shared_experts: Number of shared experts in MoE.
        top_k: Number of experts activated per token in MoE.
        expert_intermediate_size: Hidden dim per MoE expert.
        shared_intermediate_size: Hidden dim for shared MoE experts.
        kv_compression_dim: KV compression dimension (MLA only).
        query_compression_dim: Query compression dimension (MLA only).
        rope_dim: RoPE dimension (MLA only).
        window_size: Sliding window size in tokens (SWA only, 0 = full attention).
        residual: Residual connection type — "standard" or "mHC".
        mhc_expansion: Expansion factor for mHC residual.
    """

    attention: str = "GQA"  # MHA, GQA, MLA, SWA, DSA
    num_heads: int = 64
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
    # DSA params (DeepSeek Sparse Attention)
    compress_ratio: int = 0          # KV压缩比: 4(C4A), 128(C128A), 0(非DSA)
    compress_c_kv: int = 0           # 压缩KV维度 (如512)
    compress_coeff: float = 0.0      # 压缩代价系数: 1.0(C4A), 0.5(C128A)
    index_n_heads: int = 0           # Lightning Index头数
    index_head_dim: int = 0          # Lightning Index头维度 (如128)
    index_topk: int = 0              # top-K压缩KV条目数 (如512)
    q_lora_rank: int = 0             # Q LoRA秩
    o_lora_rank: int = 0             # O LoRA秩
    o_groups: int = 0                # O投影block-diagonal分组数
    # Residual
    residual: str = "standard"  # standard, mHC
    mhc_expansion: int = 4


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Attributes:
        name: Human-readable model identifier (e.g. "Llama-3-70B").
        hidden_size: Transformer hidden dimension.
        vocab_size: Vocabulary size for embedding/LM-head.
        num_layers: Total number of transformer layers.
        dtype: Weight data type — "bf16", "fp16", "fp32", or "fp8".
        default_layer: Template applied to all layers when `layers` is None.
        layers: Per-layer configs; overrides default_layer if provided.
        auxiliary: Extra model features, e.g. {"mtp_depth": 2}.
    """

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
            return [self.default_layer.model_copy() for _ in range(self.num_layers)]
        raise ValueError("Must provide default_layer or layers")

    @property
    def dtype_bytes(self) -> int:
        return {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[self.dtype]


class CalibrationConfig(BaseModel):
    """Hardware efficiency calibration factors (0.0 to 1.0).

    These scale theoretical peak throughput to realistic estimates.

    Attributes:
        compute_eff_large_gemm: Compute efficiency for large GEMMs (e.g. linear layers).
        compute_eff_small_op: Compute efficiency for small/element-wise ops.
        memory_efficiency: HBM bandwidth utilization ratio.
        comm_efficiency: Inter-/intra-node communication bandwidth utilization.
        cp_init_alpha_s: Fixed bootstrap time to build one CP communicator (s).
        cp_init_beta_s: Per-rank connection-setup time when building a CP group
            (s per rank in the group).
        cp_init_gamma_s: Coefficient of the g·log2(g) rendezvous all-gather tail
            when building a CP group of size g (s).
        cp_group_buffer_mb: Resident HBM buffer held by one CP communicator (MB).

    The four cp_* fields default to 0 (CP-group init cost disabled), so existing
    calibrated numbers are unchanged until a hardware config supplies them.
    """

    compute_eff_large_gemm: float = 0.50
    compute_eff_small_op: float = 0.20
    memory_efficiency: float = 0.70
    comm_efficiency: float = 0.70
    cp_init_alpha_s: float = 0.0
    cp_init_beta_s: float = 0.0
    cp_init_gamma_s: float = 0.0
    cp_group_buffer_mb: float = 0.0


class HardwareConfig(BaseModel):
    """Hardware specification for a single accelerator device.

    Attributes:
        name: Device identifier (e.g. "A100-80GB", "H100-SXM").
        peak_tflops_bf16: Peak BF16 throughput in TFLOPS.
        hbm_capacity_gb: Total HBM capacity in GB.
        hbm_bandwidth_tb_s: HBM bandwidth in TB/s.
        hbm_usable_ratio: Fraction of HBM available after framework overhead (0.0-1.0).
        intra_node_bw_gb_s: Intra-node interconnect bandwidth in GB/s (e.g. NVLink).
        intra_node_latency_us: Intra-node interconnect latency in microseconds (e.g. NVLink/HCCS).
        inter_node_bw_gb_s: Inter-node network bandwidth in GB/s.
        inter_node_latency_us: Inter-node latency in microseconds.
        devices_per_node: Number of accelerators per node.
        cpu_gpu_bw_gb_s: CPU↔GPU data transfer bandwidth in GB/s (PCIe or HCCS).
        calibration: Efficiency calibration factors.
    """

    name: str
    peak_tflops_bf16: float
    hbm_capacity_gb: float
    hbm_bandwidth_tb_s: float
    hbm_usable_ratio: float = 0.85
    intra_node_bw_gb_s: float = 400
    intra_node_latency_us: float = 2
    inter_node_bw_gb_s: float = 100
    inter_node_latency_us: float = 5
    devices_per_node: int = 8
    cpu_gpu_bw_gb_s: float = 32.0  # CPU↔GPU PCIe/NVLink bandwidth in GB/s
    calibration: CalibrationConfig = CalibrationConfig()

    @property
    def usable_hbm_gb(self) -> float:
        return self.hbm_capacity_gb * self.hbm_usable_ratio


class WorkloadConfig(BaseModel):
    """Workload configuration shared by inference, training and post-training.

    Attributes:
        group_size: Responses generated per prompt (for GRPO/group scoring).
        avg_prompt_len: Average prompt length in tokens.
        avg_response_len: Average response length in tokens.
        max_response_len: Maximum response length in tokens (for KV cache sizing).
        std_response_len: Std-dev of response length in tokens (optional).
        train_micro_batch_size: Micro-batch size for training (samples).
        gradient_accumulation_steps: Number of gradient accumulation steps.
        train_batch_size: Global mini-batch size.
        gen_batch_size: Batch size for generation (samples per device).
        reward_model: Whether a separate reward model is used for advantage estimation.
        reference_model: Whether a reference model is kept in memory.
        ref_offload_cpu: If True, reference model weights are offloaded to CPU.
        use_speculative_decoding: Enable speculative decoding during generation.
        mtp_acceptance_len: Expected accepted tokens per MTP step (optional).
    """

    group_size: int = 16
    avg_prompt_len: int = 2048
    max_promt_len: int = 4096
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: Optional[int] = None
    train_micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 36
    gen_batch_size: int = 18
    reward_model: bool = False
    reference_model: bool = True
    ref_offload_cpu: bool = False
    use_speculative_decoding: bool = False
    mtp_acceptance_len: Optional[int] = None


class ParallelismConfig(BaseModel):
    """Distributed parallelism configuration.

    Attributes:
        tp: Tensor parallelism degree. Must be >= 1 and divide num_heads.
        pp: Pipeline parallelism degree. Must be >= 1. Uneven layer distribution
            is supported (first N%pp stages get one extra layer).
        dp: Data parallelism degree. Must be >= 1.
        ep: Expert parallelism degree (for MoE). Must be >= 1.
        cp: Context parallelism degree. Must be >= 1.
        cp_type: Context parallelism algorithm — "ring" or "ulysses".
        sp: Whether sequence parallelism is enabled (replaces AllReduce with
            AllGather + ReduceScatter).
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3).
        pp_schedule: Pipeline schedule — "gpipe", "1f1b", "interleaved",
            "zero_bubble". GPipe holds all M micro-batch activations (peak ∝ M);
            1F1B/zero-bubble/interleaved only the warmup depth (≈ min(pp, M)).
        pp_virtual_stages: Virtual pipeline stages for interleaved schedule.
        recompute_attention: Recompute attention in backward to save activation memory.
        full_recomputation: Recompute all activations in backward pass.
        optimizer_offload: Offload optimizer states to CPU.
        param_offload: Offload model parameters to CPU when not computing.
        grad_offload: Offload gradients to CPU.
        activation_offload: Offload activations to CPU during forward pass.
    """

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
    param_offload: bool = False
    grad_offload: bool = False
    activation_offload: bool = False

    @field_validator("tp", "pp", "dp", "ep", "cp")
    @classmethod
    def must_be_positive(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {v}")
        return v

    @property
    def total_devices(self) -> int:
        return self.tp * self.pp * self.dp * self.cp


class InferenceConfig(BaseModel):
    """Inference workload configuration.

    Attributes:
        batch_size: Number of requests processed concurrently.
        avg_prompt_len: Average prompt length in tokens.
        avg_response_len: Average response length in tokens.
        max_response_len: Maximum response length in tokens (for KV cache sizing).
        std_response_len: Std-dev of response length in tokens (optional).
        use_speculative_decoding: Enable speculative decoding.
        mtp_acceptance_len: Expected accepted tokens per MTP step (optional).
    """

    batch_size: int = 64
    avg_prompt_len: int = 512
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: Optional[int] = None
    use_speculative_decoding: bool = False
    mtp_acceptance_len: Optional[int] = None


class TrainConfig(BaseModel):
    """Pretraining workload configuration.

    Attributes:
        avg_seq_len: Average sequence length in tokens (prompt + response).
        train_micro_batch_size: Micro-batch size for training.
        train_batch_size: Global training batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps.
    """

    avg_seq_len: int = 4096
    train_micro_batch_size: int = 4
    train_batch_size: int = 36
    gradient_accumulation_steps: int = 1


class RuntimeConfig(BaseModel):
    """Unified runtime configuration that bundles model, hardware, network,
    parallelism, and workload parameters for a specific deployment scenario.

    Supports three workload sections:
      - rl: RL post-training (generation + reference + training)
      - inference: Standalone inference / serving
      - train: Standalone pretraining

    Attributes:
        name: Runtime configuration name.
        description: Human-readable description.
        model: Model config name (resolved from configs/models/).
        hardware: Hardware config name (resolved from configs/hardware/).
        network: Network config name (resolved from configs/network/).
        total_devices: Total number of accelerator devices.
        parallelism: Training parallelism configuration.
        gen_parallelism: Generation parallelism (optional, defaults to parallelism).
        ref_parallelism: Reference parallelism (optional, defaults to parallelism).
        rl: RL workload configuration (optional).
        inference: Inference workload configuration (optional).
        train: Pretraining workload configuration (optional).
    """

    name: str
    description: str = ""
    model: str = ""
    hardware: str = ""
    network: Optional[str] = None
    total_devices: int = 8
    parallelism: ParallelismConfig = ParallelismConfig()
    gen_parallelism: Optional[ParallelismConfig] = None
    ref_parallelism: Optional[ParallelismConfig] = None
    rl: Optional[WorkloadConfig] = None
    inference: Optional[InferenceConfig] = None
    train: Optional[TrainConfig] = None


def _load_yaml_config(path: str, config_class):
    """Load a pydantic config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_class(**data)


def load_model_config(path: str) -> ModelConfig:
    """Load a ModelConfig from a YAML file."""
    return _load_yaml_config(path, ModelConfig)


def load_hardware_config(path: str) -> HardwareConfig:
    """Load a HardwareConfig from a YAML file."""
    return _load_yaml_config(path, HardwareConfig)


def load_runtime_config(path: str) -> RuntimeConfig:
    """Load a RuntimeConfig from a YAML file.

    The runtime YAML may contain nested dicts for parallelism, rl,
    inference, and train sections, which are automatically parsed
    into their corresponding pydantic models.
    """
    return _load_yaml_config(path, RuntimeConfig)