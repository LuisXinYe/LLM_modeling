"""Model configuration tab for the rl-perf GUI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from rl_perf.config import LayerConfig, load_model_config
from rl_perf.ui.hf_import import fetch_hf_config, hf_config_to_model_config

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Map display names to YAML file stems
_MODEL_TEMPLATES: dict[str, str] = {
    "Llama-3.1-8B": "llama3_1_8b",
    "Qwen2.5-72B": "qwen2_5_72b",
    "Mistral-7B": "mistral_7b",
    "Qwen3-235B-MoE": "qwen3_235b_moe",
    "DeepSeekV3-671B": "deepseekv3_671b",
}

_ATTENTION_TYPES = ["MHA", "GQA", "MLA", "SWA", "Mamba"]
_FFN_TYPES = ["SwiGLU", "MoE"]
_RESIDUAL_TYPES = ["standard", "mHC"]
_DTYPE_OPTIONS = ["bf16", "fp16", "fp32", "fp8"]


def build_tab() -> dict:
    """Build the Model Configuration tab and return a dict of component handles."""
    components: dict = {}

    with gr.Tab("Model"):
        with gr.Group():
            gr.Markdown("### Model Source")
            source = gr.Dropdown(
                choices=["Template", "HuggingFace", "Custom"],
                value="Template",
                label="Source",
            )
            components["source"] = source

            hf_id = gr.Textbox(
                label="HuggingFace Model ID",
                placeholder="meta-llama/Meta-Llama-3-8B",
                visible=False,
            )
            components["hf_id"] = hf_id

            template = gr.Dropdown(
                choices=list(_MODEL_TEMPLATES.keys()),
                value="Llama-3.1-8B",
                label="Template",
                visible=True,
            )
            components["template"] = template

            load_btn = gr.Button("Load Model", variant="primary")
            components["load_btn"] = load_btn

            load_status = gr.Markdown("")
            components["load_status"] = load_status

        # Toggle visibility based on source selection
        def _on_source_change(src):
            return (
                gr.update(visible=src == "HuggingFace"),
                gr.update(visible=src == "Template"),
            )

        source.change(
            fn=_on_source_change,
            inputs=[source],
            outputs=[hf_id, template],
        )

        with gr.Group():
            gr.Markdown("### Model Architecture")
            with gr.Row():
                name = gr.Textbox(label="Model Name", value="Llama-3.1-8B")
                components["name"] = name
                hidden_size = gr.Number(label="Hidden Size", value=4096, precision=0)
                components["hidden_size"] = hidden_size
            with gr.Row():
                vocab_size = gr.Number(label="Vocab Size", value=128256, precision=0)
                components["vocab_size"] = vocab_size
                num_layers = gr.Number(label="Num Layers", value=32, precision=0)
                components["num_layers"] = num_layers
                dtype = gr.Dropdown(choices=_DTYPE_OPTIONS, value="bf16", label="Dtype")
                components["dtype"] = dtype

        with gr.Group():
            gr.Markdown("### Layer Configuration")
            with gr.Row():
                attention = gr.Dropdown(
                    choices=_ATTENTION_TYPES, value="GQA", label="Attention"
                )
                components["attention"] = attention
                ffn = gr.Dropdown(choices=_FFN_TYPES, value="SwiGLU", label="FFN")
                components["ffn"] = ffn
                residual = gr.Dropdown(
                    choices=_RESIDUAL_TYPES, value="standard", label="Residual"
                )
                components["residual"] = residual

            with gr.Row():
                num_heads = gr.Number(label="Num Heads", value=32, precision=0)
                components["num_heads"] = num_heads
                num_kv_heads = gr.Number(label="Num KV Heads", value=8, precision=0)
                components["num_kv_heads"] = num_kv_heads
                head_dim = gr.Number(label="Head Dim", value=128, precision=0)
                components["head_dim"] = head_dim

            intermediate_size = gr.Number(
                label="Intermediate Size", value=14336, precision=0
            )
            components["intermediate_size"] = intermediate_size

            # MLA fields
            with gr.Row(visible=False) as mla_row:
                kv_compression_dim = gr.Number(
                    label="KV Compression Dim", value=0, precision=0
                )
                components["kv_compression_dim"] = kv_compression_dim
                query_compression_dim = gr.Number(
                    label="Query Compression Dim", value=0, precision=0
                )
                components["query_compression_dim"] = query_compression_dim
                rope_dim = gr.Number(label="RoPE Dim", value=0, precision=0)
                components["rope_dim"] = rope_dim
            components["mla_row"] = mla_row

            # SWA field
            with gr.Row(visible=False) as swa_row:
                window_size = gr.Number(
                    label="Window Size (tokens)", value=0, precision=0
                )
                components["window_size"] = window_size
            components["swa_row"] = swa_row

            # MoE fields
            with gr.Row(visible=False) as moe_row:
                num_experts = gr.Number(label="Num Experts", value=1, precision=0)
                components["num_experts"] = num_experts
                top_k = gr.Number(label="Top-K", value=1, precision=0)
                components["top_k"] = top_k
                num_shared_experts = gr.Number(
                    label="Shared Experts", value=0, precision=0
                )
                components["num_shared_experts"] = num_shared_experts
            components["moe_row"] = moe_row

            with gr.Row(visible=False) as moe_row2:
                expert_intermediate_size = gr.Number(
                    label="Expert Intermediate Size", value=0, precision=0
                )
                components["expert_intermediate_size"] = expert_intermediate_size
                shared_intermediate_size = gr.Number(
                    label="Shared Intermediate Size", value=0, precision=0
                )
                components["shared_intermediate_size"] = shared_intermediate_size
            components["moe_row2"] = moe_row2

            # mHC field
            with gr.Row(visible=False) as mhc_row:
                mhc_expansion = gr.Number(label="mHC Expansion", value=4, precision=0)
                components["mhc_expansion"] = mhc_expansion
            components["mhc_row"] = mhc_row

        # Conditional visibility for layer fields
        def _on_attention_change(attn):
            return (
                gr.update(visible=attn == "MLA"),
                gr.update(visible=attn == "SWA"),
            )

        attention.change(
            fn=_on_attention_change,
            inputs=[attention],
            outputs=[mla_row, swa_row],
        )

        def _on_ffn_change(ffn_type):
            is_moe = ffn_type == "MoE"
            return gr.update(visible=is_moe), gr.update(visible=is_moe)

        ffn.change(
            fn=_on_ffn_change,
            inputs=[ffn],
            outputs=[moe_row, moe_row2],
        )

        def _on_residual_change(res):
            return gr.update(visible=res == "mHC")

        residual.change(
            fn=_on_residual_change,
            inputs=[residual],
            outputs=[mhc_row],
        )

        # Load button handler
        def _load_model(src, hf_model_id, tpl_name):
            try:
                if src == "HuggingFace":
                    hf_cfg = fetch_hf_config(hf_model_id)
                    mc = hf_config_to_model_config(hf_cfg, name=hf_model_id)
                elif src == "Template":
                    stem = _MODEL_TEMPLATES.get(tpl_name)
                    if not stem:
                        return [gr.update()] * 20 + ["**Error:** Unknown template"]
                    yaml_path = _CONFIGS_DIR / "models" / f"{stem}.yaml"
                    mc = load_model_config(str(yaml_path))
                else:
                    # Custom: no-op, user fills manually
                    return [gr.update()] * 20 + ["Custom mode: fill fields manually."]

                layer = mc.default_layer or LayerConfig()
                is_mla = layer.attention == "MLA"
                is_swa = layer.attention == "SWA"
                is_moe = layer.ffn == "MoE"
                is_mhc = layer.residual == "mHC"

                return [
                    mc.name,
                    mc.hidden_size,
                    mc.vocab_size,
                    mc.num_layers,
                    mc.dtype,
                    layer.attention,
                    layer.ffn,
                    layer.residual,
                    layer.num_heads,
                    layer.num_kv_heads,
                    layer.head_dim,
                    layer.intermediate_size,
                    layer.kv_compression_dim,
                    layer.query_compression_dim,
                    layer.rope_dim,
                    layer.window_size,
                    layer.num_experts,
                    layer.top_k,
                    layer.num_shared_experts,
                    layer.expert_intermediate_size,
                    layer.shared_intermediate_size,
                    layer.mhc_expansion,
                    gr.update(visible=is_mla),  # mla_row
                    gr.update(visible=is_swa),  # swa_row
                    gr.update(visible=is_moe),  # moe_row
                    gr.update(visible=is_moe),  # moe_row2
                    gr.update(visible=is_mhc),  # mhc_row
                    f"**Loaded:** {mc.name}",
                ]
            except Exception as e:
                return [gr.update()] * 27 + [f"**Error:** {e}"]

        load_btn.click(
            fn=_load_model,
            inputs=[source, hf_id, template],
            outputs=[
                name,
                hidden_size,
                vocab_size,
                num_layers,
                dtype,
                attention,
                ffn,
                residual,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                kv_compression_dim,
                query_compression_dim,
                rope_dim,
                window_size,
                num_experts,
                top_k,
                num_shared_experts,
                expert_intermediate_size,
                shared_intermediate_size,
                mhc_expansion,
                mla_row,
                swa_row,
                moe_row,
                moe_row2,
                mhc_row,
                load_status,
            ],
        )

    return components
