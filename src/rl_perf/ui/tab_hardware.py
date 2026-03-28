"""Hardware & parallelism configuration tab for the rl-perf GUI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from rl_perf.config import ParallelismConfig, load_hardware_config
from rl_perf.ui.topology import build_logical_mesh_figure

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

_HW_TEMPLATES: dict[str, str] = {
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}

_ZERO_STAGES = [0, 1, 2, 3]
_PP_SCHEDULES = ["1f1b", "interleaved"]
_CP_TYPES = ["ring", "ulysses"]


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        title="Topology Preview",
    )
    return fig


def build_tab() -> dict:
    """Build the Hardware & Parallelism tab and return a dict of component handles."""
    components: dict = {}

    with gr.Tab("Hardware & Parallelism"):
        with gr.Group():
            gr.Markdown("### Hardware")
            hw_dropdown = gr.Dropdown(
                choices=list(_HW_TEMPLATES.keys()),
                value="Ascend 910C",
                label="Hardware Profile",
            )
            components["hw_dropdown"] = hw_dropdown

            with gr.Row():
                total_devices = gr.Number(label="Total Devices", value=8, precision=0)
                components["total_devices"] = total_devices
                devices_per_node = gr.Number(
                    label="Devices per Node", value=8, precision=0, interactive=False
                )
                components["devices_per_node"] = devices_per_node
                num_nodes = gr.Number(
                    label="Nodes (auto)", value=1, precision=0, interactive=False
                )
                components["num_nodes"] = num_nodes

        def _on_hw_change(hw_name):
            stem = _HW_TEMPLATES.get(hw_name)
            if not stem:
                return gr.update(), gr.update()
            hw = load_hardware_config(str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml"))
            return hw.devices_per_node, gr.update()

        hw_dropdown.change(
            fn=_on_hw_change,
            inputs=[hw_dropdown],
            outputs=[devices_per_node, num_nodes],
        )

        def _update_nodes(total, dpn):
            if dpn and dpn > 0:
                return int(total) // int(dpn)
            return 0

        total_devices.change(
            fn=_update_nodes,
            inputs=[total_devices, devices_per_node],
            outputs=[num_nodes],
        )
        devices_per_node.change(
            fn=_update_nodes,
            inputs=[total_devices, devices_per_node],
            outputs=[num_nodes],
        )

        with gr.Group():
            gr.Markdown("### 6D Parallelism")
            with gr.Row():
                tp = gr.Number(label="TP", value=1, precision=0)
                components["tp"] = tp
                pp = gr.Number(label="PP", value=1, precision=0)
                components["pp"] = pp
                dp = gr.Number(
                    label="DP (auto)", value=8, precision=0, interactive=False
                )
                components["dp"] = dp

            with gr.Row():
                ep = gr.Number(label="EP", value=1, precision=0)
                components["ep"] = ep
                cp = gr.Number(label="CP", value=1, precision=0)
                components["cp"] = cp
                cp_type = gr.Dropdown(choices=_CP_TYPES, value="ring", label="CP Type")
                components["cp_type"] = cp_type

            with gr.Row():
                sp = gr.Checkbox(label="Sequence Parallelism (SP)", value=False)
                components["sp"] = sp
                zero_stage = gr.Dropdown(
                    choices=_ZERO_STAGES, value=0, label="ZeRO Stage"
                )
                components["zero_stage"] = zero_stage
                pp_schedule = gr.Dropdown(
                    choices=_PP_SCHEDULES, value="1f1b", label="PP Schedule"
                )
                components["pp_schedule"] = pp_schedule

        # Auto-compute DP
        def _update_dp(total, tp_val, pp_val, ep_val):
            try:
                total = int(total)
                divisor = int(tp_val) * int(pp_val) * int(ep_val)
                if divisor > 0:
                    return total // divisor
            except (TypeError, ValueError):
                pass
            return 1

        for inp in [total_devices, tp, pp, ep]:
            inp.change(
                fn=_update_dp,
                inputs=[total_devices, tp, pp, ep],
                outputs=[dp],
            )

        with gr.Group():
            gr.Markdown("### Memory Optimizations")
            with gr.Row():
                recompute_attention = gr.Checkbox(
                    label="Recompute Attention", value=False
                )
                components["recompute_attention"] = recompute_attention
                full_recomputation = gr.Checkbox(
                    label="Full Recomputation", value=False
                )
                components["full_recomputation"] = full_recomputation
            with gr.Row():
                optimizer_offload = gr.Checkbox(
                    label="Optimizer Offload (CPU)", value=False
                )
                components["optimizer_offload"] = optimizer_offload
                activation_offload = gr.Checkbox(
                    label="Activation Offload (CPU)", value=False
                )
                components["activation_offload"] = activation_offload

        with gr.Group():
            gr.Markdown("### Topology Preview")
            topo_plot = gr.Plot(value=_empty_figure(), label="Logical Mesh")
            components["topo_plot"] = topo_plot
            validation_msg = gr.Markdown("")
            components["validation_msg"] = validation_msg

        # Update topology when parallelism params change
        def _update_topology(
            tp_val, pp_val, dp_val, ep_val, total, hw_name, num_layers_val=32
        ):
            try:
                tp_val = int(tp_val)
                pp_val = int(pp_val)
                dp_val = int(dp_val)
                ep_val = int(ep_val)
                total = int(total)

                stem = _HW_TEMPLATES.get(hw_name)
                if not stem:
                    return _empty_figure(), "**Warning:** Unknown hardware profile"
                hw = load_hardware_config(
                    str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml")
                )

                expected = tp_val * pp_val * dp_val * ep_val
                msg = ""
                if expected != total:
                    msg = f"**Warning:** TP*PP*DP*EP = {expected}, but total devices = {total}"

                par = ParallelismConfig(tp=tp_val, pp=pp_val, dp=dp_val, ep=ep_val)
                fig = build_logical_mesh_figure(par, hw, num_layers=num_layers_val)
                return fig, msg
            except Exception as e:
                return _empty_figure(), f"**Error:** {e}"

        components["update_topology"] = _update_topology

        for inp in [tp, pp, dp, ep, total_devices, hw_dropdown]:
            inp.change(
                fn=_update_topology,
                inputs=[tp, pp, dp, ep, total_devices, hw_dropdown],
                outputs=[topo_plot, validation_msg],
            )

    return components
