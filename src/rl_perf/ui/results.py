"""Results display area for the rl-perf GUI."""

from __future__ import annotations

import gradio as gr
import plotly.graph_objects as go


def _empty_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def build_results() -> dict:
    """Build the results display area and return a dict of component handles."""
    components: dict = {}

    with gr.Column(visible=False) as results_container:
        gr.Markdown("---")
        gr.Markdown("## Results")

        with gr.Row():
            kpi_epoch = gr.Markdown(value="**Epoch Time**\n\n--", label="Epoch Time")
            components["kpi_epoch"] = kpi_epoch
            kpi_gen_tps = gr.Markdown(value="**Gen TPS**\n\n--", label="Gen TPS")
            components["kpi_gen_tps"] = kpi_gen_tps
            kpi_train_tps = gr.Markdown(value="**Train TPS**\n\n--", label="Train TPS")
            components["kpi_train_tps"] = kpi_train_tps
            kpi_bottleneck = gr.Markdown(
                value="**Bottleneck**\n\n--", label="Bottleneck"
            )
            components["kpi_bottleneck"] = kpi_bottleneck

        with gr.Tabs():
            with gr.Tab("Timeline & Overview"):
                timeline_plot = gr.Plot(
                    value=_empty_figure("Epoch Timeline"), label="Timeline"
                )
                components["timeline_plot"] = timeline_plot

            with gr.Tab("Memory Details"):
                memory_plot = gr.Plot(
                    value=_empty_figure("Memory Breakdown"), label="Memory"
                )
                components["memory_plot"] = memory_plot

    components["results_container"] = results_container

    return components
