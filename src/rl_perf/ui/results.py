"""Results display area for the rl-perf GUI."""

from __future__ import annotations

import gradio as gr

from rl_perf.ui._theme import empty_figure, placeholder_kpi


def build_results() -> dict:
    """Build the results display area and return a dict of component handles."""
    components: dict = {}

    # Placeholder shown before any prediction is run
    with gr.Column(visible=True, elem_classes=["results-placeholder"]) as placeholder:
        gr.Markdown("Configure your model and click **Run Prediction** to see results.")

    with gr.Column(
        visible=False, elem_classes=["results-section"]
    ) as results_container:
        gr.Markdown("## Results", elem_classes=["section-header"])

        with gr.Row(elem_classes=["kpi-grid"]):
            kpi_epoch = gr.HTML(
                value=placeholder_kpi("Epoch Time"),
            )
            components["kpi_epoch"] = kpi_epoch
            kpi_gen_tps = gr.HTML(
                value=placeholder_kpi("Gen TPS"),
            )
            components["kpi_gen_tps"] = kpi_gen_tps
            kpi_train_tps = gr.HTML(
                value=placeholder_kpi("Train TPS"),
            )
            components["kpi_train_tps"] = kpi_train_tps
            kpi_bottleneck = gr.HTML(
                value=placeholder_kpi("Bottleneck"),
            )
            components["kpi_bottleneck"] = kpi_bottleneck

        with gr.Tabs():
            with gr.Tab("Timeline & Overview"):
                timeline_plot = gr.Plot(
                    value=empty_figure("Epoch Timeline"), label="Timeline"
                )
                components["timeline_plot"] = timeline_plot

            with gr.Tab("Memory Details"):
                memory_plot = gr.Plot(
                    value=empty_figure("Memory Breakdown"), label="Memory"
                )
                components["memory_plot"] = memory_plot

    components["results_container"] = results_container
    components["results_placeholder"] = placeholder

    return components
