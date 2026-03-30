"""Parameter search tab for the rl-perf GUI."""

from __future__ import annotations

import gradio as gr

from rl_perf.ui._theme import empty_figure


_OPTIMIZATION_TARGETS = ["epoch_time_hours", "gen_tps_target", "train_tps_target"]
_SWEEP_PARAMS = [
    "group_size",
    "total_prompts",
    "avg_prompt_len",
    "avg_response_len",
    "max_response_len",
    "train_micro_batch_size",
    "gen_batch_size",
    "gradient_accumulation_steps",
]


def build_tab() -> dict:
    """Build the Parameter Search tab and return a dict of component handles."""
    components: dict = {}

    with gr.Tab("Search"):
        mode = gr.Radio(
            choices=["Pareto Search", "Sensitivity Analysis"],
            value="Pareto Search",
            label="Search Mode",
        )
        components["mode"] = mode

        # Pareto options
        with gr.Group(visible=True, elem_classes=["section-group"]) as pareto_group:
            gr.Markdown("### Pareto Search", elem_classes=["section-header"])
            device_counts = gr.Textbox(
                label="Device Counts (comma separated)",
                value="8, 16, 32, 64, 128",
                placeholder="8, 16, 32, 64",
            )
            components["device_counts"] = device_counts
            optimization_target = gr.Dropdown(
                choices=_OPTIMIZATION_TARGETS,
                value="epoch_time_hours",
                label="Optimization Target",
            )
            components["optimization_target"] = optimization_target
        components["pareto_group"] = pareto_group

        # Sensitivity options
        with gr.Group(visible=False, elem_classes=["section-group"]) as sens_group:
            gr.Markdown("### Sensitivity Analysis", elem_classes=["section-header"])
            sweep_param = gr.Dropdown(
                choices=_SWEEP_PARAMS,
                value="group_size",
                label="Sweep Parameter",
            )
            components["sweep_param"] = sweep_param
            sweep_values = gr.Textbox(
                label="Sweep Values (comma separated)",
                value="4, 8, 16, 32",
                placeholder="4, 8, 16, 32",
            )
            components["sweep_values"] = sweep_values
        components["sens_group"] = sens_group

        def _on_mode_change(m):
            is_pareto = m == "Pareto Search"
            return gr.update(visible=is_pareto), gr.update(visible=not is_pareto)

        mode.change(
            fn=_on_mode_change,
            inputs=[mode],
            outputs=[pareto_group, sens_group],
        )

        search_btn = gr.Button("Run Search", variant="primary")
        components["search_btn"] = search_btn
        search_status = gr.HTML("")
        components["search_status"] = search_status

        with gr.Row():
            pareto_plot = gr.Plot(
                value=empty_figure("Pareto Frontier"), label="Pareto Plot"
            )
            components["pareto_plot"] = pareto_plot
            sens_plot = gr.Plot(
                value=empty_figure("Sensitivity"), label="Sensitivity Plot"
            )
            components["sens_plot"] = sens_plot

        comparison_df = gr.Dataframe(
            label="Results Comparison",
            headers=[
                "Devices",
                "TP",
                "PP",
                "DP",
                "EP",
                "Epoch (h)",
                "Gen TPS",
                "Train TPS",
                "Feasible",
                "Pareto",
            ],
            interactive=False,
        )
        components["comparison_df"] = comparison_df

    return components
