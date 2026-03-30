"""RL training parameters tab for the rl-perf GUI."""

from __future__ import annotations

import gradio as gr


def build_tab() -> dict:
    """Build the RL Training Parameters tab and return a dict of component handles."""
    components: dict = {}

    with gr.Tab("RL Training"):
        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Data", elem_classes=["section-header"])
            with gr.Row():
                total_prompts = gr.Number(
                    label="Total Prompts", value=10000, precision=0
                )
                components["total_prompts"] = total_prompts
                group_size = gr.Number(label="Group Size", value=8, precision=0)
                components["group_size"] = group_size
                total_responses = gr.Number(
                    label="Total Responses (auto)",
                    value=80000,
                    precision=0,
                    interactive=False,
                )
                components["total_responses"] = total_responses
            time_budget = gr.Number(
                label="Time Budget (hours)", value=24.0, precision=None
            )
            components["time_budget"] = time_budget

        def _update_responses(prompts, grp):
            try:
                return int(prompts) * int(grp)
            except (TypeError, ValueError):
                return 0

        total_prompts.change(
            fn=_update_responses,
            inputs=[total_prompts, group_size],
            outputs=[total_responses],
        )
        group_size.change(
            fn=_update_responses,
            inputs=[total_prompts, group_size],
            outputs=[total_responses],
        )

        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Sequence Lengths", elem_classes=["section-header"])
            with gr.Row():
                avg_prompt_len = gr.Number(
                    label="Avg Prompt Length", value=512, precision=0
                )
                components["avg_prompt_len"] = avg_prompt_len
                avg_response_len = gr.Number(
                    label="Avg Response Length", value=2048, precision=0
                )
                components["avg_response_len"] = avg_response_len
            with gr.Row():
                max_response_len = gr.Number(
                    label="Max Response Length", value=4096, precision=0
                )
                components["max_response_len"] = max_response_len
                std_response_len = gr.Number(
                    label="Std Response Length (optional)",
                    value=0,
                    precision=0,
                )
                components["std_response_len"] = std_response_len

        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Batch Settings", elem_classes=["section-header"])
            with gr.Row():
                train_micro_batch_size = gr.Number(
                    label="Train Micro Batch Size", value=4, precision=0
                )
                components["train_micro_batch_size"] = train_micro_batch_size
                grad_accumulation_steps = gr.Number(
                    label="Gradient Accumulation Steps", value=1, precision=0
                )
                components["grad_accumulation_steps"] = grad_accumulation_steps
                gen_batch_size = gr.Number(
                    label="Gen Batch Size", value=64, precision=0
                )
                components["gen_batch_size"] = gen_batch_size

        with gr.Group(elem_classes=["section-group"]):
            gr.Markdown("### Deployment", elem_classes=["section-header"])
            with gr.Row():
                colocated = gr.Dropdown(
                    choices=["Colocated", "Separate"],
                    value="Separate",
                    label="Deployment Mode",
                )
                components["colocated"] = colocated
                reference_model = gr.Checkbox(label="Reference Model", value=True)
                components["reference_model"] = reference_model

            with gr.Row():
                ref_offload_cpu = gr.Checkbox(label="Ref Offload CPU", value=False)
                components["ref_offload_cpu"] = ref_offload_cpu
                speculative_decoding = gr.Checkbox(
                    label="Speculative Decoding", value=False
                )
                components["speculative_decoding"] = speculative_decoding

            with gr.Row(visible=False) as mtp_row:
                mtp_acceptance_len = gr.Number(
                    label="MTP Acceptance Length", value=3, precision=0
                )
                components["mtp_acceptance_len"] = mtp_acceptance_len
            components["mtp_row"] = mtp_row

        def _on_spec_change(spec):
            return gr.update(visible=spec)

        speculative_decoding.change(
            fn=_on_spec_change,
            inputs=[speculative_decoding],
            outputs=[mtp_row],
        )

    return components
