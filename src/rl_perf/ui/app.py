"""Main Gradio application for rl-perf GUI."""

import gradio as gr


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(title="rl-perf — RL Training Performance Modeling") as app:
        gr.Markdown("# rl-perf\nRL Training Performance Modeling")
        gr.Markdown("*GUI under construction*")

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Create and launch the Gradio app."""
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )
