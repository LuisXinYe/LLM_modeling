"""Smoke tests for the Gradio UI assembly."""


def test_app_builds():
    """The Gradio app should build without errors."""
    from rl_perf.ui.app import create_app

    app = create_app()
    assert app is not None


def test_tab_model_builds():
    """tab_model.build_tab should return component handles inside a Blocks context."""
    import gradio as gr
    from rl_perf.ui.tab_model import build_tab

    with gr.Blocks():
        components = build_tab()
    assert isinstance(components, dict)
    assert "source" in components
    assert "name" in components


def test_tab_hardware_builds():
    """tab_hardware.build_tab should return component handles inside a Blocks context."""
    import gradio as gr
    from rl_perf.ui.tab_hardware import build_tab

    with gr.Blocks():
        components = build_tab()
    assert isinstance(components, dict)
    assert "tp" in components
    assert "topo_plot" in components


def test_tab_rl_builds():
    """tab_rl.build_tab should return component handles inside a Blocks context."""
    import gradio as gr
    from rl_perf.ui.tab_rl import build_tab

    with gr.Blocks():
        components = build_tab()
    assert isinstance(components, dict)
    assert "total_prompts" in components
    assert "colocated" in components


def test_tab_search_builds():
    """tab_search.build_tab should return component handles inside a Blocks context."""
    import gradio as gr
    from rl_perf.ui.tab_search import build_tab

    with gr.Blocks():
        components = build_tab()
    assert isinstance(components, dict)
    assert "search_btn" in components
    assert "pareto_plot" in components


def test_results_builds():
    """results.build_results should return component handles inside a Blocks context."""
    import gradio as gr
    from rl_perf.ui.results import build_results

    with gr.Blocks():
        components = build_results()
    assert isinstance(components, dict)
    assert "results_container" in components
    assert "kpi_epoch" in components
    assert "timeline_plot" in components
