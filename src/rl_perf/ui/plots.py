"""Plotly figure builders for the rl-perf GUI.

Each function returns a ``plotly.graph_objects.Figure`` ready to embed in a
Gradio or Dash component.  All figures use ``template="plotly_white"`` and
reasonable margins for a clean, consistent look.
"""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go

from rl_perf.report import TargetReport
from rl_perf.search import SearchResult

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
_PURPLE = "#7c3aed"
_ORANGE = "#ea580c"
_GREEN = "#16a34a"
_RED = "#dc2626"
_GRAY = "#9ca3af"
_AMBER = "#f59e0b"
_CYAN = "#06b6d4"
_LIGHT_PURPLE = "#c4b5fd"


# ---------------------------------------------------------------------------
# 1. Timeline figure
# ---------------------------------------------------------------------------


def build_timeline_figure(report: TargetReport, colocated: bool) -> go.Figure:
    """Horizontal stacked bar chart (Gantt-style) showing Gen / Train phases.

    Args:
        report: TargetReport with gen_time_hours and train_time_hours.
        colocated: When True, stack Gen then Train on a single row.
                   When False, draw Gen and Train on separate rows.

    Returns:
        A ``go.Figure`` with horizontal bar traces.
    """
    gen_h = report.gen_time_hours
    train_h = report.train_time_hours

    if colocated:
        y_vals = ["Pipeline"]
        gen_trace = go.Bar(
            name="Generation",
            x=[gen_h],
            y=y_vals,
            orientation="h",
            marker_color=_PURPLE,
            text=[f"{gen_h:.2f}h"],
            textposition="inside",
            insidetextanchor="middle",
        )
        train_trace = go.Bar(
            name="Training",
            x=[train_h],
            y=y_vals,
            orientation="h",
            marker_color=_ORANGE,
            text=[f"{train_h:.2f}h"],
            textposition="inside",
            insidetextanchor="middle",
        )
        traces = [gen_trace, train_trace]
        barmode = "stack"
    else:
        gen_trace = go.Bar(
            name="Generation",
            x=[gen_h],
            y=["Generation"],
            orientation="h",
            marker_color=_PURPLE,
            text=[f"{gen_h:.2f}h"],
            textposition="inside",
            insidetextanchor="middle",
        )
        train_trace = go.Bar(
            name="Training",
            x=[train_h],
            y=["Training"],
            orientation="h",
            marker_color=_ORANGE,
            text=[f"{train_h:.2f}h"],
            textposition="inside",
            insidetextanchor="middle",
        )
        traces = [gen_trace, train_trace]
        barmode = "overlay"

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        barmode=barmode,
        title="Epoch Timeline",
        xaxis_title="Time (hours)",
        margin=dict(l=80, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Memory figure
# ---------------------------------------------------------------------------


def build_memory_figure(report: TargetReport) -> go.Figure:
    """Stacked bar chart showing per-device memory breakdown.

    Training bar stack: Weights → Optimizer → Activations → Ref Model.
    Generation bar stack: Weights → KV Cache.
    A dashed red horizontal line marks the usable HBM limit.

    Args:
        report: TargetReport containing a MemoryProfile.

    Returns:
        A ``go.Figure`` with stacked bar traces and an HBM limit line.
    """
    mem = report.memory
    categories = ["Training", "Generation"]

    traces = [
        go.Bar(
            name="Weights",
            x=categories,
            y=[mem.weight_gb, mem.weight_gb],
            marker_color=_PURPLE,
            text=[f"{mem.weight_gb:.1f}", f"{mem.weight_gb:.1f}"],
            textposition="inside",
        ),
        go.Bar(
            name="Optimizer",
            x=categories,
            y=[mem.optimizer_gb, 0.0],
            marker_color=_LIGHT_PURPLE,
            text=[f"{mem.optimizer_gb:.1f}", ""],
            textposition="inside",
        ),
        go.Bar(
            name="Activations",
            x=categories,
            y=[mem.activation_peak_gb, 0.0],
            marker_color=_AMBER,
            text=[f"{mem.activation_peak_gb:.1f}", ""],
            textposition="inside",
        ),
        go.Bar(
            name="Ref Model",
            x=categories,
            y=[mem.ref_model_gb, 0.0],
            marker_color=_CYAN,
            text=[f"{mem.ref_model_gb:.1f}", ""],
            textposition="inside",
        ),
        go.Bar(
            name="KV Cache",
            x=categories,
            y=[0.0, mem.kv_cache_gb],
            marker_color=_GREEN,
            text=["", f"{mem.kv_cache_gb:.1f}"],
            textposition="inside",
        ),
    ]

    # HBM limit as a dashed red horizontal scatter line
    hbm_line = go.Scatter(
        name=f"HBM limit ({mem.usable_hbm_gb:.0f} GB)",
        x=["Training", "Generation"],
        y=[mem.usable_hbm_gb, mem.usable_hbm_gb],
        mode="lines",
        line=dict(color=_RED, dash="dash", width=2),
    )

    fig = go.Figure(data=traces + [hbm_line])
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        title="Per-Device Memory Breakdown",
        yaxis_title="Memory (GB)",
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Pareto figure
# ---------------------------------------------------------------------------


def build_pareto_figure(
    results: List[SearchResult],
    time_budget_hours: Optional[float] = None,
) -> go.Figure:
    """Scatter plot of (devices, epoch_time) with Pareto frontier highlighted.

    Point style:
        - Pareto-optimal feasible: large green dots
        - Feasible, non-Pareto: small gray dots
        - OOM: small red dots

    Pareto points are connected by a dashed green line (sorted by devices).
    An optional amber dashed horizontal line marks the time budget.

    Args:
        results: List of SearchResult from a pareto_search call.
        time_budget_hours: If provided, draw a dashed amber budget line.

    Returns:
        A ``go.Figure`` scatter plot.
    """
    pareto = [r for r in results if r.is_pareto]
    feasible_non_pareto = [r for r in results if r.is_feasible and not r.is_pareto]
    oom = [r for r in results if r.is_oom]

    def _hover(r: SearchResult) -> str:
        tp = r.train_parallel
        return (
            f"Devices: {r.devices}<br>"
            f"Epoch time: {r.report.epoch_time_hours:.2f}h<br>"
            f"TP={tp.tp} PP={tp.pp} DP={tp.dp} EP={tp.ep}"
        )

    traces: list[go.BaseTraceType] = []

    # OOM points
    if oom:
        traces.append(
            go.Scatter(
                name="OOM",
                x=[r.devices for r in oom],
                y=[r.report.epoch_time_hours for r in oom],
                mode="markers",
                marker=dict(color=_RED, size=8, symbol="x"),
                hovertext=[_hover(r) for r in oom],
                hoverinfo="text",
            )
        )

    # Feasible non-Pareto points
    if feasible_non_pareto:
        traces.append(
            go.Scatter(
                name="Feasible",
                x=[r.devices for r in feasible_non_pareto],
                y=[r.report.epoch_time_hours for r in feasible_non_pareto],
                mode="markers",
                marker=dict(color=_GRAY, size=8),
                hovertext=[_hover(r) for r in feasible_non_pareto],
                hoverinfo="text",
            )
        )

    # Pareto frontier line + points
    if pareto:
        pareto_sorted = sorted(pareto, key=lambda r: r.devices)
        traces.append(
            go.Scatter(
                name="Pareto frontier",
                x=[r.devices for r in pareto_sorted],
                y=[r.report.epoch_time_hours for r in pareto_sorted],
                mode="lines+markers",
                line=dict(color=_GREEN, dash="dash", width=2),
                marker=dict(color=_GREEN, size=14),
                hovertext=[_hover(r) for r in pareto_sorted],
                hoverinfo="text",
            )
        )

    # Time budget line
    if time_budget_hours is not None and results:
        all_devices = sorted({r.devices for r in results})
        traces.append(
            go.Scatter(
                name=f"Budget ({time_budget_hours:.1f}h)",
                x=[all_devices[0], all_devices[-1]],
                y=[time_budget_hours, time_budget_hours],
                mode="lines",
                line=dict(color=_AMBER, dash="dash", width=2),
            )
        )

    # If no results at all, return an empty figure with at least one empty trace
    if not traces:
        traces.append(go.Scatter(name="No data", x=[], y=[], mode="markers"))

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        title="Pareto Frontier: Devices vs Epoch Time",
        xaxis_title="Number of Devices",
        yaxis_title="Epoch Time (hours)",
        margin=dict(l=60, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Sensitivity figure
# ---------------------------------------------------------------------------


def build_sensitivity_figure(
    param_name: str,
    values: list,
    reports: List[TargetReport],
    time_budget_hours: Optional[float] = None,
) -> go.Figure:
    """Bar chart of epoch_time vs a swept parameter value.

    Bar colour:
        - Green: feasible and within budget
        - Amber: feasible but over budget
        - Red: infeasible (OOM)

    Text labels show epoch time in hours.  An optional amber dashed horizontal
    line marks the time budget.

    Args:
        param_name: Human-readable name of the swept parameter (used as X axis label).
        values: Ordered list of parameter values (x-axis categories).
        reports: Corresponding TargetReport for each value (same order as values).
        time_budget_hours: If provided, draw a dashed amber budget line.

    Returns:
        A ``go.Figure`` bar chart.
    """
    assert len(values) == len(reports), "values and reports must have equal length"

    x_labels = [str(v) for v in values]
    y_vals = [r.epoch_time_hours for r in reports]

    colors = []
    for r in reports:
        is_oom = not (r.memory.train_feasible and r.memory.gen_feasible)
        if is_oom or not r.feasible:
            colors.append(_RED)
        elif r.within_budget:
            colors.append(_GREEN)
        else:
            colors.append(_AMBER)

    bar_trace = go.Bar(
        name="Epoch time",
        x=x_labels,
        y=y_vals,
        marker_color=colors,
        text=[f"{v:.2f}h" for v in y_vals],
        textposition="outside",
    )

    traces: list[go.BaseTraceType] = [bar_trace]

    # Time budget line
    if time_budget_hours is not None and x_labels:
        traces.append(
            go.Scatter(
                name=f"Budget ({time_budget_hours:.1f}h)",
                x=[x_labels[0], x_labels[-1]],
                y=[time_budget_hours, time_budget_hours],
                mode="lines",
                line=dict(color=_AMBER, dash="dash", width=2),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        title=f"Sensitivity: {param_name}",
        xaxis_title=param_name,
        yaxis_title="Epoch Time (hours)",
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
