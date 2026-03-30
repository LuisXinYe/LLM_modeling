"""Shared Plotly styling constants and helpers for the rl-perf GUI.

Centralises chart typography, background colours, and reusable figure
builders so that plots.py, topology.py, results.py, tab_search.py, and
tab_hardware.py all draw from a single source of truth.
"""

from __future__ import annotations

import html as _html

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Plotly font constants
# ---------------------------------------------------------------------------

PLOTLY_FONT: dict = dict(family="DM Sans, system-ui, sans-serif", size=13)
"""Standard body font for Plotly charts."""

PLOTLY_TITLE_FONT: dict = dict(family="DM Sans, system-ui, sans-serif", size=16)
"""Slightly larger font used for chart titles."""

# ---------------------------------------------------------------------------
# Chart surface colours
# ---------------------------------------------------------------------------

CHART_BG: str = "#FAFAF8"
"""Warm background matching the --chart-bg CSS token."""

# ---------------------------------------------------------------------------
# Grid / hover helpers
# ---------------------------------------------------------------------------

GRID_COLOR: str = "rgba(0, 0, 0, 0.06)"
"""Subtle gridline colour shared across all charts."""

HOVERLABEL: dict = dict(
    bgcolor="white",
    font_size=12,
    font_family="DM Sans, system-ui, sans-serif",
    bordercolor="#E8E5E0",
)
"""Consistent hover-label styling for all Plotly figures."""


# ---------------------------------------------------------------------------
# Reusable figure builders
# ---------------------------------------------------------------------------


def empty_figure(title: str = "") -> go.Figure:
    """Return a minimal empty Plotly figure with optional *title*.

    Used as a placeholder in Gradio Plot components before data is available.
    """
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        font=PLOTLY_FONT,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
    )
    return fig


# ---------------------------------------------------------------------------
# KPI HTML helper
# ---------------------------------------------------------------------------


def kpi_html(label: str, value: str, detail: str = "", extra_cls: str = "") -> str:
    """Return a styled KPI card as an HTML snippet.

    All user-supplied strings (*value*, *detail*) are escaped to prevent XSS.
    *label* is also escaped for safety even though it typically comes from
    trusted code.
    """
    safe_label = _html.escape(str(label))
    safe_value = _html.escape(str(value))
    safe_detail = _html.escape(str(detail))
    return (
        f'<div class="kpi-card {extra_cls}">'
        f'<div class="kpi-label">{safe_label}</div>'
        f'<div class="kpi-value">{safe_value}</div>'
        f'<div class="kpi-detail">{safe_detail}</div>'
        f"</div>"
    )


def placeholder_kpi(label: str) -> str:
    """Return a neutral placeholder KPI card with ``--`` as value."""
    return kpi_html(label, "--", "", "kpi-neutral")
