"""Device topology visualization for the rl-perf GUI.

Provides rank-to-parallelism mapping and a 2-D logical mesh Plotly figure
that visualises how GPU ranks are assigned across TP / PP / DP / EP groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import plotly.graph_objects as go

from rl_perf.config import HardwareConfig, ParallelismConfig
from rl_perf.ui._theme import (
    CHART_BG as _CHART_BG,
    HOVERLABEL as _HOVERLABEL,
    PLOTLY_FONT as _PLOTLY_FONT,
    PLOTLY_TITLE_FONT as _PLOTLY_TITLE_FONT,
)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

# One colour per PP stage (cycles if pp > len)
# Aligned with design tokens: purple, blue, cyan, green, orange, red, amber, gray
_PP_STAGE_COLORS = [
    "#7c3aed",  # --data-purple
    "#2563eb",  # --data-blue
    "#06b6d4",  # --data-cyan
    "#16a34a",  # --status-success
    "#ea580c",  # --data-orange
    "#dc2626",  # --status-error
    "#f59e0b",  # --status-warning
    "#6b7280",  # --text-secondary
]

# One border colour per EP group (cycles if ep > len)
_EP_BORDER_COLORS = [
    "#16a34a",  # --status-success
    "#ea580c",  # --data-orange
    "#06b6d4",  # --data-cyan
    "#f59e0b",  # --status-warning
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RankInfo:
    """All coordinates associated with a single global GPU rank."""

    global_rank: int
    node: int
    local_gpu: int
    tp_rank: int
    pp_stage: int
    dp_rank: int
    ep_rank: int
    layer_start: int
    layer_end: int


# ---------------------------------------------------------------------------
# Rank mapping
# ---------------------------------------------------------------------------


def compute_rank_mapping(
    par: ParallelismConfig,
    hw: HardwareConfig,
    num_layers: int,
) -> List[RankInfo]:
    """Compute rank-to-parallelism mapping for all GPU ranks.

    Mapping order (innermost to outermost): TP → EP → PP → DP.
    This places TP ranks on the same node (intra-node fast interconnect).

    Total ranks = tp * ep * pp * dp.
    """
    tp, ep, pp, dp = par.tp, par.ep, par.pp, par.dp
    total_ranks = tp * ep * pp * dp
    layers_per_stage = num_layers // pp if pp > 0 else num_layers

    result: List[RankInfo] = []
    for global_rank in range(total_ranks):
        remainder = global_rank

        tp_rank = remainder % tp
        remainder //= tp

        ep_rank = remainder % ep
        remainder //= ep

        pp_stage = remainder % pp
        remainder //= pp

        dp_rank = remainder  # outermost

        node = global_rank // hw.devices_per_node
        local_gpu = global_rank % hw.devices_per_node

        layer_start = pp_stage * layers_per_stage
        layer_end = layer_start + layers_per_stage - 1

        result.append(
            RankInfo(
                global_rank=global_rank,
                node=node,
                local_gpu=local_gpu,
                tp_rank=tp_rank,
                pp_stage=pp_stage,
                dp_rank=dp_rank,
                ep_rank=ep_rank,
                layer_start=layer_start,
                layer_end=layer_end,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Logical mesh figure
# ---------------------------------------------------------------------------


def build_logical_mesh_figure(
    par: ParallelismConfig,
    hw: HardwareConfig,
    num_layers: int,
    max_dp_shown: int = 2,
) -> go.Figure:
    """Build a 2-D logical mesh figure.

    Axes:
      - X: TP rank (within an EP group)
      - Y: PP stage (reversed so stage 0 is at the top)

    Groups are laid out with gaps:
      - EP groups offset along X: x_base = ep_rank * (tp + 1)
      - DP groups offset along Y: y_base = dp_rank * (pp + 1)

    PP stages use different fill colours.
    EP groups use different marker border colours (when ep > 1).
    DP groups beyond the first are shown at lower opacity (0.5).
    Hover text shows: rank, node, local GPU, TP/PP/EP/DP coords, layer range.
    """
    tp, ep, pp, dp = par.tp, par.ep, par.pp, par.dp
    ranks = compute_rank_mapping(par, hw, num_layers)

    dp_groups_to_show = min(dp, max_dp_shown)

    fig = go.Figure()

    for dp_rank in range(dp_groups_to_show):
        opacity = 1.0 if dp_rank == 0 else 0.5
        y_base = dp_rank * (pp + 1)

        for ep_rank in range(ep):
            border_color = _EP_BORDER_COLORS[ep_rank % len(_EP_BORDER_COLORS)]
            x_base = ep_rank * (tp + 1)

            for pp_stage in range(pp):
                fill_color = _PP_STAGE_COLORS[pp_stage % len(_PP_STAGE_COLORS)]

                # Collect ranks for this (dp, ep, pp) slice
                slice_ranks = [
                    r
                    for r in ranks
                    if r.dp_rank == dp_rank
                    and r.ep_rank == ep_rank
                    and r.pp_stage == pp_stage
                ]

                xs = [x_base + r.tp_rank for r in slice_ranks]
                ys = [y_base + pp_stage for r in slice_ranks]
                texts = [f"r{r.global_rank}" for r in slice_ranks]
                hover_texts = [
                    (
                        f"<b>Rank {r.global_rank}</b><br>"
                        f"Node {r.node} · Local GPU {r.local_gpu}<br>"
                        f"TP {r.tp_rank} · PP {r.pp_stage} · EP {r.ep_rank} · DP {r.dp_rank}<br>"
                        f"Layers {r.layer_start}–{r.layer_end}"
                    )
                    for r in slice_ranks
                ]

                marker_line = (
                    dict(color=border_color, width=2)
                    if ep > 1
                    else dict(color="white", width=1)
                )

                trace_name = f"PP{pp_stage}"
                if ep > 1:
                    trace_name += f" EP{ep_rank}"
                if dp_groups_to_show > 1:
                    trace_name += f" DP{dp_rank}"

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+text",
                        name=trace_name,
                        text=texts,
                        textposition="middle center",
                        textfont=dict(color="white", size=9),
                        hovertext=hover_texts,
                        hoverinfo="text",
                        opacity=opacity,
                        marker=dict(
                            symbol="square",
                            size=34,
                            color=fill_color,
                            line=marker_line,
                        ),
                        showlegend=True,
                    )
                )

    _grid_color = "rgba(0, 0, 0, 0.06)"
    fig.update_layout(
        template="plotly_white",
        font=_PLOTLY_FONT,
        plot_bgcolor=_CHART_BG,
        paper_bgcolor=_CHART_BG,
        title=dict(
            text="Device Logical Mesh",
            font=_PLOTLY_TITLE_FONT,
        ),
        xaxis=dict(
            title="TP rank",
            tickmode="linear",
            dtick=1,
            gridcolor=_grid_color,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            title="PP stage",
            autorange="reversed",
            tickmode="linear",
            dtick=1,
            gridcolor=_grid_color,
            zeroline=False,
            showline=False,
        ),
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(title="Group"),
        hoverlabel=_HOVERLABEL,
    )

    return fig
