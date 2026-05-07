"""Plotting layer — one function per figure, all return matplotlib Figure objects.

Phase 3a scope: only `fig1_delivery_endzone_heatmap` and
`fig2_freeze_frame_composite` are populated (Fig 3 / Fig 4 land in Phase 3b).

Style: print-grade A4 cards. mplsoccer pitches; viridis for continuous heatmaps,
Okabe-Ito for categorical highlights (CB-safe).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch
from mplsoccer import VerticalPitch

from analysis import ASSETS_DIR
from analysis.corners import Routine, label_sequences_with_routine

log = logging.getLogger(__name__)

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "black": "#000000",
}

PRINT_DPI = 300
TITLE_FONT = {"fontsize": 13, "fontweight": "bold", "color": "#111"}
SUBTITLE_FONT = {"fontsize": 10, "color": "#444"}
ANNOT_FONT = {"fontsize": 8, "color": "#222"}
CAPTION_FONT = {"fontsize": 7.5, "color": "#666"}


def _cov_ellipse_vertical(
    xy: np.ndarray, n_std: float = 1.96, **kwargs
) -> Ellipse | None:
    """95%-coverage covariance ellipse for a `VerticalPitch` ax.

    `xy` is shape (N, 2) in (pitch_x, pitch_y) order. We swap to (pitch_y, pitch_x)
    so the resulting Ellipse renders correctly on a vertical-orientation pitch.
    """
    if len(xy) < 3:
        return None
    swapped = xy[:, [1, 0]]
    cov = np.cov(swapped, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.clip(eigvals, 0, None))
    return Ellipse(
        swapped.mean(axis=0), width=width, height=height, angle=angle, **kwargs
    )


def fig1_delivery_endzone_heatmap(
    sequences: pd.DataFrame,
    routines: list[Routine],
    headline: dict | None = None,
) -> Figure:
    """Card A Fig 1 — corner end-location KDE with top-2 routine annotations.

    `sequences` is the output of `corners.extract_corner_sequences`.
    `routines` is the list of `Routine` objects chosen by `select_top_routines`.
    `headline` is an optional dict of headline numbers for the caption.
    """
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pad_top=2,
        pad_bottom=2,
        pad_left=4,
        pad_right=4,
        goal_type="box",
        pitch_color="#fafafa",
        line_color="#222",
        linewidth=1.1,
    )
    fig, ax = pitch.draw(figsize=(7.4, 8.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.94)

    seq = sequences[
        (sequences["swing"].isin(["inswing", "outswing", "straight"]))
        & (sequences["end_zone"] != "short")
        & sequences["end_x"].notna()
        & sequences["end_y"].notna()
    ].copy()

    if len(seq) > 0:
        pitch.kdeplot(
            seq["end_x"], seq["end_y"],
            ax=ax, cmap="viridis", levels=14, fill=True, alpha=0.62,
            bw_adjust=0.55, thresh=0.05,
        )

    pitch.arrows(
        seq["start_x"], seq["start_y"],
        seq["end_x"], seq["end_y"],
        ax=ax, color="#222", width=0.55, alpha=0.06,
        headwidth=2.6, headlength=2.6,
    )

    routine_colors = [OKABE_ITO["orange"], OKABE_ITO["blue"]]
    routine_letters = ["A", "B"]
    seq_labelled = label_sequences_with_routine(seq, routines)

    annot_anchors = [
        dict(xytext_xy=(8, 88), ha="left", connector_offset=(-2, 6)),
        dict(xytext_xy=(72, 88), ha="right", connector_offset=(2, 6)),
    ]

    summary_lines = []
    for i, r in enumerate(routines):
        sub = seq_labelled[seq_labelled["routine_label"] == r.label()]
        if sub.empty:
            continue
        color = routine_colors[i % len(routine_colors)]

        median_start_x = sub["start_x"].median()
        median_start_y = sub["start_y"].median()
        median_end_x = sub["end_x"].median()
        median_end_y = sub["end_y"].median()

        pitch.arrows(
            median_start_x, median_start_y,
            median_end_x, median_end_y,
            ax=ax, color=color, width=2.6, alpha=0.95,
            headwidth=4.5, headlength=5.5, zorder=5,
        )

        end_pts = sub[["end_x", "end_y"]].to_numpy()
        ell = _cov_ellipse_vertical(
            end_pts, n_std=1.96, edgecolor=color, fill=True,
            facecolor=color, alpha=0.18, lw=1.5, zorder=4,
        )
        if ell is not None:
            ax.add_patch(ell)

        circle = plt.Circle(
            (median_end_y, median_end_x), 1.7,
            color=color, fill=True, alpha=0.96, zorder=6,
        )
        ax.add_patch(circle)
        ax.text(
            median_end_y, median_end_x,
            routine_letters[i],
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="white", zorder=7,
        )

        n_corners = len(sub)
        sum_xg = sub["seq_xg"].sum()
        n_shots = int(sub["seq_n_shots"].sum())
        n_goals = int(sub["seq_goal"].sum())
        xg_per = sum_xg / n_corners if n_corners else 0
        shot_rate = n_shots / n_corners if n_corners else 0

        anchor = annot_anchors[i]
        ax.annotate(
            (
                f"$\\bf{{Routine\\ {routine_letters[i]}}}$  {r.label().replace('_', ' ')}\n"
                f"n = {n_corners} corners · {n_shots} shots · {n_goals} goal{'s' if n_goals != 1 else ''}\n"
                f"xG / corner = {xg_per:.3f}  ·  shot rate = {shot_rate:.0%}"
            ),
            xy=(median_end_y, median_end_x),
            xytext=anchor["xytext_xy"],
            ha=anchor["ha"], va="bottom",
            fontsize=8.5, color="#111",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1, alpha=0.7,
                            connectionstyle="arc3,rad=0.12"),
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=color, lw=1.1, alpha=0.97),
            zorder=8,
        )
        summary_lines.append(
            f"{routine_letters[i]}: {r.label()} — n={n_corners}, ΣxG={sum_xg:.2f}, "
            f"{n_goals} goal{'s' if n_goals != 1 else ''}"
        )

    n_total = int(headline["n_corners"]) if headline else len(sequences)
    sum_xg_all = float(headline["sum_xg"]) if headline else float(sequences["seq_xg"].sum())
    n_shots_all = int(headline["n_shots"]) if headline else int(sequences["seq_n_shots"].sum())
    n_short = int(
        ((sequences["swing"] == "short") | (sequences["end_zone"] == "short")).sum()
    )

    fig.text(
        0.06, 0.955,
        "Leicester's attacking corners — where the deliveries land",
        ha="left", **TITLE_FONT,
    )
    fig.text(
        0.06, 0.928,
        f"PL 2015/16 · {n_total} corners · {n_shots_all} shots · ΣxG = {sum_xg_all:.2f}",
        ha="left", **SUBTITLE_FONT,
    )

    caption_lines = [
        f"Heatmap: KDE of `pass_end_location` for the {len(seq)} crossed deliveries (viridis). "
        f"Thin grey arrows = all crossed corners.",
        "Routines ranked by xG-per-corner with n≥10 floor and Wilson-95 CIs; one routine per side; "
        "ellipses = 95%-coverage covariance.",
        f"Short corners (n={n_short}) excluded from routine ranking.",
        "Source: StatsBomb open data · attacks left→right · pitch 120×80 (StatsBomb units).",
    ]
    fig.text(0.06, 0.075, "\n".join(caption_lines), ha="left", **CAPTION_FONT)
    return fig


def fig2_freeze_frame_composite(
    routine_frames: dict[str, pd.DataFrame],
    routines: list[Routine],
    routine_summaries: pd.DataFrame,
) -> Figure:
    """Card A Fig 2 — KDE composites of attacker / defender / GK positions at the
    instant of contact for the top-2 routines.

    `routine_frames` is the dict from `freeze_frames.build_routine_freeze_frames`.
    `routine_summaries` is the output of `corners.summarise_routines`.
    """
    if not routines:
        raise ValueError("Need at least one routine to plot Fig 2")

    n_panels = len(routines)
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pad_top=2,
        pad_bottom=4,
        goal_type="box",
        pitch_color="#fafafa",
        line_color="#222",
        linewidth=1.05,
    )
    fig, axes = pitch.draw(figsize=(13.5, 8.4), nrows=1, ncols=n_panels)
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.04, right=0.96, wspace=0.04)

    panel_letters = ["A", "B"]

    for i, (r, ax) in enumerate(zip(routines, axes)):
        frames = routine_frames.get(r.label(), pd.DataFrame())
        n_shots = frames["shot_id"].nunique() if not frames.empty else 0

        if not frames.empty:
            attackers = frames[(frames["teammate"]) & (~frames["is_shooter"]) & (~frames["is_gk"])]
            defenders = frames[(~frames["teammate"]) & (~frames["is_gk"])]
            gks = frames[frames["is_gk"]]
            shooters = frames[frames["is_shooter"]]

            if len(defenders) >= 5:
                pitch.kdeplot(
                    defenders["x"], defenders["y"],
                    ax=ax, cmap="Reds", levels=8, alpha=0.75, fill=True,
                    bw_adjust=0.65, thresh=0.06, zorder=2,
                )
            if len(attackers) >= 5:
                pitch.kdeplot(
                    attackers["x"], attackers["y"],
                    ax=ax, cmap="Blues", levels=8, alpha=0.55, fill=True,
                    bw_adjust=0.65, thresh=0.06, zorder=3,
                )

            pitch.scatter(
                defenders["x"], defenders["y"],
                ax=ax, s=18, color=OKABE_ITO["vermillion"], alpha=0.55,
                edgecolor="white", linewidth=0.4, zorder=4,
            )
            pitch.scatter(
                attackers["x"], attackers["y"],
                ax=ax, s=22, color=OKABE_ITO["blue"], alpha=0.7,
                edgecolor="white", linewidth=0.4, zorder=5,
            )
            if len(shooters) > 0:
                med_shooter_x = shooters["x"].median()
                med_shooter_y = shooters["y"].median()
                pitch.scatter(
                    [med_shooter_x], [med_shooter_y],
                    ax=ax, s=320, marker="*",
                    color=OKABE_ITO["yellow"], edgecolor="black",
                    linewidth=1.2, zorder=8,
                )
            if len(gks) > 0:
                med_gk_x = gks["x"].median()
                med_gk_y = gks["y"].median()
                pitch.scatter(
                    [med_gk_x], [med_gk_y],
                    ax=ax, s=180, marker="D",
                    color="#222", edgecolor="white", linewidth=1.2, zorder=8,
                )

            seq_groups = frames.groupby("shot_id").first()
            for _, row in seq_groups.iterrows():
                pass

        try:
            sumrow = routine_summaries[
                (routine_summaries["side"] == r.side)
                & (routine_summaries["swing"] == r.swing)
                & (routine_summaries["end_zone"] == r.end_zone)
            ].iloc[0]
            n_corners = int(sumrow["n_corners"])
            sum_xg = float(sumrow["sum_xg"])
            n_routine_shots = int(sumrow["n_shots"])
            n_goals = int(sumrow["n_goals"])
        except IndexError:
            n_corners = 0; sum_xg = 0.0; n_routine_shots = 0; n_goals = 0

        readable = r.label().replace("_", " ").replace("→", "→")

        ax.set_title(
            f"Panel {panel_letters[i]} — {readable}",
            loc="center", fontsize=11.5, fontweight="bold", color="#111", pad=6,
        )
        ax.text(
            40, 64,
            f"n = {n_corners} corners ({n_routine_shots} shots, {n_goals} goal{'s' if n_goals != 1 else ''}) · "
            f"ΣxG = {sum_xg:.2f} · n freeze-frames = {n_shots}",
            ha="center", va="center", fontsize=9, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", lw=0.6, alpha=0.92),
            zorder=10,
        )

        corner_y = 0 if r.side == "right" else 80
        pitch.arrows(
            120, corner_y, 109, 40,
            ax=ax, color=OKABE_ITO["green"], width=1.8, alpha=0.85,
            headwidth=4.5, headlength=5.5, zorder=2,
        )
        pitch.annotate(
            f"{r.swing} from {r.side}",
            xy=(122, corner_y), ax=ax,
            ha="center", va="center", fontsize=8.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", lw=0.5),
            zorder=10,
        )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE_ITO["blue"],
               markersize=9, label="Leicester attacker", markeredgecolor="white"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE_ITO["vermillion"],
               markersize=9, label="Defender", markeredgecolor="white"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=OKABE_ITO["yellow"],
               markersize=15, label="Shooter (median)", markeredgecolor="black"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#222",
               markersize=10, label="Goalkeeper (median)", markeredgecolor="white"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, 0.05), frameon=False, fontsize=9,
    )

    fig.text(
        0.04, 0.955,
        "Leicester corners — the marking picture at the instant of contact",
        ha="left", **TITLE_FONT,
    )
    fig.text(
        0.04, 0.928,
        "KDE composites of attackers (blue) and defenders (red) at shot for the two top-xG routines.",
        ha="left", **SUBTITLE_FONT,
    )

    caption_lines = [
        "Each panel composites every freeze-frame from the routine's shots into one cloud — "
        "blue density = Leicester attackers; red = defenders; ★ = median shooter; ◆ = median GK.",
        "Frames are populated nearest-the-ball (median ~12 players visible per shot); "
        "the absence of density at the back post does NOT mean no defender is there.",
        "Scatter at low alpha keeps every individual data point honest; KDE gives the recurring shape.",
        "Source: StatsBomb open data · attacks left→right · pitch 120×80 (StatsBomb units).",
    ]
    fig.text(0.04, 0.02, "\n".join(caption_lines), ha="left", **CAPTION_FONT)
    return fig


def fig3_goal_kick_recovery(
    sequences: pd.DataFrame,
    zone_summary: pd.DataFrame,
    long_share: float,
) -> Figure:
    """Card B Fig 3 — long goal-kick landing zones, first-contact and second-ball.

    `sequences` is the output of `goal_kicks.extract_gk_sequences`.
    `zone_summary` is the output of `goal_kicks.summarise_zones`.
    `long_share` is the long-GK fraction (the headline ~96%).
    """
    pitch = VerticalPitch(
        half=False,
        pitch_type="statsbomb",
        pad_top=2, pad_bottom=2, pad_left=4, pad_right=4,
        pitch_color="#fafafa", line_color="#222", linewidth=1.05,
    )
    fig, ax = pitch.draw(figsize=(8.0, 11.0))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.94)

    long_seqs = sequences[sequences["gk_type"] == "long"].copy()
    long_seqs = long_seqs.dropna(subset=["end_x", "end_y"])

    if len(long_seqs) >= 5:
        pitch.kdeplot(
            long_seqs["end_x"], long_seqs["end_y"],
            ax=ax, cmap="viridis", levels=12, fill=True, alpha=0.55,
            bw_adjust=0.6, thresh=0.05, zorder=2,
        )

    pitch.scatter(
        long_seqs["end_x"], long_seqs["end_y"],
        ax=ax, s=14, color="#222", alpha=0.18, edgecolor="white", linewidth=0.3,
        zorder=3,
    )

    qual = zone_summary[zone_summary["qualifies"]].copy()
    qual = qual.sort_values("n", ascending=False).head(3).reset_index(drop=True)

    zone_centroids = (
        long_seqs.groupby("zone")
        .agg(cx=("end_x", "median"), cy=("end_y", "median"))
        .reset_index()
    )

    palette = [OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["green"]]
    n_total = max(qual["n"].sum(), 1)

    callout_anchor_y_by_rank = [14.0, 40.0, 66.0]
    qual_sorted_left_to_right = qual.assign(
        cy=qual["zone"].map(lambda z: zone_centroids.set_index("zone")["cy"].get(z, 40.0))
    ).sort_values("cy").reset_index(drop=True)
    anchor_y_lookup = {
        row["zone"]: callout_anchor_y_by_rank[i % len(callout_anchor_y_by_rank)]
        for i, row in qual_sorted_left_to_right.iterrows()
    }
    callout_anchors = []
    for _, row in qual.iterrows():
        cent = zone_centroids[zone_centroids["zone"] == row["zone"]]
        if cent.empty:
            callout_anchors.append(None)
            continue
        callout_anchors.append((110.0, anchor_y_lookup.get(row["zone"], 40.0)))

    for i, row in qual.iterrows():
        cent = zone_centroids[zone_centroids["zone"] == row["zone"]]
        if cent.empty:
            continue
        cx, cy = float(cent["cx"].iloc[0]), float(cent["cy"].iloc[0])
        color = palette[i % len(palette)]
        width = 1.6 + 5.0 * (row["n"] / n_total)

        pitch.arrows(
            6, 40, cx, cy,
            ax=ax, color=color, width=width, alpha=0.92,
            headwidth=4.5, headlength=5.5, zorder=5,
        )

        recovered = long_seqs[(long_seqs["zone"] == row["zone"]) & long_seqs["second_ball_won"]]
        pitch.scatter(
            recovered["end_x"], recovered["end_y"],
            ax=ax, s=60, marker="^",
            color=OKABE_ITO["green"], alpha=0.78,
            edgecolor="white", linewidth=0.5, zorder=6,
        )

        readable = row["zone"].replace("_", " ").upper()
        anchor = callout_anchors[i]
        if anchor is None:
            continue
        pitch.annotate(
            (
                f"{readable}\n"
                f"n = {int(row['n'])} GKs\n"
                f"FC won {row['first_contact_won_rate']:.0%}\n"
                f"2nd ball {row['second_ball_won_rate']:.0%}"
            ),
            xy=(cx, cy),
            xytext=anchor,
            ax=ax, ha="center", va="center",
            fontsize=8.0, color="#111", weight="normal",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.7,
                            connectionstyle="arc3,rad=0.08"),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=1.1, alpha=0.97),
            zorder=8,
        )

    fig.text(
        0.06, 0.955,
        "Leicester's goal-kick build-up — long landing zones and recovery",
        ha="left", **TITLE_FONT,
    )
    fig.text(
        0.06, 0.928,
        f"PL 2015/16 · {len(sequences)} goal kicks · "
        f"{int((sequences['gk_type'] == 'long').sum())} long ({long_share:.0%}) · "
        f"{int((sequences['gk_type'] == 'short').sum())} short ({(1-long_share):.0%}) · "
        "first-contact and second-ball within 8 s / 5 events.",
        ha="left", **SUBTITLE_FONT,
    )

    caption_lines = [
        "Heatmap: KDE of long-GK `pass_end_location` (viridis). Black dots = individual landings.",
        "Coloured arrows from the penalty area to the median landing of each top-3 zone (n ≥ 15); arrow width ∝ frequency.",
        "Green ▲ = second-ball recovered (Leicester wins the loose ball after first contact). "
        "Wilson-95 CIs are computed per cell — see methodology appendix.",
        "Source: StatsBomb open data · attacks left→right · pitch 120×80 (StatsBomb units).",
    ]
    fig.text(0.06, 0.04, "\n".join(caption_lines), ha="left", **CAPTION_FONT)
    return fig


def fig4_post_gk_xt_chain(
    chains: list[dict],
    xt_grid: np.ndarray,
    chain_summary: pd.DataFrame,
    vardy_share: float,
    vardy_n: int,
    vardy_denom: int,
) -> Figure:
    """Card B Fig 4 — Singh xT underlay + top-3 post-GK chain templates.

    `chains` is the list from `goal_kicks.all_post_gk_chains` after xT-attachment.
    `xt_grid` is the (l, w) numpy array from the trained xT model.
    `chain_summary` is a DataFrame with one row per chain template — each row
    carries the median start/end coordinates of all three actions (a1, a2, a3),
    so each template renders as three separate arrows (not a polyline through
    only the end-points). This matters because action_k.start may differ from
    action_{k-1}.end whenever the chain spans a contested aerial.
    """
    pitch = VerticalPitch(
        half=False,
        pitch_type="statsbomb",
        pad_top=2, pad_bottom=2, pad_left=4, pad_right=4,
        pitch_color="#fafafa", line_color="#222", linewidth=1.05,
    )
    fig, ax = pitch.draw(figsize=(8.0, 11.0))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.06, right=0.94)

    l_cells, w_cells = xt_grid.shape
    cell_l = 120.0 / l_cells
    cell_w = 80.0 / w_cells
    xt_max = max(float(xt_grid.max()), 1e-6)
    for i in range(l_cells):
        for j in range(w_cells):
            v = xt_grid[i, j] / xt_max
            if v < 0.02:
                continue
            x0 = i * cell_l; y0 = j * cell_w
            cx = x0 + cell_l / 2; cy = y0 + cell_w / 2
            pitch.scatter(
                [cx], [cy], ax=ax,
                s=140 * (cell_l * cell_w / 50),
                marker="s",
                color=plt.cm.Greys(0.25 + 0.55 * v),
                alpha=0.32, zorder=1, edgecolor="none",
            )

    palette = [OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["green"]]
    template_rows = chain_summary.head(3).reset_index(drop=True)
    n_total = max(int(chain_summary["n"].sum()), 1)

    callout_xs = [105.0, 105.0, 105.0]
    callout_ys = [12.0, 40.0, 68.0]

    for i, row in template_rows.iterrows():
        color = palette[i % len(palette)]
        n_here = int(row["n"])
        width = 1.4 + 4.0 * (n_here / n_total)

        action_segments = [
            (row["a1_start_x"], row["a1_start_y"], row["a1_end_x"], row["a1_end_y"]),
            (row["a2_start_x"], row["a2_start_y"], row["a2_end_x"], row["a2_end_y"]),
            (row["a3_start_x"], row["a3_start_y"], row["a3_end_x"], row["a3_end_y"]),
        ]
        for sx, sy, ex, ey in action_segments:
            pitch.arrows(
                sx, sy, ex, ey,
                ax=ax, color=color,
                width=width, alpha=0.92,
                headwidth=4.5, headlength=5.5, zorder=5,
            )

        for k in range(2):
            sx_next = action_segments[k + 1][0]; sy_next = action_segments[k + 1][1]
            ex_prev = action_segments[k][2];     ey_prev = action_segments[k][3]
            gap = float(np.hypot(sx_next - ex_prev, sy_next - ey_prev))
            if gap > 4.0:
                pitch.lines(
                    ex_prev, ey_prev, sx_next, sy_next,
                    ax=ax, color=color, lw=1.0, alpha=0.5,
                    linestyle=":", zorder=4,
                )

        if row.get("vardy_in_chain", False):
            pitch.scatter(
                [action_segments[-1][2]], [action_segments[-1][3]], ax=ax,
                s=160, marker="*",
                color=OKABE_ITO["purple"], edgecolor="white", linewidth=0.8,
                zorder=7,
            )

        vardy_pct = float(row.get("vardy_share_in_template", 0.0))
        pitch.annotate(
            (
                f"$\\bf{{Template\\ {chr(65+i)}}}$ · n = {n_here}  ({n_here / n_total:.0%})\n"
                f"end zone: {row['end_zone_label']}\n"
                f"mean ΔxT (chain) = {row['mean_chain_xt']:+.3f}\n"
                f"Vardy in chain: {vardy_pct:.0%}"
            ),
            xy=(action_segments[-1][2], action_segments[-1][3]),
            xytext=(callout_xs[i], callout_ys[i]),
            ax=ax, ha="center", va="center",
            fontsize=8.0, color="#111",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.7,
                            connectionstyle="arc3,rad=0.10"),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.0, alpha=0.97),
            zorder=8,
        )

    fig.text(
        0.06, 0.955,
        "Post-goal-kick chain templates — where the long ball goes",
        ha="left", **TITLE_FONT,
    )
    fig.text(
        0.06, 0.928,
        f"Greys underlay = Singh xT grid (16×12, trained on Leicester-match events). "
        f"Vardy as actor or GK recipient in {vardy_n}/{vardy_denom} ({vardy_share:.0%}) "
        "post-GK 3-action chains.",
        ha="left", **SUBTITLE_FONT,
    )

    means = []
    for k in (1, 2, 3):
        col = f"a{k}_mean_xt_delta_overall"
        if col in chain_summary.columns and len(chain_summary):
            means.append(float(chain_summary[col].iloc[0]))
        else:
            means.append(0.0)
    fig.text(
        0.06, 0.905,
        f"Mean ΔxT by action position: #1 (GK pass) = {means[0]:+.3f} · "
        f"#2 (first contact / knock-down) = {means[1]:+.3f} · "
        f"#3 (exit pass / shot) = {means[2]:+.3f}",
        ha="left", **SUBTITLE_FONT,
    )

    caption_lines = [
        "Top-3 post-GK 3-action chain templates by frequency (clustering on the chain's end-zone "
        "on a coarse 3×3 grid). Arrow width ∝ template frequency. Three arrows per template — one per action.",
        "Dotted line = spatial gap between action k's end and action k+1's start (a contested aerial / "
        "second-ball recovery between two Leicester actions).",
        "★ = Vardy is the actor or GK recipient on at least one chain in this template. "
        "Source: StatsBomb open data · attacks left→right · pitch 120×80 (StatsBomb units).",
    ]
    fig.text(0.06, 0.04, "\n".join(caption_lines), ha="left", **CAPTION_FONT)
    return fig


def save_fig(fig: Figure, name: str, dpi: int = PRINT_DPI) -> Path:
    """Save figure to reports/assets/<name>.png at 300 dpi (or higher)."""
    if not name.endswith(".png"):
        name = f"{name}.png"
    out = ASSETS_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info("Wrote %s (%.1f KB)", out, out.stat().st_size / 1024)
    return out
