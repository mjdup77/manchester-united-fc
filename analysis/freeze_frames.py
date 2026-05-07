"""Card A Fig 2 — `shot_freeze_frame` parsing, normalisation, KDE compositing.

Each `shot_freeze_frame` is a JSON list of `{location, player, position, teammate}`
records (one per visible player). `teammate=True` means same team as the shooter.

Public API:
    expand_frame(shot)                -> DataFrame
    normalise_to_attack_left_to_right(...)  (no-op for StatsBomb open data — direction
                                              is already unified — kept for API symmetry)
    composite_freeze_frame(shots, role)     -> DataFrame
    average_marking_picture(shots)          -> DataFrame
    build_routine_freeze_frames(events, sequences, routines)
                                            -> dict[Routine -> DataFrame of frames]
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from analysis import LEICESTER
from analysis.corners import Routine, label_sequences_with_routine

log = logging.getLogger(__name__)


def _parse_ff(raw) -> list[dict] | None:
    if raw is None:
        return None
    if isinstance(raw, float) and pd.isna(raw):
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        return raw
    return None


def expand_frame(shot: pd.Series) -> pd.DataFrame:
    """Explode `shot.shot_freeze_frame` into rows of (player, position, x, y, teammate)."""
    ff = _parse_ff(shot["shot_freeze_frame"])
    if not ff:
        return pd.DataFrame(columns=["shot_id", "player", "position", "x", "y", "teammate"])
    rows = []
    for p in ff:
        loc = p.get("location") or [np.nan, np.nan]
        rows.append({
            "shot_id": shot["id"],
            "player": (p.get("player") or {}).get("name"),
            "position": (p.get("position") or {}).get("name"),
            "x": float(loc[0]),
            "y": float(loc[1]),
            "teammate": bool(p.get("teammate", False)),
        })
    return pd.DataFrame(rows)


def normalise_to_attack_left_to_right(
    frame: pd.DataFrame, shot_xy: tuple[float, float]
) -> pd.DataFrame:
    """No-op for StatsBomb open data (direction is unified). Kept for API symmetry.
    Adds `shot_x`, `shot_y` columns for downstream use.
    """
    out = frame.copy()
    out["shot_x"] = shot_xy[0]
    out["shot_y"] = shot_xy[1]
    return out


def build_routine_freeze_frames(
    events: pd.DataFrame,
    sequences: pd.DataFrame,
    routines: list[Routine],
    team: str = LEICESTER,
) -> dict[str, pd.DataFrame]:
    """For each routine, return a long-format DataFrame of all freeze-frame rows
    drawn from the corner shots that belong to that routine.

    Returned dict is keyed by routine label, with each value containing one row
    per (shot_id, frame_player). Adds columns:
        x, y       — frame player position (StatsBomb 120×80)
        teammate   — True if same team as the shooter
        position   — StatsBomb position string
        shot_id    — id of the shot
        shot_x, shot_y — shooter location at shot
        shot_xg
        shot_body_part
        shot_outcome
        is_shooter — True for the row representing the shooter (synthesised, not
                     in the freeze frame which shows everyone *but* the shooter)
        is_gk      — True if `position == 'Goalkeeper'`
    """
    seq_lab = label_sequences_with_routine(sequences, routines)

    shot_lookup_cols = [
        "id", "match_id", "team", "play_pattern", "type",
        "location", "shot_freeze_frame", "shot_statsbomb_xg",
        "shot_outcome", "shot_body_part", "shot_technique", "player",
    ]
    have = [c for c in shot_lookup_cols if c in events.columns]
    shots_idx = (
        events[have]
        .loc[
            (events["type"] == "Shot")
            & (events["team"] == team)
            & (events["play_pattern"] == "From Corner")
            & events["shot_freeze_frame"].notna()
        ]
        .copy()
    )
    shots_idx = shots_idx.set_index("id")

    out: dict[str, pd.DataFrame] = {}
    for r in routines:
        rows = []
        sub_seq = seq_lab[
            (seq_lab["routine_label"] == r.label())
            & (seq_lab["first_shot_id"].notna())
        ]
        for _, c in sub_seq.iterrows():
            sid = c["first_shot_id"]
            if sid not in shots_idx.index:
                continue
            shot = shots_idx.loc[sid]
            shot_loc = shot["location"]
            sx, sy = float(shot_loc[0]), float(shot_loc[1])

            frame = expand_frame(pd.Series({
                "id": sid,
                "shot_freeze_frame": shot["shot_freeze_frame"],
            }))
            if frame.empty:
                continue
            frame = normalise_to_attack_left_to_right(frame, (sx, sy))
            frame["shot_xg"] = float(shot.get("shot_statsbomb_xg") or 0.0)
            frame["shot_body_part"] = shot.get("shot_body_part")
            frame["shot_outcome"] = shot.get("shot_outcome")
            frame["shooter_name"] = shot.get("player")
            frame["is_shooter"] = False
            frame["is_gk"] = frame["position"] == "Goalkeeper"

            shooter_row = pd.DataFrame([{
                "shot_id": sid,
                "player": shot.get("player"),
                "position": "Shooter",
                "x": sx, "y": sy,
                "teammate": True,
                "shot_x": sx, "shot_y": sy,
                "shot_xg": float(shot.get("shot_statsbomb_xg") or 0.0),
                "shot_body_part": shot.get("shot_body_part"),
                "shot_outcome": shot.get("shot_outcome"),
                "shooter_name": shot.get("player"),
                "is_shooter": True,
                "is_gk": False,
            }])
            rows.append(pd.concat([frame, shooter_row], ignore_index=True))
        if rows:
            out[r.label()] = pd.concat(rows, ignore_index=True)
        else:
            out[r.label()] = pd.DataFrame()
    return out


def composite_freeze_frame(frames: pd.DataFrame, role: str = "attackers") -> pd.DataFrame:
    """Subset the long-form frame DataFrame to one of {attackers, defenders, gks}."""
    if frames.empty:
        return frames
    if role == "attackers":
        return frames[frames["teammate"] & ~frames["is_shooter"]].copy()
    if role == "defenders":
        return frames[~frames["teammate"] & ~frames["is_gk"]].copy()
    if role == "gks":
        return frames[frames["is_gk"]].copy()
    raise ValueError(f"Unknown role: {role!r}")


def average_marking_picture(frames: pd.DataFrame) -> pd.DataFrame:
    """Per StatsBomb position label, return centroid (x_mean, y_mean) and spread."""
    if frames.empty:
        return pd.DataFrame()
    g = frames.groupby("position")
    return (
        g.agg(
            n=("x", "size"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
            x_std=("x", "std"),
            y_std=("y", "std"),
            teammate_share=("teammate", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )


def kde_peak_xy(points: pd.DataFrame, grid_resolution: int = 60) -> tuple[float, float] | None:
    """Estimate the (x, y) of peak density of the (x, y) point cloud.

    Used by `corners.select_top_routines` to operationalise the
    visual-distinctiveness tiebreaker (max L2 between routines' attacker-KDE
    peaks). Falls back to None for fewer than 5 points.
    """
    if len(points) < 5:
        return None
    from scipy.stats import gaussian_kde

    xy = points[["x", "y"]].dropna().to_numpy()
    if len(xy) < 5:
        return None
    try:
        kde = gaussian_kde(xy.T, bw_method=0.35)
    except Exception:
        return None
    xs = np.linspace(80, 120, grid_resolution)
    ys = np.linspace(0, 80, grid_resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
    idx = z.argmax()
    return float(grid_x.ravel()[idx]), float(grid_y.ravel()[idx])
