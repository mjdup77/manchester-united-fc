"""Card A — corner extraction, classification, routine ranking.

All filters anchored on Leicester's *attacking* corners (the corners United
would defend). Card A's title:
    "Leicester's attacking corners: how United defends them (and counter-exploits)"

Public API:
    extract_corners(events, team) -> DataFrame
    classify_delivery(row)        -> str
    end_zone(end_xy)              -> str
    extract_corner_sequences(events, team, window_events=6, window_seconds=8.0)
        -> DataFrame  (one row per corner with post-corner shot/xG/goal flags)
    attribute_corner_xg(sequences) -> DataFrame (alias used by spec)
    summarise_routines(sequences) -> DataFrame
    select_top_routines(routines_df, k=2, kde_distinct_fn=None) -> list[tuple]

Routine key = (side, swing, end_zone). n≥10 floor for routine ranking.
xG-per-corner attribution uses the post-corner window:
    first of (a) 8 seconds, (b) next 6 events by index, (c) end of possession.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from analysis import LEICESTER

log = logging.getLogger(__name__)

CORNER_WINDOW_EVENTS = 6
CORNER_WINDOW_SECONDS = 8.0

SHORT_LENGTH_M = 12.0

LEFT_FOOTED_TAKERS = {"Christian Fuchs", "Riyad Mahrez", "Demarai Gray"}
RIGHT_FOOTED_TAKERS = {"Marc Albrighton", "Danny Drinkwater", "Danny Simpson", "Jamie Vardy"}


@dataclass(frozen=True)
class Routine:
    side: str
    swing: str
    end_zone: str

    def label(self) -> str:
        return f"{self.side}-{self.swing} → {self.end_zone}"

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.side, self.swing, self.end_zone)


def _xy(loc) -> tuple[float, float]:
    if loc is None:
        return (np.nan, np.nan)
    if isinstance(loc, float) and pd.isna(loc):
        return (np.nan, np.nan)
    return float(loc[0]), float(loc[1])


def end_zone(end_xy: tuple[float, float]) -> str:
    """Bucket pass_end_location into coach-readable zones (StatsBomb 120×80).

    Convention: corners are taken at x=120; the receiving area is x∈[102, 120].
    Within that area:
        near_post_6yd     x≥114 AND y∈[30,40]
        far_post_6yd      x≥114 AND y∈[40,50]
        penalty_spot      x∈[108,114] AND y∈[33,47]
        edge_d            x∈[102,108] AND y∈[30,50]
        wide              x≥102 BUT outside the central column above
        short             x<102 (short corner / out-of-area)
    """
    x, y = end_xy
    if pd.isna(x) or pd.isna(y):
        return "unknown"
    if x < 102:
        return "short"
    in_central = 30 <= y <= 50
    if x >= 114:
        if 30 <= y < 40:
            return "near_post_6yd"
        if 40 <= y <= 50:
            return "far_post_6yd"
        return "wide"
    if 108 <= x < 114 and 33 <= y <= 47:
        return "penalty_spot"
    if 102 <= x < 108 and in_central:
        return "edge_d"
    return "wide"


def _infer_side(start_y: float) -> str:
    """y<40 ⇒ right side of attack (StatsBomb attacks left→right)."""
    return "right" if start_y < 40 else "left"


def _infer_swing(row: pd.Series) -> str:
    """Pick the most reliable swing label, falling back to side × footedness."""
    if row.get("pass_height") == "Ground Pass" or row.get("pass_length", 99) < SHORT_LENGTH_M:
        return "short"
    if pd.notna(row.get("pass_inswinging")) and bool(row["pass_inswinging"]):
        return "inswing"
    if pd.notna(row.get("pass_outswinging")) and bool(row["pass_outswinging"]):
        return "outswing"
    if pd.notna(row.get("pass_straight")) and bool(row["pass_straight"]):
        return "straight"
    side = row["side"]
    taker = row.get("player", None)
    if taker in LEFT_FOOTED_TAKERS:
        return "inswing" if side == "right" else "outswing"
    if taker in RIGHT_FOOTED_TAKERS:
        return "outswing" if side == "right" else "inswing"
    return "unknown"


def classify_delivery(row: pd.Series) -> str:
    """One of {short, flighted, driven}: the broad delivery type from height + length."""
    if row.get("pass_height") == "Ground Pass" or row.get("pass_length", 99) < SHORT_LENGTH_M:
        return "short"
    if row.get("pass_height") == "High Pass":
        return "flighted"
    return "driven"


def extract_corners(events: pd.DataFrame, team: str = LEICESTER) -> pd.DataFrame:
    """One row per corner pass by `team`, with parsed start/end coords + classifications."""
    c = events[
        (events["type"] == "Pass")
        & (events["pass_type"] == "Corner")
        & (events["team"] == team)
    ].copy()
    starts = c["location"].apply(_xy)
    ends = c["pass_end_location"].apply(_xy)
    c["start_x"] = starts.apply(lambda t: t[0])
    c["start_y"] = starts.apply(lambda t: t[1])
    c["end_x"] = ends.apply(lambda t: t[0])
    c["end_y"] = ends.apply(lambda t: t[1])
    c["side"] = c["start_y"].apply(_infer_side)
    c["delivery_type"] = c.apply(classify_delivery, axis=1)
    c["swing"] = c.apply(_infer_swing, axis=1)
    c["end_zone"] = c.apply(lambda r: end_zone((r["end_x"], r["end_y"])), axis=1)
    return c.reset_index(drop=True)


def extract_corner_sequences(
    events: pd.DataFrame,
    team: str = LEICESTER,
    window_events: int = CORNER_WINDOW_EVENTS,
    window_seconds: float = CORNER_WINDOW_SECONDS,
) -> pd.DataFrame:
    """One row per corner with post-corner sequence outcomes summarised.

    Sequence definition: for each corner, the post-corner window is the rest
    of the same `possession` chain in the same match. We then attribute shots
    to the corner if they have `play_pattern == 'From Corner' AND team == team`
    AND share the corner's `(match_id, possession)`. This matches the project's
    headline anchor numbers (89 shots / 7.99 xG for Leicester).

    `window_events` and `window_seconds` are kept as caps for safety: if a
    possession runs longer than 6 events or 8 seconds beyond the corner we
    still cut off there to avoid second-phase / cleared-and-recovered shots.
    """
    corners = extract_corners(events, team)
    log.info("Extracted %d %s corners", len(corners), team)

    needed_cols = [
        "id", "match_id", "index", "minute", "second", "period", "possession",
        "type", "team", "pass_type", "play_pattern", "shot_statsbomb_xg",
        "shot_outcome", "shot_key_pass_id",
    ]
    have_cols = [c for c in needed_cols if c in events.columns]
    ev = events[have_cols].copy()
    shots_from_corner = ev[
        (ev["type"] == "Shot")
        & (ev["team"] == team)
        & (ev["play_pattern"] == "From Corner")
    ][["match_id", "possession", "index", "id", "shot_statsbomb_xg", "shot_outcome"]].copy()
    shots_from_corner = shots_from_corner.sort_values(["match_id", "index"]).reset_index(drop=True)

    corner_index_by_id = ev[ev["id"].isin(set(corners["id"]))][
        ["match_id", "id", "index"]
    ].rename(columns={"id": "corner_id", "index": "corner_index"})
    corners_idx = corners.merge(
        corner_index_by_id, left_on=["match_id", "id"],
        right_on=["match_id", "corner_id"], how="left",
    )

    poss_corners = corners_idx[["match_id", "possession", "corner_index"]].sort_values(
        ["match_id", "possession", "corner_index"]
    )
    poss_corners["next_corner_idx"] = poss_corners.groupby(["match_id", "possession"])["corner_index"].shift(-1)
    boundary_lookup = poss_corners.set_index(["match_id", "corner_index"])["next_corner_idx"].to_dict()

    rows = []
    for _, c in corners_idx.iterrows():
        c_idx = c["corner_index"]
        c_poss = c["possession"]
        c_match = c["match_id"]
        next_idx = boundary_lookup.get((c_match, c_idx), None)

        candidates = shots_from_corner[
            (shots_from_corner["match_id"] == c_match)
            & (shots_from_corner["possession"] == c_poss)
            & (shots_from_corner["index"] > c_idx)
        ]
        if pd.notna(next_idx):
            candidates = candidates[candidates["index"] < int(next_idx)]

        seq_n_shots = len(candidates)
        seq_xg = float(candidates["shot_statsbomb_xg"].fillna(0).sum())
        seq_goal = bool((candidates["shot_outcome"] == "Goal").any())
        if seq_n_shots:
            first = candidates.iloc[0]
            first_shot_xg = float(first.get("shot_statsbomb_xg") or 0.0)
            first_shot_id = first["id"]
            first_shot_outcome = first["shot_outcome"]
        else:
            first_shot_xg = 0.0
            first_shot_id = None
            first_shot_outcome = None

        rows.append({
            "corner_id": c["id"],
            "match_id": c["match_id"],
            "side": c["side"],
            "swing": c["swing"],
            "delivery_type": c["delivery_type"],
            "end_zone": c["end_zone"],
            "start_x": c["start_x"],
            "start_y": c["start_y"],
            "end_x": c["end_x"],
            "end_y": c["end_y"],
            "player": c["player"],
            "seq_n_shots": seq_n_shots,
            "seq_xg": seq_xg,
            "seq_goal": seq_goal,
            "first_shot_xg": first_shot_xg,
            "first_shot_id": first_shot_id,
            "first_shot_outcome": first_shot_outcome,
        })
    return pd.DataFrame(rows)


def attribute_corner_xg(sequences: pd.DataFrame) -> pd.DataFrame:
    """Spec-named alias — `extract_corner_sequences` already attributes xG inline."""
    return sequences


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion (k successes, n trials)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarise_routines(sequences: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Aggregate `sequences` by (side, swing, end_zone) → routine summary.

    Returned columns (one row per cell):
        side, swing, end_zone, n_corners, n_shots, n_goals, sum_xg,
        xg_per_corner, shot_rate, goal_rate, shot_ci_lo, shot_ci_hi,
        goal_ci_lo, goal_ci_hi, qualifies (bool)  — qualifies = n≥min_n AND
        end_zone != 'short' AND swing != 'short'.
    """
    cells = (
        sequences.groupby(["side", "swing", "end_zone"], dropna=False)
        .agg(
            n_corners=("corner_id", "size"),
            n_shots=("seq_n_shots", "sum"),
            n_goals=("seq_goal", "sum"),
            sum_xg=("seq_xg", "sum"),
        )
        .reset_index()
    )
    cells["xg_per_corner"] = cells["sum_xg"] / cells["n_corners"]
    cells["shot_rate"] = cells["n_shots"].astype(float) / cells["n_corners"]
    cells["goal_rate"] = cells["n_goals"].astype(float) / cells["n_corners"]
    shot_ci = cells.apply(lambda r: _wilson_ci(int(r["n_shots"]), int(r["n_corners"])), axis=1)
    goal_ci = cells.apply(lambda r: _wilson_ci(int(r["n_goals"]), int(r["n_corners"])), axis=1)
    cells["shot_ci_lo"] = [c[0] for c in shot_ci]
    cells["shot_ci_hi"] = [c[1] for c in shot_ci]
    cells["goal_ci_lo"] = [c[0] for c in goal_ci]
    cells["goal_ci_hi"] = [c[1] for c in goal_ci]
    cells["qualifies"] = (
        (cells["n_corners"] >= min_n)
        & (cells["end_zone"] != "short")
        & (cells["swing"] != "short")
        & (cells["swing"] != "unknown")
    )
    return cells.sort_values(["qualifies", "xg_per_corner"], ascending=[False, False]).reset_index(drop=True)


def select_top_routines(
    routines_df: pd.DataFrame,
    k: int = 2,
    min_shots: int = 8,
    kde_distinct_fn: Callable[[Routine, Routine], float] | None = None,
    ci_overlap_tol: float = 0.0,
    one_per_side: bool = False,
) -> list[Routine]:
    """Pick top-k routines by xG-per-corner.

    Filters: must have `qualifies==True` AND `n_shots >= min_shots` so that
    Fig 2 has enough freeze frames per routine for the KDE composite (QC §Fig 2
    test 6). Note that `one_per_side` — the original "one routine per side"
    preference — defaults to False because in this dataset Leicester's
    productive corners cluster on one side; forcing the other side picks a
    routine with too few shots for a stable composite.

    `kde_distinct_fn(routine_a, routine_b)` returns L2 distance between the
    routines' attacker-KDE peak coordinates on the 120×80 grid; used as
    tiebreaker when the top-2 candidates' shot-rate Wilson-95 CIs overlap.
    When None we fall back to point-estimate ranking.
    """
    pool = routines_df[
        routines_df["qualifies"] & (routines_df["n_shots"] >= min_shots)
    ].copy()
    if pool.empty:
        return []

    pool = pool.sort_values("xg_per_corner", ascending=False).reset_index(drop=True)

    chosen: list[Routine] = []
    sides_taken: set[str] = set()

    for _, row in pool.iterrows():
        if len(chosen) >= k:
            break
        if one_per_side and row["side"] in sides_taken:
            continue
        chosen.append(Routine(side=row["side"], swing=row["swing"], end_zone=row["end_zone"]))
        sides_taken.add(row["side"])

    if len(chosen) < k:
        for _, row in pool.iterrows():
            if len(chosen) >= k:
                break
            cand = Routine(side=row["side"], swing=row["swing"], end_zone=row["end_zone"])
            if cand in chosen:
                continue
            chosen.append(cand)

    if kde_distinct_fn is not None and len(chosen) == 2:
        a, b = chosen
        try:
            a_row = pool[(pool["side"] == a.side) & (pool["swing"] == a.swing) & (pool["end_zone"] == a.end_zone)].iloc[0]
            b_row = pool[(pool["side"] == b.side) & (pool["swing"] == b.swing) & (pool["end_zone"] == b.end_zone)].iloc[0]
        except IndexError:
            return chosen
        cis_overlap = (a_row["shot_ci_lo"] - ci_overlap_tol) <= b_row["shot_ci_hi"] and \
                      (b_row["shot_ci_lo"] - ci_overlap_tol) <= a_row["shot_ci_hi"]
        if cis_overlap:
            best_d = kde_distinct_fn(a, b)
            for _, row in pool.iterrows():
                cand = Routine(side=row["side"], swing=row["swing"], end_zone=row["end_zone"])
                if cand == a or cand == chosen[1]:
                    continue
                d = kde_distinct_fn(a, cand)
                if d > best_d:
                    best_d = d
                    chosen[1] = cand

    return chosen


def label_sequences_with_routine(
    sequences: pd.DataFrame, routines: Iterable[Routine]
) -> pd.DataFrame:
    """Add a `routine_label` column tagging each corner sequence with one of the
    chosen routines (or 'other')."""
    label_map: dict[tuple[str, str, str], str] = {}
    for r in routines:
        label_map[r.as_tuple()] = r.label()
    out = sequences.copy()
    out["routine_label"] = out.apply(
        lambda r: label_map.get((r["side"], r["swing"], r["end_zone"]), "other"), axis=1
    )
    return out
