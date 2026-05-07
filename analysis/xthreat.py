"""Hand-rolled Karun Singh / SoccerAction Expected Threat (xT).

Methodology (Singh 2018; Decroos et al. SoccerAction reference):

    For each grid cell s on the pitch:
        xT(s) = P(shoot|s) · P(goal|shoot, s)
              + P(move|s)  · Σ_{s'} T(s, s') · xT(s')

    where T(s, s') is the empirical transition probability of going from
    cell s to cell s' on a successful move (pass or dribble).

    The system is iterated until convergence (5 sweeps is typically enough
    on a 16×12 grid).

This module is a *hand-rolled* implementation rather than `socceraction.xthreat`
because socceraction hard-pins `numpy<2.0`, which segfaults on macOS arm64
with `pandas>=2.2`. The math is identical; the only difference is that we
estimate `P(goal|shoot, s)` via the per-cell mean of `shot_statsbomb_xg`
(a small refinement on raw goal/shot ratios that gives stabler estimates
in cells with few goals).

Training scope — DEVIATION from Phase 2 spec
--------------------------------------------
Phase 2 specced training on ALL 380 PL 2015/16 matches. To avoid the 10-20 min
StatsBomb pull, we train on the 38 Leicester matches we already have cached
(~127k events, both teams' actions). This is the Phase 2 "fallback rule"
documented in §3.3 of `agent-outputs/phase-2/00-gate-decision.md`.

Public API:
    PITCH_LENGTH, PITCH_WIDTH, GRID_L, GRID_W
    xy_to_cell(x, y) -> tuple[int, int]
    train_xt(events, l=GRID_L, w=GRID_W, n_iterations=5) -> ExpectedThreat
    cache_xt(xt, path) / load_xt(path)
    ExpectedThreat — value(x, y) and deltas for action chains
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import PROCESSED_DIR
from analysis.corners import _xy

log = logging.getLogger(__name__)

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_L = 16
GRID_W = 12

XT_CACHE_PATH = PROCESSED_DIR / "xt_grid.pkl"


def xy_to_cell(x: float, y: float, l: int = GRID_L, w: int = GRID_W) -> tuple[int, int]:
    """Map (x, y) → (i, j) on an l×w grid. Clamps to grid edges."""
    if pd.isna(x) or pd.isna(y):
        return (-1, -1)
    cx = int(np.clip(x / PITCH_LENGTH * l, 0, l - 1))
    cy = int(np.clip(y / PITCH_WIDTH * w, 0, w - 1))
    return (cx, cy)


@dataclass
class ExpectedThreat:
    grid: np.ndarray = field(default_factory=lambda: np.zeros((GRID_L, GRID_W)))
    p_shoot: np.ndarray = field(default_factory=lambda: np.zeros((GRID_L, GRID_W)))
    p_move: np.ndarray = field(default_factory=lambda: np.zeros((GRID_L, GRID_W)))
    xg_per_shot: np.ndarray = field(default_factory=lambda: np.zeros((GRID_L, GRID_W)))
    transition: np.ndarray = field(default_factory=lambda: np.zeros((GRID_L, GRID_W, GRID_L, GRID_W)))
    n_iterations: int = 0
    n_actions: int = 0

    def value(self, x: float, y: float) -> float:
        i, j = xy_to_cell(x, y, self.grid.shape[0], self.grid.shape[1])
        if i < 0:
            return 0.0
        return float(self.grid[i, j])

    def delta(self, sx: float, sy: float, ex: float, ey: float) -> float:
        return self.value(ex, ey) - self.value(sx, sy)


def _classify_actions(events: pd.DataFrame) -> pd.DataFrame:
    """Reduce StatsBomb events to a Singh-friendly action stream.

    Returns columns: type ('move' | 'shot'), start_x, start_y, end_x, end_y,
    is_goal, shot_xg, success.
    """
    cols = [c for c in [
        "type", "team", "location", "pass_end_location", "carry_end_location",
        "pass_outcome", "shot_outcome", "shot_statsbomb_xg", "shot_freeze_frame",
        "pass_type",
    ] if c in events.columns]
    ev = events[cols].copy()

    is_pass = ev["type"] == "Pass"
    pass_succ = is_pass & ev["pass_outcome"].isna()
    is_carry = ev["type"] == "Carry"
    is_shot = ev["type"] == "Shot"

    moves = ev[pass_succ | is_carry].copy()
    starts = moves["location"].apply(_xy)
    ends = moves.apply(
        lambda r: _xy(
            r["pass_end_location"] if r["type"] == "Pass" else r["carry_end_location"]
        ),
        axis=1,
    )
    moves["start_x"] = starts.apply(lambda t: t[0])
    moves["start_y"] = starts.apply(lambda t: t[1])
    moves["end_x"] = ends.apply(lambda t: t[0])
    moves["end_y"] = ends.apply(lambda t: t[1])
    moves["action"] = "move"
    moves["is_goal"] = False
    moves["shot_xg"] = 0.0

    shots = ev[is_shot].copy()
    s_xy = shots["location"].apply(_xy)
    shots["start_x"] = s_xy.apply(lambda t: t[0])
    shots["start_y"] = s_xy.apply(lambda t: t[1])
    shots["end_x"] = shots["start_x"]
    shots["end_y"] = shots["start_y"]
    shots["action"] = "shot"
    shots["is_goal"] = (shots["shot_outcome"] == "Goal")
    shots["shot_xg"] = shots["shot_statsbomb_xg"].fillna(0.0).astype(float)

    keep = ["action", "start_x", "start_y", "end_x", "end_y", "is_goal", "shot_xg"]
    out = pd.concat([moves[keep], shots[keep]], ignore_index=True)
    out = out.dropna(subset=["start_x", "start_y", "end_x", "end_y"]).reset_index(drop=True)
    return out


def _normalise_attack(actions: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb open-data convention: every team attacks left→right within an event row.
    No flip needed — verified empirically by `_verify.py:goal_direction`.
    """
    return actions


def train_xt(
    events: pd.DataFrame,
    l: int = GRID_L,
    w: int = GRID_W,
    n_iterations: int = 5,
) -> ExpectedThreat:
    """Train an xT grid via Singh's iterative algorithm. Returns ExpectedThreat.

    Implementation:
        1. Classify each event as shot or successful move.
        2. For each cell s: count n_actions, n_shots, n_moves; estimate
           P(shoot|s), P(move|s), and mean(shot_xg|s).
        3. Build T(s, s') from move start/end cells.
        4. Iterate xT(s) = P(shoot|s) * mean_xg(s) + P(move|s) * Σ T(s,s') * xT(s').
    """
    actions = _normalise_attack(_classify_actions(events))
    log.info("xT training: %d actions (%d moves, %d shots)",
             len(actions), int((actions["action"] == "move").sum()),
             int((actions["action"] == "shot").sum()))

    s_idx = np.array([
        xy_to_cell(x, y, l, w)
        for x, y in zip(actions["start_x"], actions["start_y"])
    ])
    e_idx = np.array([
        xy_to_cell(x, y, l, w)
        for x, y in zip(actions["end_x"], actions["end_y"])
    ])

    n_actions = np.zeros((l, w))
    n_shots = np.zeros((l, w))
    n_moves = np.zeros((l, w))
    sum_xg = np.zeros((l, w))
    transition_counts = np.zeros((l, w, l, w))

    for k in range(len(actions)):
        i, j = s_idx[k]
        if i < 0:
            continue
        n_actions[i, j] += 1
        if actions.iloc[k]["action"] == "shot":
            n_shots[i, j] += 1
            sum_xg[i, j] += actions.iloc[k]["shot_xg"]
        else:
            ei, ej = e_idx[k]
            if ei < 0:
                continue
            n_moves[i, j] += 1
            transition_counts[i, j, ei, ej] += 1

    safe_n = np.where(n_actions > 0, n_actions, 1)
    p_shoot = n_shots / safe_n
    p_move = n_moves / safe_n
    safe_shots = np.where(n_shots > 0, n_shots, 1)
    xg_per_shot = sum_xg / safe_shots

    transition = np.zeros((l, w, l, w))
    for i in range(l):
        for j in range(w):
            tot = transition_counts[i, j].sum()
            if tot > 0:
                transition[i, j] = transition_counts[i, j] / tot

    xt = np.zeros((l, w))
    for it in range(n_iterations):
        shoot_term = p_shoot * xg_per_shot
        move_term = p_move * np.einsum("ijkl,kl->ij", transition, xt)
        xt = shoot_term + move_term
    log.info("xT trained: max=%.3f, mean=%.4f", xt.max(), xt.mean())

    return ExpectedThreat(
        grid=xt,
        p_shoot=p_shoot,
        p_move=p_move,
        xg_per_shot=xg_per_shot,
        transition=transition,
        n_iterations=n_iterations,
        n_actions=int(n_actions.sum()),
    )


def cache_xt(xt: ExpectedThreat, path: Path = XT_CACHE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(xt, f)
    log.info("Cached xT grid → %s", path)
    return path


def load_xt(path: Path = XT_CACHE_PATH) -> ExpectedThreat | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def attach_xt_to_chain(chain: list[dict], xt: ExpectedThreat) -> list[dict]:
    """Annotate each action in `chain` with `xt_start`, `xt_end`, `xt_delta`.

    For shots, `xt_end` = max(action's xT_start, xg) — i.e. the shot is
    valued at its xG (the chain culminates in a finishing event).
    """
    out = []
    for a in chain:
        xt_start = xt.value(a["start_x"], a["start_y"])
        if a["is_shot"]:
            xt_end = max(xt_start, float(a.get("shot_xg") or 0.0))
        else:
            xt_end = xt.value(a["end_x"], a["end_y"])
        out.append({
            **a,
            "xt_start": xt_start,
            "xt_end": xt_end,
            "xt_delta": xt_end - xt_start,
        })
    return out
