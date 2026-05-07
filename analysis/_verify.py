"""Verification harness — run before any plotting.

Asserts the coordinate / direction / filter assumptions every other module relies on.
If any of these checks fail, the spec's coordinate-system assumptions are wrong and
Figs 1-4 will quietly mislead. Better to find out *now*, before sinking hours into
plotting work.

Run from repo root:
    uv run python -m analysis._verify

Outputs a pass/fail line per check, then writes the summary to docs/_verify.log.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from io import StringIO

import pandas as pd

from analysis import DOCS_DIR, LEICESTER, LEICESTER_EVENTS_PARQUET

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _load_events() -> pd.DataFrame:
    """Load consolidated Leicester season events from data/processed/."""
    if not LEICESTER_EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            f"Consolidated events parquet missing at {LEICESTER_EVENTS_PARQUET}. "
            "Run `uv run python -m analysis.characterise` first."
        )
    return pd.read_parquet(LEICESTER_EVENTS_PARQUET)


def _xy(loc) -> tuple[float, float] | tuple[float, float]:
    """Robust extractor for [x, y] location columns (numpy arrays / lists / None)."""
    if loc is None:
        return (float("nan"), float("nan"))
    if isinstance(loc, float) and pd.isna(loc):
        return (float("nan"), float("nan"))
    return float(loc[0]), float(loc[1])


def check_goal_direction(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 1: ≥95% of Leicester goal shots have shot_end_location[0] > 100.

    StatsBomb open data normalises so the attacking team always plays left→right;
    a goal must therefore land at the opponent goal-line (x ≈ 120). If this is
    less than 95%, attacking-direction assumption is broken.
    """
    goals = events[
        (events["type"] == "Shot")
        & (events["shot_outcome"] == "Goal")
        & (events["team"] == LEICESTER)
    ].copy()
    if goals.empty:
        return False, "Goal direction: NO LEICESTER GOALS FOUND (suspicious)."
    goals["end_x"] = goals["shot_end_location"].apply(lambda v: _xy(v)[0])
    pct = (goals["end_x"] > 100).mean()
    n = len(goals)
    ok = pct >= 0.95
    return ok, f"Goal direction: {pct:.1%} of {n} Leicester goals end at x>100 (need ≥95%)."


def check_corner_origins(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 2: ≥99% of `pass_type=='Corner'` rows have `location` at one of
    the four corner spots (0,0), (0,80), (120,0), (120,80) within ±1m.

    This is the filter-purity check. Anything failing means non-corner deliveries
    are leaking into the corner filter.
    """
    corners = events[(events["type"] == "Pass") & (events["pass_type"] == "Corner")].copy()
    n = len(corners)
    if n == 0:
        return False, "Corner origins: NO CORNERS FOUND (suspicious)."
    corners[["sx", "sy"]] = corners["location"].apply(lambda v: pd.Series(_xy(v)))
    spots = [(0.0, 0.0), (0.0, 80.0), (120.0, 0.0), (120.0, 80.0)]
    in_spot = corners.apply(
        lambda r: any(abs(r["sx"] - x) <= 1.0 and abs(r["sy"] - y) <= 1.0 for x, y in spots),
        axis=1,
    )
    pct = in_spot.mean()
    ok = pct >= 0.99
    return ok, f"Corner origins: {pct:.1%} of {n} corner passes start at a corner spot ±1m (need ≥99%)."


def check_long_high_passes(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 3 (Phase 2 §3.6 wording): among `pass_type=='Goal Kick'` rows with
    `pass_height=='High Pass'`, ≥95% have `pass_length>=30`.

    Anchors the canonical "long GK" definition: pass_height == 'High Pass'. Note
    this assertion is scoped to goal kicks; across all High Pass rows the share
    drops to ~57% (many short floated lobs in open play). The spec only relies
    on the GK-scoped version, which is what this checks.
    """
    gk_high = events[
        (events["type"] == "Pass")
        & (events["pass_type"] == "Goal Kick")
        & (events["pass_height"] == "High Pass")
    ].copy()
    n = len(gk_high)
    if n == 0:
        return False, "Long high-pass GK: NO HIGH-PASS GOAL KICKS FOUND."
    pct = (gk_high["pass_length"] >= 30).mean()
    ok = pct >= 0.95
    return (
        ok,
        f"Long high-pass GK: {pct:.1%} of {n} High-Pass goal kicks have length≥30m "
        f"(need ≥95%).",
    )


def check_freeze_frame_density(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 4: median `len(shot_freeze_frame)` ≥8 across Leicester corner shots,
    min ≥4. Frames sparser than this can't support KDE compositing.
    """
    shots_from_corners = events[
        (events["type"] == "Shot")
        & (events["play_pattern"] == "From Corner")
        & (events["team"] == LEICESTER)
        & (events["shot_freeze_frame"].notna())
    ]
    if shots_from_corners.empty:
        return False, "Freeze-frame density: NO Leicester corner shots with freeze frames."
    import json

    sizes = shots_from_corners["shot_freeze_frame"].apply(
        lambda s: len(json.loads(s)) if isinstance(s, str) else (len(s) if s is not None else 0)
    )
    median = float(sizes.median())
    minimum = int(sizes.min())
    n = len(sizes)
    ok = median >= 8 and minimum >= 4
    return (
        ok,
        f"Freeze-frame density: median={median:.1f}, min={minimum} across {n} corner shots "
        f"(need median≥8 AND min≥4).",
    )


def check_team_id_stability(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 5: `team == 'Leicester City'` resolves to a stable team_id across
    all 38 matches. Catches name-variant bugs (e.g. 'Leicester City FC').
    """
    lei = events[events["team"] == LEICESTER]
    n_matches = lei["match_id"].nunique()
    n_team_ids = lei["team_id"].nunique()
    ok = n_matches == 38 and n_team_ids == 1
    return (
        ok,
        f"Team-id stability: {n_matches} matches map to {n_team_ids} team_id(s) "
        f"(need 38 AND 1).",
    )


def check_corner_count(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 6: anchor sanity — Leicester corners ≈ 196 (per data_characterisation.md).

    Hard-pass on exact 196; soft-pass within ±3.
    """
    n = ((events["type"] == "Pass")
         & (events["pass_type"] == "Corner")
         & (events["team"] == LEICESTER)).sum()
    ok = abs(int(n) - 196) <= 3
    return ok, f"Anchor sanity: Leicester corners n={n} (expected 196 ±3)."


def check_corner_shots_count(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 7: anchor sanity — shots from Leicester corners ≈ 89."""
    n = (
        (events["type"] == "Shot")
        & (events["play_pattern"] == "From Corner")
        & (events["team"] == LEICESTER)
    ).sum()
    ok = abs(int(n) - 89) <= 3
    return ok, f"Anchor sanity: Leicester shots from corners n={n} (expected 89 ±3)."


def check_goal_kick_count(events: pd.DataFrame) -> tuple[bool, str]:
    """Assertion 8: anchor sanity — Leicester goal kicks ≈ 329."""
    n = (
        (events["type"] == "Pass")
        & (events["pass_type"] == "Goal Kick")
        & (events["team"] == LEICESTER)
    ).sum()
    ok = abs(int(n) - 329) <= 3
    return ok, f"Anchor sanity: Leicester goal kicks n={n} (expected 329 ±3)."


CHECKS = [
    ("goal_direction", check_goal_direction),
    ("corner_origins", check_corner_origins),
    ("long_high_passes", check_long_high_passes),
    ("freeze_frame_density", check_freeze_frame_density),
    ("team_id_stability", check_team_id_stability),
    ("anchor_corners", check_corner_count),
    ("anchor_corner_shots", check_corner_shots_count),
    ("anchor_goal_kicks", check_goal_kick_count),
]


def run_checks() -> tuple[bool, str]:
    buf = StringIO()
    print(f"# Verification harness — {datetime.now().isoformat(timespec='seconds')}", file=buf)
    print("# (analysis/_verify.py — coordinate/direction/filter assumptions)", file=buf)

    t0 = time.time()
    events = _load_events()
    print(
        f"\nLoaded {len(events):,} events from {LEICESTER_EVENTS_PARQUET.name} "
        f"({events['match_id'].nunique()} matches) in {time.time()-t0:.2f}s\n",
        file=buf,
    )

    all_ok = True
    for name, fn in CHECKS:
        ok, msg = fn(events)
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}: {msg}"
        print(line, file=buf)
        all_ok = all_ok and ok

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}", file=buf)
    return all_ok, buf.getvalue()


def main() -> int:
    ok, log_text = run_checks()
    print(log_text)
    out = DOCS_DIR / "_verify.log"
    out.write_text(log_text)
    print(f"\nWrote log → {out.relative_to(DOCS_DIR.parent)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
