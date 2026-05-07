"""Pull all Leicester City PL 2015/16 events and produce a data-characterisation summary.

Run from repo root:
    uv run python -m analysis.characterise
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.data import (
    PROCESSED_DIR,
    load_events,
    load_team_matches,
    parse_freeze_frame,
)

LEICESTER = "Leicester City"
OUT = PROCESSED_DIR / "leicester_all_events.parquet"


def main() -> None:
    matches = load_team_matches(LEICESTER)
    print(f"[1] Leicester PL 2015/16 matches: {len(matches)}")

    print(f"[2] Pulling event data for all {len(matches)} matches...")
    frames = []
    for _, m in tqdm(matches.iterrows(), total=len(matches)):
        ev = load_events(m["match_id"])
        ev = ev.assign(
            match_id=m["match_id"],
            match_date=m["match_date"],
            home_team=m["home_team"],
            away_team=m["away_team"],
            home_score=m["home_score"],
            away_score=m["away_score"],
        )
        frames.append(ev)

    all_ev = pd.concat(frames, ignore_index=True)
    all_ev.to_parquet(OUT)
    print(f"[3] Saved consolidated events → {OUT.name} ({len(all_ev):,} rows)")

    print("\n=== DATA CHARACTERISATION ===")
    print(f"Matches: {all_ev['match_id'].nunique()}")
    print(f"Total events: {len(all_ev):,}")
    print(f"Date range: {all_ev['match_date'].min()} to {all_ev['match_date'].max()}")

    # Leicester-specific slices
    lei = all_ev[all_ev["team"] == LEICESTER]
    print(f"\nLeicester-possession events: {len(lei):,}")

    lei_corners = lei[(lei["type"] == "Pass") & (lei["pass_type"] == "Corner")]
    print(f"Leicester corners taken: {len(lei_corners)}")

    # Shots resulting from corners (Leicester or opponent)
    shots = all_ev[all_ev["type"] == "Shot"]
    from_corner = shots[shots["play_pattern"] == "From Corner"]
    lei_shots_from_corner = from_corner[from_corner["team"] == LEICESTER]
    opp_shots_from_corner = from_corner[from_corner["team"] != LEICESTER]
    print(f"Shots from corners — Leicester: {len(lei_shots_from_corner)}, opponents: {len(opp_shots_from_corner)}")
    print(
        f"  xG sum from corners — Leicester: {lei_shots_from_corner['shot_statsbomb_xg'].sum():.2f}, "
        f"opponents: {opp_shots_from_corner['shot_statsbomb_xg'].sum():.2f}"
    )

    # Goals from corners
    goals_from_corner = from_corner[from_corner["shot_outcome"] == "Goal"]
    print(f"Goals from corners (both teams): {len(goals_from_corner)}")

    # Freeze-frame availability sanity check
    shots_with_ff = shots[shots["shot_freeze_frame"].notna()].copy()
    print(f"\nShots with freeze frames: {len(shots_with_ff)} / {len(shots)} ({len(shots_with_ff)/len(shots):.0%})")
    if len(shots_with_ff):
        sample_ff = parse_freeze_frame(shots_with_ff.iloc[0]["shot_freeze_frame"])
        print(f"Sample freeze-frame size: {len(sample_ff)} players")

    # Goal kicks (for build-up analysis candidate)
    gks = all_ev[(all_ev["type"] == "Pass") & (all_ev["pass_type"] == "Goal Kick")]
    lei_gks = gks[gks["team"] == LEICESTER]
    print(f"\nGoal kicks — Leicester: {len(lei_gks)}, total across all matches: {len(gks)}")
    if len(lei_gks):
        print(
            f"  Leicester GK: {(lei_gks['pass_height']=='High Pass').mean():.0%} long / "
            f"{(lei_gks['pass_height']=='Ground Pass').mean():.0%} short"
        )

    # Play pattern mix for Leicester possessions
    print("\nLeicester play-pattern mix (possessions starting):")
    pp = lei.groupby("possession")["play_pattern"].first().value_counts(normalize=True)
    for pat, share in pp.head(8).items():
        print(f"  {pat:18s} {share:>6.1%}")


if __name__ == "__main__":
    main()
