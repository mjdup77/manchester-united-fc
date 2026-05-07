"""End-to-end pipeline: pull data → run analyses → render figures → assemble report.

Usage:
    uv run python -m analysis            # full run (cached after the first time)
    uv run python -m analysis --no-report  # skip HTML report, just regenerate figures

A single command produces:
    reports/assets/fig1_corner_delivery_heatmap.png
    reports/assets/fig2_freeze_frame_composite.png
    reports/assets/fig3_goal_kick_recovery.png
    reports/assets/fig4_post_gk_xt_chain.png
    reports/leicester-1516.html
    data/processed/snapshots.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from analysis import (
    ASSETS_DIR,
    LEICESTER,
    LEICESTER_EVENTS_PARQUET,
    PROCESSED_DIR,
    REPORTS_DIR,
)
from analysis.corners import (
    extract_corner_sequences,
    select_top_routines,
    summarise_routines,
)
from analysis.data import load_team_events
from analysis.freeze_frames import build_routine_freeze_frames, kde_peak_xy
from analysis.goal_kicks import (
    all_post_gk_chains,
    extract_goal_kicks,
    extract_gk_sequences,
    summarise_post_gk_chains,
    summarise_zones,
)
from analysis.viz import (
    fig1_delivery_endzone_heatmap,
    fig2_freeze_frame_composite,
    fig3_goal_kick_recovery,
    fig4_post_gk_xt_chain,
    save_fig,
)
from analysis.xthreat import (
    XT_CACHE_PATH,
    attach_xt_to_chain,
    cache_xt,
    load_xt,
    train_xt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analysis")


def _load_or_build_events() -> pd.DataFrame:
    if LEICESTER_EVENTS_PARQUET.exists():
        log.info("Loading cached Leicester events parquet…")
        return pd.read_parquet(LEICESTER_EVENTS_PARQUET)
    log.info("Pulling 38 matches from StatsBomb (~5 min on first run)…")
    ev = load_team_events(LEICESTER)
    ev.to_parquet(LEICESTER_EVENTS_PARQUET)
    return ev


def run_corners(events: pd.DataFrame) -> dict:
    log.info("=== Card A — corners ===")
    sequences = extract_corner_sequences(events, LEICESTER)
    routines_df = summarise_routines(sequences, min_n=10)

    def kde_distinct(a, b):
        rf = build_routine_freeze_frames(events, sequences, [a, b])
        peak_a = kde_peak_xy(rf.get(a.label(), pd.DataFrame()).query("teammate and not is_shooter"))
        peak_b = kde_peak_xy(rf.get(b.label(), pd.DataFrame()).query("teammate and not is_shooter"))
        if peak_a is None or peak_b is None:
            return 0.0
        import numpy as _np
        return float(_np.hypot(peak_a[0] - peak_b[0], peak_a[1] - peak_b[1]))

    top = select_top_routines(routines_df, k=2, min_shots=8, kde_distinct_fn=kde_distinct)
    log.info("Top routines: %s", [r.label() for r in top])

    headline = {
        "n_corners": int(len(sequences)),
        "n_shots": int(sequences["seq_n_shots"].sum()),
        "n_goals": int(sequences["seq_goal"].sum()),
        "sum_xg": float(sequences["seq_xg"].sum()),
    }

    fig1 = fig1_delivery_endzone_heatmap(sequences, top, headline=headline)
    save_fig(fig1, "fig1_corner_delivery_heatmap")

    routine_frames = build_routine_freeze_frames(events, sequences, top)
    fig2 = fig2_freeze_frame_composite(routine_frames, top, routines_df)
    save_fig(fig2, "fig2_freeze_frame_composite")

    return {
        "headline": headline,
        "routines": [r.as_tuple() for r in top],
        "routine_summaries": routines_df.to_dict(orient="records"),
    }


def run_goal_kicks(events: pd.DataFrame) -> dict:
    log.info("=== Card B — goal kicks + xT ===")
    gks = extract_goal_kicks(events, LEICESTER)
    long_share = float((gks["gk_type"] == "long").sum() / len(gks))
    sequences = extract_gk_sequences(events, LEICESTER)
    zone_summary = summarise_zones(sequences)

    fig3 = fig3_goal_kick_recovery(sequences, zone_summary, long_share)
    save_fig(fig3, "fig3_goal_kick_recovery")

    xt = load_xt(XT_CACHE_PATH)
    if xt is None:
        log.info("Training xT on %d events (Leicester-match scope)…", len(events))
        t0 = time.time()
        xt = train_xt(events, n_iterations=5)
        log.info("xT trained in %.1fs", time.time() - t0)
        cache_xt(xt, XT_CACHE_PATH)
    else:
        log.info("Loaded cached xT grid from %s", XT_CACHE_PATH)

    chains = all_post_gk_chains(events, gks, max_actions=3, team=LEICESTER)
    for c in chains:
        c["actions"] = attach_xt_to_chain(c["actions"], xt)
    log.info("Post-GK chains: %d (long+3-action+complete)", len(chains))

    templates, headline_chains = summarise_post_gk_chains(chains, xt, max_actions=3)
    log.info("Top-3 templates: %s", templates.head(3)[["n", "end_zone_label"]].to_dict(orient="records"))

    fig4 = fig4_post_gk_xt_chain(
        chains=chains,
        xt_grid=xt.grid,
        chain_summary=templates,
        vardy_share=headline_chains["vardy_share"],
        vardy_n=headline_chains["vardy_n"],
        vardy_denom=headline_chains["vardy_denom"],
    )
    save_fig(fig4, "fig4_post_gk_xt_chain")

    return {
        "headline": {
            "n_gks": int(len(gks)),
            "long_share": long_share,
            "n_long": int((gks["gk_type"] == "long").sum()),
            "n_short": int((gks["gk_type"] == "short").sum()),
            **headline_chains,
        },
        "zone_summary": zone_summary.to_dict(orient="records"),
        "templates": templates.head(5).to_dict(orient="records"),
    }


def write_snapshot(corners_out: dict, gk_out: dict) -> None:
    snap = {
        "corners": corners_out["headline"]["n_corners"],
        "shots_from_corners": corners_out["headline"]["n_shots"],
        "corner_xg": round(corners_out["headline"]["sum_xg"], 3),
        "corner_goals": corners_out["headline"]["n_goals"],
        "goal_kicks": gk_out["headline"]["n_gks"],
        "gk_long_pct": round(gk_out["headline"]["long_share"], 3),
        "vardy_chain_n": gk_out["headline"]["vardy_n"],
        "vardy_chain_denom": gk_out["headline"]["vardy_denom"],
    }
    out = PROCESSED_DIR / "snapshots.json"
    out.write_text(json.dumps(snap, indent=2))
    log.info("Snapshot persisted: %s", snap)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Leicester 2015/16 — full pipeline")
    p.add_argument("--no-report", action="store_true", help="skip HTML render")
    args = p.parse_args(argv)

    events = _load_or_build_events()
    corners_out = run_corners(events)
    gk_out = run_goal_kicks(events)
    write_snapshot(corners_out, gk_out)

    if not args.no_report:
        from analysis.report import render_briefing
        render_briefing(corners_out, gk_out, REPORTS_DIR)
        log.info("Report rendered → %s", REPORTS_DIR / "leicester-1516.html")

    log.info("Done. Assets in %s", ASSETS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
