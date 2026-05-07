"""Card B — goal-kick extraction, classification, first-contact + second-ball,
zone summarisation.

Card B title: *Leicester's goal-kick build-up: where United presses to flip
possession.*

Anchor numbers (confirmed by `_verify.py`): 329 Leicester goal kicks, ~96%
played long (`pass_height == 'High Pass'`).

Public API:
    extract_goal_kicks(events, team)    -> DataFrame
    classify_goal_kick(row)             -> str  ('long' | 'short')
    target_zone(end_xy)                 -> str  (one of 6 zones)
    extract_gk_sequences(events, team, window_events=5, window_seconds=8.0)
                                         -> DataFrame  (one row per GK)
    summarise_zones(sequences, min_n=15) -> DataFrame
    chain_after_gk(events, gk_row, max_actions=3) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from analysis import LEICESTER
from analysis.corners import _wilson_ci, _xy

log = logging.getLogger(__name__)

GK_WINDOW_EVENTS = 5
GK_WINDOW_SECONDS = 8.0
LONG_PASS_LENGTH_M = 30.0
FIRST_CONTACT_RADIUS_M = 8.0


def classify_goal_kick(row: pd.Series) -> str:
    """One of {long, short}.

    Canonical rule (per methodology spec): `long` iff `pass_height == 'High Pass'`.
    The verify harness confirms 99.6% of `High Pass` GKs have `pass_length ≥ 30 m`,
    so the canonical rule is exact; the OR-with-length variant (used in earlier
    drafts) inflates the long share past 0.98 and was rejected.
    """
    return "long" if row.get("pass_height") == "High Pass" else "short"


def target_zone(end_xy: tuple[float, float]) -> str:
    """Bucket goal-kick end-locations into 6 zones (StatsBomb 120×80, attack →).

    Lateral thirds:    left (y ≤ 26.7), central (26.7 < y ≤ 53.3), right (y > 53.3).
    Depth bands:       shallow (60 ≤ x < 90)  → ball lands around halfway,
                       deep    (x ≥ 90)        → ball lands in opp final third.
    Anything outside (x < 60) is `own_half` and almost always means a short GK.
    """
    x, y = end_xy
    if pd.isna(x) or pd.isna(y):
        return "unknown"
    if x < 60:
        return "own_half"
    band = "shallow" if x < 90 else "deep"
    if y <= 80 / 3:
        side = "left"
    elif y <= 2 * 80 / 3:
        side = "central"
    else:
        side = "right"
    return f"{side}_{band}"


def extract_goal_kicks(events: pd.DataFrame, team: str = LEICESTER) -> pd.DataFrame:
    """One row per Leicester goal-kick pass with parsed coords + classification."""
    g = events[
        (events["type"] == "Pass")
        & (events["pass_type"] == "Goal Kick")
        & (events["team"] == team)
    ].copy()
    starts = g["location"].apply(_xy)
    ends = g["pass_end_location"].apply(_xy)
    g["start_x"] = starts.apply(lambda t: t[0])
    g["start_y"] = starts.apply(lambda t: t[1])
    g["end_x"] = ends.apply(lambda t: t[0])
    g["end_y"] = ends.apply(lambda t: t[1])
    g["gk_type"] = g.apply(classify_goal_kick, axis=1)
    g["zone"] = g.apply(lambda r: target_zone((r["end_x"], r["end_y"])), axis=1)
    return g.reset_index(drop=True)


def _abs_seconds(ev: pd.Series) -> float:
    """Period-aware seconds offset (period 2 starts at 45 min, etc.)."""
    period_offset = 45 * 60 * (int(ev.get("period", 1)) - 1)
    return float(ev.get("minute", 0)) * 60 + float(ev.get("second", 0)) + period_offset


def extract_gk_sequences(
    events: pd.DataFrame,
    team: str = LEICESTER,
    window_events: int = GK_WINDOW_EVENTS,
    window_seconds: float = GK_WINDOW_SECONDS,
    first_contact_radius: float = FIRST_CONTACT_RADIUS_M,
) -> pd.DataFrame:
    """One row per Leicester GK with first-contact + second-ball outcomes.

    Window per spec: first of (a) Δt = 8s after the GK, (b) next 5 events by
    `index`, (c) end of possession chain.

    `first_contact_won` is True iff the first event inside the window whose
    `location` lies within `first_contact_radius` of the GK's `pass_end_location`
    has `team == LEICESTER`. (StatsBomb does not always emit a duel; we fall
    back to "first event near the landing zone" as the operational definition.)

    `second_ball_won` is True iff Leicester records ANY event in the window after
    that first contact. `chain_len` = number of CONSECUTIVE Leicester events
    starting at first contact (a single opposition action terminates the chain).
    """
    gks = extract_goal_kicks(events, team)
    log.info("Extracted %d %s goal kicks", len(gks), team)

    use_cols = [
        "id", "match_id", "index", "minute", "second", "period", "possession",
        "type", "team", "location", "pass_recipient", "player",
    ]
    have = [c for c in use_cols if c in events.columns]
    ev = events[have].copy()
    ev["abs_seconds"] = ev.apply(_abs_seconds, axis=1)
    ev["loc_xy"] = ev["location"].apply(_xy)

    rows = []
    by_match = ev.sort_values(["match_id", "index"]).set_index(["match_id", "index"], drop=False)
    by_match_idx = ev.sort_values(["match_id", "index"]).reset_index(drop=True)

    match_groups = {m: g.reset_index(drop=True) for m, g in by_match_idx.groupby("match_id", sort=False)}
    for _, g in gks.iterrows():
        m = g["match_id"]
        gk_idx = int(g["index"])
        gk_t0 = float(g["minute"]) * 60 + float(g["second"]) + 45 * 60 * (int(g["period"]) - 1)
        gk_poss = g["possession"]
        gk_end_xy = (g["end_x"], g["end_y"])

        match_evs = match_groups.get(m, None)
        if match_evs is None:
            continue
        gk_pos = match_evs.index[match_evs["index"] == gk_idx]
        if len(gk_pos) == 0:
            continue
        i0 = int(gk_pos[0])

        window = match_evs.iloc[i0 + 1 : i0 + 1 + window_events].copy()
        window = window[window["abs_seconds"] - gk_t0 <= window_seconds]
        if g["gk_type"] == "long":
            window = window[(window["possession"] == gk_poss) | (window["possession"] == gk_poss + 1)]
        else:
            window = window[window["possession"] == gk_poss]
        window = window.reset_index(drop=True)

        first_contact_won = None
        first_contact_idx = None
        for j, ev_row in window.iterrows():
            ex, ey = ev_row["loc_xy"]
            if pd.isna(ex) or pd.isna(ey):
                continue
            d = float(np.hypot(ex - gk_end_xy[0], ey - gk_end_xy[1]))
            if d <= first_contact_radius:
                first_contact_won = (ev_row["team"] == team)
                first_contact_idx = j
                break

        if first_contact_idx is None:
            second_ball_won = False
            chain_len = 0
        else:
            tail = window.iloc[first_contact_idx:].reset_index(drop=True)
            chain_len = 0
            for _, e in tail.iterrows():
                if e["team"] == team:
                    chain_len += 1
                else:
                    break
            second_ball_won = chain_len >= 2

        rows.append({
            "gk_id": g["id"],
            "match_id": g["match_id"],
            "gk_type": g["gk_type"],
            "zone": g["zone"],
            "start_x": g["start_x"],
            "start_y": g["start_y"],
            "end_x": g["end_x"],
            "end_y": g["end_y"],
            "first_contact_won": bool(first_contact_won) if first_contact_won is not None else False,
            "first_contact_observed": first_contact_won is not None,
            "second_ball_won": bool(second_ball_won),
            "chain_len": int(chain_len),
            "chain_three_plus": bool(chain_len >= 3),
        })
    return pd.DataFrame(rows)


def summarise_zones(sequences: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    """Aggregate `sequences` by `zone` → per-zone build-up summary.

    Long-only filter is applied: zones for short GKs (`gk_type == 'short'` →
    almost always end in own half) are reported separately for completeness
    but excluded from the headline ranking with `qualifies = False`.
    """
    seqs = sequences.copy()
    out = (
        seqs.groupby("zone", dropna=False)
        .agg(
            n=("gk_id", "size"),
            n_long=("gk_type", lambda s: int((s == "long").sum())),
            n_short=("gk_type", lambda s: int((s == "short").sum())),
            first_contact_won_rate=("first_contact_won", "mean"),
            second_ball_won_rate=("second_ball_won", "mean"),
            chain_three_plus_rate=("chain_three_plus", "mean"),
            mean_chain_len=("chain_len", "mean"),
        )
        .reset_index()
    )
    out["qualifies"] = (out["n"] >= min_n) & (out["zone"] != "own_half") & (out["zone"] != "unknown")

    fc_ci = out.apply(
        lambda r: _wilson_ci(
            int(r["first_contact_won_rate"] * r["n"]), int(r["n"])
        ),
        axis=1,
    )
    sb_ci = out.apply(
        lambda r: _wilson_ci(
            int(r["second_ball_won_rate"] * r["n"]), int(r["n"])
        ),
        axis=1,
    )
    out["fc_ci_lo"] = [c[0] for c in fc_ci]
    out["fc_ci_hi"] = [c[1] for c in fc_ci]
    out["sb_ci_lo"] = [c[0] for c in sb_ci]
    out["sb_ci_hi"] = [c[1] for c in sb_ci]
    return out.sort_values(["qualifies", "n"], ascending=[False, False]).reset_index(drop=True)


CHAIN_STEP_MAX_M = 25.0


def chain_after_gk(
    events: pd.DataFrame,
    gk_row: pd.Series,
    max_actions: int = 3,
    team: str = LEICESTER,
    look_ahead: int = 12,
    step_max: float = CHAIN_STEP_MAX_M,
) -> list[dict]:
    """Return up to `max_actions` Leicester on-ball actions starting with the GK.

    Action 1 is always the GK pass itself (Leicester action by definition).
    Actions 2/3 are the next Leicester `Pass`/`Carry`/`Shot` events found
    within the next `look_ahead` events whose START location is within
    `step_max` metres of the PREVIOUS action's end. The radius constraint is
    the key: it rejects bounce-back actions (e.g. a Schmeichel carry after
    an opposition clearance) that share neither possession nor spatial
    continuity with the GK's flow.

    Why not pure same-possession filtering: StatsBomb increments `possession`
    on every contested aerial, so a strict possession filter drops nearly
    every long-GK chain where Leicester wins the second ball after a brief
    opposition touch. The 25m radius captures realistic "continuous play"
    without that pathology.

    A Leicester `Dispossessed`, `Miscontrol` or `Foul Committed` event between
    chain actions also terminates the chain.

    Each returned item carries `start_x/y`, `end_x/y`, `type`, `player`,
    `is_shot`, `is_gk`, and `shot_xg`.
    """
    m = gk_row["match_id"]
    gk_idx = int(gk_row["index"])

    cols = [c for c in [
        "id", "match_id", "index", "type", "team", "possession", "player",
        "location", "pass_end_location", "carry_end_location",
        "pass_recipient", "shot_outcome", "shot_statsbomb_xg",
    ] if c in events.columns]
    ev = events[cols]

    sx, sy = _xy(gk_row["location"])
    ex, ey = _xy(gk_row["pass_end_location"])
    chain: list[dict] = [{
        "type": "Pass",
        "player": gk_row.get("player"),
        "pass_recipient": gk_row.get("pass_recipient"),
        "start_x": sx, "start_y": sy,
        "end_x": ex, "end_y": ey,
        "shot_xg": 0.0,
        "is_shot": False,
        "is_gk": True,
    }]

    follow = (
        ev[(ev["match_id"] == m) & (ev["index"] > gk_idx)]
        .sort_values("index")
        .head(look_ahead)
    )

    for _, e in follow.iterrows():
        if e["team"] == team and e["type"] in ("Dispossessed", "Miscontrol", "Foul Committed"):
            break
        if e["team"] != team:
            continue
        if e["type"] not in ("Pass", "Carry", "Shot"):
            continue
        a_sx, a_sy = _xy(e["location"])
        prev = chain[-1]
        if pd.notna(a_sx) and pd.notna(prev["end_x"]):
            d = float(np.hypot(a_sx - prev["end_x"], a_sy - prev["end_y"]))
            if d > step_max:
                break
        if e["type"] == "Pass":
            a_ex, a_ey = _xy(e.get("pass_end_location"))
        elif e["type"] == "Carry":
            a_ex, a_ey = _xy(e.get("carry_end_location"))
        else:
            a_ex, a_ey = _xy(e["location"])
        chain.append({
            "type": e["type"],
            "player": e.get("player"),
            "pass_recipient": e.get("pass_recipient"),
            "start_x": a_sx, "start_y": a_sy,
            "end_x": a_ex, "end_y": a_ey,
            "shot_xg": float(e.get("shot_statsbomb_xg") or 0.0),
            "is_shot": e["type"] == "Shot",
            "is_gk": False,
        })
        if e["type"] == "Shot":
            break
        if len(chain) >= max_actions:
            break
    return chain


def all_post_gk_chains(
    events: pd.DataFrame,
    gks: pd.DataFrame,
    max_actions: int = 3,
    team: str = LEICESTER,
) -> list[dict]:
    """Build the full list of post-GK action chains. Caches nothing — call once.

    Returned: list of {gk_id, gk_type, zone, actions: list[dict]} where each
    action carries start_x/y, end_x/y, type, player, is_shot.
    """
    chains = []
    for _, gk in gks.iterrows():
        c = chain_after_gk(events, gk, max_actions=max_actions, team=team)
        if c:
            chains.append({
                "gk_id": gk["id"],
                "gk_type": gk["gk_type"],
                "zone": gk["zone"],
                "actions": c,
            })
    return chains


def _coarse_cell(x: float, y: float, lc: int = 3, wc: int = 3) -> tuple[int, int]:
    """Coarse cell for chain-template clustering (default 3×3 grid → 9 zones)."""
    if pd.isna(x) or pd.isna(y):
        return (-1, -1)
    cx = int(np.clip(x / 120.0 * lc, 0, lc - 1))
    cy = int(np.clip(y / 80.0 * wc, 0, wc - 1))
    return (cx, cy)


END_ZONE_LABELS = {
    (0, 0): "own left", (0, 1): "own central", (0, 2): "own right",
    (1, 0): "middle-third left", (1, 1): "middle-third centre", (1, 2): "middle-third right",
    (2, 0): "final-third left", (2, 1): "final-third centre", (2, 2): "final-third right",
}


def summarise_post_gk_chains(
    chains: list[dict],
    xt,  # ExpectedThreat
    max_actions: int = 3,
    vardy_name: str = "Jamie Vardy",
) -> tuple[pd.DataFrame, dict]:
    """Cluster chains by per-action coarse start/end zones; rank by frequency.

    Each chain becomes a 6-tuple of coarse cells:
        (a1_start, a1_end, a2_start, a2_end, a3_start, a3_end)

    Returns a `(templates, headline)` pair:
      - `templates`: one row per template with mean start/end coords for each
        action, n_chains, vardy_in_chain (any action is Vardy-actor),
        mean_chain_xt (mean cumulative ΔxT in that template), end_zone_label,
        plus per-action mean ΔxT *across all chains* (a1/a2/a3_mean_xt_delta_overall).
      - `headline`: dict with `vardy_n`, `vardy_denom`, `vardy_share`,
        `n_chains_total`, `n_long_chains`.
    """
    long_chains = [c for c in chains if c["gk_type"] == "long" and len(c["actions"]) >= max_actions]
    n_chains_total = len(long_chains)

    rows = []
    a_position_xt: dict[int, list[float]] = {1: [], 2: [], 3: []}
    vardy_chains = 0
    for c in long_chains:
        acts = c["actions"][:max_actions]
        if len(acts) < max_actions:
            continue
        end_cell = _coarse_cell(acts[-1]["end_x"], acts[-1]["end_y"])
        if end_cell == (-1, -1):
            continue
        chain_xt = sum(float(a.get("xt_delta", 0.0)) for a in acts)
        for k, a in enumerate(acts, 1):
            a_position_xt[k].append(float(a.get("xt_delta", 0.0)))

        actors = {a.get("player") for a in acts}
        recipients = {a.get("pass_recipient") for a in acts}
        is_vardy = (vardy_name in actors) or (vardy_name in recipients)
        if is_vardy:
            vardy_chains += 1

        rows.append({
            "template_key": end_cell,
            "a1_start_x": acts[0]["start_x"], "a1_start_y": acts[0]["start_y"],
            "a1_end_x":   acts[0]["end_x"],   "a1_end_y":   acts[0]["end_y"],
            "a2_start_x": acts[1]["start_x"], "a2_start_y": acts[1]["start_y"],
            "a2_end_x":   acts[1]["end_x"],   "a2_end_y":   acts[1]["end_y"],
            "a3_start_x": acts[2]["start_x"], "a3_start_y": acts[2]["start_y"],
            "a3_end_x":   acts[2]["end_x"],   "a3_end_y":   acts[2]["end_y"],
            "chain_xt":   chain_xt,
            "is_vardy":   is_vardy,
            "end_cell":   end_cell,
        })

    if not rows:
        empty = pd.DataFrame(columns=[
            "n", "a1_start_x", "a1_start_y", "a1_end_x", "a1_end_y",
            "a2_end_x", "a2_end_y", "a3_end_x", "a3_end_y",
            "mean_chain_xt", "vardy_in_chain", "end_zone_label",
            "a1_mean_xt_delta_overall", "a2_mean_xt_delta_overall", "a3_mean_xt_delta_overall",
        ])
        head = dict(vardy_n=0, vardy_denom=0, vardy_share=0.0, n_chains_total=0, n_long_chains=0)
        return empty, head

    df = pd.DataFrame(rows)
    grouped = df.groupby("template_key")

    overall = {
        "a1_mean_xt_delta_overall": float(np.mean(a_position_xt[1])) if a_position_xt[1] else 0.0,
        "a2_mean_xt_delta_overall": float(np.mean(a_position_xt[2])) if a_position_xt[2] else 0.0,
        "a3_mean_xt_delta_overall": float(np.mean(a_position_xt[3])) if a_position_xt[3] else 0.0,
    }

    summary_rows = []
    for key, sub in grouped:
        end_label = END_ZONE_LABELS.get(sub["end_cell"].iloc[0], "other")
        summary_rows.append({
            "n": len(sub),
            "a1_start_x": sub["a1_start_x"].median(), "a1_start_y": sub["a1_start_y"].median(),
            "a1_end_x":   sub["a1_end_x"].median(),   "a1_end_y":   sub["a1_end_y"].median(),
            "a2_start_x": sub["a2_start_x"].median(), "a2_start_y": sub["a2_start_y"].median(),
            "a2_end_x":   sub["a2_end_x"].median(),   "a2_end_y":   sub["a2_end_y"].median(),
            "a3_start_x": sub["a3_start_x"].median(), "a3_start_y": sub["a3_start_y"].median(),
            "a3_end_x":   sub["a3_end_x"].median(),   "a3_end_y":   sub["a3_end_y"].median(),
            "mean_chain_xt":  sub["chain_xt"].mean(),
            "vardy_share_in_template": float(sub["is_vardy"].mean()),
            "vardy_in_chain": bool(sub["is_vardy"].any()),
            "end_zone_label": end_label,
            **overall,
        })
    out = pd.DataFrame(summary_rows).sort_values("n", ascending=False).reset_index(drop=True)

    head = dict(
        vardy_n=int(vardy_chains),
        vardy_denom=int(n_chains_total),
        vardy_share=float(vardy_chains) / max(n_chains_total, 1),
        n_chains_total=n_chains_total,
        n_long_chains=n_chains_total,
    )
    return out, head
