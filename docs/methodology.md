# Methodology — Leicester City 2015/16 briefing

This document captures every filter, threshold, and modelling choice referenced in the briefing
cards. It is the single source of truth for what the cards claim and how those claims were computed.

---

## 0. Verification harness (run before any plotting)

`analysis/_verify.py` checks 8 anchor-numbers from `data_characterisation.md` against a freshly
loaded parquet. Output in `docs/_verify.log`. All 8 currently pass:

- 100% of Leicester goals end at `x > 100` (attacking-direction unification).
- 100% of corner passes start at `(0, 0)` / `(0, 80)` / `(120, 0)` / `(120, 80)` ± 1 m.
- 99.6% of `pass_height == 'High Pass'` goal kicks have `pass_length ≥ 30 m`.
- Median `len(shot_freeze_frame)` = 15, min = 10 across the 89 corner shots.
- 38 matches map to a single Leicester `team_id` (no name-variant bugs).
- 196 corners, 89 shots from corners, 329 goal kicks.

If any of these break, the briefing-card claims silently mislead. The harness is the first line of defence.

---

## 1. Card A — corners

### 1.1 Filters

| Step | Filter |
|---|---|
| Delivery population | `pass_type == 'Corner' AND team == 'Leicester City'` (n = 196) |
| Shot population | `type == 'Shot' AND team == 'Leicester City' AND play_pattern == 'From Corner'` (n = 89) |
| Routine ranking pool | n_corners ≥ 10 AND end_zone ≠ 'short' AND swing ≠ 'short' AND swing ≠ 'unknown' |
| Top-k pick | n_shots ≥ 8 (so Fig 2 has a stable freeze-frame composite) |

### 1.2 Sequence-window definition (post-corner)

Each corner pass starts a **possession chain** (StatsBomb's `possession` integer). Shots are
attributed to that corner if they:

1. Share `(match_id, possession)` with the corner.
2. Have `play_pattern == 'From Corner' AND team == 'Leicester City'`.
3. Are bounded by the next Leicester corner in the same possession (rare; cap of 1 corner per
   chain in practice).

The original Phase 2 spec capped the window at 6 events / 8 seconds; the parquet shows that the
StatsBomb `possession` increment occurs naturally on every cleared corner, so the cap effectively
never bites. The headline anchors (89 shots, 7.99 xG, 5 goals in tight window) are reproduced
exactly.

> **The "5 vs 12 goals" gap.** Public records list 12 Leicester goals "from corners" in 2015/16;
> our tight-window count is 5. The 7-goal gap is goals from cleared-and-recovered second-phase
> shots that fall outside the same-possession window. This is a deliberate methodological choice —
> we want **direct** corner threat, not "any goal eventually traceable to a corner clearance".

### 1.3 Routine taxonomy

Each corner is keyed by `(side, swing, end_zone)`:

- **side** ∈ {left, right} from `start_y`: `start_y < 40 ⇒ right` (StatsBomb attacks left → right).
- **swing** ∈ {inswing, outswing, straight, short, unknown} — preferring StatsBomb's
  `pass_inswinging` / `pass_outswinging` / `pass_straight` flags when present (sparse), otherwise
  inferred from `(side, taker_footedness)`.
- **delivery_type** ∈ {short, flighted, driven} from `pass_height` and `pass_length`.
- **end_zone** ∈ {near_post_6yd, far_post_6yd, penalty_spot, edge_d, wide, short} from
  `pass_end_location`. Bins are explicit (see `corners.py:end_zone()`).

### 1.4 Routine ranking

Routines ranked by **xG / corner**, with Wilson-95 CIs on shot rate and goal rate. Tiebreaker
when CIs overlap: maximise the L2 distance between the routines' attacker-KDE peak coordinates on
the 120×80 grid (the visual-distinctiveness tiebreaker, codified in `select_top_routines`).

### 1.5 Freeze-frame composite (Fig 2)

For each top-2 routine: collect every `shot_freeze_frame` (Phase 0 verified 99% population) for
shots in that routine. Explode each frame into one row per (shot_id, frame_player). Group by
role (attacker / defender / shooter / GK) using the `teammate` flag and `position == 'Goalkeeper'`.
Per role, plot a 2D-Gaussian KDE on the 120×80 grid; overlay a low-α scatter for individual data
points so the underlying frame count is honest. The shooter and GK markers are the **median**
location across all shots in the routine.

---

## 2. Card B — goal kicks

### 2.1 Filters

| Step | Filter |
|---|---|
| GK population | `pass_type == 'Goal Kick' AND team == 'Leicester City'` (n = 329) |
| Long classification | `pass_height == 'High Pass'` (the canonical 96%) |
| First contact | first event after the GK with `location` within 8 m of `pass_end_location` |
| Second-ball | first Leicester event after first contact (chain length ≥ 2) |
| Chain ranking pool | qualifies = n ≥ 15 AND zone ∉ {short, own_half, unknown} |

### 2.2 Why `pass_height == 'High Pass'` (and not `pass_length ≥ 30 m`)

The verify harness shows 99.6% of `High Pass` GKs have `pass_length ≥ 30 m`. We use **only**
`pass_height` for the long classification because the OR-with-length variant inflates the long
share past 0.98 (it picks up short ground passes that happen to travel ≥ 30 m, which is a
qualitatively different action). The canonical 96% holds with the `pass_height` rule alone.

### 2.3 Zone taxonomy (Fig 3)

The end-location (60 ≤ x < 120) is bucketed into 6 zones:

- **shallow** (60 ≤ x < 90) — ball lands around the halfway line.
- **deep** (x ≥ 90) — ball lands in the offensive final third.

× three lateral thirds (left / central / right) → 6 zones. Zones with x < 60 are labelled
`own_half` and treated separately (almost all are short GKs).

Per zone we report: n, first-contact-won rate, second-ball-recovery rate, chain-≥-3-actions rate,
all with Wilson-95 CIs.

### 2.4 Post-GK chain (Fig 4)

Action 1 is the GK pass itself (always). Actions 2 and 3 are the next two Leicester
`Pass`/`Carry`/`Shot` events found within the next 12 events whose **start location is within 25 m
of the previous action's end**. The radius constraint is the key — it filters bounce-backs (e.g.
a Schmeichel carry after an opposition clearance) that share neither possession nor spatial
continuity with the GK's flow.

> **Why not pure same-possession filtering?** StatsBomb increments `possession` on every contested
> aerial, so a strict same-possession filter drops nearly every long-GK chain where Leicester
> wins the second ball after a brief opposition touch (n ≈ 6 chains under that filter — too few).
> The 25 m radius captures realistic continuous play without that pathology (n = 115 complete
> 3-action chains — adequate sample).

A Leicester `Dispossessed` / `Miscontrol` / `Foul Committed` event between actions also
terminates the chain.

### 2.5 xT — hand-rolled Karun Singh

`analysis/xthreat.py` implements Singh's iterative algorithm on a 16 × 12 grid:

```
xT(s) = P(shoot|s) · mean(xG | shoot, s)
      + P(move|s)  · Σ_{s'} T(s, s') · xT(s')
```

5 sweeps to convergence. Trained on the **38 cached Leicester matches** (~127 k actions, both
teams) — a deviation from the Phase 2 spec, which had specced training on all 380 PL 2015/16
matches. The deviation was forced by a `socceraction>=1.4` ↔ `numpy>=2.0` conflict (socceraction
hard-pins numpy<2.0; pandas>=2.2 needs numpy≥2.0 for binary compatibility on macOS arm64). We
hand-rolled the iteration rather than work around the segfault.

The Phase 2 fallback rule (§3.3 of the gate decision) explicitly allows this: train on
Leicester-only with an appendix caveat. The **math is identical** (Singh, 2018); the trade-off is
a slight bias toward Leicester's possession patterns. Per-action ΔxT *deltas* (the relevant
quantity for chain ranking) are not visibly biased on inspection of the trained grid.

### 2.6 Chain templating (Fig 4)

Chains clustered by their **end-zone** on a coarse 3 × 3 grid (so the templates surface "where
does the post-GK build-up arrive", which is the tactically useful question). Top-3 templates by
frequency:

1. **Template A** — final-third right (n = 32, mean ΔxT = +0.027, Vardy share 50%).
2. **Template B** — middle-third right (n = 28, mean ΔxT = +0.009, Vardy share 25%).
3. **Template C** — middle-third centre (n = 15, mean ΔxT = +0.011, Vardy share 40%).

### 2.7 Vardy denominator

The headline "Vardy in 36% of chains" uses **all 115 complete post-GK 3-action chains** as the
denominator (not chains-with-Vardy filtered to chains-with-Vardy). Vardy is counted as "in chain"
if he is either the **actor** of any of the 3 actions OR the **`pass_recipient`** of the GK pass
(action #1). Both flags are tracked; the union is used.

---

## 3. Coordinate convention

- StatsBomb 120 × 80 throughout (length × width). Attacks left → right.
- All figures use `mplsoccer.VerticalPitch(pitch_type='statsbomb')` so the goal is at the top.
- The vertical orientation swaps the *axes* but not the *data*: we always pass pitch coords
  `(x, y)` directly to mplsoccer's helpers (`pitch.scatter`, `pitch.kdeplot`, `pitch.arrows`,
  `pitch.annotate`); mplsoccer handles the transformation. Direct `ax.text` / `ax.annotate`
  on a `VerticalPitch` axes need `(y, x)` — we explicitly avoid that pattern.

## 4. Reproducibility

- One command (`uv run python -m analysis`) regenerates all 4 PNGs and the HTML.
- `random_state=42` everywhere randomness enters (it doesn't, currently — KDE uses scipy's
  default Silverman, no randomisation; xT is deterministic).
- `data/processed/snapshots.json` persists the anchor numbers; CI / smoke test fails on drift.
