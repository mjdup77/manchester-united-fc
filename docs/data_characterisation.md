# Data Characterisation — Leicester City, PL 2015/16

*Produced automatically by `analysis/pull_leicester.py`. Regenerate after any data refresh.*

## Source

**StatsBomb Open Data** — `competition_id=2, season_id=27` (Premier League 2015/16). Pulled via `statsbombpy` and cached locally in `data/raw/`. Consolidated Leicester events in `data/processed/leicester_all_events.parquet`.

## Coverage

| Metric | Value |
|---|---|
| Matches | 38 (full Leicester PL season) |
| Total events | 126,916 |
| Leicester-possession events | 56,567 |
| Date range | 2015-08-08 → 2016-05-15 |
| Leicester vs Man Utd fixtures | 2 (`match_id` 3754186, 3754165) |

The two United fixtures are:

| Date | Home | Away | Score |
|---|---|---|---|
| 2015-11-28 | Leicester City | Manchester United | 1 – 1 |
| 2016-05-01 | Manchester United | Leicester City | 1 – 1 |

The second is the famous May 1 draw that set up Leicester clinching the title the next day.

## What we can do with this data

### 1. Set pieces — Leicester's corners (defensive perspective for United)

| Metric | Value |
|---|---|
| Leicester corners taken | 196 |
| Shots by Leicester from corners | 89 |
| xG total from corners (Leicester) | 7.99 |
| xG total conceded from corners | 9.61 |
| Goals from corners (both teams) | 12 |

**Critical unlock: `shot_freeze_frame` is populated on 99% (1,027 / 1,042) of shots.** Each freeze-frame is a list of players with `location`, `position` label, and `teammate` flag. That means for every shot from a corner, we see the positions of (typically) 10–18 surrounding players at the exact instant of the shot — sufficient for real spatial analysis:

- Where Leicester's deliveries land (`pass_end_location` on the corner pass)
- Which zones of the 6-yard / 18-yard / penalty spot area they target
- Who attacks (shooter position + role from freeze-frame)
- Marking picture at shot time (inferred from teammate/opponent relative positions)
- xG-weighted outcomes by routine, delivery side, and delivery type (inswing / outswing / short)

**Caveat:** freeze-frame size varies (median ~10, max ~22). Some shots have sparse frames (only players closest to the ball). Filter to dense frames for marking-picture work; use everything for ball-location / shot-location work.

### 2. Build-up phase — Leicester's goal kicks

| Metric | Value |
|---|---|
| Leicester goal kicks | 329 |
| % played long (`pass_height == 'High Pass'`) | **96%** |
| % played short (`pass_height == 'Ground Pass'`) | 4% |

The 96% long figure is the single loudest tactical signal in the dataset. Ranieri's Leicester explicitly refuses to play out from the back — the goal kick is a *launch mechanism* for Vardy / Okazaki to contest in the opposition half and win second balls. This is rare in the modern PL and a clear coach-relevant finding.

**What we can derive:**
- Target zones of long goal kicks (pass end locations — central, left channel, right channel)
- First-contact outcomes (Leicester wins aerial duel vs. concedes possession)
- Second-ball recovery rates (next Leicester event within N seconds of the GK)
- Which Leicester players are the designated target / second-ball runners
- How possession is retained or recycled after the GK

### 3. Leicester play-pattern mix (possessions starting from)

| Pattern | Share |
|---|---|
| Regular Play | 37.4% |
| From Throw-In | 26.6% |
| From Free Kick | 14.0% |
| From Goal Kick | 9.5% |
| From Corner | 5.0% |
| From Kick Off | 3.6% |
| From Counter | 2.3% |
| From Keeper | 1.4% |

## Key StatsBomb fields in use

| Field | Notes |
|---|---|
| `type` | Top-level event type (Pass, Shot, Carry, Duel, …) |
| `play_pattern` | Where the possession originated — key for filtering "From Corner" shots, etc. |
| `team`, `team_id` | Which side is in possession at this event |
| `location` | `[x, y]` in 120 × 80 pitch coordinates |
| `pass_type` | Corner, Free Kick, Goal Kick, Throw-in, … |
| `pass_height` | Ground Pass / Low Pass / High Pass |
| `pass_length`, `pass_angle` | Geometric attributes |
| `pass_cross`, `pass_inswinging`, `pass_outswinging` | Cross-specific qualifiers |
| `pass_end_location` | `[x, y]` of where the pass ended (delivered-to zone for corners) |
| `pass_recipient`, `pass_recipient_id` | Intended receiver |
| `pass_outcome` | Incomplete / Out / Unknown — absent when pass was successful |
| `shot_statsbomb_xg` | StatsBomb's xG for the shot |
| `shot_outcome` | Goal / Saved / Off T / Blocked / … |
| `shot_body_part`, `shot_technique` | Head / Right / Left, Volley / Half-Volley / Normal |
| `shot_freeze_frame` | **List of {location, player, position, teammate}** for surrounding players at the instant of the shot |
| `under_pressure` | Was the actor pressed |
| `possession`, `possession_team` | Possession-chain grouping |

## What is *not* available (be explicit about it in the final report)

- No full tracking (25 Hz player XY + ball XY). Open data has only freeze-frames at shots.
- No StatsBomb 360 (per-event freeze frames) for 2015/16 — that's limited to 2020+ competitions in the open set.
- No player physical metrics (distance, high-speed running).
- No video links / moments.
- Some qualifier fields are sparse (e.g. `pass_inswinging` only flagged on a subset of crosses — cannot assume "not flagged" means "not inswinging"; treat as *asserted* label only).

## Implications for MVP scope

1. **Defensive corners on Leicester's deliveries (89 shots, 7.99 xG, 99% freeze-frame coverage)** is the right set-piece phase. Large enough sample, rich enough data.
2. **Leicester long goal kicks (329 observations, 96% long)** is the cleanest build-up candidate — the long/short split alone is a headline insight, and we can layer second-ball analysis on top.
3. Everything else (wide overloads, turnover transitions, Vardy's channel runs) is defensible for v2/v3 but heavier to implement cleanly in one evening without tracking data.
