# Manchester United FC — Tactical Intelligence

**A football-data portfolio piece for the Senior Football Data Analyst role at Manchester United.**

A pair of single-side A4 briefing cards profiling Leicester City 2015/16 — the most recent freely available season of Premier League event-level data — written from a United-facing tactical perspective. The methodology is data-source-agnostic and plugs directly into modern Opta / StatsBomb commercial feeds.

> **Live report:** _to be deployed at https://manchester-united-fc.vercel.app/leicester-1516_
> _(see [`DEPLOY.md`](./DEPLOY.md) for the deploy steps; build is one command, `uv run python -m analysis`.)_

---

## The findings, in one paragraph

Leicester's 2015/16 corner threat is concentrated in **two right-side outswinger routines** — Routine A (Fuchs to Huth at the front-post 6-yard band, n = 18, 78% shot rate, 1 goal from 1.35 xG) and Routine B (the wide variant that drags defenders out, n = 14, 2 goals from 0.6 xG). The freeze-frame composite shows the Routine-A contest happens in the same square metre as the defender block — winning the first jump on Huth is the single highest-leverage assignment when defending a Fuchs corner. Their goal-kick build-up is the inverse: Schmeichel plays **96% long**, exclusively as a launch mechanism (4% short). Almost half of all GKs (47%) target the right-shallow zone around the halfway line; Leicester wins 87% of first contacts there, but only 44% of second balls. Vardy is the actor or GK recipient on **36% of all post-GK 3-action chains** (denominator = ALL chains, not a Vardy-filtered subset). The dominant chain template ends in the final-third right and produces +0.027 xT — twice the average. **The press starts at the landing zone, not at the keeper; the third action is where the chain dies; bias the right-back's positioning toward Vardy.**

## What's in the cards

- **Card A — Leicester's attacking corners: how United defends them (and counter-exploits).**
  Two figures: KDE of corner end-locations with the top-2 routines highlighted (Fig 1), and side-by-side freeze-frame composites of the marking picture at the instant of contact for each routine (Fig 2). 196 corners → 89 shots from corner sequences → ΣxG = 7.99.

- **Card B — Leicester's goal-kick build-up: where United presses to flip possession.**
  Two figures: long-GK landing-zone heatmap with first-contact and second-ball outcomes per zone (Fig 3), and the top-3 post-GK 3-action chain templates with a hand-rolled Karun Singh xT grid as underlay (Fig 4). 329 goal kicks, 96% long, 115 complete 3-action chains, Vardy in 36%.

The two cards are wired into a single static HTML at [`reports/leicester-1516.html`](./reports/leicester-1516.html). Both are A4-portrait, print-ready (Cmd-P → Save as PDF in any browser).

## Methodology and what this data can't see

Each analysis ships with:

1. **The football question** and why it matters to a coach.
2. **The data and its limits** — what we can and cannot see (event-only data; freeze frames only at shots; no tracking; squad turnover since 2015/16).
3. **The method** in plain prose with named filters and thresholds (e.g. "long" = `pass_height == 'High Pass'`).
4. **The findings** in coach-facing language with explicit "if X, then Y" exploitation triggers.
5. **What internal data would refine** — every finding is footnoted with the gap between public and commercial feeds.

Per-figure caveats live on the briefing cards themselves; deeper details are in [`docs/methodology.md`](./docs/methodology.md), [`agent-outputs/phase-3/corner_findings.md`](./agent-outputs/phase-3/corner_findings.md), and [`agent-outputs/phase-3/goal_kick_findings.md`](./agent-outputs/phase-3/goal_kick_findings.md).

## Reproduce the build

```bash
# Install Python deps (uv handles the venv)
uv sync

# Run the full pipeline (data pull → 4 figures → static HTML)
uv run python -m analysis

# Outputs:
#   reports/assets/fig1_corner_delivery_heatmap.png
#   reports/assets/fig2_freeze_frame_composite.png
#   reports/assets/fig3_goal_kick_recovery.png
#   reports/assets/fig4_post_gk_xt_chain.png
#   reports/leicester-1516.html
#   data/processed/snapshots.json   (anchor-number test fixture)
```

The first run pulls 38 matches from StatsBomb (~5 min). Subsequent runs are <10 s thanks to parquet caching in `data/raw/` and `data/processed/`.

## Stack and project structure

```
analysis/
  __init__.py            shared paths and constants
  __main__.py            end-to-end orchestrator
  _verify.py             coordinate / direction / filter sanity harness
  data.py                StatsBomb fetch + parquet cache
  characterise.py        regenerates docs/data_characterisation.md
  corners.py             Card A — corner extraction, classification, ranking
  freeze_frames.py       Card A — shot_freeze_frame parsing + KDE composite
  goal_kicks.py          Card B — GK extraction, first-contact, chain build
  xthreat.py             hand-rolled Karun Singh xT (16×12 grid)
  viz.py                 four figures, one function each
  report.py              Jinja templates → static HTML

reports/
  index.html             portfolio hub
  leicester-1516.html    the two A4 cards
  styles.css             A4 portrait, print-ready
  assets/                committed PNGs (Vercel-served)

data/
  raw/                   gitignored; events_<match_id>.parquet
  processed/             gitignored; xt_grid.pkl, snapshots.json

docs/
  data_characterisation.md
  methodology.md         filters, sequence definitions, xT version, caveats
  _verify.log            harness output (8 anchor checks)

agent-outputs/           (gitignored) multi-agent design workflow outputs
```

## Limitations (the honest surface)

- Open event data lags the current season. The methodology is data-source-agnostic and plugs into Opta / StatsBomb commercial feeds.
- No 25 Hz tracking — set-piece spatial analysis is limited to event-level + shot freeze frames. With tracking, every "first contact" and "second-ball recovery" claim could be tightened, and the Vardy chain question (does he win the first contact, or does he run onto knock-downs?) becomes answerable.
- Squad has turned over since 2018. Individual names (Fuchs, Mahrez, Albrighton, Vardy) are illustrative; the **methodology** is the deliverable.
- Only two head-to-head fixtures vs United in 2015/16. Claims are structural ("if your back-line resembles X, the route is Y"), never match-specific.

## Author

**MJ du Plessis** — building football analytics portfolio pieces. Eight years of Python and SQL, three at Veo (millions of grass-roots / amateur match recordings — strong opinions about how the public-data → tracking-data ladder bends in 2026). [github.com/mjdup77](https://github.com/mjdup77).
