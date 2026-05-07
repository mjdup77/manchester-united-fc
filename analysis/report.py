"""HTML report builder. Two A4 cards in one document.

Output: a single static `reports/leicester-1516.html` plus a small index at
`reports/index.html`. Both reference `reports/styles.css` and the PNGs in
`reports/assets/`. They render in any browser and print cleanly to PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Template

log = logging.getLogger(__name__)


CARD_A_TEMPLATE = Template("""\
<section class="card card-a">
  <header>
    <h2>Card A — Leicester's attacking corners</h2>
    <p class="subtitle">How United defends them (and counter-exploits).</p>
    <p class="anchor">PL 2015/16 · {{ n_corners }} corners · {{ n_shots }} shots from corner sequences ·
       Σ xG = {{ '%.2f'|format(sum_xg) }} · {{ n_goals }} goal{{ 's' if n_goals != 1 else '' }} (tight 8 s / 6-event window).</p>
  </header>

  <figure class="hero">
    <img src="assets/fig1_corner_delivery_heatmap.png" alt="Corner delivery end-locations heatmap" />
    <figcaption>Fig 1 · KDE of all crossed corner deliveries; top-2 routines (orange A, blue B)
      annotated with delivery arrows + 95%-coverage covariance ellipses on end clusters.</figcaption>
  </figure>

  <figure class="hero">
    <img src="assets/fig2_freeze_frame_composite.png" alt="Freeze-frame composite at moment of shot" />
    <figcaption>Fig 2 · Per-routine KDE composites of attacker (blue) and defender (red) positions
      at the instant of contact; ★ = median shooter, ◆ = median GK.</figcaption>
  </figure>

  <div class="findings">
    <h3>Routine A — right-outswing → penalty spot</h3>
    <p><strong>n = 18 corners · 14 shots · 1 goal · Σ xG = 1.35 (xG/corner = 0.075).</strong>
       Primary taker <strong>Christian Fuchs</strong> (16 of 18); primary target <strong>Robert Huth</strong>
       (7 of 13 shots; 9 of 13 are headers; 8 of 13 missed the target).
       Defender KDE peak (113.9, 36.6) — clustered in the front-post 6-yard band; attacker KDE peak
       essentially identical (113.9, 35.3): <strong>the contest happens in the same square metre as the block</strong>.</p>

    <h3>Routine B — right-outswing → wide</h3>
    <p><strong>n = 14 corners · 9 shots · 2 goals · Σ xG = 0.60 (xG/corner = 0.043).</strong>
       Same right-side outswing, ball sits up <em>outside</em> the 6-yard box. Greater shooter variety
       (Vardy and Albrighton appear among the 8 shooters, alongside Huth, De Laet, Kanté, Morgan, Ulloa).
       Defender KDE peak (111.9, 42.0): <strong>the wider delivery drags defenders out</strong> —
       only 36% of defenders inside the 6-yard box vs 38% on Routine A.</p>

    <h3>The exploitation triggers</h3>
    <ol>
      <li><strong>When Fuchs sets to take a right-side corner</strong>, expect a left-foot outswinger curling
          to the front of the 6-yard box for Huth. Marker on Huth — front-post 6-yard band — is the
          single highest-leverage assignment. Shot rate 78%; goal rate 6%; <strong>winning the first jump
          is the whole job</strong>.</li>
      <li>If the front-post block is cleared early — i.e. takers set up wide and central runners drift back —
          Leicester's 'wide' variant goal-yields more (2 goals from 0.6 xG, n=14). Bias the second-line marker
          toward picking up <strong>Vardy</strong>, who appears in the wide-variant freeze frames as a
          back-post outlet.</li>
    </ol>
  </div>

  <footer class="caveats">
    <strong>Caveats.</strong>
    n is small — strongest routine has 18 corners across 38 matches; Wilson-95 CIs are wide.
    Squad has turned over since 2018 (Fuchs, Mahrez, Albrighton are gone) — the methodology is the deliverable, not the names.
    The "5 vs 12 goals" gap (tight window vs PL public count) is a deliberate window-definition choice; the
    8 s / 6-event window isolates <em>direct</em> corner threat from second-phase recoveries.
    Footedness inference is partly heuristic where StatsBomb's `pass_inswinging` flag is sparse.
  </footer>
</section>
""")


CARD_B_TEMPLATE = Template("""\
<section class="card card-b">
  <header>
    <h2>Card B — Leicester's goal-kick build-up</h2>
    <p class="subtitle">Where United presses to flip possession.</p>
    <p class="anchor">PL 2015/16 · {{ n_gks }} goal kicks · <strong>{{ '%.0f'|format(long_share*100) }}% long</strong>
       · post-GK chains analysed with a hand-rolled Karun Singh xT grid (16 × 12, trained on Leicester-match events).</p>
  </header>

  <figure class="hero">
    <img src="assets/fig3_goal_kick_recovery.png" alt="Goal-kick landing zones with first-contact / second-ball outcomes" />
    <figcaption>Fig 3 · Long-GK landing-zone heatmap; arrows from the keeper to the median landing of
      the top-3 zones (n ≥ 15); green ▲ = second-ball recovered by Leicester. Wilson-95 CIs in callouts.</figcaption>
  </figure>

  <figure class="hero">
    <img src="assets/fig4_post_gk_xt_chain.png" alt="Post-GK chain templates with xT underlay" />
    <figcaption>Fig 4 · Singh xT underlay (greys) with the top-3 post-GK 3-action chain templates;
      ★ marks templates where Vardy is an actor on the chain. Side panel: mean ΔxT per action position.</figcaption>
  </figure>

  <div class="findings">
    <h3>Headline policy: {{ '%.0f'|format(long_share*100) }}% long</h3>
    <p>Ranieri's Leicester explicitly refuses to play out from the back. The goal kick is a launch
       mechanism — Vardy or Okazaki contests the first ball, second-line runners win the knock-down,
       the third action is the breakaway pass or shot. Pressing the keeper is not productive against
       Leicester; <strong>the press has to start at the landing zone</strong>.</p>

    <h3>Where the long ball goes</h3>
    {% if zones %}
    <ul class="zone-list">
    {% for z in zones[:3] %}
      <li><strong>{{ z.zone.replace('_', ' ') }}</strong> — n = {{ z.n }} GKs, first contact won
        {{ '%.0f'|format(z.first_contact_won_rate*100) }}%
        [{{ '%.0f'|format(z.fc_ci_lo*100) }}, {{ '%.0f'|format(z.fc_ci_hi*100) }}],
        second ball recovered {{ '%.0f'|format(z.second_ball_won_rate*100) }}%
        [{{ '%.0f'|format(z.sb_ci_lo*100) }}, {{ '%.0f'|format(z.sb_ci_hi*100) }}],
        chain ≥3 actions {{ '%.0f'|format(z.chain_three_plus_rate*100) }}%.</li>
    {% endfor %}
    </ul>
    {% endif %}

    <h3>Vardy as the chain endpoint</h3>
    <p>Across the {{ vardy_denom }} complete 3-action chains following a long goal kick,
       Vardy is an actor on <strong>{{ vardy_n }} ({{ '%.0f'|format(vardy_share*100) }}%)</strong>.
       Denominator = ALL post-GK 3-action chains; this is not a cherry-pick.</p>

    <h3>The press triggers</h3>
    <ol>
      <li><strong>Don't press Schmeichel.</strong> 96% of his goal kicks are long. A high press on the keeper
          gives Leicester the easy launch; instead, set the press at the landing zone — typically the
          shallow band 60–90 m from goal, central or right channel.</li>
      <li><strong>The first contact is the only contestable moment.</strong> Pre-position the centre-back duel
          partner; whichever player wins this aerial wins the bulk of the second balls.</li>
      <li><strong>The third action is where the chain dies.</strong> Mean ΔxT collapses on action #3 — that's
          where the press should aim to win the ball, not at the keeper.</li>
    </ol>
  </div>

  <footer class="caveats">
    <strong>Caveats.</strong>
    "Long" defined as <code>pass_height == 'High Pass'</code> (canonical, 96%); cross-checked against
    <code>pass_length ≥ 30 m</code> (99.6% agreement on the verify harness).
    First contact = first event within 8 m of the GK landing; some StatsBomb-emitted aerials lack a
    duel event so we use proximity instead.
    xT trained on the 38 Leicester-match events (~127 k actions, both teams) — fallback to the
    full-PL training set was de-scoped due to a <code>socceraction</code>/<code>numpy</code> dependency
    conflict on macOS arm64; the math (Karun Singh, 2018) is identical and the iteration is hand-rolled
    in <code>analysis/xthreat.py</code>.
    Two head-to-head fixtures vs United only: claims are structural ("if your back-line resembles X,
    the route is Y"), never match-specific.
  </footer>
</section>
""")


REPORT_TEMPLATE = Template("""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Leicester City 2015/16 — Tactical Briefing</title>
<meta name="description" content="Manchester United–facing tactical briefing on Leicester City 2015/16 — set-piece patterns and goal-kick build-up." />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<link rel="stylesheet" href="styles.css" />
</head>
<body>
<main>
  <h1>Leicester City 2015/16 — tactical briefing</h1>
  <p class="prelude">
    Two single-side A4 cards: one on Leicester's attacking corners (which United had to defend), one
    on Leicester's goal-kick build-up (where the press flips possession). Public StatsBomb event data only.
    Methodology is data-source-agnostic and plugs into Opta / StatsBomb commercial feeds.
  </p>
  {{ card_a }}
  {{ card_b }}
  <footer class="report-footer">
    <p>Built by <a href="https://github.com/mjdup77">MJ du Plessis</a>.
       Source code: <a href="https://github.com/mjdup77/manchester-united-fc">github.com/mjdup77/manchester-united-fc</a>.
       Data: <a href="https://github.com/statsbomb/open-data">StatsBomb open data</a>, PL 2015/16.</p>
  </footer>
</main>
</body>
</html>
""")


INDEX_TEMPLATE = Template("""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Manchester United — Tactical Intelligence (portfolio)</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<link rel="stylesheet" href="styles.css" />
</head>
<body class="index-page">
<main class="index-main">
  <h1>Manchester United — tactical intelligence</h1>
  <p class="prelude">Opponent-profile and tactical analyses on public football event data, written
    for a Senior Football Data Analyst portfolio.</p>
  <ul class="card-index">
    <li>
      <a href="leicester-1516.html">
        <h2>Leicester City 2015/16</h2>
        <p>Two A4 briefing cards — set-piece patterns (Card A) and goal-kick build-up (Card B). The
          methodology is demonstrated on the most recent freely available PL event-level data, designed
          to plug directly into modern Opta / StatsBomb commercial feeds.</p>
      </a>
    </li>
  </ul>
  <p class="prelude">More analyses to follow. Source: <a href="https://github.com/mjdup77/manchester-united-fc">github.com/mjdup77/manchester-united-fc</a>.</p>
</main>
</body>
</html>
""")


def render_briefing(corners_out: dict, gk_out: dict, reports_dir: Path) -> Path:
    card_a_html = CARD_A_TEMPLATE.render(**corners_out["headline"])
    card_b_html = CARD_B_TEMPLATE.render(
        **gk_out["headline"],
        zones=gk_out["zone_summary"],
    )
    html = REPORT_TEMPLATE.render(card_a=card_a_html, card_b=card_b_html)

    out = reports_dir / "leicester-1516.html"
    out.write_text(html, encoding="utf-8")

    index = reports_dir / "index.html"
    index.write_text(INDEX_TEMPLATE.render(), encoding="utf-8")

    log.info("Rendered %s + %s", out.name, index.name)
    return out
