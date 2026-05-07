"""Analysis package — Leicester City 2015/16 profile for the United briefing cards.

Module-level constants and shared paths. Other modules import these via
`from analysis import RAW_DIR, LEICESTER, ...`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
DOCS_DIR = ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, ASSETS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LEICESTER = "Leicester City"
PL_COMPETITION_ID = 2
PL_2015_16_SEASON_ID = 27

LEICESTER_EVENTS_PARQUET = PROCESSED_DIR / "leicester_all_events.parquet"
