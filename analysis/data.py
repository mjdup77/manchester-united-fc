"""Data loading helpers for StatsBomb open data — Premier League 2015/16.

Uses statsbombpy to fetch competitions, matches, and events, and caches them
as parquet files in data/raw/ to avoid repeated HTTP calls.

Key finding: while tracking data isn't available for 2015/16 PL, every
Shot event DOES carry a `shot_freeze_frame` — the position of every player
on the pitch at the moment of the shot. That enables real spatial analysis
of set pieces (who's where when the ball meets head/foot).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PL_COMPETITION_ID = 2
PL_2015_16_SEASON_ID = 27


def load_competitions() -> pd.DataFrame:
    return sb.competitions()


def _coerce_mixed_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet (pyarrow) rejects columns that mix int and str. Cast to str."""
    for col in df.select_dtypes(include="object").columns:
        # keep nested list/dict columns alone — they are handled separately
        sample = next((v for v in df[col] if v is not None and not (isinstance(v, float) and pd.isna(v))), None)
        if isinstance(sample, (list, dict)):
            continue
        df[col] = df[col].where(df[col].notna(), None).astype("string")
    return df


def load_pl_matches() -> pd.DataFrame:
    """All PL 2015/16 matches, cached."""
    cache = RAW_DIR / "pl_2015_16_matches.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    matches = sb.matches(
        competition_id=PL_COMPETITION_ID, season_id=PL_2015_16_SEASON_ID
    )
    matches = _coerce_mixed_object_columns(matches)
    matches.to_parquet(cache)
    return matches


def load_team_matches(team_name: str) -> pd.DataFrame:
    """All PL 2015/16 matches for a given team, sorted by date."""
    matches = load_pl_matches()
    mask = (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    return matches.loc[mask].sort_values("match_date").reset_index(drop=True)


def _serialise_nested(ev: pd.DataFrame) -> pd.DataFrame:
    """Convert list-of-dict columns to JSON strings for parquet compatibility."""
    nested_cols = [
        "shot_freeze_frame",
        "tactics",
        "goalkeeper_saved_to_post",
        "related_events",
    ]
    for col in nested_cols:
        if col in ev.columns:
            ev[col] = ev[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else None
            )
    return ev


def load_events(match_id: int) -> pd.DataFrame:
    """Events for one match, cached to parquet."""
    cache = RAW_DIR / f"events_{match_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    ev = sb.events(match_id=match_id, fmt="dataframe")
    ev = _serialise_nested(ev)
    ev = _coerce_mixed_object_columns(ev)
    ev.to_parquet(cache)
    return ev


def load_team_events(team_name: str) -> pd.DataFrame:
    """Concatenate events for every match a team played."""
    matches = load_team_matches(team_name)
    frames = []
    for _, m in matches.iterrows():
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
    return pd.concat(frames, ignore_index=True)


def parse_freeze_frame(ff_json: str | None) -> list[dict] | None:
    """Inverse of the JSON-stringify done during save."""
    if ff_json is None or (isinstance(ff_json, float) and pd.isna(ff_json)):
        return None
    return json.loads(ff_json)


if __name__ == "__main__":
    print("Loaded competitions; PL 15/16 ready.")
    print(load_pl_matches().head(3)[["match_date", "home_team", "away_team"]].to_string(index=False))
