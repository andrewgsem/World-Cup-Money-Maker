"""
data_loaders.py
---------------
The data layer for the World Cup model. Each source is a function with a
common shape; results are cached to local parquet so you never re-pull on a
notebook restart. Keep the fragile/network parts isolated here so the model
file stays clean.

Sources implemented:
  - load_results():  international match results (the backbone), 1872->present,
                     incl. the scheduled (unscored) 2026 World Cup fixtures.
  - load_xg():       OPTIONAL xG via the `soccerdata` package (FBref). Network
                     on your machine required; safe no-op if unavailable.

Everything is cached under ./data_cache/ as parquet.
"""

from __future__ import annotations
import os
import urllib.request
import pandas as pd

CACHE_DIR = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def _ensure_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def load_results(force_refresh: bool = False) -> pd.DataFrame:
    """
    The backbone dataset: every international match with score, tournament,
    venue and a neutral-ground flag. Future (unplayed) fixtures have NaN scores
    and are kept -- that is how you get the 2026 World Cup fixture list.

    Returns a DataFrame with columns:
        date (datetime), home_team, away_team, home_score, away_score,
        tournament, city, country, neutral (bool)
    """
    _ensure_cache()
    path = _cache_path("results")

    if os.path.exists(path) and not force_refresh:
        df = pd.read_parquet(path)
    else:
        # raw.githubusercontent.com mirror of martj42/international_results
        local_csv = os.path.join(CACHE_DIR, "results.csv")
        urllib.request.urlretrieve(RESULTS_URL, local_csv)
        df = pd.read_csv(local_csv)
        df.to_parquet(path)

    df["date"] = pd.to_datetime(df["date"])
    # normalise the neutral flag to a real bool (the CSV ships "TRUE"/"FALSE")
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def split_played_future(df: pd.DataFrame):
    """Separate completed matches (have scores) from scheduled fixtures."""
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    future = df[df["home_score"].isna()].copy()
    return played, future


def load_xg(force_refresh: bool = False) -> pd.DataFrame | None:
    """
    OPTIONAL. Pull international xG via soccerdata/FBref. Returns None if the
    package isn't installed or the network call fails -- the model runs fine
    without it (you just lose the xG upgrade). Wire it in later.
    """
    _ensure_cache()
    path = _cache_path("xg")
    if os.path.exists(path) and not force_refresh:
        return pd.read_parquet(path)
    try:
        import soccerdata as sd  # noqa: F401
        # Example wiring -- adjust season/league to the international comps you want.
        # fbref = sd.FBref(leagues="INT-World Cup", seasons=2022)
        # xg = fbref.read_team_match_stats(stat_type="schedule")
        # xg.to_parquet(path); return xg
        print("[load_xg] soccerdata available -- fill in the FBref call for your comps.")
        return None
    except Exception as e:  # ImportError or network/scrape failure
        print(f"[load_xg] xG source unavailable ({e}). Continuing without xG.")
        return None


if __name__ == "__main__":
    d = load_results()
    played, future = split_played_future(d)
    print(f"results: {len(d):,} rows  |  played: {len(played):,}  |  future: {len(future):,}")
    print(f"date range: {d.date.min().date()} -> {d.date.max().date()}")
