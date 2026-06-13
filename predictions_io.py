"""
predictions_io.py
-----------------
Bridge between your private updater (the notebook) and the public viewer
(app.py). The updater LOCKS each day's predictions into a log the moment you
predict -- before results are known -- then fills in actual scores as games
finish. The website only READS this log; it never fits or edits anything.

Flow in your notebook each matchday:
    1. preds = predict_matchday(model, day_fixtures)   # your existing call
    2. lock_predictions(preds)                          # locks them (once)
    3. ... games are played ...
    4. record_result(...) for each                      # your existing flow
    5. sync_actuals(PLAYED)                             # fills real scores in
    6. push the updated log file to GitHub              # friends see it

Scoring buckets (each played match falls in exactly one, hierarchical):
    - correct_score    : predicted most-likely score == actual score
    - correct_outcome  : predicted W/D/L matches, but score wrong
    - nothing          : predicted outcome wrong
"""

from __future__ import annotations
import os
import pandas as pd

PREDLOG_PATH = "data_cache/prediction_log.parquet"

_COLUMNS = ["date", "home", "away", "p_home", "p_draw", "p_away",
            "pred_score", "home_score", "away_score"]


def _empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def load_log(path: str = PREDLOG_PATH) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return _empty_log()


def lock_predictions(preds_df: pd.DataFrame, path: str = PREDLOG_PATH) -> pd.DataFrame:
    """Lock predictions into the log. Unplayed matches are UPDATED with the
    latest prediction; matches that already have a real result are frozen and
    never overwritten."""
    log = load_log(path)
    p = preds_df.rename(columns={"P(home)": "p_home", "P(draw)": "p_draw",
                                 "P(away)": "p_away"})
    has_outcome = "outcome_score" in p.columns
    for r in p.itertuples(index=False):
        d = pd.Timestamp(r.date).normalize()
        m = (log["date"] == d) & (log["home"] == r.home) & (log["away"] == r.away)
        if m.any():
            if pd.notna(log.loc[m, "home_score"]).any():
                continue                     # already played -> freeze, don't touch
            log = log[~m]                     # unplayed -> drop old, re-add fresh below
        log = pd.concat([log, pd.DataFrame([{
            "date": d, "home": r.home, "away": r.away,
            "p_home": r.p_home, "p_draw": r.p_draw, "p_away": r.p_away,
            "pred_score": r.likely_score,
            "pred_score_outcome": r.outcome_score if has_outcome else pd.NA,
            "home_score": pd.NA, "away_score": pd.NA,
        }])], ignore_index=True)
    log.to_parquet(path)
    return log


def sync_actuals(played_df: pd.DataFrame, path: str = PREDLOG_PATH) -> pd.DataFrame:
    """Fill in real scores for any locked prediction whose match has now been
    played. Matches on date + team names."""
    log = load_log(path)
    if log.empty:
        return log
    pl = played_df.copy()
    pl["d"] = pd.to_datetime(pl["date"]).dt.normalize()
    lookup = {(row.d, row.home_team, row.away_team): (row.home_score, row.away_score)
              for row in pl.itertuples(index=False)}
    for i, row in log.iterrows():
        if pd.notna(row["home_score"]):
            continue
        key = (pd.Timestamp(row["date"]).normalize(), row["home"], row["away"])
        if key in lookup:
            hs, as_ = lookup[key]
            log.at[i, "home_score"] = int(hs)
            log.at[i, "away_score"] = int(as_)
    log.to_parquet(path)
    return log


def _headline_score(row) -> str:
    """The score the site actually headlines: the outcome-conditional score
    when it exists, the overall most-likely score for rows locked before
    that upgrade."""
    oc = row.get("pred_score_outcome", None)
    if oc is not None and not pd.isna(oc):
        return oc
    return row["pred_score"]


def _bucket(row) -> str:
    probs = {"home": row["p_home"], "draw": row["p_draw"], "away": row["p_away"]}
    pred_outcome = max(probs, key=probs.get)
    hs, as_ = int(row["home_score"]), int(row["away_score"])
    actual_outcome = "home" if hs > as_ else ("draw" if hs == as_ else "away")
    if _headline_score(row) == f"{hs}-{as_}":
        return "correct_score"
    if pred_outcome == actual_outcome:
        return "correct_outcome"
    return "nothing"


def compute_scorecard(log: pd.DataFrame):
    """Return (played_df_with_bucket_column, counts_dict) for played matches.
    Evaluation is against the HEADLINE score (outcome-conditional when
    available), so the scorecard judges exactly what the site displayed."""
    played = log[log["home_score"].notna()].copy()
    if played.empty:
        return played, {"correct_score": 0, "correct_outcome": 0, "nothing": 0, "total": 0}
    played["eval_score"] = played.apply(_headline_score, axis=1)
    played["bucket"] = played.apply(_bucket, axis=1)
    counts = played["bucket"].value_counts().to_dict()
    return played, {
        "correct_score": counts.get("correct_score", 0),
        "correct_outcome": counts.get("correct_outcome", 0),
        "nothing": counts.get("nothing", 0),
        "total": len(played),
    }