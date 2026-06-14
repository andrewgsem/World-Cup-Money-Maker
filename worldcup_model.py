"""
worldcup_model.py
-----------------
2026 World Cup match-prediction model.

This file uses `# %%` cell markers: open it in VS Code with the Python +
Jupyter extensions and you get a notebook experience (run cell-by-cell with
Shift+Enter) in one clean, version-controllable file.

Model summary
  - Dixon-Coles: each team has latent attack & defence strengths; expected
    goals come from attack vs opponent defence (+ home/venue term); goals are
    Poisson with the DC low-score correction.
  - Fitted by maximum likelihood with (a) exponential time decay, (b) per-match
    competition weighting, (c) an L2 (ridge) penalty = Gaussian shrinkage prior.
  - Match engine turns two teams + a date into a full scoreline grid -> W/D/L.
  - Evaluated with Ranked Probability Score and log loss vs an Elo baseline.
  - Real-time loop: re-fit each matchday, predict that day's fixtures, log score.

Build order matches the cells below. Run top to bottom the first time.
"""

# %% [imports & data]
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import poisson

from data_loaders import load_results, split_played_future
from predictions_io import lock_predictions, sync_actuals

RAW = load_results()

_LIVE_P = "data_cache/played_live.parquet"
_LIVE_F = "data_cache/future_live.parquet"
if os.path.exists(_LIVE_P) and os.path.exists(_LIVE_F):
    PLAYED = pd.read_parquet(_LIVE_P)
    FUTURE = pd.read_parquet(_LIVE_F)
    print(f"Resumed saved progress: {len(PLAYED):,} played, {len(FUTURE)} future remaining")
else:
    PLAYED, FUTURE = split_played_future(RAW)
    print(f"Fresh start: {len(PLAYED):,} played, {len(FUTURE)} future")

# %% [competition weights]
# Not all matches inform team strength equally. Competitive matches count more
# than friendlies. These multipliers stack on top of the time-decay weight.
# Tune them; they are deliberately simple. Matched by substring on `tournament`.
COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "FIFA World Cup qualification": 0.85,
    "UEFA Nations League": 0.85,
    "CONCACAF Nations League": 0.80,
    "Copa América": 0.90,
    "UEFA Euro": 0.90,
    "African Cup of Nations": 0.85,
    "AFC Asian Cup": 0.80,
    "Confederations Cup": 0.85,
    "qualification": 0.75,   # generic catch-all for other qualifiers
    "Friendly": 0.45,
}
DEFAULT_COMP_WEIGHT = 0.60


def competition_weight(tournament: str) -> float:
    if tournament in COMPETITION_WEIGHTS:
        return COMPETITION_WEIGHTS[tournament]
    for key, w in COMPETITION_WEIGHTS.items():       # substring fallback
        if key.lower() in str(tournament).lower():
            return w
    return DEFAULT_COMP_WEIGHT


# %% [the Dixon-Coles model]
@dataclass
class DCModel:
    """A fitted Dixon-Coles model. Holds team strengths and global params."""
    teams: list
    attack: dict        # team -> attack strength (mean 0)
    defence: dict       # team -> defence strength (mean 0)
    intercept: float    # baseline log scoring rate
    home_adv: float     # additive home/venue advantage (log scale)
    rho: float          # DC low-score correction
    max_goals: int = 10

    # -- expected goals for a single fixture -------------------------------
    def _lambdas(self, home, away, neutral=False):
        a_h = self.attack.get(home, 0.0)
        d_h = self.defence.get(home, 0.0)
        a_a = self.attack.get(away, 0.0)
        d_a = self.defence.get(away, 0.0)
        h = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + h + a_h - d_a)   # home expected goals
        mu = np.exp(self.intercept + a_a - d_h)        # away expected goals
        return lam, mu

    # -- full scoreline probability grid -----------------------------------
    def score_matrix(self, home, away, neutral=False):
        lam, mu = self._lambdas(home, away, neutral)
        gx = np.arange(self.max_goals + 1)
        px = poisson.pmf(gx, lam)
        py = poisson.pmf(gx, mu)
        grid = np.outer(px, py)                        # independent Poisson
        grid *= _dc_correction_matrix(lam, mu, self.rho, self.max_goals)
        grid /= grid.sum()                             # renormalise after tau + truncation
        return grid

    # -- collapse grid to W/D/L --------------------------------------------
    def predict(self, home, away, neutral=False):
        grid = self.score_matrix(home, away, neutral)
        p_home = np.tril(grid, -1).sum()               # home goals > away goals
        p_draw = np.trace(grid)
        p_away = np.triu(grid, 1).sum()
        return {"home": float(p_home), "draw": float(p_draw), "away": float(p_away)}

    def most_likely_score(self, home, away, neutral=False):
        grid = self.score_matrix(home, away, neutral)
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        return int(i), int(j), float(grid[i, j])


def _dc_correction_matrix(lam, mu, rho, max_goals):
    """Dixon-Coles tau: corrects the four low-score cells Poisson misprices."""
    m = np.ones((max_goals + 1, max_goals + 1))
    m[0, 0] = 1.0 - lam * mu * rho
    m[0, 1] = 1.0 + lam * rho
    m[1, 0] = 1.0 + mu * rho
    m[1, 1] = 1.0 - rho
    return np.clip(m, 1e-10, None)                     # keep strictly positive


# %% [the fit -- maximum likelihood with time decay + comp weight + shrinkage]
def _tau_terms(hs, as_, lam, mu, rho):
    """Vectorised tau for the four corrected cells, per observed scoreline."""
    tau = np.ones_like(lam)
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return np.clip(tau, 1e-10, None)


def fit_dixon_coles(
    matches: pd.DataFrame,
    ref_date=None,
    xi: float = 0.30,        # time-decay rate per YEAR (half-life = ln2/xi years)
    reg: float = 0.05,       # ridge/shrinkage strength (Gaussian prior toward mean)
    max_goals: int = 10,
    min_matches: int = 3,    # drop teams with too little history to estimate
) -> DCModel:
    """
    Fit attack/defence/intercept/home/rho by maximising the weighted DC
    log-likelihood. Each match is weighted by exp(-xi * years_ago) * comp_weight.
    A ridge penalty shrinks strengths toward the mean (small-sample stabiliser).
    """
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    if ref_date is None:
        ref_date = df["date"].max()
    ref_date = pd.Timestamp(ref_date)
    df = df[df["date"] <= ref_date]

    # keep teams with enough matches to estimate; others fall back to mean (0)
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    keep = set(counts[counts >= min_matches].index)
    df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)]
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)

    # per-match weights: time decay (years) x competition importance
    years_ago = (ref_date - df["date"]).dt.days.values / 365.25
    w_time = np.exp(-xi * years_ago)
    w_comp = df["tournament"].map(competition_weight).values
    weights = w_time * w_comp

    hi = df["home_team"].map(idx).values
    ai = df["away_team"].map(idx).values
    hs = df["home_score"].values.astype(int)
    as_ = df["away_score"].values.astype(int)
    neutral = df["neutral"].values.astype(bool)

    # param vector: [attack(n), defence(n), intercept, home, rho]
    def unpack(p):
        attack = p[:n]
        defence = p[n:2 * n]
        intercept, home, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        # enforce identifiability: centre attack & defence to mean 0
        attack = attack - attack.mean()
        defence = defence - defence.mean()
        return attack, defence, intercept, home, rho

    def neg_loglik(p):
        attack, defence, intercept, home, rho = unpack(p)
        h = np.where(neutral, 0.0, home)
        log_lam = intercept + h + attack[hi] - defence[ai]
        log_mu = intercept + attack[ai] - defence[hi]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)
        # Poisson log-pmf for both scorelines
        ll = (hs * log_lam - lam - gammaln_int(hs)) + (as_ * log_mu - mu - gammaln_int(as_))
        tau = _tau_terms(hs, as_, lam, mu, rho)
        ll = ll + np.log(tau)
        wll = np.sum(weights * ll)
        penalty = reg * (np.sum(attack ** 2) + np.sum(defence ** 2))  # shrinkage
        return -(wll - penalty)

    # init: zeros for strengths, sensible baseline scoring rate & home edge
    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = np.log(1.35)   # intercept ~ avg goals
    p0[2 * n + 1] = 0.25       # home advantage
    p0[2 * n + 2] = -0.05      # rho

    # bound rho away from values that make tau non-positive
    bounds = [(None, None)] * (2 * n) + [(None, None), (None, None), (-0.2, 0.2)]
    res = minimize(neg_loglik, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500})

    attack, defence, intercept, home, rho = unpack(res.x)
    return DCModel(
        teams=teams,
        attack={t: float(attack[idx[t]]) for t in teams},
        defence={t: float(defence[idx[t]]) for t in teams},
        intercept=float(intercept), home_adv=float(home), rho=float(rho),
        max_goals=max_goals,
    )


# small helper: log(k!) for non-negative integer counts (Poisson normaliser)
from scipy.special import gammaln
def gammaln_int(k):
    return gammaln(k + 1)


# %% [Elo baseline -- the model you must beat]
def fit_elo(matches: pd.DataFrame, k=30, home_adv=65, base=1500, scale=400):
    """Plain Elo over match history. Returns final ratings + a predict fn."""
    df = matches.dropna(subset=["home_score", "away_score"]).sort_values("date")
    r = {}
    for row in df.itertuples(index=False):
        rh = r.get(row.home_team, base)
        ra = r.get(row.away_team, base)
        ha = 0 if row.neutral else home_adv
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / scale))
        if row.home_score > row.away_score:
            s = 1.0
        elif row.home_score < row.away_score:
            s = 0.0
        else:
            s = 0.5
        # margin-of-victory multiplier (mild)
        mov = np.log1p(abs(row.home_score - row.away_score))
        r[row.home_team] = rh + k * mov * (s - exp_h)
        r[row.away_team] = ra + k * mov * ((1 - s) - (1 - exp_h))

    def predict(home, away, neutral=False, draw_frac=0.26):
        rh, ra = r.get(home, base), r.get(away, base)
        ha = 0 if neutral else home_adv
        p_home_raw = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / scale))
        # carve a draw probability out of the win/loss split (simple baseline)
        p_draw = draw_frac
        p_home = p_home_raw * (1 - p_draw)
        p_away = (1 - p_home_raw) * (1 - p_draw)
        return {"home": p_home, "draw": p_draw, "away": p_away}

    return r, predict

# %% [Tier 1: blend fitted strengths toward an Elo-implied prior]
def blend_elo_prior(model, train, cutoff, blend=0.6):
    """Pull DC attack/defence toward an Elo-implied prior. blend=0 is pure DC,
    blend=1 is pure Elo-implied. ~0.5-0.6 scored best in backtest."""
    ratings, _ = fit_elo(train[train.date < cutoff])
    teams = model.teams
    elos = np.array([ratings.get(t, 1500.0) for t in teams])
    z = (elos - elos.mean()) / (elos.std() + 1e-9)        # standardised Elo
    att = np.array([model.attack[t] for t in teams])
    deff = np.array([model.defence[t] for t in teams])
    ap = z * att.std()                                     # match the fitted scale
    dp = z * deff.std()
    na = {t: (1-blend)*model.attack[t]  + blend*ap[i] for i, t in enumerate(teams)}
    nd = {t: (1-blend)*model.defence[t] + blend*dp[i] for i, t in enumerate(teams)}
    return DCModel(teams, na, nd, model.intercept, model.home_adv, model.rho, model.max_goals)


# %% [evaluation -- proper scoring rules]
def outcome_vector(home_score, away_score):
    """One-hot [home, draw, away] for an observed result."""
    if home_score > away_score:
        return np.array([1.0, 0.0, 0.0])
    if home_score == away_score:
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def rps(pred: dict, obs_vec: np.ndarray) -> float:
    """Ranked Probability Score for ordered W/D/L. Lower is better."""
    p = np.array([pred["home"], pred["draw"], pred["away"]])
    cum_p = np.cumsum(p)
    cum_o = np.cumsum(obs_vec)
    return float(np.sum((cum_p[:-1] - cum_o[:-1]) ** 2) / (len(p) - 1))


def log_loss_one(pred: dict, obs_vec: np.ndarray) -> float:
    p = np.clip(np.array([pred["home"], pred["draw"], pred["away"]]), 1e-12, 1)
    return float(-np.sum(obs_vec * np.log(p)))


# %% [walk-forward backtest -- honest out-of-sample evaluation]
def backtest_tournament(all_matches, tournament_name, year, **fit_kwargs):
    """
    Fit on everything BEFORE a tournament starts, then predict each of its
    games (no leakage). Compares DC vs Elo with RPS and log loss.
    """
    df = all_matches.dropna(subset=["home_score", "away_score"]).copy()
    mask = (df["tournament"] == tournament_name) & (df["date"].dt.year == year)
    test = df[mask].sort_values("date")
    if test.empty:
        raise ValueError(f"No matches for {tournament_name} {year}")
    cutoff = test["date"].min()
    train = df[df["date"] < cutoff]

    dc = fit_dixon_coles(train, ref_date=cutoff, **fit_kwargs)
    _, elo_pred = fit_elo(train)

    rows = []
    for m in test.itertuples(index=False):
        obs = outcome_vector(m.home_score, m.away_score)
        dcp = dc.predict(m.home_team, m.away_team, neutral=m.neutral)
        elp = elo_pred(m.home_team, m.away_team, neutral=m.neutral)
        rows.append({
            "dc_rps": rps(dcp, obs), "elo_rps": rps(elp, obs),
            "dc_ll": log_loss_one(dcp, obs), "elo_ll": log_loss_one(elp, obs),
        })
    r = pd.DataFrame(rows)
    return {
        "n_games": len(r),
        "dc_rps": r.dc_rps.mean(), "elo_rps": r.elo_rps.mean(),
        "dc_logloss": r.dc_ll.mean(), "elo_logloss": r.elo_ll.mean(),
    }


# %% [predict_matchday: probabilities + model expected goals]
from scipy.stats import poisson
from scipy.optimize import brentq

def _probs_from_goals(lam, mu, mg=10):
    gx = np.arange(mg + 1)
    g = np.outer(poisson.pmf(gx, lam), poisson.pmf(gx, mu))
    return float(np.tril(g, -1).sum()), float(np.trace(g)), float(np.triu(g, 1).sum())

def predict_matchday(model, fixtures):
    out = []
    for m in fixtures.itertuples(index=False):
        lam, mu = model._lambdas(m.home_team, m.away_team, neutral=bool(m.neutral))
        ph, pdr, pa = _probs_from_goals(lam, mu)
        out.append({"date": m.date.date() if hasattr(m.date, "date") else m.date,
                    "home": m.home_team, "away": m.away_team,
                    "P(home)": round(ph, 3), "P(draw)": round(pdr, 3), "P(away)": round(pa, 3),
                    "exp_home": round(lam, 2), "exp_away": round(mu, 2),
                    "total_goals": round(lam + mu, 3),
                    "likely_score": f"{int(round(lam))}-{int(round(mu))}",
                    "outcome_score": f"{int(round(lam))}-{int(round(mu))}"})
    return pd.DataFrame(out)

# %% [market blend: 1/3 model, 1/3 market, 1/3 your predicted score]
def _goals_from_probs(ph, pdr, pa, total):
    """Invert blended probabilities to expected goals, holding total fixed."""
    target_diff = ph - pa
    def f(split):
        lam = max(total * (1 + split) / 2, 0.05); mu = max(total - lam, 0.05)
        h, _, a = _probs_from_goals(lam, mu)
        return (h - a) - target_diff
    try:
        s = brentq(f, -0.98, 0.98)
    except ValueError:
        s = float(np.clip(target_diff, -0.9, 0.9))
    lam = max(total * (1 + s) / 2, 0.05); mu = max(total - lam, 0.05)
    return lam, mu

def blend_market(preds_df, odds, my_scores=None, w_model=1/3, w_market=1/3, w_mine=1/3):
    """Probabilities: blend model + market (renormalised to the two non-mine weights).
    Scoreline: 1/3 model expected goals + 1/3 market-implied + 1/3 your predicted score.
    my_scores: {(home, away): (home_goals, away_goals)}  -- your own scoreline guess.
    A match needs odds to be blended; your score is optional per match."""
    my_scores = my_scores or {}
    df = preds_df.copy()
    for i, r in df.iterrows():
        key = (r["home"], r["away"])
        if key not in odds:
            continue
        oh, od, oa = odds[key]
        raw = np.array([1/oh, 1/od, 1/oa]); mkt = raw / raw.sum()

        # --- probabilities: model vs market, split by their relative weights ---
        pm = w_model / (w_model + w_market)   # model share of the prob blend
        model_p = np.array([r["P(home)"], r["P(draw)"], r["P(away)"]])
        blend = pm * model_p + (1 - pm) * mkt
        blend = blend / blend.sum()
        ph, pdr, pa = [float(x) for x in blend]
        df.at[i, "P(home)"], df.at[i, "P(draw)"], df.at[i, "P(away)"] = round(ph,3), round(pdr,3), round(pa,3)

        # --- scoreline: three-way average of expected goals ---
        total = float(r["total_goals"])
        lam_model, mu_model = float(r["exp_home"]), float(r["exp_away"])
        lam_mkt, mu_mkt = _goals_from_probs(*mkt, total)
        if key in my_scores:
            my_h, my_a = my_scores[key]
            eh = w_model*lam_model + w_market*lam_mkt + w_mine*my_h
            ea = w_model*mu_model  + w_market*mu_mkt  + w_mine*my_a
        else:  # no score from you -> just model+market on the score
            eh = (lam_model + lam_mkt) / 2
            ea = (mu_model + mu_mkt) / 2
        df.at[i, "exp_home"], df.at[i, "exp_away"] = round(eh, 2), round(ea, 2)

        rh, ra = int(round(eh)), int(round(ea))
        if rh == ra:                      # tie-break a draw toward the favourite
            if eh - ea > 0.10: rh += 1
            elif ea - eh > 0.10: ra += 1
        df.at[i, "likely_score"] = df.at[i, "outcome_score"] = f"{rh}-{ra}"
    return df


# %% [predict one day]
import warnings; warnings.filterwarnings("ignore")

target = pd.Timestamp("2026-06-14")          # <-- change each matchday

ODDS = {
    ("Germany", "Curaçao"): (1.05, 15.00, 30.00),
    ("Ivory Coast", "Ecuador"): (3.25, 2.65, 2.70),
    ("Netherlands", "Japan"): (2.10, 3.50, 3.45),
    ("Sweden", "Tunisia"): (2.00, 3.35, 3.95),
}
MY_SCORES = {
    ("Germany", "Curaçao"): (3, 0),     # your own scoreline call, per match
    ("Ivory Coast", "Ecuador"): (2, 1),
    ("Netherlands", "Japan"): (1, 1),
    ("Sweden", "Tunisia"): (2, 0),
}

def recent(df, cutoff, years=18):
    return df[(df.date < cutoff) & (df.date >= cutoff - pd.Timedelta(days=365*years))]

day_fixtures = FUTURE[FUTURE.date == target]
if len(day_fixtures) == 0:
    print("No games scheduled on", target.date())
else:
    dc = fit_dixon_coles(recent(PLAYED, target), ref_date=target, xi=0.30, reg=0.05)
    model = blend_elo_prior(dc, recent(PLAYED, target), target, blend=0.6)
    preds = predict_matchday(model, day_fixtures)
    preds = blend_market(preds, ODDS, my_scores=MY_SCORES)
    lock_predictions(preds)
    print(preds.to_string(index=False))


# %% [record real match results as they finish]
def record_result(home, away, home_score, away_score, date=None):
    """Fill in a finished match's score and move it from FUTURE into PLAYED,
    so the next matchday's re-fit learns from it. Team names must match the
    dataset's spelling exactly (e.g. 'South Korea', 'Czech Republic')."""
    global PLAYED, FUTURE
    mask = (FUTURE.home_team == home) & (FUTURE.away_team == away)
    if date is not None:
        mask &= (FUTURE.date == pd.Timestamp(date))
    hits = FUTURE[mask]
    if len(hits) == 0:
        print(f"No scheduled fixture found for: {home} vs {away}")
        for typed in (home, away):
            near = FUTURE[(FUTURE.home_team.str.contains(typed, case=False, na=False)) |
                          (FUTURE.away_team.str.contains(typed, case=False, na=False))]
            if len(near):
                print(f'  matching "{typed}":', sorted(set(near.home_team) | set(near.away_team)))
        return
    if len(hits) > 1:
        print("Multiple fixtures match - pass date= to disambiguate:")
        print(hits[["date", "home_team", "away_team"]].to_string(index=False)); return
    row = hits.iloc[[0]].copy()
    row["home_score"] = int(home_score); row["away_score"] = int(away_score)
    PLAYED = pd.concat([PLAYED, row], ignore_index=True)
    PLAYED["home_score"] = PLAYED["home_score"].astype(int)
    PLAYED["away_score"] = PLAYED["away_score"].astype(int)
    FUTURE = FUTURE.drop(hits.index[0])
    print(f"Recorded: {home} {home_score}-{away_score} {away}   |   played={len(PLAYED):,}  future={len(FUTURE)}")

# %% [record results -- run, then run the save block]
record_result("Qatar", "Switzerland", 1, 1)
record_result("Brazil", "Morocco", 1, 1)
record_result("Haiti", "Scotland", 0, 1)
record_result("Australia", "Turkey", 2, 0)


# %% [save progress -- run after recording results]
PLAYED.to_parquet("data_cache/played_live.parquet")
FUTURE.to_parquet("data_cache/future_live.parquet")
sync_actuals(PLAYED)
print(f"Saved: {len(PLAYED):,} played, {len(FUTURE)} future remaining")


# %%
FUTURE[FUTURE.date == pd.Timestamp("2026-06-14")][["home_team", "away_team"]]
# %%
