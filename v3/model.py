"""v3: market-anchored margin model with prior-season carryover.

Skeleton logic (what gets assessed, in order):
  1. Market anchor   : closing spread = expected home margin (strongest signal).
  2. Team strength   : season-to-date stat differences, blended toward each team's
                       PRIOR-season level (persistence rho) instead of toward zero:
                       blend = (GP*current + K*rho*prior) / (GP + K)
  3. Situation       : rest days, back-to-backs, sample size.
  4. Residual model  : ridge regression predicts (margin - spread). Final margin is
                       Normal(spread + residual + judgment_adj, sigma).
  5. Judgment layer  : judgment_adj (points, home-positive) for things the data
                       can't see — injuries, minutes limits, lineup news. Logged
                       separately so the model and the final call are graded apart.

One distribution gives win prob, ATS cover prob and a projected score.
"""
import math, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
import build_dataset as b2  # noqa: E402

STATS = b2.DIFF_STATS
K = 12.0      # games of weight given to the prior
RHO = 0.65    # how much of last season's edge persists (roster churn, aging)
SIGMA = 12.0  # sd of margin around expectation; re-estimated in fit()
CTX = ["rest_diff", "b2b_home", "b2b_away", "gp_min", "spread", "total"]


def season_of(d):
    return np.where(d.dt.month >= 8, d.dt.year + 1, d.dt.year)


def team_long(g):
    cols = STATS + ["GP"]
    h = g[["Date", "TEAM_NAME"] + cols].rename(columns={"TEAM_NAME": "team"})
    a = g[["Date", "TEAM_NAME.1"] + [c + ".1" for c in cols]].rename(
        columns={"TEAM_NAME.1": "team", **{c + ".1": c for c in cols}})
    L = pd.concat([h, a])
    for c in cols:
        L[c] = pd.to_numeric(L[c], errors="coerce")
    L["season"] = season_of(L.Date)
    return L


def prior_table(L):
    """Each team's last available row per season, centered on that season's league mean."""
    last = L.sort_values("Date").groupby(["team", "season"]).tail(1).copy()
    for c in STATS:
        last[c] = last[c] - last.groupby("season")[c].transform("mean")
    last["season"] = last.season + 1          # becomes the prior for next season
    return last.set_index(["team", "season"])[STATS]


def build():
    con = sqlite3.connect(ROOT / "Data" / "dataset.sqlite")
    g = pd.read_sql('select * from "dataset_2012-26"', con)
    g["Date"] = pd.to_datetime(g.Date, errors="coerce").dt.normalize()
    g = b2.fix_same_day_leak(g)
    L = team_long(g)
    prior = prior_table(L)
    # league mean per date-season for centering current stats
    for c in STATS:
        g[c] = pd.to_numeric(g[c], errors="coerce"); g[c + ".1"] = pd.to_numeric(g[c + ".1"], errors="coerce")
    g["season"] = season_of(g.Date)
    g["GP"] = pd.to_numeric(g.GP, errors="coerce").fillna(0); g["GP.1"] = pd.to_numeric(g["GP.1"], errors="coerce").fillna(0)
    ph = prior.reindex(pd.MultiIndex.from_arrays([g.TEAM_NAME, g.season])).fillna(0).values
    pa = prior.reindex(pd.MultiIndex.from_arrays([g["TEAM_NAME.1"], g.season])).fillna(0).values
    gh, ga = g.GP.values[:, None], g["GP.1"].values[:, None]
    # center current stats on the PREVIOUS season's league mean (known before tip),
    # so unequal GP between the two teams can't leak the league mean into the diff
    lm = L.groupby("season")[STATS].mean()
    prev = lm.shift(1).fillna(lm)
    mu = prev.reindex(g.season).values
    cur_h = g[STATS].values - mu
    cur_a = g[[s + ".1" for s in STATS]].values - mu
    bh = (gh * cur_h + K * RHO * ph) / (gh + K)
    ba = (ga * cur_a + K * RHO * pa) / (ga + K)
    # when GP=0 the current row is zeros; use prior only (diff of centered priors)
    feats = pd.DataFrame(bh - ba, columns=[f"b_{s}" for s in STATS], index=g.index)
    feats.loc[(g.GP == 0) | (g["GP.1"] == 0)] = (RHO * (ph - pa))[((g.GP == 0) | (g["GP.1"] == 0)).values]
    feats["Date"], feats["Home"], feats["Away"] = g.Date, g.TEAM_NAME, g["TEAM_NAME.1"]
    v2 = pd.read_parquet(ROOT / "v2" / "v2_games.parquet")
    D = v2.merge(feats, on=["Date", "Home", "Away"], how="inner")
    D["rest_diff"] = D.rest_home - D.rest_away
    D["b2b_home"] = (D.rest_home <= 1).astype(int)
    D["b2b_away"] = (D.rest_away <= 1).astype(int)
    D["resid"] = D.margin - D.spread
    D.to_parquet(ROOT / "v3" / "v3_games.parquet")
    return D


# Backtest 2017-26: adding spread/total/rest/GP to the residual model overfit (worse MAE than
# the market); team-strength features alone beat the line by 1+ pt on 53.6% of 1,233 games.
FEATS = [f"b_{s}" for s in STATS]
LEAN_PTS, PLAY_PTS = 1.0, 1.5   # residual size that flags a lean / a small play


def fit(D, alpha=200.0):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(D[FEATS].values, D.resid.values)
    sigma = float(np.std(D.resid.values - m.predict(D[FEATS].values)))
    return m, sigma


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def distribution(spread, resid_hat, judgment_adj, sigma, home_spread_line=None):
    """spread: expected home margin (+ = home favored). Returns win / cover probs."""
    mu = spread + resid_hat + judgment_adj
    p_win = norm_cdf(mu / sigma)
    line = spread if home_spread_line is None else -home_spread_line
    p_cover = norm_cdf((mu - line) / sigma)
    return mu, p_win, p_cover


if __name__ == "__main__":
    D = build()
    print(len(D), "games;", D.season.min(), "-", D.season.max())
