"""v2 nightly scorer — the 'skeleton' layer.

Outputs per game:
  market_home_p : win prob implied by the closing spread (the anchor)
  v2_home_p     : market + shrunk stat diffs + rest (backtest: ~= market)
  ats_lean      : XGB home-cover prob; only flagged if |p-0.5| > 0.05
                  (backtest 2022-26: 53.1% on 1,151 flagged bets, +1.3% ROI — weak)
  flags         : judgment checklist items for the human/Claude override
Totals are NOT modeled for betting: every backtested variant lost money.

Input CSV columns (one row per team): TEAM, GP, W, L, PTS, OPP_PTS, FG_PCT,
FG3_PCT, FT_PCT, FG3A, FTA, OREB, DREB, REB, AST  [TOV, STL, BLK optional]
Games JSON: list of {away, home, home_spread, total, rest_away, rest_home}
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
import backtest as bt  # reuses feature engineering + K shrinkage

SPREAD_SD = 12.5


def spread_to_p(home_spread_line):
    """home_spread_line uses betting convention: negative = home favored."""
    exp_margin = -home_spread_line
    return 0.5 * (1 + math.erf(exp_margin / SPREAD_SD / math.sqrt(2)))


def team_features(t):
    t = t.copy()
    t["W_PCT"] = t.W / (t.W + t.L).replace(0, np.nan)
    t["PLUS_MINUS"] = t.PTS - t.OPP_PTS
    for c in ["TOV", "STL", "BLK"]:
        if c not in t:
            t[c] = np.nan  # unknown -> zero diff
    return t.set_index("TEAM")


def build_rows(teams, games):
    rows = []
    for gm in games:
        h, a = teams.loc[gm["home"]], teams.loc[gm["away"]]
        r = {"Home": gm["home"], "Away": gm["away"]}
        for s in bt.DIFFS:
            stat = s[2:]
            d = h[stat] - a[stat]
            r[s] = 0.0 if pd.isna(d) else float(d)
        r["gp_min"] = float(min(h.GP, a.GP))
        r["rest_home"], r["rest_away"] = gm["rest_home"], gm["rest_away"]
        r["spread"] = -gm["home_spread"]           # dataset convention: + = home favored
        r["total"] = gm["total"]
        r["mkt_home_p"] = spread_to_p(gm["home_spread"])
        rows.append(r)
    X = pd.DataFrame(rows)
    for c in bt.DIFFS:
        X[c + "_s"] = X[c] * X.gp_min / (X.gp_min + bt.K)
    X["mkt_logit"] = np.log(X.mkt_home_p / (1 - X.mkt_home_p))
    X["rest_diff"] = X.rest_home - X.rest_away
    X["b2b_home"] = (X.rest_home <= 1).astype(int)
    X["b2b_away"] = (X.rest_away <= 1).astype(int)
    return X


def train_final():
    D = bt.D
    F = bt.FEATS["v2"]
    win = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    win.fit(D[F].values, D.home_win.values)
    Dc = D.dropna(subset=["home_cover"])
    ats = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                            tree_method="hist", verbosity=0)
    ats.fit(Dc[F].values, Dc.home_cover.values)
    return win, ats


def checklist(X):
    flags = []
    for _, r in X.iterrows():
        f = []
        if r.gp_min < 10:
            f.append(f"small sample (GP {int(r.gp_min)}): stats shrunk to {r.gp_min/(r.gp_min+bt.K):.0%}")
        if r.b2b_home:
            f.append("home on back-to-back")
        if r.b2b_away:
            f.append("away on back-to-back")
        f.append("CHECK: injuries/rest/minutes limits not in model")
        flags.append("; ".join(f))
    return flags


def main(teams_csv, games_json, out_csv):
    teams = team_features(pd.read_csv(teams_csv))
    games = json.loads(Path(games_json).read_text())
    X = build_rows(teams, games)
    win, ats = train_final()
    F = bt.FEATS["v2"]
    X["v2_home_p"] = win.predict_proba(X[F].values)[:, 1]
    X["ats_home_cover_p"] = ats.predict_proba(X[F].values)[:, 1]
    X["ats_lean"] = np.where((X.ats_home_cover_p - 0.5).abs() > 0.05,
                             np.where(X.ats_home_cover_p > 0.5, X.Home, X.Away), "none")
    X["flags"] = checklist(X)
    X["home_spread"] = [g["home_spread"] for g in games]
    cols = ["Away", "Home", "home_spread", "total", "mkt_home_p", "v2_home_p",
            "ats_home_cover_p", "ats_lean", "gp_min", "flags"]
    X[cols].to_csv(out_csv, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print(X[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
