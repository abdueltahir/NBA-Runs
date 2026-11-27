"""Walk-forward backtest: train on all seasons before S, test on S.

Compares, for game winner / ATS cover / total over:
  stats_only : repo-style features (team stat diffs, rest, GP) without market
  market     : the closing line alone (de-vigged ML prob; 50% for ATS/OU)
  v2         : market + early-season-shrunk stat diffs + rest + GP
Betting ROI assumes -110 and only bets when model confidence > threshold.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
D = pd.read_parquet(ROOT / "v2" / "v2_games.parquet")
DIFFS = [c for c in D.columns if c.startswith("d_")]
K = 12.0  # shrinkage: weight = GP / (GP + K); at 5 games ~29% weight, at 30 games ~71%
for c in DIFFS:
    D[c + "_s"] = D[c] * D.gp_min / (D.gp_min + K)
SHRUNK = [c + "_s" for c in DIFFS]
D["mkt_logit"] = np.log(D.mkt_home_p / (1 - D.mkt_home_p))
D["rest_diff"] = D.rest_home - D.rest_away
D["b2b_home"] = (D.rest_home <= 1).astype(int)
D["b2b_away"] = (D.rest_away <= 1).astype(int)
CTX = ["rest_diff", "b2b_home", "b2b_away", "gp_min"]
FEATS = {
    "stats_only": DIFFS + ["rest_home", "rest_away", "gp_min"],
    "v2": ["mkt_logit", "spread", "total"] + SHRUNK + CTX,
}


def fit_predict(model_kind, Xtr, ytr, Xte):
    if model_kind == "lr":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                          tree_method="hist", verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def roi(p, y, thr):
    pick = np.abs(p - 0.5) > thr
    if pick.sum() == 0:
        return 0, np.nan, np.nan
    won = ((p[pick] > 0.5) == (y[pick] == 1))
    return int(pick.sum()), won.mean(), (won * (100 / 110) - (~won)).mean()


def run(test_seasons=(2022, 2023, 2024, 2025, 2026)):
    rows, bets = [], []
    for S in test_seasons:
        tr, te = D[D.season < S], D[D.season == S]
        # --- game winner ---
        y = te.home_win.values
        res = {"season": S, "n": len(te),
               "win_acc_market": ((te.mkt_home_p > .5) == y).mean(),
               "win_ll_market": log_loss(y, te.mkt_home_p.clip(.01, .99))}
        for name, kind in [("stats_only", "xgb"), ("v2", "lr")]:
            p = fit_predict(kind, tr[FEATS[name]].values, tr.home_win.values, te[FEATS[name]].values)
            res[f"win_acc_{name}"] = ((p > .5) == y).mean()
            res[f"win_ll_{name}"] = log_loss(y, p)
        # --- ATS and totals with v2 ---
        for tgt in ["home_cover", "over"]:
            trv, tev = tr.dropna(subset=[tgt]), te.dropna(subset=[tgt])
            for kind in ["lr", "xgb"]:
                p = fit_predict(kind, trv[FEATS["v2"]].values, trv[tgt].values, tev[FEATS["v2"]].values)
                for thr in [0.0, 0.03, 0.05]:
                    n, hit, r = roi(p, tev[tgt].values, thr)
                    bets.append(dict(season=S, target=tgt, model=kind, thr=thr, bets=n, hit=hit, roi=r))
        rows.append(res)
    W = pd.DataFrame(rows)
    B = pd.DataFrame(bets)
    pd.set_option("display.width", 200)
    print("=== GAME WINNER (accuracy / log loss; lower log loss = better) ===")
    print(W.round(3).to_string(index=False))
    print("\nmean:", W.drop(columns=["season", "n"]).mean().round(4).to_dict())
    print("\n=== ATS / TOTALS betting at -110 (break-even hit rate 52.4%) ===")
    agg = B.groupby(["target", "model", "thr"]).apply(
        lambda g: pd.Series({"bets": g.bets.sum(),
                             "hit": np.average(g.hit.fillna(0), weights=g.bets) if g.bets.sum() else np.nan,
                             "roi": np.average(g.roi.fillna(0), weights=g.bets) if g.bets.sum() else np.nan,
                             "seasons_profitable": int((g.roi > 0).sum())}), include_groups=False)
    print(agg.round(3).to_string())
    W.to_csv(ROOT / "v2" / "backtest_winner.csv", index=False)
    B.to_csv(ROOT / "v2" / "backtest_bets.csv", index=False)


if __name__ == "__main__":
    run()
