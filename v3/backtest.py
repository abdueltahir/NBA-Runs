"""Walk-forward backtest of v3 vs the market (train on seasons < S, test on S)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v3"))
import model as v3  # noqa: E402

D = pd.read_parquet(ROOT / "v3" / "v3_games.parquet")
rows, bets = [], []
for S in (2022, 2023, 2024, 2025, 2026):
    tr, te = D[D.season < S], D[D.season == S]
    m, sig = v3.fit(tr)
    r = m.predict(te[v3.FEATS].values)
    mu = te.spread.values + r
    from scipy.stats import norm
    p_win = norm.cdf(mu / sig)
    p_cov = norm.cdf(r / sig)
    y = te.home_win.values
    early = te.gp_min.values < 15
    res = dict(season=S, n=len(te),
               acc_mkt=((te.mkt_home_p > .5) == y).mean(), acc_v3=((p_win > .5) == y).mean(),
               ll_mkt=log_loss(y, te.mkt_home_p.clip(.01, .99)), ll_v3=log_loss(y, p_win.clip(.01, .99)),
               ll_mkt_early=log_loss(y[early], te.mkt_home_p.clip(.01, .99)[early]),
               ll_v3_early=log_loss(y[early], p_win.clip(.01, .99)[early]),
               mae_mkt=np.abs(te.margin - te.spread).mean(), mae_v3=np.abs(te.margin - mu).mean(),
               sigma=sig)
    rows.append(res)
    c = te.home_cover.values
    ok = ~np.isnan(c)
    for thr in (0.0, 0.03, 0.05, 0.08):
        pick = ok & (np.abs(p_cov - .5) > thr)
        won = ((p_cov[pick] > .5) == (c[pick] == 1))
        bets.append(dict(season=S, thr=thr, bets=int(pick.sum()), wins=int(won.sum())))
W = pd.DataFrame(rows)
B = pd.DataFrame(bets).groupby("thr")[["bets", "wins"]].sum()
B["hit"] = B.wins / B.bets
B["roi_at_-110"] = (B.wins * 100 / 110 - (B.bets - B.wins)) / B.bets
pd.set_option("display.width", 220)
print(W.round(4).to_string(index=False))
print("\nmean:", W.drop(columns=["season", "n"]).mean().round(4).to_dict())
print("\nATS (break-even 52.4%):\n", B.round(4).to_string())
W.to_csv(ROOT / "v3" / "backtest_v3.csv", index=False)
B.to_csv(ROOT / "v3" / "backtest_v3_ats.csv")
