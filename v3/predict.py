"""v3 nightly scorer.

usage: python v3/predict.py <teams.csv> <games.json> <out.csv>
teams.csv : TEAM,GP,W,L,PTS,OPP_PTS,FG_PCT,FG3_PCT,FT_PCT,FG3A,FTA,OREB,DREB,REB,AST[,TOV,STL,BLK]
games.json: [{away, home, home_spread (betting sign: - = home fav), total,
              rest_away, rest_home, judgment_adj (pts, + = home), notes}]
"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v3")); sys.path.insert(0, str(ROOT / "v2"))
import model as v3  # noqa: E402
import build_dataset as b2  # noqa: E402

SEASON = 2027  # 2026-27


def priors():
    con = sqlite3.connect(ROOT / "Data" / "dataset.sqlite")
    g = pd.read_sql('select * from "dataset_2012-26"', con)
    g["Date"] = pd.to_datetime(g.Date, errors="coerce").dt.normalize()
    g = b2.fix_same_day_leak(g)
    L = v3.team_long(g)
    lm = L.groupby("season")[v3.STATS].mean()
    return v3.prior_table(L), lm.loc[SEASON - 1]


def team_rows(path):
    t = pd.read_csv(path)
    t["W_PCT"] = t.W / (t.W + t.L)
    t["PLUS_MINUS"] = t.PTS - t.OPP_PTS
    for c in v3.STATS:
        if c not in t:
            t[c] = np.nan
    return t.set_index("TEAM")


def blended(team, t, prior, league_mean):
    gp = float(t.loc[team, "GP"])
    pr = prior.loc[(team, SEASON)] if (team, SEASON) in prior.index else pd.Series(0.0, index=v3.STATS)
    out = {}
    for s in v3.STATS:
        cur = t.loc[team, s]
        if pd.isna(cur):                       # stat not available this season -> prior only
            out[s] = v3.RHO * pr[s]
        else:
            out[s] = (gp * (cur - league_mean[s]) + v3.K * v3.RHO * pr[s]) / (gp + v3.K)
    return out


def main(teams_csv, games_json, out_csv):
    D = pd.read_parquet(ROOT / "v3" / "v3_games.parquet")
    model, sigma = v3.fit(D)
    prior, lm = priors()
    t = team_rows(teams_csv)
    games = json.loads(Path(games_json).read_text())
    rows = []
    for gm in games:
        bh, ba = blended(gm["home"], t, prior, lm), blended(gm["away"], t, prior, lm)
        x = np.array([[bh[s] - ba[s] for s in v3.STATS]])
        resid = float(model.predict(x)[0])
        spread = -gm["home_spread"]                      # expected home margin
        adj = float(gm.get("judgment_adj", 0.0))
        mu_model, p_win_model, p_cov_model = v3.distribution(spread, resid, 0.0, sigma)
        mu_final, p_win_final, p_cov_final = v3.distribution(spread, resid, adj, sigma)
        mkt_p = v3.norm_cdf(spread / sigma)
        edge = resid + adj
        tier = "play" if abs(edge) >= v3.PLAY_PTS else "lean" if abs(edge) >= v3.LEAN_PTS else "pass"
        side = gm["home"] if edge > 0 else gm["away"]
        gp_min = min(t.loc[gm["home"], "GP"], t.loc[gm["away"], "GP"])
        flags = [f"small sample: stats weighted {gp_min/(gp_min+v3.K):.0%} vs prior" if gp_min < 15 else ""]
        if gm["rest_home"] <= 1: flags.append("home B2B")
        if gm["rest_away"] <= 1: flags.append("away B2B")
        rows.append(dict(
            game=f"{gm['away']} @ {gm['home']}", away=gm["away"], home=gm["home"],
            home_spread=gm["home_spread"], total=gm["total"],
            proj_home=round((gm["total"] + mu_final) / 2, 1), proj_away=round((gm["total"] - mu_final) / 2, 1),
            mkt_home_p=round(mkt_p, 3), model_home_p=round(p_win_model, 3), final_home_p=round(p_win_final, 3),
            model_resid_pts=round(resid, 2), judgment_adj=adj, ats_side=side if tier != "pass" else "",
            ats_cover_p=round(max(p_cov_final, 1 - p_cov_final), 3), tier=tier,
            flags="; ".join(f for f in flags if f), notes=gm.get("notes", "")))
    R = pd.DataFrame(rows)
    R.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
    print(f"sigma {sigma:.2f}")
    print(R.drop(columns=["away", "home", "notes"]).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
