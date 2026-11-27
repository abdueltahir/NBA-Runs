"""v2 dataset: repo team-stat rows joined to closing market lines.

Each row = one game, with home-minus-away stat differences, market spread /
total / de-vigged moneyline probability, rest, and sample size (GP).
Spread convention in OddsData: expected home margin (positive = home favored).
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIFF_STATS = ["W_PCT", "PLUS_MINUS", "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "FG3A", "FTA",
              "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK"]


def ml_to_prob(ml):
    ml = ml.astype(float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def load_odds():
    con = sqlite3.connect(ROOT / "Data" / "OddsData.sqlite")
    names = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    frames = []
    for n in names:
        df = pd.read_sql(f'select * from "{n}"', con)
        need = {"Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"}
        if need <= set(df.columns):
            frames.append(df[list(need)])
    odds = pd.concat(frames).drop_duplicates(["Date", "Home", "Away"])
    odds["Date"] = pd.to_datetime(odds.Date, errors="coerce").dt.normalize()
    return odds.dropna(subset=["Date"])


LEAKY_SEASONS = (2025, 2026)  # repo rows here include the predicted game's own result


def fix_same_day_leak(g):
    """For leaky seasons, replace each team's stats with its previous game's row.

    Verified: in 2013-2024, the change in W between a team's consecutive rows
    equals the PREVIOUS game's result 100% of the time (pre-game stats). In
    2025-2026 it equals the CURRENT game's result 100% of the time (leak).
    """
    g = g.copy()
    g["_season"] = np.where(g.Date.dt.month >= 8, g.Date.dt.year + 1, g.Date.dt.year)
    cols = DIFF_STATS + ["GP"]
    long = pd.concat([
        g[["Date", "_season", "TEAM_NAME"] + cols].assign(side="h", idx=g.index).rename(columns={"TEAM_NAME": "team"}),
        g[["Date", "_season", "TEAM_NAME.1"] + [c + ".1" for c in cols]]
        .rename(columns={"TEAM_NAME.1": "team", **{c + ".1": c for c in cols}}).assign(side="a", idx=g.index),
    ])
    for c in cols:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long = long.sort_values(["team", "_season", "Date"])
    leaky = long._season.isin(LEAKY_SEASONS)
    shifted = long.groupby(["team", "_season"])[cols].shift(1)
    long.loc[leaky, cols] = shifted.loc[leaky, cols]
    long.loc[leaky & long.GP.isna(), "GP"] = 0
    for side, suf in (("h", ""), ("a", ".1")):
        part = long[long.side == side].set_index("idx")
        for c in cols:
            g.loc[part.index, c + suf] = part[c]
    for c in DIFF_STATS:  # first game of season: no prior stats -> zero diff
        g[c] = pd.to_numeric(g[c], errors="coerce")
        g[c + ".1"] = pd.to_numeric(g[c + ".1"], errors="coerce")
        both_nan = g[c].isna() | g[c + ".1"].isna()
        g.loc[both_nan, [c, c + ".1"]] = 0.0
    return g.drop(columns="_season")


def build():
    con = sqlite3.connect(ROOT / "Data" / "dataset.sqlite")
    g = pd.read_sql('select * from "dataset_2012-26"', con)
    g["Date"] = pd.to_datetime(g.Date, errors="coerce").dt.normalize()
    g = fix_same_day_leak(g)
    odds = load_odds()
    m = g.merge(odds, left_on=["Date", "TEAM_NAME", "TEAM_NAME.1"],
                right_on=["Date", "Home", "Away"], how="inner")
    out = pd.DataFrame({"Date": m.Date, "Home": m.Home, "Away": m.Away})
    for s in DIFF_STATS:
        out[f"d_{s}"] = m[s].astype(float) - m[f"{s}.1"].astype(float)
    out["gp_min"] = np.minimum(m.GP.astype(float), m["GP.1"].astype(float))
    out["rest_home"] = m["Days-Rest-Home"].astype(float).clip(0, 5)
    out["rest_away"] = m["Days-Rest-Away"].astype(float).clip(0, 5)
    out["spread"] = pd.to_numeric(m.Spread.replace({"PK": 0, "pk": 0}), errors="coerce")
    out["total"] = pd.to_numeric(m.OU_y if "OU_y" in m else m.OU, errors="coerce")
    ph, pa = ml_to_prob(pd.to_numeric(m.ML_Home, errors="coerce")), ml_to_prob(pd.to_numeric(m.ML_Away, errors="coerce"))
    out["mkt_home_p"] = ph / (ph + pa)
    # pre-2023 tables store |spread|; sign it by the moneyline favorite
    out["spread"] = out.spread.abs() * np.sign(out.mkt_home_p - 0.5)
    out["margin"] = pd.to_numeric(m.Win_Margin, errors="coerce")
    out["points"] = pd.to_numeric(m.Points, errors="coerce")
    out["home_win"] = (out.margin > 0).astype(int)
    out["home_cover"] = np.where(out.margin + 0 == out.spread, np.nan, (out.margin > out.spread).astype(float))
    out["over"] = np.where(out.points == out.total, np.nan, (out.points > out.total).astype(float))
    out["season"] = np.where(out.Date.dt.month >= 8, out.Date.dt.year + 1, out.Date.dt.year)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["spread", "mkt_home_p", "total"])
    out = out[(out.mkt_home_p > 0.01) & (out.mkt_home_p < 0.99)]
    out.to_parquet(ROOT / "v2" / "v2_games.parquet")
    print(f"games joined: {len(out)} / {len(g)} | seasons {out.season.min()}-{out.season.max()}")
    print(out.groupby("season").size().to_string())
    # sanity: spread sign
    print("corr(spread, margin) =", round(out.spread.corr(out.margin), 3))
    return out


if __name__ == "__main__":
    build()
