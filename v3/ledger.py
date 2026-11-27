"""Prediction ledger: log picks before tip, grade them after the final.

  python v3/ledger.py log      <date> <predictions.csv> <games.json>   -> daily/<date>/rows.json
  python v3/ledger.py grade    <rows.json> <results.json>              -> grades rows.json in place
  python v3/ledger.py assemble <out ledger.json> <rows.json> [...]     -> one ledger for the dashboard

results.json: [{"away": "...", "home": "...", "away_pts": 110, "home_pts": 105}]
Row schema (matches the dashboard): id, date, model, market, game, pick, prob,
market_prob, outcome (pending|hit|miss|push|close|ungraded), result, grade, note,
plus line (home spread) and edge_pts for ATS rows.
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABBR = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS"}


def tag(away, home):
    return f"{ABBR[away]}@{ABBR[home]}"


def row(date, model, market, game, pick, prob, mkt=None, note="", **extra):
    return dict(id=f"{date}|{market}|{model}|{game}|{pick}", date=date, model=model, market=market, game=game,
                pick=pick, prob=None if prob is None else round(float(prob), 3),
                market_prob=None if mkt is None else round(float(mkt), 3),
                outcome="pending", result="", grade="", note=note, **extra)


def log(date, pred_csv, games_json):
    P = pd.read_csv(pred_csv)
    G = {(g["away"], g["home"]): g for g in json.loads(Path(games_json).read_text())}
    rows = []
    for _, r in P.iterrows():
        g = G[(r.away, r.home)]
        gm = tag(r.away, r.home)
        fav_mkt = r.home if r.mkt_home_p >= .5 else r.away
        rows.append(row(date, "market", "winner", gm, ABBR[fav_mkt], max(r.mkt_home_p, 1 - r.mkt_home_p),
                        note=f"spread {r.home_spread:+g} (home)"))
        fav_v3 = r.home if r.model_home_p >= .5 else r.away
        rows.append(row(date, "v3", "winner", gm, ABBR[fav_v3], max(r.model_home_p, 1 - r.model_home_p),
                        max(r.mkt_home_p, 1 - r.mkt_home_p) if fav_v3 == fav_mkt else min(r.mkt_home_p, 1 - r.mkt_home_p),
                        note=f"proj {ABBR[r.away]} {r.proj_away}–{ABBR[r.home]} {r.proj_home}"))
        if r.tier != "pass":
            side_line = r.home_spread if r.ats_side == r.home else -r.home_spread
            rows.append(row(date, "v3", "ats", gm, f"{ABBR[r.ats_side]} {side_line:+g}", r.ats_cover_p, 0.5,
                            note=f"{r.tier}: residual {r.model_resid_pts:+.2f} pts", line=r.home_spread,
                            edge_pts=r.model_resid_pts))
        fin = g.get("final_ats", "pass")
        if fin and fin != "pass":
            side_line = r.home_spread if fin == r.home else -r.home_spread
            rows.append(row(date, "judgment", "ats", gm, f"{ABBR[fin]} {side_line:+g}", r.ats_cover_p, 0.5,
                            note=g.get("final_reason", ""), line=r.home_spread))
        elif r.tier != "pass":   # model flagged it and judgment passed: log the pass so it's graded too
            rows.append(row(date, "judgment", "ats_pass", gm, "pass", None, None,
                            note=g.get("final_reason", "passed on model lean")))
    out = ROOT / "daily" / date / "rows.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"logged {len(rows)} rows -> {out}")


def grade(rows_json, results_json):
    rows = json.loads(Path(rows_json).read_text())
    res = {tag(x["away"], x["home"]): x for x in json.loads(Path(results_json).read_text())}
    n = 0
    for r in rows:
        if r["outcome"] not in ("pending", "ungraded") or r["game"] not in res:
            continue
        x = res[r["game"]]
        a, h = r["game"].split("@")
        margin = x["home_pts"] - x["away_pts"]
        r["result"] = f"{a} {x['away_pts']}–{h} {x['home_pts']}"
        if r["market"] == "winner":
            won = h if margin > 0 else a
            r["outcome"] = "hit" if r["pick"] == won else "miss"
        elif r["market"] in ("ats", "ats_pass"):
            line = r.get("line")
            if line is None:  # ats_pass rows don't carry the line; borrow it from the v3 ats row for this game
                line = next((o.get("line") for o in rows if o["game"] == r["game"] and o.get("line") is not None), None)
            if line is None:
                continue
            cover = margin + line                       # >0 home covers
            if r["market"] == "ats_pass":
                r["outcome"] = "ungraded"; r["grade"] = f"home ATS margin {cover:+g}"; n += 1; continue
            team = r["pick"].split()[0]
            side = cover if team == h else -cover
            r["outcome"] = "push" if side == 0 else "hit" if side > 0 else "miss"
            r["grade"] = f"covered by {side:+g}"
        n += 1
    Path(rows_json).write_text(json.dumps(rows, indent=1))
    print(f"graded {n} rows in {rows_json}")


def assemble(out, *paths):
    allrows = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()).get("rows", []) if p.endswith("ledger.json") else json.loads(Path(p).read_text()):
            allrows[r["id"]] = r
    rows = sorted(allrows.values(), key=lambda r: (r["date"], r["game"], r["model"]))
    Path(out).write_text(json.dumps({"rows": rows}, indent=1))
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    {"log": log, "grade": grade, "assemble": assemble}[sys.argv[1]](*sys.argv[2:])
