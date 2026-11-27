"""Top performer v2 — pick each team's highest PTS + 1.25*REB + 1.5*AST player, only when confident.

Projection = season per-game averages (players listed as out/doubtful removed).
Confidence = gap between the #1 and #2 projected players on that team.
Backtest 2016-24 (18,315 team-games, season-avg projection, players who actually played):
  all picks 57%; gap >= 10: 76%; gap >= 14: 83% (19% of team-games); gap >= 16: 86%.
  Early season (< 12 games) at gap >= 14: 79%.
Tiers: PICK gap >= 14 (counts toward the 80% target), lean 10-14 (tracked separately), else skip.
Questionable players: if the pick is questionable -> skip; if a questionable player is the runner-up,
the pick is graded on the gap WITH him playing (conservative) and the note says what changes if he sits.

players.csv columns: game,team,player,gp,pts,reb,ast,status   (status: active | Q | out)
boxes.json: [{"team","player","pts","reb","ast"}] for every player who played (final box score)

Usage:
  python v3/top_perf.py log <date> daily/<date>/players.csv daily/<date>/rows.json
  python v3/top_perf.py grade daily/<date>/rows.json daily/<date>/boxes.json
"""
import json, os, sys
import pandas as pd

PICK, LEAN = 14.0, 10.0
def score(p, r, a): return p + 1.25 * r + 1.5 * a
def prob(gap): return 0.86 if gap >= 16 else 0.80 if gap >= 14 else 0.72 if gap >= 10 else 0.55

def log(date, players_csv, rows_json):
    d = pd.read_csv(players_csv)
    d = d[d.status.str.lower() != 'out'].copy()
    d['proj'] = score(d.pts, d.reb, d.ast)
    rows = json.load(open(rows_json)) if os.path.exists(rows_json) else []
    rows = [r for r in rows if r.get('model') != 'top_perf_v2']
    for (game, team), t in d.groupby(['game', 'team']):
        t = t.sort_values('proj', ascending=False).reset_index(drop=True)
        p1, p2 = t.iloc[0], t.iloc[1]
        gap = p1.proj - p2.proj
        note = f"proj {p1.proj:.1f} vs {p2.player} {p2.proj:.1f} (gap {gap:.1f})"
        if str(p1.status).upper() == 'Q':
            tier = 'skip'; note += '; pick is questionable'
        else:
            tier = 'PICK' if gap >= PICK else 'lean' if gap >= LEAN else 'skip'
            if str(p2.status).upper() == 'Q' and len(t) > 2:
                alt = p1.proj - t[t.player != p2.player].iloc[1].proj
                note += f"; {p2.player} questionable — gap {alt:.1f} if he sits"
        if tier == 'skip':
            continue
        rows.append(dict(id=f"{date}|top_performer|top_perf_v2|{game}|{p1.player} ({team})", date=date,
                         model='top_perf_v2', market='top_performer', game=game, pick=f"{p1.player} ({team})",
                         prob=prob(gap), market_prob=None, outcome='pending', result='', grade='',
                         note=f"{tier}: {note}", tier=tier))
    json.dump(rows, open(rows_json, 'w'), indent=1)
    for r in rows:
        if r['model'] == 'top_perf_v2': print(r['tier'], r['game'], r['pick'], r['note'])

def grade(rows_json, boxes_json):
    rows = json.load(open(rows_json)); box = pd.DataFrame(json.load(open(boxes_json)))
    box['s'] = score(box.pts, box.reb, box.ast)
    for r in rows:
        if r.get('model') != 'top_perf_v2' or r['outcome'] != 'pending': continue
        name, team = r['pick'].rsplit(' (', 1); team = team.rstrip(')')
        t = box[box.team == team]
        if t.empty: continue
        top = t.loc[t.s.idxmax()]; me = t[t.player == name]
        if me.empty:
            r['outcome'] = 'ungraded'; r['result'] = f"{name} did not play; top {top.player} {top.s:.2f}"; continue
        m = me.iloc[0]
        r['outcome'] = 'hit' if m.s >= top.s - 1e-9 else 'miss'
        r['result'] = f"{name} {int(m.pts)}/{int(m.reb)}/{int(m.ast)} = {m.s:.2f}; top {top.player} {top.s:.2f}"
        if r['outcome'] == 'miss' and top.s - m.s <= 3: r['grade'] = 'close (within 3)'
    json.dump(rows, open(rows_json, 'w'), indent=1)
    print('graded', rows_json)

if __name__ == '__main__':
    {'log': log, 'grade': grade}[sys.argv[1]](*sys.argv[2:])
