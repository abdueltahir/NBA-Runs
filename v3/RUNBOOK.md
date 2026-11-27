# NBA model — daily runbook (v3)

Goal: every day, (1) grade yesterday's pending picks against final scores, (2) score today's slate with v3,
(3) apply judgment (injuries/rest/news the data can't see), (4) log everything, (5) refresh the dashboard.

Dashboard: https://claude.ai/artifact/MDwaJ4GBcUoJgQWNLZvRRu  (Dashboard type; write with ArtifactData)
Store layout: code/v3_bundle {text} = this bundle; ledger_days/<YYYY-MM-DD> {rows:[...]} = source of truth;
datasets/ledger + datasets/backtest = file datasets the page reads (re-upload asset, update url + updated).

## Restore (fresh session)
1. git clone --depth 1 https://github.com/abdueltahir/nba-runs nba-runs   (our code + daily/ history)
2. git clone --depth 1 https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting nba-ml   (read-only data source)
3. cd nba-runs && ln -s ../nba-ml/Data Data        (Data is gitignored; never commit or push to kyleskom/...)
4. pip install --break-system-packages -q xgboost scikit-learn pandas pyarrow scipy
5. python v2/build_dataset.py && python v3/model.py        (seconds to ~2 min; builds leak-fixed data + prior table)
6. ArtifactData list ledger_days (out_dir) -> compare with daily/<date>/rows.json; the store is the source of truth,
   so overwrite any repo file that differs. Note each doc's version (writes need if_version).
(Fallback if the repo is unreachable: ArtifactData get code/v3_bundle and split on '=== FILE: <path> ===' lines.)

## Commit (end of every run)
git add daily/ && git commit -m "Daily run <date>" && git -c push.negotiate=true push origin main

## Grade
For each date with pending rows: get final scores (WebSearch "NBA scores <date>", ESPN/CBS recaps).
Write daily/<date>/results.json [{away, home, away_pts, home_pts}] using full team names
(ledger game tags are ABBR, e.g. CHA@DET). Run: python v3/ledger.py grade daily/<date>/rows.json daily/<date>/results.json
Older hand-logged top_perf_v1/v1.1 rows are already graded. First-basket picks are RETIRED (2026-11-13; archived in
archive/first_basket) — never make or log first-basket picks or parlays.
Top performer v2 rows: fetch each game's ESPN box score, write daily/<date>/boxes.json [{team (ABBR), player, pts, reb, ast}]
for both teams, run: python v3/top_perf.py grade daily/<date>/rows.json daily/<date>/boxes.json. Misses within 3 = close.
Grade notes for misses: say whether process or luck (e.g., "line moved 3 pts after injury news = process").

## Score today
1. Lines: WebSearch "NBA odds <date>" -> fanduel.com/research/nba-predictions-odds-<m>-<d>-<yyyy> works.
   FanDuel lists "X vs. Y" as HOME vs AWAY; confirm home team with a second source (Covers ATS picks page).
2. Team stats (all 30): landofbasketball.com/team_rankings_by_year/2027_most_points_pg_rs.htm (shooting),
   ..._most_rebounds_pg_rs.htm, ..._most_assists_pg_rs.htm, champsorchumps.us/summary/nba/2027 (W, L, PPG, allowed).
   Search for each URL first (WebFetch needs it surfaced by WebSearch). Check GP == W+L.
   teams.csv columns: TEAM,GP,W,L,PTS,OPP_PTS,FG_PCT,FG3_PCT,FT_PCT,FG3A,FTA,OREB,DREB,REB,AST
3. Injuries/rest: search "NBA injury report <date>", team news. Back-to-back = played yesterday.
4. games.json: [{away, home, home_spread (- = home fav), total, tip, rest_away, rest_home,
   judgment_adj (pts, + = home; only for news NOT already in the line), notes, final_ats (team name or "pass"), final_reason}]
5. python v3/predict.py daily/<date>/teams.csv daily/<date>/games.json daily/<date>/predictions.csv
6. python v3/ledger.py log <date> daily/<date>/predictions.csv daily/<date>/games.json
7. Top performer v2 (after step 6, which rewrites rows.json): for every team on the slate fetch
   espn.com/nba/team/stats/_/name/<abbr> (per-game GP, PTS, REB, AST for the top ~6 players). Mark status from the
   injury report: out/doubtful -> out, questionable -> Q. Write daily/<date>/players.csv
   (game,team,player,gp,pts,reb,ast,status; team = ledger ABBR) and run
   python v3/top_perf.py log <date> daily/<date>/players.csv daily/<date>/rows.json
   PICK = gap >= 14 between the #1 and #2 projected (backtest 83%, target 80%); lean = 10-14 (tracked, not counted).
   Report PICKs with the runner-up and any questionable player that changes the gap.

## Judgment rules (Claude has final say; log every override with a reason)
- Market line is the anchor. v3 only flags ATS leans: |residual| >= 1.0 lean, >= 1.5 small play.
  Backtest 2017-26: 53.6% ATS on 1,233 lean games (+2.3% ROI at -110), not statistically proven. Totals: no model edge.
- Early season (GP < 15) the residual is mostly last season's team. Discount leans for teams with big roster
  turnover or a star now out (stats built with him); upgrade only for news that broke after the line.
- Never bet a lean against fresh injury news the line already moved on. Small plays only; most nights = pass.

## Publish
1. python v3/ledger.py assemble daily/ledger.json daily/*/rows.json
2. ArtifactData batch: set ledger_days/<date> {rows} for each changed date (pin if_version).
3. Artifact publish asset:true url=<dashboard> file_path=daily/ledger.json -> new /_blob url;
   ArtifactData update datasets/ledger {source:{kind:file,url,name:"ledger.json"}, updated:{at,by:"Claude"}}.
4. Check: ArtifactData get data/users/me doc check (re-read while "checking", max 5).
5. Send the user a short summary: yesterday's record (model vs market vs judgment, top performer v2 PICK record
   vs the 80% target), today's leans/plays and top-performer PICKs.
