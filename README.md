# NBA-Runs

Daily NBA prediction model and scorecard.

- `v3/` – game model (winners + ATS leans vs the market), top performer v2 (`top_perf.py`), ledger tools. **`v3/RUNBOOK.md` is the daily procedure.**
- `v2/` – dataset builder (leak-fixed features) and the older v2 model.
- `daily/<date>/` – per-day inputs and ledger rows; `daily/ledger.json` is the assembled ledger.

Data comes from [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting), cloned beside this repo and symlinked as `Data/` (not committed).

Dashboard: https://claude.ai/artifact/MDwaJ4GBcUoJgQWNLZvRRu
