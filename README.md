[![token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/c/token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/mmr9759/token-radar-01-v9/rank.svg)](https://make.fun/c/token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/mmr9759/token-radar-01-v9/reward.svg)](https://make.fun/c/token-radar-01)

# Token Launch Radar — radar-9

Entry for make.fun `token-radar-01` ($1,325, top 10, deadline 14 Dec 2026),
built as `@mmr9759`.

## What it is

`main.py` (Python 3, standard library only, deterministic) reads the sealed
snapshot at `/input/snapshot.json` and writes a ranked board of the **198
clean pre-token protocols** to `/output/board.json`, plus the dashboard,
three screenshots and `dossier.json`.

## The model

`log(mcap / TVL)` fitted on the 128 protocols whose token already trades:

- per-category multiple (median, shrunk K=10 toward the global median
  0.2136×), log-multiples winsorised at p5–p95;
- a **half-strength Huber size slope** (−0.431 per log10 TVL), fitted on
  residuals *after* the category effect, so a protocol 10× the average
  size keeps about 65% of its multiple — the steep unanchored curve that
  sank radar-6 is not repeated;
- small fitted growth and age terms, shrunk by half and capped at ±0.15
  log points (7d growth enters slightly negative: in the comps a hot
  week usually means the TVL denominator just jumped).

## Blind tests (computed by the program, shown on the board)

| Test | Model | Flat global median |
|---|---|---|
| Leave-one-out, geometric mean | **5.492×** | 6.092× |
| Leave-one-out, median | **4.062×** | 5.070× |
| Walk-forward, train earliest 50% | **5.660×** | 6.280× |
| Walk-forward, train earliest 60% | **6.092×** | 7.094× |

Bootstrap: 200 refits (seed 20261118) give every row a P10–P90 implied
value and a rank range.

## Verification (local)

- Two runs with the real input write byte-identical output.
- All seven gate checks pass (board, dashboard, screenshots, dossier).
- `dossier.json` is byte-identical to the kit vendor library's
  `buildDossier` output on the same files.
- The dashboard's farm-plan generator produces exactly the same steps
  as `board.json` for all 198 rows.

## Run it

```sh
python3 main.py --input <snapshot.json> --output <dir>
# or: MAKEFUN_INPUT=… MAKEFUN_OUTPUT_DIR=… python3 main.py
```
