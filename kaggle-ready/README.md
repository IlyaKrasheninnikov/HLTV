# HLTV CS2 — Kaggle bundle

Self-contained dataset + training script. Drop the folder into a Kaggle dataset, attach it to a notebook, and run a single cell.

## Quickstart (Kaggle)

```python
!pip install -q -r /kaggle/input/hltv-cs2/requirements.txt
!python /kaggle/input/hltv-cs2/run.py --input /kaggle/input/hltv-cs2/ --output /kaggle/working/
```

Replace `hltv-cs2` with whatever slug you used when creating the Kaggle dataset.

## Quickstart (local)

```bash
pip install -r requirements.txt
python run.py --input ./ --output ./out/
```

## What `run.py` does

Loads three parquet files, trains a LightGBM model for every task below, prints metrics, saves models + per-task feature importances + a single `all_metrics.json`.

You can skip specific tasks: `python run.py --skip pistol_r1,total_rounds`

## Tasks & current test metrics

| Task | Metric | Result | Baseline |
|---|---|---|---|
| 1. Match winner | AUC / log-loss | **0.714 / 0.605** | 0.675 |
| 2. Map winner (after OT) | AUC / log-loss | **0.661 / 0.643** | 0.680 |
| 3. Map winner regulation (3-class: t1 / t2 / tie) | acc / log-loss | **0.548 / 0.934** | 0.965 |
| 4a. Pistol R1 | AUC / log-loss | **0.530 / 0.689** | 0.689 |
| 4b. Pistol R13 | AUC / log-loss | **0.524 / 0.692** | 0.693 |
| 5a. Team 1 rounds | MAE / RMSE | **2.75 / 3.53** | 2.86 / 3.61 |
| 5b. Team 2 rounds | MAE / RMSE | **3.24 / 4.00** | 3.43 / 4.18 |
| 5c. Total rounds | MAE / RMSE | **3.46 / 4.82** | 3.47 / 4.83 |
| 6. Live (in-game, optional) | AUC overall / AUC at round 24 | **0.842 / 0.92** | — |

## Datasets

| File | Rows | Cols | Granularity |
|---|---|---|---|
| `prematch_features.parquet` | 3,245 | 63 | one row per match (pre-match info only) |
| `permap_features.parquet` | 7,120 | 120 | one row per played map (used for tasks 2-5) |
| `round_features.parquet` | ~155k | 30 | one row per played round (live in-game model) |

All splits are chronological 70/15/15. The round dataset splits by `match_id` so all rounds of a match stay in the same fold.

## Targets in `permap_features.parquet`

| Column | Type | Meaning |
|---|---|---|
| `y_team1_wins_map` | 0/1 | team1 won this map (after any OT) |
| `y_regulation_winner` | t1/t2/tie | who led after the 24 regulation rounds; `tie` = 12-12, went to OT |
| `is_overtime_map` | 0/1 | did this map go to OT |
| `y_pistol_r1_t1_wins` | 0/1 or NaN | team1 won round 1 (first pistol) |
| `y_pistol_r13_t1_wins` | 0/1 or NaN | team1 won round 13 (second pistol) |
| `t1_rounds`, `t2_rounds`, `total_rounds` | int | round counts |
| `reg_score_t1`, `reg_score_t2` | int | rounds won by each team in regulation only |

## Source

Scraped from hltv.org. 3,288 CS2 matches with ≥1 star (Majors, IEM, BLAST, ESL Pro League, etc.). Date range 2023-10-05 → 2026-05-31.

Each match contributes ~2-5 played maps, each map contributes ~16-30 rounds.

## What to try next on Kaggle

1. **Optuna sweep** on `task_match_winner` — easy +0.005-0.015 AUC.
2. **Multi-task head** sharing a trunk across all tasks — should help the noisy tasks (pistols, total rounds).
3. **Transformer over rounds** for the live model — current LightGBM round model is at AUC 0.84; sequence context might push toward 0.87.
4. **Per-player pistol-W%** from demo parsing — addresses the main weakness of the pistol task.
5. **Betting backtest** if you obtain historical odds (HLTV doesn't carry them — try Pinnacle, bo3.gg).
