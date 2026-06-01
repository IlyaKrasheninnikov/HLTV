# HLTV CS2 — Kaggle bundle

Self-contained dataset + training script. Drop the folder into a Kaggle dataset, attach it to a notebook, and run a single cell.

## Quickstart (Kaggle)

```python
# 1) LightGBM baseline (8 task-specific models + optional live model)
!pip install -q -r /kaggle/input/hltv-cs2/requirements.txt
!python /kaggle/input/hltv-cs2/run.py --input /kaggle/input/hltv-cs2/ --output /kaggle/working/

# 2) Multi-task neural net (one shared trunk + 8 heads). Auto-uses GPU if available.
#    torch is preinstalled on Kaggle; if you're somewhere else use:
#    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!python /kaggle/input/hltv-cs2/mtl.py --input /kaggle/input/hltv-cs2/ --output /kaggle/working/
```

Replace `hltv-cs2` with whatever slug you used when creating the Kaggle dataset.
**Pick a forward slash output path** like `/kaggle/working/out`, not `out\`.

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

## Modern-architecture experiments (Kaggle GPU)

Two scripts that try modern architectures aimed at beating the LightGBM + meta-ensemble pipeline:

```python
# TabPFN v2 — foundation model for small tabular. Often beats GBDT at this scale.
!pip install -q tabpfn   # ~700MB download on first run
!python /kaggle/input/<slug>/tabpfn_run.py \
    --input /kaggle/input/<slug>/ --output /kaggle/working/out

# Round-Transformer — 4-layer encoder over the round timeline. Aims to beat
# the round-LightGBM on the live in-game task (current best AUC 0.842).
!python /kaggle/input/<slug>/transformer_rounds.py \
    --input /kaggle/input/<slug>/ --output /kaggle/working/out \
    --epochs 30 --batch-size 64
```

Honest expectations:
- **TabPFN**: at our row counts (3k pre-match, 7k per-map) it has a real chance of beating tuned LightGBM by 0.01-0.03 AUC. If it does, drop it in as the new task-1 model. If it doesn't, LightGBM stays the champion.
- **Round-Transformer**: the round-LightGBM hits 0.842 AUC overall but is dominated by `score_diff`. A transformer attends across the round history, which *should* capture momentum and clutch streaks better. Target: 0.86+. If it lands at 0.85, marginal. If 0.87+, real win.

These scripts compete with the LightGBM/ensemble pipeline; results go side-by-side in their own `*_metrics.json` so you can compare directly.

## Best-result recipe (use this!)

Run these in order. Final results live in `/kaggle/working/out/`.

```python
# install
!pip install -q -r /kaggle/input/<slug>/requirements.txt
!pip install -q optuna

# 1) baseline (~1 min)
!python /kaggle/input/<slug>/run.py --input /kaggle/input/<slug>/ --output /kaggle/working/out

# 2) Optuna sweep per task. Deep-tune the two tasks with proven headroom
#    (match_winner, t1_rounds) at 200 trials each, the rest at 80.
!python /kaggle/input/<slug>/tune.py --input /kaggle/input/<slug>/ --output /kaggle/working/out \
    --n-trials 80 \
    --n-trials-task match_winner=200 \
    --n-trials-task t1_rounds=200

# 3) Meta-ensemble: OOF tuned-LGB preds -> small GPU NN with 7 heads.
#    Auto-detects the tuned params and uses them.
!python /kaggle/input/<slug>/ensemble.py --input /kaggle/input/<slug>/ --output /kaggle/working/out

# 4) Predict for an existing match in the dataset (uses the tuned + ensemble models)
!python /kaggle/input/<slug>/predict.py --input /kaggle/input/<slug>/ \
    --model-dir /kaggle/working/out --match-id 2393895
```

The `ensemble.py` output includes a **side-by-side test comparison of meta-NN vs the standalone tuned LightGBM** on the same rows so you can verify the lift is real.

`tune.py` supports `--only task1,task2` to run a single subset, and `--skip` to exclude one. `--n-trials-task name=N` is repeatable for per-task overrides.

## What's "best so far" (measured on chronological test split)

| Task | Best model | Test metric |
|---|---|---|
| 1. Match winner (series) | Tuned LGB (`task_match_winner_lgbm_tuned.txt`) | **AUC 0.730 / log-loss 0.595** |
| 2. Map winner (total, after OT) | Meta-NN | **AUC 0.671** |
| 3. Map winner regulation (3-class) | Meta-NN | **acc 0.568 / log-loss 0.918** |
| 4a. Pistol R1 | Meta-NN | **AUC 0.567** |
| 4b. Pistol R13 | Meta-NN | **AUC 0.550** |
| 5a. Team 1 rounds | Tuned LGB | **MAE 2.70 / RMSE 3.53** |
| 5b. Team 2 rounds | Meta-NN | MAE 3.27 / RMSE 4.02 |
| 5c. Total rounds | Meta-NN | **MAE 3.42 / RMSE 4.81** |
| Bonus. Live in-game | `rounds_lgbm.txt` | **AUC 0.842 overall, 0.93 at R24** |

## Multi-task neural net (`mtl.py`)

Single PyTorch model with a shared MLP trunk and 8 task-specific heads, jointly trained on `permap_features.parquet`. NaN labels (pistols missing on some maps) are masked per-batch so each head only sees real examples. Losses: BCE for binary heads, cross-entropy for the 3-class regulation winner, Poisson NLL for the round-count regressions. Auxiliary "is_overtime_map" head helps the trunk shape useful representations even when it's not the target.

What to expect vs the LightGBM baselines:
- **Pistols, totals, regulation**: where MTL most often helps. Shared trunk regularizes the noisy tasks.
- **Map winner total**: should match or slightly beat LightGBM (~0.66).
- **Match winner is NOT in `mtl.py`** — it lives in `prematch_features.parquet`, a different table. Keep using `run.py` for that one task.

## What to try next on Kaggle

1. **Optuna sweep** on `task_match_winner` — easy +0.005-0.015 AUC.
2. **Tune MTL loss weights** — currently `{map_total:1.0, map_reg:0.7, pistol1:1.3, pistol13:1.3, ot:0.3, t1r/t2r/totalr:0.4}`. Optuna these per-head weights.
3. **Transformer over rounds** for the live model — current LightGBM round model is at AUC 0.84; sequence context might push toward 0.87.
4. **Per-player pistol-W%** from demo parsing — addresses the main weakness of the pistol task.
5. **Betting backtest** if you obtain historical odds (HLTV doesn't carry them — try Pinnacle, bo3.gg).
