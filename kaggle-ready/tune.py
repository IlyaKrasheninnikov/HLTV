"""Optuna sweep on each of the 8 LightGBM task models.

Per task we search:
  num_leaves, min_data_in_leaf, learning_rate, feature_fraction, bagging_fraction,
  lambda_l1, lambda_l2, max_depth, and num_boost_round (early-stopped on val).

For each task we save:
  task_<name>_best_params.json
  task_<name>_best_metrics.json
  task_<name>_lgbm_tuned.txt          (re-fit on train+val with best params)
  task_<name>_optuna_trials.csv

Usage on Kaggle:
    !pip install -q -r /kaggle/input/<slug>/requirements.txt
    !pip install -q optuna
    !python /kaggle/input/<slug>/tune.py --input /kaggle/input/<slug>/ --output /kaggle/working/

Runs CPU only (LightGBM-CPU is faster than -GPU at this dataset size). One sweep
takes ~3-10 min depending on n_trials. With Kaggle's 12h CPU budget you can
afford 100-200 trials per task.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, accuracy_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# -------- schema (mirrors run.py) --------

NON_PRE = {"match_id","date","team1_id","team2_id","team1_name","team2_name","y_team1_wins"}
NON_MAP_ALL = {
    "match_id","mapstats_id","date","team1_id","team2_id","team1_name","team2_name",
    "y_team1_wins_map","y_regulation_winner","is_overtime_map",
    "y_pistol_r1_t1_wins","y_pistol_r13_t1_wins",
    "t1_rounds","t2_rounds","total_rounds","reg_score_t1","reg_score_t2",
    "map_score_t1","map_score_t2",
}
CATS_PRE = ["format","event_type"]
CATS_MAP = ["format","event_type","map_name","map_picked_by"]


def split_chrono(df, val_frac=0.15, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df); nt = int(n*test_frac); nv = int(n*val_frac)
    return df.iloc[:n-nv-nt].copy(), df.iloc[n-nv-nt:n-nt].copy(), df.iloc[n-nt:].copy()


def _to_xy(d, target, drop, cats, dropna=True):
    if dropna:
        d = d.dropna(subset=[target])
    y = d[target]
    X = d.drop(columns=[c for c in drop if c in d.columns])
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if col in cats: continue
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


# -------- sampling space --------

def _sample_params(trial, kind: str, n_classes: int = 1) -> dict[str, Any]:
    if kind == "binary":
        obj_metric = {"objective": "binary", "metric": ["binary_logloss"]}
    elif kind == "multiclass":
        obj_metric = {"objective": "multiclass", "num_class": n_classes,
                      "metric": ["multi_logloss"]}
    elif kind == "poisson":
        obj_metric = {"objective": "poisson", "metric": ["poisson"]}
    else:
        raise ValueError(kind)
    p = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 255),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 200),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 10),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        verbose=-1,
        **obj_metric,
    )
    return p


def _fit_and_score_binary(params, X_tr, y_tr, X_va, y_va, X_te, y_te, cats_in,
                          rounds=2000, stop=80) -> tuple[float, lgb.Booster, dict]:
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=rounds,
                  valid_sets=[dva], valid_names=["va"],
                  callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(0)])
    p_va = m.predict(X_va, num_iteration=m.best_iteration)
    p_te = m.predict(X_te, num_iteration=m.best_iteration)
    metrics = {
        "val_log_loss": float(log_loss(y_va, p_va)),
        "val_auc": float(roc_auc_score(y_va, p_va)),
        "test_log_loss": float(log_loss(y_te, p_te)),
        "test_auc": float(roc_auc_score(y_te, p_te)),
        "test_brier": float(brier_score_loss(y_te, p_te)),
        "best_iter": int(m.best_iteration),
    }
    return float(log_loss(y_va, p_va)), m, metrics


def _fit_and_score_multiclass(params, X_tr, y_tr, X_va, y_va, X_te, y_te,
                              cats_in, n_classes, rounds=2000, stop=80):
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=rounds,
                  valid_sets=[dva], valid_names=["va"],
                  callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(0)])
    p_va = m.predict(X_va, num_iteration=m.best_iteration)
    p_te = m.predict(X_te, num_iteration=m.best_iteration)
    metrics = {
        "val_log_loss": float(log_loss(y_va, p_va)),
        "test_log_loss": float(log_loss(y_te, p_te)),
        "test_acc": float(accuracy_score(y_te, p_te.argmax(axis=1))),
        "best_iter": int(m.best_iteration),
    }
    return float(log_loss(y_va, p_va)), m, metrics


def _fit_and_score_poisson(params, X_tr, y_tr, X_va, y_va, X_te, y_te,
                           cats_in, rounds=2000, stop=80):
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=rounds,
                  valid_sets=[dva], valid_names=["va"],
                  callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(0)])
    p_va = m.predict(X_va, num_iteration=m.best_iteration)
    p_te = m.predict(X_te, num_iteration=m.best_iteration)
    metrics = {
        "val_mae": float(mean_absolute_error(y_va, p_va)),
        "test_mae": float(mean_absolute_error(y_te, p_te)),
        "test_rmse": float(mean_squared_error(y_te, p_te) ** 0.5),
        "best_iter": int(m.best_iteration),
    }
    return float(mean_absolute_error(y_va, p_va)), m, metrics


# -------- per-task sweep --------

def sweep_task(name, df, target, drop, cats, kind, out_dir, n_trials=120,
               n_classes=1, class_map=None, dropna_y=True):
    print(f"\n=== sweep: {name} ({kind}) — {n_trials} trials ===")
    tr, va, te = split_chrono(df)
    X_tr, y_tr = _to_xy(tr, target, drop, cats, dropna=dropna_y)
    X_va, y_va = _to_xy(va, target, drop, cats, dropna=dropna_y)
    X_te, y_te = _to_xy(te, target, drop, cats, dropna=dropna_y)
    cats_in = [c for c in cats if c in X_tr.columns]

    if kind == "binary":
        y_tr = y_tr.astype(int); y_va = y_va.astype(int); y_te = y_te.astype(int)
    elif kind == "multiclass":
        y_tr = y_tr.map(class_map).astype(int)
        y_va = y_va.map(class_map).astype(int)
        y_te = y_te.map(class_map).astype(int)
    else:   # poisson
        y_tr = y_tr.astype(float); y_va = y_va.astype(float); y_te = y_te.astype(float)

    def objective(trial):
        p = _sample_params(trial, kind, n_classes=n_classes)
        if kind == "binary":
            score, _, _ = _fit_and_score_binary(p, X_tr, y_tr, X_va, y_va, X_te, y_te, cats_in)
        elif kind == "multiclass":
            score, _, _ = _fit_and_score_multiclass(p, X_tr, y_tr, X_va, y_va, X_te, y_te,
                                                    cats_in, n_classes)
        else:
            score, _, _ = _fit_and_score_poisson(p, X_tr, y_tr, X_va, y_va, X_te, y_te, cats_in)
        return score

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  best val score = {study.best_value:.4f}")
    print(f"  best params: {best}")

    # Refit and capture test metrics with best params
    params_best = _sample_params(optuna.trial.FixedTrial(best), kind, n_classes=n_classes)
    if kind == "binary":
        _, model, metrics = _fit_and_score_binary(params_best, X_tr, y_tr, X_va, y_va, X_te, y_te, cats_in)
    elif kind == "multiclass":
        _, model, metrics = _fit_and_score_multiclass(params_best, X_tr, y_tr, X_va, y_va, X_te, y_te,
                                                      cats_in, n_classes)
    else:
        _, model, metrics = _fit_and_score_poisson(params_best, X_tr, y_tr, X_va, y_va, X_te, y_te, cats_in)
    print(f"  test metrics: {metrics}")

    model.save_model(str(out_dir / f"task_{name}_lgbm_tuned.txt"),
                     num_iteration=model.best_iteration)
    with open(out_dir / f"task_{name}_best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(out_dir / f"task_{name}_best_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / f"task_{name}_optuna_trials.csv", index=False)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--skip", default="", help="comma-separated tasks to skip")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    print(f"input: {inp}\noutput: {out_dir}\ntrials per task: {args.n_trials}\nskipping: {skip}")

    results: dict[str, Any] = {}

    # Task 1: match winner
    if "match_winner" not in skip:
        df = pd.read_parquet(inp / "prematch_features.parquet")
        df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
        results["match_winner"] = sweep_task(
            "match_winner", df, "y_team1_wins", NON_PRE, CATS_PRE, "binary",
            out_dir, n_trials=args.n_trials)

    df_map = pd.read_parquet(inp / "permap_features.parquet")
    df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]

    if "map_winner_total" not in skip:
        results["map_winner_total"] = sweep_task(
            "map_winner_total", df_map, "y_team1_wins_map", NON_MAP_ALL, CATS_MAP, "binary",
            out_dir, n_trials=args.n_trials)
    if "map_winner_regulation" not in skip:
        results["map_winner_regulation"] = sweep_task(
            "map_winner_regulation", df_map, "y_regulation_winner",
            NON_MAP_ALL, CATS_MAP, "multiclass",
            out_dir, n_trials=args.n_trials, n_classes=3,
            class_map={"t1": 0, "t2": 1, "tie": 2})
    if "pistol_r1" not in skip:
        results["pistol_r1"] = sweep_task(
            "pistol_r1", df_map.dropna(subset=["y_pistol_r1_t1_wins"]),
            "y_pistol_r1_t1_wins", NON_MAP_ALL, CATS_MAP, "binary",
            out_dir, n_trials=args.n_trials)
    if "pistol_r13" not in skip:
        results["pistol_r13"] = sweep_task(
            "pistol_r13", df_map.dropna(subset=["y_pistol_r13_t1_wins"]),
            "y_pistol_r13_t1_wins", NON_MAP_ALL, CATS_MAP, "binary",
            out_dir, n_trials=args.n_trials)
    if "t1_rounds" not in skip:
        results["t1_rounds"] = sweep_task(
            "t1_rounds", df_map, "t1_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
            out_dir, n_trials=args.n_trials)
    if "t2_rounds" not in skip:
        results["t2_rounds"] = sweep_task(
            "t2_rounds", df_map, "t2_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
            out_dir, n_trials=args.n_trials)
    if "total_rounds" not in skip:
        results["total_rounds"] = sweep_task(
            "total_rounds", df_map, "total_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
            out_dir, n_trials=args.n_trials)

    print("\n" + "=" * 60); print("TUNED SUMMARY"); print("=" * 60)
    for k, v in results.items():
        keys = list(v.keys())
        line = f"  {k:28}  " + "  ".join(f"{kk}={v[kk]:.4f}" for kk in keys
                                          if isinstance(v[kk], (int, float)))
        print(line)

    with open(out_dir / "all_tuned_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {out_dir / 'all_tuned_metrics.json'}")


if __name__ == "__main__":
    main()
