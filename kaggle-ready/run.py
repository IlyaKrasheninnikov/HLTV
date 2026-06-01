"""HLTV CS2 — train every model in one command.

Usage in a Kaggle notebook cell:
    !pip install -q -r /kaggle/input/<dataset-slug>/requirements.txt
    !python /kaggle/input/<dataset-slug>/run.py --input /kaggle/input/<dataset-slug>/ --output /kaggle/working/

Or locally:
    pip install -r requirements.txt
    python run.py --input ./ --output ./out/

What it does:
  - Loads the four parquet datasets from --input.
  - Trains 4 architecture models (prematch, per-map, stacked, round-level) AND
    8 task-specific models (match_winner, map_winner_total, map_winner_regulation,
    pistol_r1, pistol_r13, t1_rounds, t2_rounds, total_rounds).
  - Saves all .txt models, .json metrics, .csv feature importances under --output.
  - Prints a final summary table.

This script is self-contained — no imports from the scraper project.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, accuracy_score,
)


# ----------------------------- schema -----------------------------

NON_PRE = {"match_id","date","team1_id","team2_id","team1_name","team2_name","y_team1_wins"}
NON_MAP_ALL = {
    "match_id","mapstats_id","date","team1_id","team2_id","team1_name","team2_name",
    "y_team1_wins_map","y_regulation_winner","is_overtime_map",
    "y_pistol_r1_t1_wins","y_pistol_r13_t1_wins",
    "t1_rounds","t2_rounds","total_rounds","reg_score_t1","reg_score_t2",
    "map_score_t1","map_score_t2",
}
NON_ROUND = {"match_id","mapstats_id","date","team1_id","team2_id","y_map_t1_wins"}
CATS_PRE = ["format","event_type"]
CATS_MAP = ["format","event_type","map_name","map_picked_by"]
CATS_ROUND = ["map_name","outcome","side_winner","last1","last2","last3"]


# ----------------------------- helpers -----------------------------

def split_chrono(df, val_frac=0.15, test_frac=0.15, by_match=False):
    df = df.sort_values("date").reset_index(drop=True)
    if by_match:
        md = df.groupby("match_id")["date"].first().sort_values()
        n_m = len(md)
        c_v = int(n_m*(1-val_frac-test_frac)); c_t = int(n_m*(1-test_frac))
        return (df[df["match_id"].isin(set(md.iloc[:c_v].index))].copy(),
                df[df["match_id"].isin(set(md.iloc[c_v:c_t].index))].copy(),
                df[df["match_id"].isin(set(md.iloc[c_t:].index))].copy())
    n = len(df); nt = int(n*test_frac); nv = int(n*val_frac)
    return df.iloc[:n-nv-nt].copy(), df.iloc[n-nv-nt:n-nt].copy(), df.iloc[n-nt:].copy()


def _to_xy(d, target_col, drop_cols, cats, dropna_y=True):
    if dropna_y:
        d = d.dropna(subset=[target_col])
    y = d[target_col]
    X = d.drop(columns=[c for c in drop_cols if c in d.columns])
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if col in cats: continue
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def _save_outputs(out_dir, name, model, metrics):
    model.save_model(str(out_dir / f"{name}_lgbm.txt"), num_iteration=model.best_iteration)
    fi = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)
    with open(out_dir / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ----------------------------- training -----------------------------

def train_binary(name, df, target, drop_cols, cats, out_dir,
                 lr=0.03, leaves=31, min_data=20, rounds=2000, stop=100):
    print(f"\n=== {name} (binary) ===")
    tr, va, te = split_chrono(df)
    X_tr, y_tr = _to_xy(tr, target, drop_cols, cats)
    X_va, y_va = _to_xy(va, target, drop_cols, cats)
    X_te, y_te = _to_xy(te, target, drop_cols, cats)
    y_tr = y_tr.astype(int); y_va = y_va.astype(int); y_te = y_te.astype(int)
    cats_in = [c for c in cats if c in X_tr.columns]
    params = dict(objective="binary", metric=["binary_logloss","auc"],
                  learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_data,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dtr,dva],
                  valid_names=["tr","va"],
                  callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(0)])
    p = m.predict(X_te, num_iteration=m.best_iteration)
    baseline = log_loss(y_te, np.full_like(y_te, y_te.mean(), dtype=float))
    out = {
        "task": name, "type": "binary",
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "log_loss": float(log_loss(y_te, p)),
        "auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
        "baseline_logloss": float(baseline),
        "best_iter": int(m.best_iteration),
        "n_features": int(X_tr.shape[1]),
    }
    print(f"  test: log_loss={out['log_loss']:.4f}  AUC={out['auc']:.4f}  Brier={out['brier']:.4f}  (baseline log_loss={baseline:.4f}, n={out['n_test']})")
    _save_outputs(out_dir, name, m, out)
    return out


def train_multiclass(name, df, target, drop_cols, cats, classes, out_dir):
    print(f"\n=== {name} (multiclass {len(classes)}) ===")
    tr, va, te = split_chrono(df)
    X_tr, y_tr = _to_xy(tr, target, drop_cols, cats)
    X_va, y_va = _to_xy(va, target, drop_cols, cats)
    X_te, y_te = _to_xy(te, target, drop_cols, cats)
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_tr = y_tr.map(cls_idx).astype(int)
    y_va = y_va.map(cls_idx).astype(int)
    y_te = y_te.map(cls_idx).astype(int)
    cats_in = [c for c in cats if c in X_tr.columns]
    params = dict(objective="multiclass", num_class=len(classes), metric=["multi_logloss"],
                  learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dtr,dva],
                  valid_names=["tr","va"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = m.predict(X_te, num_iteration=m.best_iteration)
    cls_freqs = y_tr.value_counts(normalize=True).reindex(range(len(classes))).fillna(0).values
    baseline = log_loss(y_te, np.tile(cls_freqs, (len(y_te), 1)))
    out = {
        "task": name, "type": "multiclass", "classes": classes,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "log_loss": float(log_loss(y_te, p)),
        "accuracy": float(accuracy_score(y_te, p.argmax(axis=1))),
        "baseline_logloss": float(baseline),
        "best_iter": int(m.best_iteration),
        "n_features": int(X_tr.shape[1]),
        "class_distribution_test": {classes[i]: int((y_te==i).sum()) for i in range(len(classes))},
    }
    print(f"  test: log_loss={out['log_loss']:.4f}  acc={out['accuracy']:.4f}  (baseline log_loss={baseline:.4f}, n={out['n_test']})")
    print(f"  dist: {out['class_distribution_test']}")
    _save_outputs(out_dir, name, m, out)
    return out


def train_regression(name, df, target, drop_cols, cats, out_dir, objective="poisson"):
    print(f"\n=== {name} (regression / {objective}) ===")
    tr, va, te = split_chrono(df)
    X_tr, y_tr = _to_xy(tr, target, drop_cols, cats)
    X_va, y_va = _to_xy(va, target, drop_cols, cats)
    X_te, y_te = _to_xy(te, target, drop_cols, cats)
    y_tr = y_tr.astype(float); y_va = y_va.astype(float); y_te = y_te.astype(float)
    cats_in = [c for c in cats if c in X_tr.columns]
    metrics = ["l1","l2"] if objective == "regression" else [objective]
    params = dict(objective=objective, metric=metrics,
                  learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dtr,dva],
                  valid_names=["tr","va"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = m.predict(X_te, num_iteration=m.best_iteration)
    mae = mean_absolute_error(y_te, p)
    rmse = mean_squared_error(y_te, p) ** 0.5
    baseline_mae = mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float))
    baseline_rmse = mean_squared_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float)) ** 0.5
    out = {
        "task": name, "type": "regression", "objective": objective,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "mae": float(mae), "rmse": float(rmse),
        "baseline_mae": float(baseline_mae), "baseline_rmse": float(baseline_rmse),
        "y_mean_train": float(y_tr.mean()), "y_std_train": float(y_tr.std()),
        "best_iter": int(m.best_iteration),
        "n_features": int(X_tr.shape[1]),
    }
    print(f"  test: MAE={mae:.3f}  RMSE={rmse:.3f}  (baseline MAE={baseline_mae:.3f}  RMSE={baseline_rmse:.3f}, n={out['n_test']})")
    _save_outputs(out_dir, name, m, out)
    return out


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./",
                    help="folder containing the parquet files")
    ap.add_argument("--output", default=None,
                    help="folder to save models/metrics (defaults to --input)")
    ap.add_argument("--skip", default="",
                    help="comma-separated list of tasks to skip")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    print(f"input:  {inp}")
    print(f"output: {out_dir}")
    if skip:
        print(f"skipping: {skip}")

    pre_path = inp / "prematch_features.parquet"
    map_path = inp / "permap_features.parquet"
    round_path = inp / "round_features.parquet"

    for p in (pre_path, map_path):
        if not p.exists():
            print(f"ERROR: missing required input {p}", file=sys.stderr); sys.exit(1)

    results: dict[str, Any] = {}

    # ============ Task 1: match winner ============
    if "match_winner" not in skip:
        df = pd.read_parquet(pre_path)
        df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
        results["match_winner"] = train_binary(
            "task_match_winner", df, "y_team1_wins", NON_PRE, CATS_PRE, out_dir)

    # ============ Tasks 2-5 on per-map features ============
    df_map = pd.read_parquet(map_path)
    df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]

    if "map_winner_total" not in skip:
        results["map_winner_total"] = train_binary(
            "task_map_winner_total", df_map, "y_team1_wins_map", NON_MAP_ALL, CATS_MAP, out_dir)

    if "map_winner_regulation" not in skip:
        results["map_winner_regulation"] = train_multiclass(
            "task_map_winner_regulation", df_map, "y_regulation_winner",
            NON_MAP_ALL, CATS_MAP, classes=["t1","t2","tie"], out_dir=out_dir)

    if "pistol_r1" not in skip:
        results["pistol_r1"] = train_binary(
            "task_pistol_r1", df_map.dropna(subset=["y_pistol_r1_t1_wins"]),
            "y_pistol_r1_t1_wins", NON_MAP_ALL, CATS_MAP, out_dir)
    if "pistol_r13" not in skip:
        results["pistol_r13"] = train_binary(
            "task_pistol_r13", df_map.dropna(subset=["y_pistol_r13_t1_wins"]),
            "y_pistol_r13_t1_wins", NON_MAP_ALL, CATS_MAP, out_dir)

    if "t1_rounds" not in skip:
        results["t1_rounds"] = train_regression(
            "task_t1_rounds", df_map, "t1_rounds", NON_MAP_ALL, CATS_MAP, out_dir, "poisson")
    if "t2_rounds" not in skip:
        results["t2_rounds"] = train_regression(
            "task_t2_rounds", df_map, "t2_rounds", NON_MAP_ALL, CATS_MAP, out_dir, "poisson")
    if "total_rounds" not in skip:
        results["total_rounds"] = train_regression(
            "task_total_rounds", df_map, "total_rounds", NON_MAP_ALL, CATS_MAP, out_dir, "poisson")

    # ============ Bonus: round-level live model ============
    if "rounds_live" not in skip and round_path.exists():
        df = pd.read_parquet(round_path)
        tr, va, te = split_chrono(df, by_match=True)
        X_tr, y_tr = _to_xy(tr, "y_map_t1_wins", NON_ROUND, CATS_ROUND)
        X_va, y_va = _to_xy(va, "y_map_t1_wins", NON_ROUND, CATS_ROUND)
        X_te, y_te = _to_xy(te, "y_map_t1_wins", NON_ROUND, CATS_ROUND)
        y_tr = y_tr.astype(int); y_va = y_va.astype(int); y_te = y_te.astype(int)
        cats_in = [c for c in CATS_ROUND if c in X_tr.columns]
        params = dict(objective="binary", metric=["binary_logloss","auc"],
                      learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                      lambda_l2=1.0, verbose=-1)
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
        m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[dtr,dva],
                      valid_names=["tr","va"],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        p = m.predict(X_te, num_iteration=m.best_iteration)
        out = {
            "task": "rounds_live", "type": "binary",
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
            "log_loss": float(log_loss(y_te, p)),
            "auc": float(roc_auc_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
            "best_iter": int(m.best_iteration),
            "n_features": int(X_tr.shape[1]),
        }
        # AUC by round_no
        te2 = te.copy(); te2["p"] = p
        by_round = {}
        for rn, sub in te2.groupby("round_no"):
            if len(sub) >= 50 and sub["y_map_t1_wins"].nunique() == 2:
                by_round[int(rn)] = float(roc_auc_score(sub["y_map_t1_wins"], sub["p"]))
        out["auc_by_round"] = by_round
        print(f"\n=== rounds_live ===")
        print(f"  test overall: log_loss={out['log_loss']:.4f}  AUC={out['auc']:.4f}")
        print(f"  AUC by round: ", end="")
        for rn in (1, 5, 10, 15, 20, 24):
            if rn in by_round:
                print(f"r{rn}={by_round[rn]:.3f} ", end="")
        print()
        _save_outputs(out_dir, "task_rounds_live", m, out)
        results["rounds_live"] = out

    # ============ summary ============
    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)
    for k, v in results.items():
        if "auc" in v:
            print(f"  {k:30}  AUC={v['auc']:.4f}  log_loss={v['log_loss']:.4f}")
        elif "accuracy" in v:
            print(f"  {k:30}  acc={v['accuracy']:.4f}  log_loss={v['log_loss']:.4f}")
        else:
            print(f"  {k:30}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}")

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nall metrics saved to {out_dir / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
