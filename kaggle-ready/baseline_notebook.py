"""HLTV CS2 baseline notebook (Kaggle-ready).

Loads the four parquet datasets, reproduces all four LightGBM models, prints
metrics. Use as the starting point for new Kaggle experiments.

Usage:
  Kaggle:  set INPUT to /kaggle/input/<dataset-slug>/
  Local:   python baseline_notebook.py --input ./
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


NON_PRE   = {"match_id","date","team1_id","team2_id","team1_name","team2_name","y_team1_wins"}
NON_MAP   = {"match_id","mapstats_id","date","team1_id","team2_id","team1_name","team2_name",
             "y_team1_wins_map","map_score_t1","map_score_t2"}
NON_ROUND = {"match_id","mapstats_id","date","team1_id","team2_id","y_map_t1_wins"}


def _split(df, val_frac=0.15, test_frac=0.15, by_match=False):
    df = df.sort_values("date").reset_index(drop=True)
    if by_match:
        md = df.groupby("match_id")["date"].first().sort_values()
        n_m = len(md)
        c_v = int(n_m*(1-val_frac-test_frac)); c_t = int(n_m*(1-test_frac))
        return (df[df["match_id"].isin(set(md.iloc[:c_v].index))].copy(),
                df[df["match_id"].isin(set(md.iloc[c_v:c_t].index))].copy(),
                df[df["match_id"].isin(set(md.iloc[c_t:].index))].copy())
    n = len(df); n_t = int(n*test_frac); n_v = int(n*val_frac)
    return df.iloc[:n-n_v-n_t].copy(), df.iloc[n-n_v-n_t:n-n_t].copy(), df.iloc[n-n_t:].copy()


def _metrics(y, p, label):
    m = {
        "log_loss": float(log_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "baseline_logloss": float(log_loss(y, np.full_like(y, y.mean(), dtype=float))),
    }
    print(f"  {label}: log_loss={m['log_loss']:.4f}  auc={m['auc']:.4f}  brier={m['brier']:.4f}  (baseline={m['baseline_logloss']:.4f}, n={m['n']})")
    return m


def _to_xy(d, target_col, drop_cols, cat_cols):
    y = d[target_col].astype(int)
    X = d.drop(columns=[c for c in drop_cols if c in d.columns])
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if col in cat_cols: continue
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def _train(X_tr, y_tr, X_va, y_va, cats, lr=0.03, leaves=31, min_data=20, rounds=2000, stop=100):
    params = dict(objective="binary", metric=["binary_logloss","auc"],
                  learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_data,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    cats_in = [c for c in cats if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    return lgb.train(params, dtr, num_boost_round=rounds,
                     valid_sets=[dtr, dva], valid_names=["train","val"],
                     callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    args = ap.parse_args()
    inp = Path(args.input)
    out: dict = {}

    print("=" * 60); print("PRE-MATCH MODEL"); print("=" * 60)
    df = pd.read_parquet(inp / "prematch_features.parquet")
    df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
    tr, va, te = _split(df)
    cats = ["format","event_type"]
    X_tr, y_tr = _to_xy(tr, "y_team1_wins", NON_PRE, cats)
    X_va, y_va = _to_xy(va, "y_team1_wins", NON_PRE, cats)
    X_te, y_te = _to_xy(te, "y_team1_wins", NON_PRE, cats)
    m = _train(X_tr, y_tr, X_va, y_va, cats)
    out["prematch"] = _metrics(y_te, m.predict(X_te, num_iteration=m.best_iteration), "test")

    print("\n" + "=" * 60); print("PER-MAP MODEL"); print("=" * 60)
    df = pd.read_parquet(inp / "permap_features.parquet")
    df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
    tr, va, te = _split(df)
    cats = ["format","event_type","map_name","map_picked_by"]
    X_tr, y_tr = _to_xy(tr, "y_team1_wins_map", NON_MAP, cats)
    X_va, y_va = _to_xy(va, "y_team1_wins_map", NON_MAP, cats)
    X_te, y_te = _to_xy(te, "y_team1_wins_map", NON_MAP, cats)
    m = _train(X_tr, y_tr, X_va, y_va, cats, min_data=30)
    out["per_map"] = _metrics(y_te, m.predict(X_te, num_iteration=m.best_iteration), "test")

    print("\n" + "=" * 60); print("STACKED MODEL (best pre-match)"); print("=" * 60)
    df = pd.read_parquet(inp / "stacked_features.parquet")
    df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
    tr, va, te = _split(df)
    cats = ["format","event_type"]
    X_tr, y_tr = _to_xy(tr, "y_team1_wins", NON_PRE, cats)
    X_va, y_va = _to_xy(va, "y_team1_wins", NON_PRE, cats)
    X_te, y_te = _to_xy(te, "y_team1_wins", NON_PRE, cats)
    m = _train(X_tr, y_tr, X_va, y_va, cats)
    out["stacked"] = _metrics(y_te, m.predict(X_te, num_iteration=m.best_iteration), "test")

    print("\n" + "=" * 60); print("ROUND-LEVEL LIVE MODEL"); print("=" * 60)
    df = pd.read_parquet(inp / "round_features.parquet")
    tr, va, te = _split(df, by_match=True)
    cats = ["map_name","outcome","side_winner","last1","last2","last3"]
    X_tr, y_tr = _to_xy(tr, "y_map_t1_wins", NON_ROUND, cats)
    X_va, y_va = _to_xy(va, "y_map_t1_wins", NON_ROUND, cats)
    X_te, y_te = _to_xy(te, "y_map_t1_wins", NON_ROUND, cats)
    m = _train(X_tr, y_tr, X_va, y_va, cats,
               lr=0.05, leaves=63, min_data=200, rounds=4000, stop=150)
    p_te = m.predict(X_te, num_iteration=m.best_iteration)
    out["rounds"] = _metrics(y_te, p_te, "test")
    te_with_p = te.copy(); te_with_p["p"] = p_te
    by_round = {}
    for rn, sub in te_with_p.groupby("round_no"):
        if len(sub) >= 50 and sub["y_map_t1_wins"].nunique() == 2:
            by_round[int(rn)] = float(roc_auc_score(sub["y_map_t1_wins"], sub["p"]))
    print("  AUC by round:")
    for rn in (1, 5, 10, 15, 20, 24):
        if rn in by_round:
            print(f"    round {rn:>2}: AUC={by_round[rn]:.3f}")
    out["rounds_by_round_auc"] = by_round

    print("\n=== Summary ===")
    for k, v in out.items():
        if isinstance(v, dict) and "auc" in v:
            print(f"  {k:12} test_auc={v['auc']:.4f}  log_loss={v['log_loss']:.4f}")
    Path("baseline_results.json").write_text(json.dumps(out, indent=2))
    print("\nresults saved to baseline_results.json")


if __name__ == "__main__":
    main()
