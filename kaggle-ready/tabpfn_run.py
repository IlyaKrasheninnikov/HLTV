"""TabPFN v2 across our 8 task targets.

TabPFN is a foundation-model classifier/regressor for small tabular data
(< ~10k rows). On Kaggle T4 it inference-trains in seconds per task. Should
match or beat tuned LightGBM on most of our tasks at this dataset size.

Reference: https://github.com/PriorLabs/TabPFN (TabPFN v2, ICLR 2025 Notable)

Usage:
    !pip install -q -r /kaggle/input/<slug>/requirements.txt
    !pip install -q tabpfn  # downloads ~700MB checkpoint on first run
    !python /kaggle/input/<slug>/tabpfn_run.py \
        --input /kaggle/input/<slug>/ --output /kaggle/working/out

Notes:
  - GPU is auto-used. CPU works but ~30x slower.
  - TabPFN max 500-2000 train rows in classic mode; we use the "many-class" mode
    with subsample if needed.
  - Categoricals are passed as ordinal integers (TabPFN handles them internally).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, accuracy_score,
)


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


def split_chrono(df, val_frac=0.0, test_frac=0.15):
    """TabPFN doesn't need a val set (in-context learning), so we use train+test."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df); nt = int(n*test_frac)
    return df.iloc[:n-nt].copy(), df.iloc[n-nt:].copy()


def _ordinal_encode(X_tr: pd.DataFrame, X_te: pd.DataFrame, cats):
    """Replace categoricals with int codes (train vocab; OOV -> -1).
    TabPFN treats numeric input; categoricals via integer codes is fine."""
    X_tr = X_tr.copy(); X_te = X_te.copy()
    for c in cats:
        if c not in X_tr.columns: continue
        vals = pd.Series(X_tr[c].astype(object).fillna("__nan__").unique()).reset_index(drop=True)
        vocab = {v: i for i, v in enumerate(vals)}
        X_tr[c] = X_tr[c].astype(object).fillna("__nan__").map(vocab).astype(np.int32)
        X_te[c] = X_te[c].astype(object).fillna("__nan__").map(lambda v: vocab.get(v, -1)).astype(np.int32)
    # coerce remaining object columns
    for col in X_tr.columns:
        if X_tr[col].dtype == object:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce")
    return X_tr.to_numpy(dtype=np.float32), X_te.to_numpy(dtype=np.float32)


def _to_xy(d, target, drop, cats, dropna=True):
    if dropna:
        d = d.dropna(subset=[target])
    y = d[target]
    X = d.drop(columns=[c for c in drop if c in d.columns])
    return X, y


def _maybe_subsample(X, y, max_rows: int, seed: int = 42):
    """TabPFN handles 1k-10k rows well. If we have more, subsample."""
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_rows, replace=False)
    return X[idx], y[idx]


def run_binary(name, df, target, drop, cats, out_dir, device, max_rows=8000):
    from tabpfn import TabPFNClassifier
    print(f"\n=== {name} (binary, TabPFN) ===")
    tr, te = split_chrono(df)
    X_tr_df, y_tr = _to_xy(tr, target, drop, cats)
    X_te_df, y_te = _to_xy(te, target, drop, cats)
    y_tr = y_tr.astype(int).to_numpy(); y_te = y_te.astype(int).to_numpy()
    X_tr, X_te = _ordinal_encode(X_tr_df, X_te_df, cats)
    X_tr, y_tr = _maybe_subsample(X_tr, y_tr, max_rows)
    print(f"  train rows: {len(X_tr)}  test rows: {len(X_te)}  feats: {X_tr.shape[1]}")
    t0 = time.time()
    clf = TabPFNClassifier(device=device, n_estimators=8, ignore_pretraining_limits=True)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    elapsed = time.time() - t0
    baseline = log_loss(y_te, np.full_like(y_te, y_te.mean(), dtype=float))
    out = {
        "task": name, "type": "binary",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "log_loss": float(log_loss(y_te, p)),
        "auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
        "baseline_logloss": float(baseline),
        "elapsed_s": float(elapsed),
    }
    print(f"  test: log_loss={out['log_loss']:.4f}  AUC={out['auc']:.4f}  Brier={out['brier']:.4f}  (baseline_logloss={baseline:.4f}, n={out['n_test']}, {elapsed:.1f}s)")
    with open(out_dir / f"tabpfn_{name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_multiclass(name, df, target, drop, cats, classes, out_dir, device, max_rows=8000):
    from tabpfn import TabPFNClassifier
    print(f"\n=== {name} (multiclass {len(classes)}, TabPFN) ===")
    tr, te = split_chrono(df)
    X_tr_df, y_tr = _to_xy(tr, target, drop, cats)
    X_te_df, y_te = _to_xy(te, target, drop, cats)
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_tr = y_tr.map(cls_idx).astype(int).to_numpy()
    y_te = y_te.map(cls_idx).astype(int).to_numpy()
    X_tr, X_te = _ordinal_encode(X_tr_df, X_te_df, cats)
    X_tr, y_tr = _maybe_subsample(X_tr, y_tr, max_rows)
    print(f"  train: {len(X_tr)}  test: {len(X_te)}  feats: {X_tr.shape[1]}")
    t0 = time.time()
    clf = TabPFNClassifier(device=device, n_estimators=8, ignore_pretraining_limits=True)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)
    elapsed = time.time() - t0
    cls_freqs = (np.bincount(y_tr, minlength=len(classes)) / len(y_tr)).astype(float)
    baseline = log_loss(y_te, np.tile(cls_freqs, (len(y_te), 1)))
    out = {
        "task": name, "type": "multiclass", "classes": classes,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "log_loss": float(log_loss(y_te, p)),
        "accuracy": float(accuracy_score(y_te, p.argmax(axis=1))),
        "baseline_logloss": float(baseline),
        "elapsed_s": float(elapsed),
    }
    print(f"  test: log_loss={out['log_loss']:.4f}  acc={out['accuracy']:.4f}  (baseline={baseline:.4f}, n={out['n_test']}, {elapsed:.1f}s)")
    with open(out_dir / f"tabpfn_{name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_regression(name, df, target, drop, cats, out_dir, device, max_rows=8000):
    from tabpfn import TabPFNRegressor
    print(f"\n=== {name} (regression, TabPFN) ===")
    tr, te = split_chrono(df)
    X_tr_df, y_tr = _to_xy(tr, target, drop, cats)
    X_te_df, y_te = _to_xy(te, target, drop, cats)
    y_tr = y_tr.astype(float).to_numpy(); y_te = y_te.astype(float).to_numpy()
    X_tr, X_te = _ordinal_encode(X_tr_df, X_te_df, cats)
    X_tr, y_tr = _maybe_subsample(X_tr, y_tr, max_rows)
    print(f"  train: {len(X_tr)}  test: {len(X_te)}  feats: {X_tr.shape[1]}")
    t0 = time.time()
    reg = TabPFNRegressor(device=device, n_estimators=8, ignore_pretraining_limits=True)
    reg.fit(X_tr, y_tr)
    p = reg.predict(X_te)
    elapsed = time.time() - t0
    mae = mean_absolute_error(y_te, p)
    rmse = mean_squared_error(y_te, p) ** 0.5
    baseline_mae = mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float))
    out = {
        "task": name, "type": "regression",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "mae": float(mae), "rmse": float(rmse),
        "baseline_mae": float(baseline_mae),
        "elapsed_s": float(elapsed),
    }
    print(f"  test: MAE={mae:.3f}  RMSE={rmse:.3f}  (baseline MAE={baseline_mae:.3f}, n={out['n_test']}, {elapsed:.1f}s)")
    with open(out_dir / f"tabpfn_{name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip", default="")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto)")
    ap.add_argument("--max-train-rows", type=int, default=8000,
                    help="cap training rows fed to TabPFN")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = args.device or "cpu"
    print(f"input: {inp}\noutput: {out_dir}\ndevice: {device}\nmax_train_rows: {args.max_train_rows}")

    results: dict = {}

    # match_winner — pre-match
    if "match_winner" not in skip:
        df = pd.read_parquet(inp / "prematch_features.parquet")
        df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
        results["match_winner"] = run_binary(
            "match_winner", df, "y_team1_wins", NON_PRE, CATS_PRE, out_dir, device,
            args.max_train_rows)

    # per-map tasks
    df_map = pd.read_parquet(inp / "permap_features.parquet")
    df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]

    if "map_winner_total" not in skip:
        results["map_winner_total"] = run_binary(
            "map_winner_total", df_map, "y_team1_wins_map", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)
    if "map_winner_regulation" not in skip:
        results["map_winner_regulation"] = run_multiclass(
            "map_winner_regulation", df_map, "y_regulation_winner",
            NON_MAP_ALL, CATS_MAP, ["t1","t2","tie"], out_dir, device, args.max_train_rows)
    if "pistol_r1" not in skip:
        results["pistol_r1"] = run_binary(
            "pistol_r1", df_map.dropna(subset=["y_pistol_r1_t1_wins"]),
            "y_pistol_r1_t1_wins", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)
    if "pistol_r13" not in skip:
        results["pistol_r13"] = run_binary(
            "pistol_r13", df_map.dropna(subset=["y_pistol_r13_t1_wins"]),
            "y_pistol_r13_t1_wins", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)
    if "t1_rounds" not in skip:
        results["t1_rounds"] = run_regression(
            "t1_rounds", df_map, "t1_rounds", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)
    if "t2_rounds" not in skip:
        results["t2_rounds"] = run_regression(
            "t2_rounds", df_map, "t2_rounds", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)
    if "total_rounds" not in skip:
        results["total_rounds"] = run_regression(
            "total_rounds", df_map, "total_rounds", NON_MAP_ALL, CATS_MAP,
            out_dir, device, args.max_train_rows)

    print("\n" + "=" * 60); print("TABPFN SUMMARY"); print("=" * 60)
    for k, v in results.items():
        if "auc" in v:
            print(f"  {k:30}  AUC={v['auc']:.4f}  log_loss={v['log_loss']:.4f}  t={v['elapsed_s']:.0f}s")
        elif "accuracy" in v:
            print(f"  {k:30}  acc={v['accuracy']:.4f}  log_loss={v['log_loss']:.4f}  t={v['elapsed_s']:.0f}s")
        else:
            print(f"  {k:30}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}  t={v['elapsed_s']:.0f}s")

    with open(out_dir / "tabpfn_all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved tabpfn_all_metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
