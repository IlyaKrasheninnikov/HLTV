"""Predict-from-features: takes a pre-built feature row and emits all 8 task predictions.

Two modes:

  1) `--match-id <id>`: look up a single match in prematch_features.parquet AND
     permap_features.parquet and print predictions for that match's series winner
     and each played map's per-map tasks.

  2) `--from-csv path.csv`: bulk-predict from any CSV that has the same columns as
     the corresponding feature parquets. Useful for batch inference.

Models used (in priority order):
  - task_match_winner_lgbm_tuned.txt  (or _lgbm.txt if untuned)  -> match winner
  - ensemble_model.pt + tuned LGB OOF models for the per-map tasks

Note: to predict a brand-new match that ISN'T already in the feature parquet,
you must first extract its features using the scraper project (hltv/features.py
+ hltv/features_map.py). This script does NOT scrape — it operates on
pre-extracted feature rows.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


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


def _lgb_predict(model_path: Path, df: pd.DataFrame, drop, cats) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_path))
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if col in cats: continue
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return booster.predict(X)


def _maybe_tuned(model_dir: Path, base: str) -> Path:
    tuned = model_dir / f"{base}_lgbm_tuned.txt"
    if tuned.exists(): return tuned
    untuned = model_dir / f"{base}_lgbm.txt"
    if untuned.exists(): return untuned
    raise FileNotFoundError(f"no model found for {base} in {model_dir}")


def predict_match(df_pre: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Series winner probabilities."""
    path = _maybe_tuned(model_dir, "task_match_winner")
    p = _lgb_predict(path, df_pre, NON_PRE, CATS_PRE)
    out = df_pre[["match_id","date","team1_id","team2_id","team1_name","team2_name"]].copy()
    out["p_team1_wins_series"] = p
    return out


def predict_map(df_map: pd.DataFrame, model_dir: Path,
                use_ensemble: bool = True) -> pd.DataFrame:
    """Per-map task predictions. If ensemble_model.pt exists, prefer it;
    else fall back to standalone tuned LGB heads."""
    out = df_map[["match_id","mapstats_id","date","map_name",
                  "team1_id","team2_id","team1_name","team2_name"]].copy()

    # Standalone LGB heads (always compute — also the inputs to ensemble)
    head_models = [
        ("task_map_winner_total",      "p_team1_wins_map_total",   "binary"),
        ("task_map_winner_regulation", "p_regulation",             "multiclass"),
        ("task_pistol_r1",             "p_team1_wins_pistol_r1",   "binary"),
        ("task_pistol_r13",            "p_team1_wins_pistol_r13",  "binary"),
        ("task_t1_rounds",             "expected_t1_rounds",       "poisson"),
        ("task_t2_rounds",             "expected_t2_rounds",       "poisson"),
        ("task_total_rounds",          "expected_total_rounds",    "poisson"),
    ]
    lgb_preds: dict[str, np.ndarray] = {}
    for base, col, kind in head_models:
        try:
            path = _maybe_tuned(model_dir, base)
        except FileNotFoundError:
            continue
        p = _lgb_predict(path, df_map, NON_MAP_ALL, CATS_MAP)
        lgb_preds[col] = p
        if kind == "multiclass":
            out["p_reg_t1"]  = p[:, 0]
            out["p_reg_t2"]  = p[:, 1]
            out["p_reg_tie"] = p[:, 2]
        else:
            out[col] = p

    if not use_ensemble:
        return out

    # Ensemble overlay
    ens_path = model_dir / "ensemble_model.pt"
    if not ens_path.exists():
        return out
    try:
        import torch
    except ImportError:
        print("[predict] torch missing; skipping ensemble. Standalone LGB preds returned.")
        return out

    ckpt = torch.load(ens_path, map_location="cpu", weights_only=False)
    spec = ckpt["spec"]
    feat_cols = spec["feat_cols"]
    mu = np.asarray(spec["mu"], dtype=np.float32)
    sd = np.asarray(spec["sd"], dtype=np.float32)
    arch = spec["arch"]

    # Build feature matrix in the same column order
    # Map our LGB pred columns to the ensemble feature names.
    rename_to_ens = {
        "p_team1_wins_map_total": "lgb_map_total",
        "p_reg_t1":                "lgb_map_reg_t1",
        "p_reg_t2":                "lgb_map_reg_t2",
        "p_reg_tie":               "lgb_map_reg_tie",
        "p_team1_wins_pistol_r1":  "lgb_pistol1",
        "p_team1_wins_pistol_r13": "lgb_pistol13",
        "expected_t1_rounds":      "lgb_t1r",
        "expected_t2_rounds":      "lgb_t2r",
        "expected_total_rounds":   "lgb_totalr",
    }
    src = out.rename(columns=rename_to_ens).copy()
    # Pull the raw extra features from df_map
    for c in feat_cols:
        if c not in src.columns and c in df_map.columns:
            src[c] = df_map[c].values
    # Coerce numeric
    Xm = src[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xm = (Xm - mu) / sd
    Xm = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)

    # Rebuild the NN architecture and load weights
    import torch.nn as nn
    class MetaNN(nn.Module):
        def __init__(self, in_dim, trunk=(32, 16), dropout=0.25):
            super().__init__()
            prev = in_dim
            layers = []
            for h in trunk:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.h_map_total = nn.Linear(prev, 1)
            self.h_map_reg = nn.Linear(prev, 3)
            self.h_pistol1 = nn.Linear(prev, 1)
            self.h_pistol13 = nn.Linear(prev, 1)
            self.h_t1r = nn.Linear(prev, 1)
            self.h_t2r = nn.Linear(prev, 1)
            self.h_totalr = nn.Linear(prev, 1)
        def forward(self, x):
            h = self.trunk(x)
            return {
                "map_total": self.h_map_total(h).squeeze(-1),
                "map_reg":   self.h_map_reg(h),
                "pistol1":   self.h_pistol1(h).squeeze(-1),
                "pistol13":  self.h_pistol13(h).squeeze(-1),
                "t1r":       self.h_t1r(h).squeeze(-1),
                "t2r":       self.h_t2r(h).squeeze(-1),
                "totalr":    self.h_totalr(h).squeeze(-1),
            }

    model = MetaNN(in_dim=Xm.shape[1],
                   trunk=tuple(arch.get("trunk", [32, 16])),
                   dropout=arch.get("dropout", 0.25))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(Xm)
        outs = model(x)
        sig = lambda z: 1.0 / (1.0 + np.exp(-z.numpy()))
        out["p_team1_wins_map_total_ens"]   = sig(outs["map_total"])
        out["p_team1_wins_pistol_r1_ens"]   = sig(outs["pistol1"])
        out["p_team1_wins_pistol_r13_ens"]  = sig(outs["pistol13"])
        # multiclass softmax
        logits = outs["map_reg"].numpy()
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        sm = ex / ex.sum(axis=1, keepdims=True)
        out["p_reg_t1_ens"]  = sm[:, 0]
        out["p_reg_t2_ens"]  = sm[:, 1]
        out["p_reg_tie_ens"] = sm[:, 2]
        out["expected_t1_rounds_ens"]   = np.exp(np.clip(outs["t1r"].numpy(),    -20, 6))
        out["expected_t2_rounds_ens"]   = np.exp(np.clip(outs["t2r"].numpy(),    -20, 6))
        out["expected_total_rounds_ens"]= np.exp(np.clip(outs["totalr"].numpy(), -20, 6))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--match-id", type=int, default=None)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--out", default=None,
                    help="path to save the predictions CSV; defaults to stdout")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="use only standalone LGB heads, skip the meta-NN overlay")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else inp

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        # try both interpretations
        if "y_team1_wins" in df.columns and "y_team1_wins_map" not in df.columns:
            preds = predict_match(df, model_dir)
        else:
            preds = predict_map(df, model_dir, use_ensemble=not args.no_ensemble)
    elif args.match_id is not None:
        # Predict series winner from prematch_features
        df_pre = pd.read_parquet(inp / "prematch_features.parquet")
        df_pre = df_pre[df_pre["match_id"] == args.match_id]
        if len(df_pre) == 0:
            raise SystemExit(f"match_id {args.match_id} not found in prematch_features.parquet")
        series = predict_match(df_pre, model_dir)
        print("\n=== Match-level (series winner) ===")
        print(series.to_string(index=False))

        # Predict per-map for the same match
        df_map = pd.read_parquet(inp / "permap_features.parquet")
        df_map = df_map[df_map["match_id"] == args.match_id]
        if len(df_map):
            maps = predict_map(df_map, model_dir, use_ensemble=not args.no_ensemble)
            print("\n=== Per-map predictions ===")
            print(maps.to_string(index=False))
            if args.out:
                series.to_csv(Path(args.out).with_suffix(".series.csv"), index=False)
                maps.to_csv(Path(args.out).with_suffix(".maps.csv"), index=False)
                print(f"\nsaved to {args.out}.series.csv and {args.out}.maps.csv")
        else:
            print(f"\n[note] no rows in permap_features.parquet for match_id {args.match_id}")
            if args.out:
                series.to_csv(args.out, index=False)
        return

    else:
        # Default: predict on the test slice of each parquet (chronological tail)
        df_pre = pd.read_parquet(inp / "prematch_features.parquet")
        df_pre = df_pre[df_pre[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
        df_pre = df_pre.sort_values("date").iloc[-200:]   # last 200 matches
        df_map = pd.read_parquet(inp / "permap_features.parquet")
        df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
        df_map = df_map.sort_values("date").iloc[-600:]
        preds_series = predict_match(df_pre, model_dir)
        preds_maps   = predict_map(df_map, model_dir, use_ensemble=not args.no_ensemble)
        print(f"[predict] series rows: {len(preds_series)}")
        print(preds_series.head().to_string(index=False))
        print(f"\n[predict] map rows: {len(preds_maps)}")
        print(preds_maps.head().to_string(index=False))
        if args.out:
            preds_series.to_csv(Path(args.out).with_suffix(".series.csv"), index=False)
            preds_maps.to_csv(Path(args.out).with_suffix(".maps.csv"), index=False)
            print(f"\nsaved to {args.out}.series.csv and {args.out}.maps.csv")


if __name__ == "__main__":
    main()
