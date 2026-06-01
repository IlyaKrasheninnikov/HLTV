"""Meta-ensemble: tuned LightGBM out-of-fold predictions -> small NN multi-head.

Pipeline:
  1. Time-aware OOF cross-validation across the per-map dataset (and prematch
     for the match-winner task). For each fold, train LightGBM on past data
     with the BEST params found by tune.py and predict the fold.
  2. Aggregate OOF predictions into an N x K feature matrix where each column
     is a task probability or expected value.
  3. Train a small PyTorch NN (32->16 trunk) that takes OOF preds + a SUBSET
     of raw features (rank diff, points diff, rolling rating) and emits
     refined predictions for each task.
  4. Compare test metrics: tuned LGB only vs LGB + meta-NN.

Why this works: the LGB models already capture most of the signal. The meta-NN
only has to *correct* their residuals across tasks where one task's prediction
informs another (e.g. team1 favored on map => higher P(pistol1) too).

Usage on Kaggle:
    !pip install -q -r /kaggle/input/<slug>/requirements.txt
    !pip install -q optuna torch
    # First run tune.py to produce task_<name>_best_params.json files.
    !python /kaggle/input/<slug>/tune.py --input /kaggle/input/<slug>/ --output /kaggle/working/ --n-trials 80
    # Then run ensemble.py — it reads the tuned params from --output
    !python /kaggle/input/<slug>/ensemble.py --input /kaggle/input/<slug>/ --output /kaggle/working/

The script auto-uses CUDA if available.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, accuracy_score,
)


# -------- schema --------

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

# Raw extra features the meta-NN gets in addition to LGB OOF preds.
EXTRA_FEATS = [
    "diff_rank", "diff_points", "diff_rating", "diff_avg_player_rating",
    "diff_ct_wr", "diff_t_wr", "diff_rest_days", "stars",
    "diff_pistol_wr",  # only present in per-map after extension
]


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


def _params(best: dict, kind: str, n_classes: int = 1) -> dict:
    base = dict(best)
    base["verbose"] = -1
    if kind == "binary":
        base["objective"] = "binary"; base["metric"] = ["binary_logloss"]
    elif kind == "multiclass":
        base["objective"] = "multiclass"; base["num_class"] = n_classes
        base["metric"] = ["multi_logloss"]
    elif kind == "poisson":
        base["objective"] = "poisson"; base["metric"] = ["poisson"]
    return base


def _load_best(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    print(f"  [warn] no best_params at {path.name}, using defaults")
    return dict(default)


# -------- OOF predictions per task --------

def time_aware_oof(df, target, drop, cats, kind, params, cats_in_fn,
                   n_folds=5, n_classes=1, class_map=None, dropna_y=True):
    """Returns a Series of OOF predictions aligned with df.index (chronological).

    For binary -> P(class=1). For multiclass -> N x C array; we keep argmax-prob
    of class index 0 (t1) as a scalar feature. For poisson -> expected value.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    if dropna_y:
        mask = df[target].notna().to_numpy()
    else:
        mask = np.ones(len(df), dtype=bool)
    n = len(df)
    # OOF arrays
    if kind == "multiclass":
        oof = np.full((n, n_classes), np.nan, dtype=np.float32)
    else:
        oof = np.full(n, np.nan, dtype=np.float32)

    fold_edges = np.linspace(0, n, n_folds + 1, dtype=int)
    cats_in = None

    for f in range(1, n_folds + 1):
        train_idx = np.arange(0, fold_edges[f-1])
        pred_idx = np.arange(fold_edges[f-1], fold_edges[f])
        if len(train_idx) < 80 or len(pred_idx) == 0:
            continue
        # train/val inside the training slice
        val_split = int(len(train_idx) * 0.85)
        tr_idx = train_idx[:val_split]
        va_idx = train_idx[val_split:]
        # apply target-NaN mask
        tr_idx = tr_idx[mask[tr_idx]]
        va_idx = va_idx[mask[va_idx]]
        if len(tr_idx) < 50 or len(va_idx) < 20:
            continue

        tr = df.iloc[tr_idx]; va = df.iloc[va_idx]; pr = df.iloc[pred_idx]
        X_tr, y_tr = _to_xy(tr, target, drop, cats)
        X_va, y_va = _to_xy(va, target, drop, cats)
        X_pr, _ = _to_xy(pr, target, drop, cats, dropna=False)
        if cats_in is None:
            cats_in = [c for c in cats if c in X_tr.columns]
        if kind == "binary":
            y_tr = y_tr.astype(int); y_va = y_va.astype(int)
        elif kind == "multiclass":
            y_tr = y_tr.map(class_map).astype(int)
            y_va = y_va.map(class_map).astype(int)
        else:
            y_tr = y_tr.astype(float); y_va = y_va.astype(float)
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
        m = lgb.train(params, dtr, num_boost_round=2000,
                      valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        preds = m.predict(X_pr, num_iteration=m.best_iteration)
        if kind == "multiclass":
            oof[pred_idx] = preds
        else:
            oof[pred_idx] = preds
    return df, oof, mask


# -------- meta NN --------

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


REG_CLASSES = ["t1", "t2", "tie"]


def masked_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_ce(logits, target, mask):
    ce = F.cross_entropy(logits, target.clamp_min(0), reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def masked_poisson(log_pred, target, mask):
    nll = F.poisson_nll_loss(log_pred, target, log_input=True, full=False, reduction="none")
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"input: {inp}\noutput: {out_dir}\ndevice: {device}")

    df_map = pd.read_parquet(inp / "permap_features.parquet")
    df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]

    # ---------- OOF preds for each task using tuned LGB params ----------
    print("\n[1/3] Computing OOF predictions per task ...")

    def best_or_default(name, default):
        return _load_best(out_dir / f"task_{name}_best_params.json", default)

    df_sorted, oof_map_total, _ = time_aware_oof(
        df_map, "y_team1_wins_map", NON_MAP_ALL, CATS_MAP, "binary",
        _params(best_or_default("map_winner_total", {}), "binary"),
        cats_in_fn=None, n_folds=args.n_folds)
    print(f"  map_winner_total OOF coverage: {(~np.isnan(oof_map_total)).mean():.3f}")

    _, oof_map_reg, _ = time_aware_oof(
        df_map, "y_regulation_winner", NON_MAP_ALL, CATS_MAP, "multiclass",
        _params(best_or_default("map_winner_regulation", {}), "multiclass", n_classes=3),
        cats_in_fn=None, n_folds=args.n_folds, n_classes=3,
        class_map={"t1": 0, "t2": 1, "tie": 2})
    print(f"  map_winner_regulation OOF coverage: {(~np.isnan(oof_map_reg).any(axis=1)).mean():.3f}")

    _, oof_p1, _ = time_aware_oof(
        df_map, "y_pistol_r1_t1_wins", NON_MAP_ALL, CATS_MAP, "binary",
        _params(best_or_default("pistol_r1", {}), "binary"),
        cats_in_fn=None, n_folds=args.n_folds)
    _, oof_p13, _ = time_aware_oof(
        df_map, "y_pistol_r13_t1_wins", NON_MAP_ALL, CATS_MAP, "binary",
        _params(best_or_default("pistol_r13", {}), "binary"),
        cats_in_fn=None, n_folds=args.n_folds)
    _, oof_t1r, _ = time_aware_oof(
        df_map, "t1_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
        _params(best_or_default("t1_rounds", {}), "poisson"),
        cats_in_fn=None, n_folds=args.n_folds)
    _, oof_t2r, _ = time_aware_oof(
        df_map, "t2_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
        _params(best_or_default("t2_rounds", {}), "poisson"),
        cats_in_fn=None, n_folds=args.n_folds)
    _, oof_totalr, _ = time_aware_oof(
        df_map, "total_rounds", NON_MAP_ALL, CATS_MAP, "poisson",
        _params(best_or_default("total_rounds", {}), "poisson"),
        cats_in_fn=None, n_folds=args.n_folds)

    df = df_sorted.copy()
    df["lgb_map_total"] = oof_map_total
    df["lgb_map_reg_t1"] = oof_map_reg[:, 0]
    df["lgb_map_reg_t2"] = oof_map_reg[:, 1]
    df["lgb_map_reg_tie"] = oof_map_reg[:, 2]
    df["lgb_pistol1"] = oof_p1
    df["lgb_pistol13"] = oof_p13
    df["lgb_t1r"] = oof_t1r
    df["lgb_t2r"] = oof_t2r
    df["lgb_totalr"] = oof_totalr

    # ---------- build meta features ----------
    meta_cols = [
        "lgb_map_total","lgb_map_reg_t1","lgb_map_reg_t2","lgb_map_reg_tie",
        "lgb_pistol1","lgb_pistol13","lgb_t1r","lgb_t2r","lgb_totalr",
    ] + [c for c in EXTRA_FEATS if c in df.columns]
    df_meta = df.dropna(subset=["lgb_map_total"]).copy()  # use rows with OOF
    print(f"\n[2/3] Meta training rows: {len(df_meta)}  feature columns: {len(meta_cols)}")

    tr, va, te = split_chrono(df_meta)
    # Standardize meta features on train
    Xn_cols = meta_cols
    mu = tr[Xn_cols].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=np.float32)
    sd = tr[Xn_cols].apply(pd.to_numeric, errors="coerce").std().to_numpy(dtype=np.float32)
    sd[sd < 1e-6] = 1.0
    def enc(d):
        x = d[Xn_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        x = (x - mu) / sd
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    Xn_tr, Xn_va, Xn_te = enc(tr), enc(va), enc(te)

    def targets(d):
        return {
            "map_total":   torch.tensor(d["y_team1_wins_map"].astype(np.float32).to_numpy()),
            "map_total_m": torch.tensor(d["y_team1_wins_map"].notna().astype(np.float32).to_numpy()),
            "map_reg":     torch.tensor(d["y_regulation_winner"]
                                          .map({c:i for i,c in enumerate(REG_CLASSES)})
                                          .fillna(-1).astype(int).to_numpy(), dtype=torch.long),
            "map_reg_m":   torch.tensor(d["y_regulation_winner"].notna().astype(np.float32).to_numpy()),
            "pistol1":     torch.tensor(d["y_pistol_r1_t1_wins"].astype(np.float32).fillna(0).to_numpy()),
            "pistol1_m":   torch.tensor(d["y_pistol_r1_t1_wins"].notna().astype(np.float32).to_numpy()),
            "pistol13":    torch.tensor(d["y_pistol_r13_t1_wins"].astype(np.float32).fillna(0).to_numpy()),
            "pistol13_m":  torch.tensor(d["y_pistol_r13_t1_wins"].notna().astype(np.float32).to_numpy()),
            "t1r":         torch.tensor(d["t1_rounds"].astype(np.float32).to_numpy()),
            "t1r_m":       torch.tensor(d["t1_rounds"].notna().astype(np.float32).to_numpy()),
            "t2r":         torch.tensor(d["t2_rounds"].astype(np.float32).to_numpy()),
            "t2r_m":       torch.tensor(d["t2_rounds"].notna().astype(np.float32).to_numpy()),
            "totalr":      torch.tensor(d["total_rounds"].astype(np.float32).to_numpy()),
            "totalr_m":    torch.tensor(d["total_rounds"].notna().astype(np.float32).to_numpy()),
        }

    t_tr = targets(tr.reset_index(drop=True))
    t_va = targets(va.reset_index(drop=True))
    t_te = targets(te.reset_index(drop=True))

    in_dim = Xn_tr.shape[1]
    model = MetaNN(in_dim=in_dim, trunk=(32, 16), dropout=0.25).to(device)
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    weights = {"map_total": 1.0, "map_reg": 0.7,
               "pistol1": 1.2, "pistol13": 1.2,
               "t1r": 0.5, "t2r": 0.5, "totalr": 0.4}

    def loss_fn(out, tgt):
        return (
            weights["map_total"] * masked_bce(out["map_total"], tgt["map_total"], tgt["map_total_m"])
            + weights["map_reg"] * masked_ce(out["map_reg"], tgt["map_reg"], tgt["map_reg_m"])
            + weights["pistol1"] * masked_bce(out["pistol1"], tgt["pistol1"], tgt["pistol1_m"])
            + weights["pistol13"] * masked_bce(out["pistol13"], tgt["pistol13"], tgt["pistol13_m"])
            + weights["t1r"] * masked_poisson(out["t1r"], tgt["t1r"], tgt["t1r_m"])
            + weights["t2r"] * masked_poisson(out["t2r"], tgt["t2r"], tgt["t2r_m"])
            + weights["totalr"] * masked_poisson(out["totalr"], tgt["totalr"], tgt["totalr_m"])
        )

    def to_dev(t): return {k: v.to(device) for k, v in t.items()}

    def eval_meta(Xn, df_eval, t):
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(Xn).to(device))
            out = {k: v.cpu().numpy() for k, v in out.items()}
        metrics = {}
        def _bin(name, logits, ycol):
            m = df_eval[ycol].notna().to_numpy()
            if m.sum() < 10: return
            y = df_eval.loc[m, ycol].astype(int).to_numpy()
            p = 1 / (1 + np.exp(-logits[m]))
            metrics[name] = {"n": int(m.sum()),
                             "auc": float(roc_auc_score(y, p)),
                             "log_loss": float(log_loss(y, p, labels=[0,1])),
                             "brier": float(brier_score_loss(y, p))}
        _bin("map_winner_total", out["map_total"], "y_team1_wins_map")
        _bin("pistol_r1", out["pistol1"], "y_pistol_r1_t1_wins")
        _bin("pistol_r13", out["pistol13"], "y_pistol_r13_t1_wins")
        # multiclass
        m = df_eval["y_regulation_winner"].notna().to_numpy()
        if m.sum() >= 10:
            y = df_eval.loc[m, "y_regulation_winner"].map({c:i for i,c in enumerate(REG_CLASSES)}).astype(int).to_numpy()
            logits = out["map_reg"][m]
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = ex / ex.sum(axis=1, keepdims=True)
            metrics["map_winner_regulation"] = {"n": int(m.sum()),
                                                "log_loss": float(log_loss(y, p, labels=[0,1,2])),
                                                "accuracy": float(accuracy_score(y, p.argmax(axis=1)))}
        # regression
        for name, key, col in [("t1_rounds","t1r","t1_rounds"),
                                ("t2_rounds","t2r","t2_rounds"),
                                ("total_rounds","totalr","total_rounds")]:
            m = df_eval[col].notna().to_numpy()
            if m.sum() < 10: continue
            y = df_eval.loc[m, col].astype(float).to_numpy()
            yhat = np.exp(np.clip(out[key][m], -20, 6))
            metrics[name] = {"n": int(m.sum()),
                             "mae": float(mean_absolute_error(y, yhat)),
                             "rmse": float(mean_squared_error(y, yhat) ** 0.5)}
        return metrics

    print("\n[3/3] Training meta-NN ...")
    best_signal = -1.0; best_state = None; no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = np.arange(len(Xn_tr)); np.random.shuffle(idx)
        running = 0.0; nb = 0
        for s in range(0, len(idx), args.batch_size):
            sel = idx[s:s+args.batch_size]
            xn = torch.from_numpy(Xn_tr[sel]).to(device)
            tgt = {k: v[sel].to(device) for k, v in t_tr.items()}
            opt.zero_grad()
            out = model(xn)
            loss = loss_fn(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1:
            val_m = eval_meta(Xn_va, va.reset_index(drop=True), t_va)
            aucs = [v["auc"] for k, v in val_m.items() if "auc" in v]
            signal = float(np.mean(aucs)) if aucs else 0.0
            print(f"  ep {ep:3d}  train_loss={running/nb:.4f}  val_avg_AUC={signal:.4f}  "
                  f"map={val_m.get('map_winner_total',{}).get('auc',float('nan')):.3f}  "
                  f"p1={val_m.get('pistol_r1',{}).get('auc',float('nan')):.3f}  "
                  f"p13={val_m.get('pistol_r13',{}).get('auc',float('nan')):.3f}",
                  flush=True)
            if signal > best_signal:
                best_signal = signal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 4:
                    print(f"  early stop at ep {ep}, best val avg AUC={best_signal:.4f}")
                    break

    if best_state is not None: model.load_state_dict(best_state)
    test_m = eval_meta(Xn_te, te.reset_index(drop=True), t_te)
    print("\n" + "=" * 60); print("ENSEMBLE (meta-NN) TEST METRICS"); print("=" * 60)
    for k, v in test_m.items():
        if "auc" in v:
            print(f"  {k:30}  AUC={v['auc']:.4f}  log_loss={v['log_loss']:.4f}")
        elif "accuracy" in v:
            print(f"  {k:30}  acc={v['accuracy']:.4f}  log_loss={v['log_loss']:.4f}")
        else:
            print(f"  {k:30}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}")

    # Also print the standalone LGB baselines on the same test rows for comparison
    print("\n=== Standalone LGB on same test rows (for comparison) ===")
    def _lgb_metrics():
        out: dict[str, Any] = {}
        # binary
        for col, label in [("lgb_map_total", "y_team1_wins_map"),
                            ("lgb_pistol1", "y_pistol_r1_t1_wins"),
                            ("lgb_pistol13", "y_pistol_r13_t1_wins")]:
            m = te[label].notna().to_numpy()
            if m.sum() < 10: continue
            y = te.loc[m, label].astype(int).to_numpy()
            p = te.loc[m, col].to_numpy()
            p = np.clip(p, 1e-6, 1 - 1e-6)
            out[col] = {"auc": float(roc_auc_score(y, p)),
                        "log_loss": float(log_loss(y, p, labels=[0, 1]))}
        # multiclass
        m = te["y_regulation_winner"].notna().to_numpy()
        if m.sum() >= 10:
            y = te.loc[m, "y_regulation_winner"].map({c:i for i,c in enumerate(REG_CLASSES)}).astype(int).to_numpy()
            p = te.loc[m, ["lgb_map_reg_t1","lgb_map_reg_t2","lgb_map_reg_tie"]].to_numpy()
            p = p / p.sum(axis=1, keepdims=True).clip(1e-6)
            out["lgb_map_reg"] = {"log_loss": float(log_loss(y, p, labels=[0,1,2])),
                                  "accuracy": float(accuracy_score(y, p.argmax(axis=1)))}
        # regression — OOF preds are already expected values for Poisson
        for col, lab in [("lgb_t1r","t1_rounds"),("lgb_t2r","t2_rounds"),("lgb_totalr","total_rounds")]:
            m = te[lab].notna().to_numpy()
            if m.sum() < 10: continue
            y = te.loc[m, lab].astype(float).to_numpy()
            yhat = te.loc[m, col].to_numpy()
            out[col] = {"mae": float(mean_absolute_error(y, yhat)),
                        "rmse": float(mean_squared_error(y, yhat) ** 0.5)}
        return out

    lgb_te = _lgb_metrics()
    for k, v in lgb_te.items():
        if "auc" in v:
            print(f"  {k:25}  AUC={v['auc']:.4f}  log_loss={v['log_loss']:.4f}")
        elif "accuracy" in v:
            print(f"  {k:25}  acc={v['accuracy']:.4f}  log_loss={v['log_loss']:.4f}")
        else:
            print(f"  {k:25}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}")

    with open(out_dir / "ensemble_metrics.json", "w") as f:
        json.dump({"meta_nn_test": test_m, "lgb_oof_test_baseline": lgb_te}, f, indent=2)
    torch.save({"state_dict": model.state_dict(),
                "spec": {"feat_cols": Xn_cols, "mu": mu.tolist(), "sd": sd.tolist(),
                         "arch": {"trunk": [32, 16], "dropout": 0.25}}},
               out_dir / "ensemble_model.pt")
    print(f"\nsaved ensemble_metrics.json and ensemble_model.pt to {out_dir}")


if __name__ == "__main__":
    main()
