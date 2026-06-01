"""Multi-task neural network — shared trunk + 8 task heads.

Trained on permap_features.parquet. Predicts simultaneously:
  - map_winner_total      (binary)
  - map_winner_regulation (3-class: t1 / t2 / tie)
  - pistol_r1             (binary, may be NaN -> masked)
  - pistol_r13            (binary, may be NaN -> masked)
  - is_overtime_map       (binary, free auxiliary task)
  - t1_rounds             (regression, Poisson NLL)
  - t2_rounds             (regression, Poisson NLL)
  - total_rounds          (regression, Poisson NLL)

Usage:
    !pip install -q -r requirements.txt
    !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    !python mtl.py --input /kaggle/input/<slug>/ --output /kaggle/working/

If GPU is available it will be used automatically.
Match winner stays in run.py since it uses a different feature table.

Single file, no external imports from the parent project.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    mean_absolute_error, mean_squared_error, accuracy_score,
)


NON_MAP = {"match_id","mapstats_id","date","team1_id","team2_id","team1_name","team2_name"}
TARGETS = {
    "y_team1_wins_map","y_regulation_winner","is_overtime_map",
    "y_pistol_r1_t1_wins","y_pistol_r13_t1_wins",
    "t1_rounds","t2_rounds","total_rounds","reg_score_t1","reg_score_t2",
    "map_score_t1","map_score_t2",
}
CATS = ["format","event_type","map_name","map_picked_by"]


# ----------------------------- data -----------------------------

@dataclass
class FeatureSpec:
    num_cols: list[str]
    cat_cols: list[str]
    cat_vocabs: dict[str, dict[str, int]]   # value -> idx; index 0 reserved for unknown/missing
    num_mean: np.ndarray
    num_std: np.ndarray

    @property
    def num_dim(self) -> int: return len(self.num_cols)
    @property
    def cat_sizes(self) -> list[int]: return [len(self.cat_vocabs[c]) + 1 for c in self.cat_cols]


def build_spec(df_train: pd.DataFrame, df_full: pd.DataFrame) -> FeatureSpec:
    drop = NON_MAP | TARGETS
    num_cols, cat_cols = [], []
    for c in df_full.columns:
        if c in drop: continue
        if c in CATS:
            cat_cols.append(c)
        else:
            # numeric (we'll coerce object->numeric below)
            num_cols.append(c)

    cat_vocabs: dict[str, dict[str, int]] = {}
    for c in cat_cols:
        vals = sorted({str(v) for v in df_full[c].dropna().unique()})
        cat_vocabs[c] = {v: i + 1 for i, v in enumerate(vals)}  # 0 = missing/unknown

    # numeric mean/std from TRAIN only
    Xn = df_train[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mu = np.nanmean(Xn, axis=0)
    sd = np.nanstd(Xn, axis=0)
    sd[sd < 1e-6] = 1.0
    return FeatureSpec(num_cols, cat_cols, cat_vocabs, mu, sd)


def encode(df: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    Xn = df[spec.num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xn = (Xn - spec.num_mean) / spec.num_std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = np.zeros((len(df), len(spec.cat_cols)), dtype=np.int64)
    for j, c in enumerate(spec.cat_cols):
        vocab = spec.cat_vocabs[c]
        Xc[:, j] = df[c].astype(object).map(lambda v: vocab.get(str(v), 0)).fillna(0).astype(int).to_numpy()
    return Xn, Xc


def split_chrono(df, val_frac=0.15, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df); nt = int(n*test_frac); nv = int(n*val_frac)
    return df.iloc[:n-nv-nt].copy(), df.iloc[n-nv-nt:n-nt].copy(), df.iloc[n-nt:].copy()


# ----------------------------- model -----------------------------

class TrunkHead(nn.Module):
    """Shared MLP trunk + per-task heads."""

    def __init__(self, num_dim: int, cat_sizes: list[int], emb_dim: int = 8,
                 trunk_dims: tuple[int, ...] = (256, 128, 64), dropout: float = 0.15):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, emb_dim) for n in cat_sizes])
        in_dim = num_dim + emb_dim * len(cat_sizes)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in trunk_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        # heads
        self.h_map_total = nn.Linear(prev, 1)
        self.h_map_reg = nn.Linear(prev, 3)         # 3 classes
        self.h_pistol1 = nn.Linear(prev, 1)
        self.h_pistol13 = nn.Linear(prev, 1)
        self.h_ot = nn.Linear(prev, 1)              # aux: did map go to OT
        self.h_t1r = nn.Linear(prev, 1)             # log(mean) for Poisson
        self.h_t2r = nn.Linear(prev, 1)
        self.h_totalr = nn.Linear(prev, 1)

    def forward(self, x_num, x_cat):
        embs = [self.emb[j](x_cat[:, j]) for j in range(x_cat.shape[1])]
        x = torch.cat([x_num] + embs, dim=1)
        h = self.trunk(x)
        return {
            "map_total": self.h_map_total(h).squeeze(-1),
            "map_reg":   self.h_map_reg(h),
            "pistol1":   self.h_pistol1(h).squeeze(-1),
            "pistol13":  self.h_pistol13(h).squeeze(-1),
            "ot":        self.h_ot(h).squeeze(-1),
            "t1r":       self.h_t1r(h).squeeze(-1),
            "t2r":       self.h_t2r(h).squeeze(-1),
            "totalr":    self.h_totalr(h).squeeze(-1),
        }


# ----------------------------- targets -----------------------------

REG_CLASSES = ["t1", "t2", "tie"]


def make_targets(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Build target + mask tensors for every head."""
    def _f(s): return torch.tensor(s.to_numpy(dtype=np.float32))
    out = {
        "map_total":    _f(df["y_team1_wins_map"].astype(float)),
        "map_total_m":  _f(df["y_team1_wins_map"].notna().astype(float)),
        "map_reg":      torch.tensor(df["y_regulation_winner"].map({c: i for i, c in enumerate(REG_CLASSES)})
                                       .fillna(-1).astype(int).to_numpy(), dtype=torch.long),
        "map_reg_m":    _f(df["y_regulation_winner"].notna().astype(float)),
        "pistol1":      _f(df["y_pistol_r1_t1_wins"].astype(float)),
        "pistol1_m":    _f(df["y_pistol_r1_t1_wins"].notna().astype(float)),
        "pistol13":     _f(df["y_pistol_r13_t1_wins"].astype(float)),
        "pistol13_m":   _f(df["y_pistol_r13_t1_wins"].notna().astype(float)),
        "ot":           _f(df["is_overtime_map"].astype(float)),
        "ot_m":         _f(df["is_overtime_map"].notna().astype(float)),
        "t1r":          _f(df["t1_rounds"].astype(float)),
        "t1r_m":        _f(df["t1_rounds"].notna().astype(float)),
        "t2r":          _f(df["t2_rounds"].astype(float)),
        "t2r_m":        _f(df["t2_rounds"].notna().astype(float)),
        "totalr":       _f(df["total_rounds"].astype(float)),
        "totalr_m":     _f(df["total_rounds"].notna().astype(float)),
    }
    # NaN -> 0 for regression targets (will be masked anyway)
    for k in ("map_total","pistol1","pistol13","ot","t1r","t2r","totalr"):
        out[k] = torch.nan_to_num(out[k], 0.0)
    return out


# ----------------------------- losses -----------------------------

def masked_bce(logits, target, mask) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def masked_ce(logits, target, mask) -> torch.Tensor:
    # target is long with -1 where missing; replace with 0 for safe gather, then mask out
    safe_target = target.clamp_min(0)
    ce = F.cross_entropy(logits, safe_target, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (ce * mask).sum() / denom


def masked_poisson_nll(log_pred, target, mask) -> torch.Tensor:
    # PyTorch Poisson NLL with log_input=True is the right one for log-rate output
    nll = F.poisson_nll_loss(log_pred, target, log_input=True, full=False, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom


def total_loss(out: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor],
               weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    parts = {
        "map_total": masked_bce(out["map_total"], tgt["map_total"], tgt["map_total_m"]),
        "map_reg":   masked_ce(out["map_reg"], tgt["map_reg"], tgt["map_reg_m"]),
        "pistol1":   masked_bce(out["pistol1"], tgt["pistol1"], tgt["pistol1_m"]),
        "pistol13":  masked_bce(out["pistol13"], tgt["pistol13"], tgt["pistol13_m"]),
        "ot":        masked_bce(out["ot"], tgt["ot"], tgt["ot_m"]),
        "t1r":       masked_poisson_nll(out["t1r"], tgt["t1r"], tgt["t1r_m"]),
        "t2r":       masked_poisson_nll(out["t2r"], tgt["t2r"], tgt["t2r_m"]),
        "totalr":    masked_poisson_nll(out["totalr"], tgt["totalr"], tgt["totalr_m"]),
    }
    total = sum(weights.get(k, 1.0) * v for k, v in parts.items())
    return total, {k: float(v.item()) for k, v in parts.items()}


# ----------------------------- training loop -----------------------------

def to_device(t, device): return {k: v.to(device) for k, v in t.items()}


def iterate_minibatches(x_num, x_cat, tgt, batch_size, shuffle=True, device="cpu"):
    n = x_num.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        sel = idx[start:start+batch_size]
        xn = torch.from_numpy(x_num[sel]).to(device)
        xc = torch.from_numpy(x_cat[sel]).to(device)
        t = {k: v[sel].to(device) for k, v in tgt.items()}
        yield xn, xc, t


def evaluate(model, x_num, x_cat, df, device, batch_size=2048) -> dict[str, Any]:
    model.eval()
    preds: dict[str, list[np.ndarray]] = {k: [] for k in
        ("map_total","map_reg","pistol1","pistol13","ot","t1r","t2r","totalr")}
    with torch.no_grad():
        for start in range(0, x_num.shape[0], batch_size):
            xn = torch.from_numpy(x_num[start:start+batch_size]).to(device)
            xc = torch.from_numpy(x_cat[start:start+batch_size]).to(device)
            out = model(xn, xc)
            for k in preds:
                preds[k].append(out[k].cpu().numpy())
    for k in preds:
        preds[k] = np.concatenate(preds[k], axis=0)

    metrics: dict[str, Any] = {}

    def _binary_metrics(name, logits, y_col):
        m = df[y_col].notna().to_numpy()
        if m.sum() < 10: return
        y = df.loc[m, y_col].astype(int).to_numpy()
        p = 1 / (1 + np.exp(-logits[m]))
        metrics[name] = {
            "n": int(m.sum()),
            "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
        }

    _binary_metrics("map_winner_total", preds["map_total"], "y_team1_wins_map")
    _binary_metrics("pistol_r1", preds["pistol1"], "y_pistol_r1_t1_wins")
    _binary_metrics("pistol_r13", preds["pistol13"], "y_pistol_r13_t1_wins")
    _binary_metrics("is_overtime_map", preds["ot"], "is_overtime_map")

    # map_winner_regulation (3-class)
    mask = df["y_regulation_winner"].notna().to_numpy()
    if mask.sum() >= 10:
        y = df.loc[mask, "y_regulation_winner"].map({c: i for i, c in enumerate(REG_CLASSES)}).astype(int).to_numpy()
        logits = preds["map_reg"][mask]
        # softmax
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        p = ex / ex.sum(axis=1, keepdims=True)
        metrics["map_winner_regulation"] = {
            "n": int(mask.sum()),
            "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
        }

    # regression: exp(log_pred) is the Poisson rate / expected value
    def _reg_metrics(name, log_pred, y_col):
        m = df[y_col].notna().to_numpy()
        if m.sum() < 10: return
        y = df.loc[m, y_col].astype(float).to_numpy()
        yhat = np.exp(np.clip(log_pred[m], -20, 6))   # safety clamp
        metrics[name] = {
            "n": int(m.sum()),
            "mae": float(mean_absolute_error(y, yhat)),
            "rmse": float(mean_squared_error(y, yhat) ** 0.5),
        }
    _reg_metrics("t1_rounds", preds["t1r"], "t1_rounds")
    _reg_metrics("t2_rounds", preds["t2r"], "t2_rounds")
    _reg_metrics("total_rounds", preds["totalr"], "total_rounds")

    return metrics, preds


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"input:  {inp}")
    print(f"output: {out_dir}")
    print(f"device: {device}")

    df = pd.read_parquet(inp / "permap_features.parquet")
    df = df[df[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0].copy()
    df_tr, df_va, df_te = split_chrono(df)
    print(f"train: {len(df_tr)}  val: {len(df_va)}  test: {len(df_te)}")

    spec = build_spec(df_tr, df)
    Xn_tr, Xc_tr = encode(df_tr, spec)
    Xn_va, Xc_va = encode(df_va, spec)
    Xn_te, Xc_te = encode(df_te, spec)
    t_tr = make_targets(df_tr.reset_index(drop=True))
    t_va = make_targets(df_va.reset_index(drop=True))
    t_te = make_targets(df_te.reset_index(drop=True))

    model = TrunkHead(num_dim=spec.num_dim, cat_sizes=spec.cat_sizes).to(device)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Loss weights: lift the noisy tasks a bit; downweight totalr which dominated raw values
    weights = {"map_total": 1.0, "map_reg": 0.7, "pistol1": 1.3, "pistol13": 1.3,
               "ot": 0.3, "t1r": 0.4, "t2r": 0.4, "totalr": 0.4}

    best_val_signal = -1.0   # average AUC across binary heads, for early stop
    best_state = None
    no_improve = 0
    patience = 25

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        n_batches = 0
        for xn, xc, t in iterate_minibatches(Xn_tr, Xc_tr, t_tr, args.batch_size, device=device):
            opt.zero_grad()
            out = model(xn, xc)
            loss, parts = total_loss(out, t, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            val_m, _ = evaluate(model, Xn_va, Xc_va, df_va.reset_index(drop=True), device)
            aucs = [v["auc"] for k, v in val_m.items() if "auc" in v]
            signal = float(np.mean(aucs)) if aucs else 0.0
            avg_train_loss = sum(running.values()) / max(1, n_batches)
            print(f"ep {epoch:3d}  train_loss={avg_train_loss:.4f}  val_avg_AUC={signal:.4f}  "
                  f"map_total={val_m.get('map_winner_total',{}).get('auc',float('nan')):.3f}  "
                  f"pistol1={val_m.get('pistol_r1',{}).get('auc',float('nan')):.3f}  "
                  f"pistol13={val_m.get('pistol_r13',{}).get('auc',float('nan')):.3f}",
                  flush=True)
            if signal > best_val_signal:
                best_val_signal = signal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience // 5:
                    print(f"  early stop at epoch {epoch} (best val avg AUC={best_val_signal:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, _ = evaluate(model, Xn_te, Xc_te, df_te.reset_index(drop=True), device)

    print("\n" + "=" * 60); print("MTL TEST METRICS"); print("=" * 60)
    for k, v in test_m.items():
        if "auc" in v:
            print(f"  {k:30}  AUC={v['auc']:.4f}  log_loss={v['log_loss']:.4f}  n={v['n']}")
        elif "accuracy" in v:
            print(f"  {k:30}  acc={v['accuracy']:.4f}  log_loss={v['log_loss']:.4f}  n={v['n']}")
        else:
            print(f"  {k:30}  MAE={v['mae']:.3f}  RMSE={v['rmse']:.3f}  n={v['n']}")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "spec": {
            "num_cols": spec.num_cols, "cat_cols": spec.cat_cols,
            "cat_vocabs": spec.cat_vocabs,
            "num_mean": spec.num_mean.tolist(), "num_std": spec.num_std.tolist(),
        },
        "arch": {"emb_dim": 8, "trunk_dims": [256, 128, 64], "dropout": 0.15},
        "test_metrics": test_m,
    }, out_dir / "mtl_model.pt")
    with open(out_dir / "mtl_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    print(f"\nsaved {out_dir / 'mtl_model.pt'} and {out_dir / 'mtl_metrics.json'}")


if __name__ == "__main__":
    main()
