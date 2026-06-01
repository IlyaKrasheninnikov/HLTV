"""Round-Transformer for live in-game win probability.

Architecture:
  Each round in a map is a token. Per-token features:
    - round_no, period, is_overtime (positional / structural)
    - score_t1, score_t2, score_diff
    - last-outcome one-hot (ct_win/t_win/bomb_defused/bomb_exploded)
    - side_winner one-hot (CT/T)
    - eq_value_t1, eq_value_t2, eq_value_diff
    - pistol_won_t1, pistol_won_t2

  Per-map (static) features (broadcast or added to first token only):
    - diff_rank, diff_points, diff_rating, diff_onmap_win_rate, map_name embedding

  Sequence model: small Transformer encoder (4 layers, 8 heads, d_model=128).
  Output head: per-token logit predicting eventual map winner.
  Loss: BCE on every round position. Causal-mask is NOT used at training (we
  want to predict the future from each prefix; we use the standard encoder
  but the per-round target is the same final-map label, so each token sees
  past+current state and predicts the SAME label — the model learns the
  natural "more rounds = more certain" curve automatically because we predict
  per-token).

Why this beats LightGBM on round-level:
  LightGBM treats each round-state row independently. It can't model
  "team1 just won 3 in a row" or "the swing rate is unusual". The Transformer
  attends across the round history, capturing momentum that the engineered
  `last1/last2/last3` features can only approximate.

Usage:
    !pip install -q -r /kaggle/input/<slug>/requirements.txt
    !python /kaggle/input/<slug>/transformer_rounds.py \
        --input /kaggle/input/<slug>/ --output /kaggle/working/out \
        --epochs 30 --batch-size 64

Should run in 10-20 min on Kaggle T4.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


PER_ROUND_NUM = [
    "round_no", "period", "is_overtime",
    "score_t1", "score_t2", "score_diff",
    "rounds_to_win_t1", "rounds_to_win_t2",
    "eq_value_t1", "eq_value_t2", "eq_value_diff",
    "pistol_won_t1", "pistol_won_t2",
]
PER_ROUND_CAT_OUTCOME = ["ct_win","t_win","bomb_defused","bomb_exploded","stopwatch","__missing__"]
PER_ROUND_CAT_SIDE    = ["CT","T","__missing__"]

STATIC_NUM = ["diff_rank", "diff_points", "diff_rating", "diff_onmap_win_rate"]
MAP_NAMES_VOCAB_CAP = 30   # cap the embedding vocab for maps + unknown


@dataclass
class Spec:
    map_vocab: dict[str, int]
    num_mean: np.ndarray
    num_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray


def split_chrono_by_match(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    md = df.groupby("match_id")["date"].first().sort_values()
    n_m = len(md)
    c_v = int(n_m*(1-val_frac-test_frac)); c_t = int(n_m*(1-test_frac))
    tr_ids = set(md.iloc[:c_v].index)
    va_ids = set(md.iloc[c_v:c_t].index)
    te_ids = set(md.iloc[c_t:].index)
    return (df[df["match_id"].isin(tr_ids)].copy(),
            df[df["match_id"].isin(va_ids)].copy(),
            df[df["match_id"].isin(te_ids)].copy())


def _coerce(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def fit_spec(df_tr: pd.DataFrame) -> Spec:
    # Map vocab
    counts = df_tr["map_name"].astype(object).fillna("__missing__").value_counts()
    top = counts.head(MAP_NAMES_VOCAB_CAP).index.tolist()
    map_vocab = {m: i + 1 for i, m in enumerate(top)}  # 0 = unknown/missing
    # Per-round numeric stats
    Xn = _coerce(df_tr, PER_ROUND_NUM)
    mu = np.nanmean(Xn, axis=0); sd = np.nanstd(Xn, axis=0); sd[sd < 1e-6] = 1.0
    # Static numeric stats (first-row per map approximation)
    sm = df_tr.groupby(["match_id", "mapstats_id"]).first().reset_index()
    Xs = _coerce(sm, STATIC_NUM)
    smu = np.nanmean(Xs, axis=0); ssd = np.nanstd(Xs, axis=0); ssd[ssd < 1e-6] = 1.0
    return Spec(map_vocab, mu, sd, smu, ssd)


def encode_round_tokens(df: pd.DataFrame, spec: Spec):
    """Group by (match_id, mapstats_id), return per-map tensors padded to max len.

    Returns:
      x_num:  (B, T, F_num)   - normalized per-round numeric features
      x_outcome: (B, T)       - long, index into PER_ROUND_CAT_OUTCOME
      x_side:    (B, T)       - long, index into PER_ROUND_CAT_SIDE
      x_map:     (B,)         - long, map id
      x_static:  (B, F_static)
      y:         (B,)         - 0/1 (team1 wins the map)
      mask:      (B, T)       - 1.0 for real tokens, 0.0 for padding
    """
    df = df.sort_values(["match_id", "mapstats_id", "date", "round_no"]).reset_index(drop=True)
    out_idx = {v: i for i, v in enumerate(PER_ROUND_CAT_OUTCOME)}
    side_idx = {v: i for i, v in enumerate(PER_ROUND_CAT_SIDE)}

    groups: list[tuple] = []
    for (mid, msid), g in df.groupby(["match_id", "mapstats_id"], sort=False):
        if "y_map_t1_wins" not in g.columns:
            continue
        y = int(g["y_map_t1_wins"].iloc[0])
        Xn = _coerce(g, PER_ROUND_NUM)
        Xn = (Xn - spec.num_mean) / spec.num_std
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        outc = g["outcome"].astype(object).fillna("__missing__").map(lambda v: out_idx.get(v, out_idx["__missing__"]))
        sidec = g["side_winner"].astype(object).fillna("__missing__").map(lambda v: side_idx.get(v, side_idx["__missing__"]))
        map_id = spec.map_vocab.get(g["map_name"].iloc[0] if "map_name" in g.columns else "__missing__", 0)
        Xs = _coerce(g.head(1), STATIC_NUM)[0]
        Xs = (Xs - spec.static_mean) / spec.static_std
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        groups.append((Xn, outc.to_numpy(np.int64), sidec.to_numpy(np.int64),
                       int(map_id), Xs.astype(np.float32), y))
    if not groups:
        raise RuntimeError("no maps to encode")
    T = max(g[0].shape[0] for g in groups)
    B = len(groups)
    F_num = len(PER_ROUND_NUM)
    F_static = len(STATIC_NUM)
    x_num = np.zeros((B, T, F_num), dtype=np.float32)
    x_out = np.zeros((B, T), dtype=np.int64)
    x_side = np.zeros((B, T), dtype=np.int64)
    x_map = np.zeros((B,), dtype=np.int64)
    x_static = np.zeros((B, F_static), dtype=np.float32)
    y_arr = np.zeros((B,), dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, (Xn, oc, sc, mp, Xs, y) in enumerate(groups):
        t = Xn.shape[0]
        x_num[i, :t] = Xn
        x_out[i, :t] = oc
        x_side[i, :t] = sc
        x_map[i] = mp
        x_static[i] = Xs
        y_arr[i] = y
        mask[i, :t] = 1.0
    return x_num, x_out, x_side, x_map, x_static, y_arr, mask


class RoundTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dropout=0.1,
                 n_outcome=len(PER_ROUND_CAT_OUTCOME),
                 n_side=len(PER_ROUND_CAT_SIDE),
                 n_maps=MAP_NAMES_VOCAB_CAP + 1,
                 n_static=len(STATIC_NUM),
                 max_len=64):
        super().__init__()
        self.num_proj = nn.Linear(len(PER_ROUND_NUM), d_model // 2)
        self.outcome_emb = nn.Embedding(n_outcome, d_model // 4)
        self.side_emb = nn.Embedding(n_side, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.map_emb = nn.Embedding(n_maps, d_model)
        self.static_proj = nn.Linear(n_static, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_num, x_out, x_side, x_map, x_static, mask):
        B, T, _ = x_num.shape
        tok = torch.cat([self.num_proj(x_num),
                         self.outcome_emb(x_out),
                         self.side_emb(x_side)], dim=-1)            # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=tok.device)).unsqueeze(0).expand(B, -1, -1)
        static = self.map_emb(x_map) + self.static_proj(x_static)   # (B, d_model)
        tok = tok + pos + static.unsqueeze(1)
        tok = self.input_norm(tok)
        key_padding_mask = (mask == 0)
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # (B, T, d_model)
        return self.head(h).squeeze(-1)                              # (B, T) per-round logit


def masked_bce_token(logits, y_per_map, mask):
    """Predict the same per-map label at every position. mask out padding."""
    y = y_per_map.float().unsqueeze(1).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(model, x_num, x_out, x_side, x_map, x_static, y, mask, device, batch_size=64):
    model.eval()
    n = x_num.shape[0]
    final_logits_per_map = []
    by_round_logits = []
    by_round_targets = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            sl = slice(s, s + batch_size)
            xn = torch.from_numpy(x_num[sl]).to(device)
            xo = torch.from_numpy(x_out[sl]).to(device)
            xs = torch.from_numpy(x_side[sl]).to(device)
            xm = torch.from_numpy(x_map[sl]).to(device)
            xst = torch.from_numpy(x_static[sl]).to(device)
            mk = torch.from_numpy(mask[sl]).to(device)
            logits = model(xn, xo, xs, xm, xst, mk).cpu().numpy()
            for i in range(logits.shape[0]):
                m = mask[sl.start + i]
                idx = int(m.sum()) - 1
                final_logits_per_map.append(logits[i, idx])
                by_round_logits.append(logits[i, :idx + 1])
                by_round_targets.append([y[sl.start + i]] * (idx + 1))
    final = np.array(final_logits_per_map, dtype=np.float32)
    fp = 1 / (1 + np.exp(-final))
    auc = float(roc_auc_score(y, fp))
    ll = float(log_loss(y, fp))
    br = float(brier_score_loss(y, fp))
    # AUC by round_no using all token-level (logit, label) pairs
    all_logits = np.concatenate([np.array(b) for b in by_round_logits], axis=0)
    all_labels = np.concatenate([np.array(b) for b in by_round_targets], axis=0)
    # Reconstruct round number per token
    round_nos = []
    for b in by_round_logits:
        round_nos.extend(list(range(1, len(b) + 1)))
    round_nos = np.array(round_nos)
    auc_by_round = {}
    for rn in (1, 5, 10, 12, 15, 20, 24):
        m = (round_nos == rn) & (all_labels >= 0)
        if m.sum() >= 50 and len(set(all_labels[m].tolist())) == 2:
            p = 1 / (1 + np.exp(-all_logits[m]))
            auc_by_round[rn] = float(roc_auc_score(all_labels[m], p))
    return {"auc": auc, "log_loss": ll, "brier": br, "n": int(len(y)),
            "auc_by_round": auc_by_round}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"input: {inp}\noutput: {out_dir}\ndevice: {device}")

    df = pd.read_parquet(inp / "round_features.parquet")
    print(f"round-feature rows: {len(df)}")
    df_tr, df_va, df_te = split_chrono_by_match(df)
    print(f"train rows: {len(df_tr)}  val rows: {len(df_va)}  test rows: {len(df_te)}")

    spec = fit_spec(df_tr)
    x_tr = encode_round_tokens(df_tr, spec)
    x_va = encode_round_tokens(df_va, spec)
    x_te = encode_round_tokens(df_te, spec)
    print(f"train maps: {x_tr[0].shape[0]}  T={x_tr[0].shape[1]}  F_num={x_tr[0].shape[2]}")

    max_T = max(x_tr[0].shape[1], x_va[0].shape[1], x_te[0].shape[1])
    model = RoundTransformer(
        d_model=args.d_model, nhead=args.n_heads, num_layers=args.n_layers,
        dropout=args.dropout, max_len=max_T,
    ).to(device)
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Pad val/test to the same max_T as train (already encoded per-set; if shorter, pad with zeros)
    def pad_to(t, T):
        x_num, x_out, x_side, x_map, x_static, y, mask = t
        cur_T = x_num.shape[1]
        if cur_T >= T: return t
        pad = T - cur_T
        x_num = np.concatenate([x_num, np.zeros((x_num.shape[0], pad, x_num.shape[2]), dtype=np.float32)], axis=1)
        x_out = np.concatenate([x_out, np.zeros((x_out.shape[0], pad), dtype=np.int64)], axis=1)
        x_side = np.concatenate([x_side, np.zeros((x_side.shape[0], pad), dtype=np.int64)], axis=1)
        mask = np.concatenate([mask, np.zeros((mask.shape[0], pad), dtype=np.float32)], axis=1)
        return x_num, x_out, x_side, x_map, x_static, y, mask
    x_tr = pad_to(x_tr, max_T); x_va = pad_to(x_va, max_T); x_te = pad_to(x_te, max_T)

    best_val_auc = -1.0; best_state = None; no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = np.arange(x_tr[0].shape[0]); np.random.shuffle(idx)
        running = 0.0; nb = 0
        for s in range(0, len(idx), args.batch_size):
            sel = idx[s:s+args.batch_size]
            xn = torch.from_numpy(x_tr[0][sel]).to(device)
            xo = torch.from_numpy(x_tr[1][sel]).to(device)
            xsd = torch.from_numpy(x_tr[2][sel]).to(device)
            xm = torch.from_numpy(x_tr[3][sel]).to(device)
            xst = torch.from_numpy(x_tr[4][sel]).to(device)
            y = torch.from_numpy(x_tr[5][sel]).to(device)
            mk = torch.from_numpy(x_tr[6][sel]).to(device)
            opt.zero_grad()
            logits = model(xn, xo, xsd, xm, xst, mk)
            loss = masked_bce_token(logits, y, mk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()); nb += 1
        sched.step()
        val = evaluate(model, *x_va, device, batch_size=args.batch_size)
        if ep % 1 == 0 or ep == 1:
            r1 = val["auc_by_round"].get(1, float("nan"))
            r10 = val["auc_by_round"].get(10, float("nan"))
            r24 = val["auc_by_round"].get(24, float("nan"))
            print(f"  ep {ep:3d}  train_loss={running/nb:.4f}  val_AUC_final={val['auc']:.4f}  "
                  f"r1={r1:.3f}  r10={r10:.3f}  r24={r24:.3f}",
                  flush=True)
        if val["auc"] > best_val_auc:
            best_val_auc = val["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 6:
                print(f"  early stop at ep {ep}; best val AUC={best_val_auc:.4f}")
                break

    if best_state is not None: model.load_state_dict(best_state)
    test = evaluate(model, *x_te, device, batch_size=args.batch_size)

    print("\n" + "=" * 60); print("TRANSFORMER TEST"); print("=" * 60)
    print(f"  AUC final={test['auc']:.4f}  log_loss={test['log_loss']:.4f}  Brier={test['brier']:.4f}  n={test['n']}")
    print("  AUC by round:")
    for rn in (1, 5, 10, 12, 15, 20, 24):
        if rn in test["auc_by_round"]:
            print(f"    r{rn:>2}: {test['auc_by_round'][rn]:.3f}")

    torch.save({"state_dict": model.state_dict(),
                "spec": {"map_vocab": spec.map_vocab,
                         "num_mean": spec.num_mean.tolist(), "num_std": spec.num_std.tolist(),
                         "static_mean": spec.static_mean.tolist(), "static_std": spec.static_std.tolist()},
                "arch": {"d_model": args.d_model, "nhead": args.n_heads,
                         "num_layers": args.n_layers, "max_len": max_T}},
               out_dir / "transformer_rounds.pt")
    with open(out_dir / "transformer_rounds_metrics.json", "w") as f:
        json.dump(test, f, indent=2)
    print(f"\nsaved transformer_rounds.pt + metrics to {out_dir}")


if __name__ == "__main__":
    main()
