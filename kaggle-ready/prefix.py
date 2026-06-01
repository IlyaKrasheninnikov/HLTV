"""Predict the map winner using only the first N rounds.

For every N in 0, 1, 2, ..., 30 we:
  1. Aggregate the round-level data into one row per map containing:
     - pre-match features (rank diff, points diff, team rolling stats — joined
       from permap_features.parquet)
     - prefix summary features computed from the FIRST N rounds:
         * current_score_t1, current_score_t2, score_diff
         * count of CT/T wins, bomb defuses, bomb explosions
         * pistol_r1_t1_wins (if N >= 1), pistol_r13_t1_wins (if N >= 13)
         * last-3 outcome indicators
         * rolling side win rate in the prefix
         * mean equipment value t1/t2 over the prefix
         * rounds_played (= min(N, total_played_rounds_on_map))
  2. Train LightGBM binary classifier on chronological split.
  3. Report test AUC / log-loss / Brier.

Outputs:
  prefix_lgbm_N{n}.txt        — booster per N (saves only key Ns to save disk)
  prefix_summary.csv          — one row per N with metrics
  prefix_summary.json         — same

Usage on Kaggle:
    !python /kaggle/input/<slug>/prefix.py \
        --input /kaggle/input/<slug>/ --output /kaggle/working/out \
        --ns 0,1,2,3,4,6,8,10,12,15,18,21,24

The script auto-detects whether each N has enough samples (some maps end
before N rounds are played; we use min(N, actual_played) per map).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# Static per-match features we carry forward from permap_features.
PRE_FEATS = [
    "diff_rank", "diff_points", "diff_rating", "diff_onmap_win_rate",
    "diff_avg_player_rating", "diff_avg_player_adr",
    "diff_ct_wr", "diff_t_wr", "diff_rest_days",
    "stars", "t1_points", "t2_points", "t1_rank", "t2_rank",
    "t1_pistol_wr", "t2_pistol_wr", "diff_pistol_wr",
    "t1_avg_rounds_for", "t2_avg_rounds_for",
    "diff_avg_rounds_for", "diff_avg_rounds_against",
    "event_prize_pool_usd", "event_num_attending",
    "event_has_prize", "is_overtime_map_hint",  # last one is NOT a leak; computed below as 0 (we don't know yet)
]
CAT_FEATS = ["format", "event_type", "map_name", "map_picked_by"]


def split_chrono(df, val_frac=0.15, test_frac=0.15):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df); nt = int(n*test_frac); nv = int(n*val_frac)
    return df.iloc[:n-nv-nt].copy(), df.iloc[n-nv-nt:n-nt].copy(), df.iloc[n-nt:].copy()


def build_prefix_dataset(df_round: pd.DataFrame, df_map: pd.DataFrame, N: int) -> pd.DataFrame:
    """For each (match_id, mapstats_id), aggregate the first N rounds and join
    with the pre-match feature row."""
    # Sort rounds correctly
    df_round = df_round.sort_values(["match_id", "mapstats_id", "round_no"])

    # Take first N rounds per map (or fewer if the map ended sooner)
    prefix = df_round.groupby(["match_id", "mapstats_id"]).head(N) if N > 0 else \
             df_round.groupby(["match_id", "mapstats_id"]).head(0)

    # Per-map prefix aggregates
    def agg(g: pd.DataFrame) -> pd.Series:
        n_rounds_observed = len(g)
        if n_rounds_observed == 0:
            return pd.Series({
                "prefix_n_observed": 0,
                "prefix_score_t1": 0, "prefix_score_t2": 0, "prefix_score_diff": 0,
                "prefix_ct_wins": 0, "prefix_t_wins": 0,
                "prefix_bomb_defused": 0, "prefix_bomb_exploded": 0,
                "prefix_pistol_r1_t1": np.nan, "prefix_pistol_r13_t1": np.nan,
                "prefix_eq_t1_mean": np.nan, "prefix_eq_t2_mean": np.nan,
                "prefix_eq_diff_mean": np.nan,
                "prefix_last1_t1_wins": np.nan,
                "prefix_last3_t1_winrate": np.nan,
                "prefix_t1_ct_wr": np.nan, "prefix_t1_t_wr": np.nan,
                "prefix_period_max": 0,
            })
        # current cumulative score = score after last observed round
        last = g.iloc[-1]
        s1 = last["score_t1"] or 0
        s2 = last["score_t2"] or 0
        # win counts by outcome type
        ct = int((g["side_winner"] == "CT").sum())
        t = int((g["side_winner"] == "T").sum())
        defused = int((g["outcome"] == "bomb_defused").sum())
        exploded = int((g["outcome"] == "bomb_exploded").sum())
        # pistol round outcomes if present
        p1 = np.nan
        if (g["round_no"] == 1).any():
            r1 = g[g["round_no"] == 1].iloc[0]
            p1 = float(r1.get("pistol_won_t1", 0))
        p13 = np.nan
        if (g["round_no"] == 13).any():
            r13 = g[g["round_no"] == 13].iloc[0]
            p13 = float(r13.get("pistol_won_t1", 0))
        # equipment averages (skip missing)
        eq1 = pd.to_numeric(g["eq_value_t1"], errors="coerce")
        eq2 = pd.to_numeric(g["eq_value_t2"], errors="coerce")
        # last 1 / last 3 round outcomes from team1's perspective: did t1 win?
        # We derive "team1 won the round" by score deltas of cumulative score.
        s1s = pd.to_numeric(g["score_t1"], errors="coerce").to_numpy()
        s2s = pd.to_numeric(g["score_t2"], errors="coerce").to_numpy()
        # delta_s1[i] = s1s[i] - s1s[i-1] (1 if t1 won that round else 0); first round uses raw value
        if len(s1s) > 0:
            prev1 = np.concatenate([[0], s1s[:-1]])
            prev2 = np.concatenate([[0], s2s[:-1]])
            t1_won_each = (s1s - prev1).astype(int)   # 0/1
            t2_won_each = (s2s - prev2).astype(int)
        else:
            t1_won_each = t2_won_each = np.array([], dtype=int)
        last1_t1 = int(t1_won_each[-1]) if len(t1_won_each) >= 1 else np.nan
        last3_wr = float(t1_won_each[-3:].mean()) if len(t1_won_each) >= 3 else \
                   (float(t1_won_each.mean()) if len(t1_won_each) > 0 else np.nan)
        # CT/T win rate from team1's perspective is awkward to derive without team-side info,
        # so we approximate it via halves: in first half (round_no <= 12), team1 plays one side;
        # second half they swap. We don't know which side team1 started, so leave these as
        # half-rate approximations:
        t1_first_half_wr = float(t1_won_each[: min(12, len(t1_won_each))].mean()) \
                            if len(t1_won_each) > 0 else np.nan
        t1_second_half_wr = float(t1_won_each[12:24].mean()) if len(t1_won_each) > 12 else np.nan
        return pd.Series({
            "prefix_n_observed": int(n_rounds_observed),
            "prefix_score_t1": int(s1),
            "prefix_score_t2": int(s2),
            "prefix_score_diff": int(s1 - s2),
            "prefix_ct_wins": ct,
            "prefix_t_wins": t,
            "prefix_bomb_defused": defused,
            "prefix_bomb_exploded": exploded,
            "prefix_pistol_r1_t1": p1,
            "prefix_pistol_r13_t1": p13,
            "prefix_eq_t1_mean": float(eq1.mean()) if eq1.notna().any() else np.nan,
            "prefix_eq_t2_mean": float(eq2.mean()) if eq2.notna().any() else np.nan,
            "prefix_eq_diff_mean": float((eq1 - eq2).mean()) if (eq1.notna() & eq2.notna()).any() else np.nan,
            "prefix_last1_t1_wins": last1_t1,
            "prefix_last3_t1_winrate": last3_wr,
            "prefix_first_half_t1_wr": t1_first_half_wr,
            "prefix_second_half_t1_wr": t1_second_half_wr,
            "prefix_period_max": int(g["period"].max()) if "period" in g.columns else 0,
        })

    if N == 0:
        # Build all-zero/NaN prefix dataframe with the right shape using the
        # full set of map keys.
        keys = df_round[["match_id", "mapstats_id"]].drop_duplicates()
        n_keys = len(keys)
        zeros = pd.DataFrame(0, index=range(n_keys), columns=[
            "prefix_n_observed","prefix_score_t1","prefix_score_t2","prefix_score_diff",
            "prefix_ct_wins","prefix_t_wins","prefix_bomb_defused","prefix_bomb_exploded",
        ])
        nans = pd.DataFrame(np.nan, index=range(n_keys), columns=[
            "prefix_pistol_r1_t1","prefix_pistol_r13_t1",
            "prefix_eq_t1_mean","prefix_eq_t2_mean","prefix_eq_diff_mean",
            "prefix_last1_t1_wins","prefix_last3_t1_winrate",
            "prefix_first_half_t1_wr","prefix_second_half_t1_wr",
        ])
        zeros["prefix_period_max"] = 0
        agg_df = pd.concat([keys.reset_index(drop=True), zeros, nans], axis=1)
    else:
        agg_df = prefix.groupby(["match_id", "mapstats_id"]).apply(agg).reset_index()

    # Join with pre-match features on (match_id, mapstats_id)
    keep_cols = ["match_id", "mapstats_id", "date", "y_team1_wins_map"] + \
                CAT_FEATS + PRE_FEATS
    keep_cols = [c for c in keep_cols if c in df_map.columns]
    base = df_map[keep_cols].drop_duplicates(["match_id", "mapstats_id"])
    out = base.merge(agg_df, on=["match_id", "mapstats_id"], how="inner")
    return out


def to_xy(d: pd.DataFrame, cats):
    drop = {"match_id","mapstats_id","date","y_team1_wins_map"}
    y = d["y_team1_wins_map"].astype(int)
    X = d.drop(columns=[c for c in drop if c in d.columns])
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if col in cats: continue
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def train_one_N(df_full: pd.DataFrame, N: int, out_dir: Path, save_model: bool = False):
    tr, va, te = split_chrono(df_full)
    X_tr, y_tr = to_xy(tr, CAT_FEATS)
    X_va, y_va = to_xy(va, CAT_FEATS)
    X_te, y_te = to_xy(te, CAT_FEATS)
    cats_in = [c for c in CAT_FEATS if c in X_tr.columns]
    params = dict(objective="binary", metric=["binary_logloss","auc"],
                  learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
                  feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                  lambda_l2=1.0, verbose=-1)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats_in)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cats_in)
    m = lgb.train(params, dtr, num_boost_round=2000,
                  valid_sets=[dva], valid_names=["va"],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p_te = m.predict(X_te, num_iteration=m.best_iteration)
    baseline = log_loss(y_te, np.full_like(y_te, y_te.mean(), dtype=float))
    out = {
        "N": int(N),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "log_loss": float(log_loss(y_te, p_te)),
        "auc": float(roc_auc_score(y_te, p_te)),
        "brier": float(brier_score_loss(y_te, p_te)),
        "baseline_logloss": float(baseline),
        "best_iter": int(m.best_iteration),
        "n_features": int(X_tr.shape[1]),
    }
    if save_model:
        m.save_model(str(out_dir / f"prefix_lgbm_N{N}.txt"), num_iteration=m.best_iteration)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./")
    ap.add_argument("--output", default=None)
    ap.add_argument("--ns", default="0,1,2,3,4,5,6,8,10,12,14,16,18,20,22,24",
                    help="comma-separated prefix lengths")
    ap.add_argument("--save-models", action="store_true",
                    help="save the LightGBM booster per N (default: only save metrics)")
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    out_dir = Path(args.output).resolve() if args.output else inp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"input: {inp}\noutput: {out_dir}")

    df_round = pd.read_parquet(inp / "round_features.parquet")
    df_map = pd.read_parquet(inp / "permap_features.parquet")
    df_map = df_map[df_map[["t1_n","t2_n"]].fillna(0).max(axis=1) > 0]
    print(f"rounds: {len(df_round)}  maps: {len(df_map)}")

    ns = sorted({int(x) for x in args.ns.split(",")})
    results = []
    print(f"\n{'N':>3}  {'AUC':>6}  {'LogLoss':>7}  {'Brier':>6}  {'BaseLL':>6}  {'best_iter':>9}  {'n_test':>6}")
    print("-" * 65)
    for N in ns:
        df_pref = build_prefix_dataset(df_round, df_map, N)
        res = train_one_N(df_pref, N, out_dir, save_model=args.save_models)
        results.append(res)
        print(f"{N:>3}  {res['auc']:.4f}  {res['log_loss']:.4f}  "
              f"{res['brier']:.4f}  {res['baseline_logloss']:.4f}  "
              f"{res['best_iter']:>9}  {res['n_test']:>6}", flush=True)

    pd.DataFrame(results).to_csv(out_dir / "prefix_summary.csv", index=False)
    with open(out_dir / "prefix_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved prefix_summary.csv + .json to {out_dir}")


if __name__ == "__main__":
    main()
