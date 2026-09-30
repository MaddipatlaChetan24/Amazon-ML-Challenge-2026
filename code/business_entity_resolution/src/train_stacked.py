"""
Second-stage (stacked) classifier.

Stage 1 is the model from train_model.py. Stage 2 sees every stage-1 feature
plus how each candidate's stage-1 probability compares with the other
candidates of the same Source-1 entity (rank, gap to the best, how many clear
0.5, ...). Stage-1 probabilities on the stage-2 training rows are out-of-fold
(GroupKFold by entity), so stage 2 never learns from probabilities the stage-1
model produced on its own training rows.

The validation split comes from train_model.split_indices, so scores are
directly comparable with train_model.py.
"""
import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, DATASET_DIR, macro_f05_at_threshold, split_indices

PROB_GROUP_FEATURES = ["p1", "p1_rank", "p1_max", "p1_gap_to_max", "p1_top_gap",
                       "p1_n_above_50", "p1_sum", "p1_share"]


def add_prob_group_features(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    df["p1"] = p.astype(np.float32)
    g = df.groupby("source1_entity_id")["p1"]
    df["p1_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    df["p1_max"] = g.transform("max")
    df["p1_gap_to_max"] = df["p1_max"] - df["p1"]
    # for the top candidate: its lead over the runner-up; for others: 0
    second = df["p1"].where(df["p1_rank"] == 2).groupby(df["source1_entity_id"]).transform("max").fillna(0.0)
    df["p1_top_gap"] = np.where(df["p1_rank"] == 1, df["p1"] - second, 0.0).astype(np.float32)
    df["p1_n_above_50"] = (df["p1"] >= 0.5).groupby(df["source1_entity_id"]).transform("sum").astype(np.float32)
    df["p1_sum"] = g.transform("sum")
    df["p1_share"] = (df["p1"] / df["p1_sum"].clip(lower=1e-6)).astype(np.float32)
    return df


def _hgb(args):
    return HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=args.learning_rate, max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes, l2_regularization=1.0, random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="matching_model_v5.joblib",
                    help="stage-1 model trained by train_model.py on the same split")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--learning-rate", type=float, default=0.12)
    ap.add_argument("--max-leaf-nodes", type=int, default=127)
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--out", default="matching_model_stacked.joblib")
    args = ap.parse_args()
    print(f"config: {vars(args)}", flush=True)

    stage1 = joblib.load(f"{CACHE_DIR}/{args.stage1}")
    feats1 = stage1["features"]
    df = pd.read_parquet(f"{CACHE_DIR}/train_features.parquet")
    df = add_posthoc_features(df, "train")
    groups = df["source1_entity_id"].values
    X, y = df[feats1].values, df["label"].values
    train_idx, val_idx = split_indices(groups)

    # out-of-fold stage-1 probabilities for the stage-2 training rows
    p = np.zeros(len(df), dtype=np.float32)
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(train_idx, groups=groups[train_idx])):
        t0 = time.time()
        m = _hgb(args).fit(X[train_idx[a]], y[train_idx[a]])
        p[train_idx[b]] = m.predict_proba(X[train_idx[b]])[:, 1]
        print(f"  fold {k+1}/{args.folds}: {m.n_iter_} iters, {time.time()-t0:.0f}s", flush=True)
    # validation rows: the real stage-1 model (trained on all of train_idx)
    p[val_idx] = stage1["model"].predict_proba(X[val_idx])[:, 1]
    del X

    df = add_prob_group_features(df, p)
    feats2 = feats1 + PROB_GROUP_FEATURES
    X2 = df[feats2].values
    t0 = time.time()
    m2 = _hgb(args).fit(X2[train_idx], y[train_idx])
    print(f"stage 2 trained: {m2.n_iter_} iters, {time.time()-t0:.0f}s", flush=True)

    val = df.iloc[val_idx][["source1_entity_id", "label"]].copy()
    gt = pd.read_csv(os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"), sep="\t",
                     keep_default_na=False)
    n_true = dict(zip(gt["source1_entity_id"],
                      gt["matched_entity_ids"].map(lambda x: len(x.split(",")) if x else 0)))
    val["_n_true"] = val["source1_entity_id"].map(n_true).fillna(0).astype(int)
    val["p_stage1"] = p[val_idx]
    val["p_stage2"] = m2.predict_proba(X2[val_idx])[:, 1]

    best = {}
    for col in ("p_stage1", "p_stage2"):
        bt, bf = 0.5, -1.0
        for t in np.arange(0.30, 0.96, 0.02):
            f = macro_f05_at_threshold(val, col, "label", t)
            if f > bf:
                bf, bt = f, t
        best[col] = (float(bt), bf)
        print(f"{col}: best threshold {bt:.2f}  macro_F0.5={bf:.4f}", flush=True)

    joblib.dump({"stage1": stage1, "model": m2, "features": feats2, "stage1_features": feats1,
                 "threshold": best["p_stage2"][0], "bucket_thresholds": None,
                 "validation_f05": best["p_stage2"][1]}, f"{CACHE_DIR}/{args.out}")
    print(f"saved {CACHE_DIR}/{args.out}", flush=True)


if __name__ == "__main__":
    main()
