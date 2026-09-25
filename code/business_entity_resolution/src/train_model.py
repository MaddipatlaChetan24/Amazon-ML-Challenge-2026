"""
Train the matching classifier and tune decision thresholds for macro F0.5.

One HistGradientBoostingClassifier (sklearn, already available -- no new
dependency, no install risk). Thresholds are tuned PER CANDIDATE-COUNT
BUCKET, not one single global cutoff and not a per-entity sweep. The
per-entity version (fitting precision/recall from a single entity's own
handful of candidates) is statistically broken -- massively overfits to
noise, since a typical entity has only 2-5 true matches to estimate from.
Bucketing by candidate count instead pools ALL entities that share a
similar amount of blocking "competition" (an entity with 3 candidates faces
a very different precision/recall tradeoff than one with 80), giving each
bucket a real sample size to tune from -- hundreds to hundreds of thousands
of entities, not one. Buckets with too few entities fall back to the global
threshold rather than fitting an unstable estimate from a handful of cases.

Split is by source1_entity_id (GroupShuffleSplit), not by row, so all
candidates for a given S1 entity land on the same side of the split --
otherwise validation would leak information about an entity's other
candidates into "unseen" data.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES, AMBIGUITY_FEATURE_NAMES

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
ALL_FEATURES = FEATURE_NAMES + AMBIGUITY_FEATURE_NAMES

# Candidate-count bucket edges (inclusive lower, exclusive upper), chosen to
# give each bucket a meaningfully different "how much competition does this
# entity have" profile while keeping enough entities per bucket for a
# statistically sound threshold fit. A bucket with fewer than
# MIN_BUCKET_ENTITIES falls back to the global threshold instead of fitting
# an unstable estimate.
CANDIDATE_COUNT_BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 20), (20, 50), (50, float("inf"))]
MIN_BUCKET_ENTITIES = 500


def bucket_for_count(n: int) -> tuple:
    for lo, hi in CANDIDATE_COUNT_BUCKETS:
        if lo <= n < hi:
            return (lo, hi)
    return CANDIDATE_COUNT_BUCKETS[-1]


def macro_f05_at_threshold(df: pd.DataFrame, prob_col: str, label_col: str, threshold: float) -> float:
    """Macro F0.5 per report Section 1: computed per source1_entity_id, then
    averaged across all entities in the group -- including entities with 0
    true matches (score 1.0 if predicted empty, 0.0 if any false positive)."""
    pred = (df[prob_col] >= threshold).astype(int)
    g = df.groupby("source1_entity_id")
    tp = (pred & df[label_col]).groupby(df["source1_entity_id"]).sum()
    pred_pos = pred.groupby(df["source1_entity_id"]).sum()
    true_pos = df[label_col].groupby(df["source1_entity_id"]).sum()

    precision = (tp / pred_pos).fillna(1.0)  # no predictions made -> vacuous precision 1
    precision[pred_pos == 0] = 1.0
    recall = (tp / true_pos).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    recall[true_pos == 0] = 1.0  # no true matches: recall vacuously 1

    # entities with true_pos==0 (singletons): score is 1.0 if pred_pos==0 else 0.0
    f05 = pd.Series(0.0, index=precision.index)
    denom = 0.25 * precision + recall
    nonzero = denom > 0
    f05[nonzero] = (1.25 * precision[nonzero] * recall[nonzero]) / denom[nonzero]

    is_singleton = true_pos == 0
    f05[is_singleton] = np.where(pred_pos[is_singleton] == 0, 1.0, 0.0)

    return f05.mean()


def macro_f05_at_threshold_precomputed(df: pd.DataFrame, pred_col: str = "_pred", label_col: str = "label") -> float:
    """Same aggregation as macro_f05_at_threshold, but takes an already-
    computed 0/1 prediction column instead of a single scalar threshold --
    needed once predictions come from per-bucket thresholds (a different
    cutoff per row) rather than one global cutoff."""
    pred = df[pred_col]
    tp = (pred & df[label_col]).groupby(df["source1_entity_id"]).sum()
    pred_pos = pred.groupby(df["source1_entity_id"]).sum()
    true_pos = df[label_col].groupby(df["source1_entity_id"]).sum()

    precision = (tp / pred_pos).fillna(1.0)
    precision[pred_pos == 0] = 1.0
    recall = (tp / true_pos).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    recall[true_pos == 0] = 1.0

    f05 = pd.Series(0.0, index=precision.index)
    denom = 0.25 * precision + recall
    nonzero = denom > 0
    f05[nonzero] = (1.25 * precision[nonzero] * recall[nonzero]) / denom[nonzero]

    is_singleton = true_pos == 0
    f05[is_singleton] = np.where(pred_pos[is_singleton] == 0, 1.0, 0.0)

    return f05.mean()


def main():
    print("loading train_features.parquet...", flush=True)
    df = pd.read_parquet(f"{CACHE_DIR}/train_features.parquet")
    print(f"  {len(df):,} rows, label balance: {df['label'].mean()*100:.3f}% positive", flush=True)

    X = df[ALL_FEATURES].values
    y = df["label"].values
    groups = df["source1_entity_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    print(f"  train pairs: {len(train_idx):,}  val pairs: {len(val_idx):,}", flush=True)

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X[train_idx], y[train_idx])
    print(f"  trained in {time.time()-t0:.1f}s, {model.n_iter_} iterations", flush=True)

    val_df = df.iloc[val_idx][["source1_entity_id", "candidate_entity_id", "label"]].copy()
    val_df["prob"] = model.predict_proba(X[val_idx])[:, 1]

    print("\nsweeping GLOBAL threshold for macro F0.5 on validation...", flush=True)
    best_t, best_f05 = 0.5, -1
    for t in np.arange(0.10, 0.96, 0.02):
        f05 = macro_f05_at_threshold(val_df, "prob", "label", t)
        marker = ""
        if f05 > best_f05:
            best_f05, best_t = f05, t
            marker = "  <-- best so far"
        print(f"  threshold={t:.2f}  macro_F0.5={f05:.4f}{marker}", flush=True)

    print(f"\nBEST GLOBAL: threshold={best_t:.2f}  macro_F0.5={best_f05:.4f}", flush=True)

    print("\ntuning PER-CANDIDATE-COUNT-BUCKET thresholds...", flush=True)
    group_size = val_df.groupby("source1_entity_id")["label"].transform("size")
    val_df["_bucket"] = group_size.apply(bucket_for_count)
    bucket_thresholds = {}
    for bucket in CANDIDATE_COUNT_BUCKETS:
        bucket_df = val_df[val_df["_bucket"] == bucket]
        n_entities = bucket_df["source1_entity_id"].nunique()
        if n_entities < MIN_BUCKET_ENTITIES:
            bucket_thresholds[bucket] = best_t
            print(f"  bucket {bucket}: only {n_entities:,} entities (<{MIN_BUCKET_ENTITIES}) "
                  f"-- falling back to global threshold {best_t:.2f}", flush=True)
            continue
        bt, bf05 = 0.5, -1
        for t in np.arange(0.10, 0.96, 0.02):
            f05 = macro_f05_at_threshold(bucket_df, "prob", "label", t)
            if f05 > bf05:
                bf05, bt = f05, t
        bucket_thresholds[bucket] = float(bt)
        print(f"  bucket {bucket}: {n_entities:,} entities, threshold={bt:.2f}, macro_F0.5={bf05:.4f}", flush=True)

    # Combined: apply each entity's bucket-specific threshold, measure overall
    val_df["_threshold"] = val_df["_bucket"].map(bucket_thresholds)
    val_df["_pred"] = (val_df["prob"] >= val_df["_threshold"]).astype(int)
    combined_f05 = macro_f05_at_threshold_precomputed(val_df)
    print(f"\nCOMBINED (per-bucket) macro_F0.5={combined_f05:.4f} vs. GLOBAL-only={best_f05:.4f}", flush=True)

    # Breakdown: singleton-only and non-singleton-only, at the chosen threshold
    true_pos_per_entity = val_df.groupby("source1_entity_id")["label"].sum()
    singleton_ids = true_pos_per_entity[true_pos_per_entity == 0].index
    val_singletons = val_df[val_df["source1_entity_id"].isin(singleton_ids)]
    val_nonsingletons = val_df[~val_df["source1_entity_id"].isin(singleton_ids)]
    if len(val_singletons):
        print(f"singleton-only macro F0.5 @ {best_t:.2f}: "
              f"{macro_f05_at_threshold(val_singletons, 'prob', 'label', best_t):.4f} "
              f"(n={val_singletons['source1_entity_id'].nunique():,} entities)", flush=True)
    if len(val_nonsingletons):
        print(f"non-singleton macro F0.5 @ {best_t:.2f}: "
              f"{macro_f05_at_threshold(val_nonsingletons, 'prob', 'label', best_t):.4f} "
              f"(n={val_nonsingletons['source1_entity_id'].nunique():,} entities)", flush=True)
    if len(val_singletons):
        print(f"singleton-only macro F0.5, per-bucket thresholds: "
              f"{macro_f05_at_threshold_precomputed(val_singletons):.4f}", flush=True)
    if len(val_nonsingletons):
        print(f"non-singleton macro F0.5, per-bucket thresholds: "
              f"{macro_f05_at_threshold_precomputed(val_nonsingletons):.4f}", flush=True)

    import joblib
    joblib.dump(
        {
            "model": model,
            "threshold": float(best_t),  # kept for backward compatibility / fallback
            "bucket_thresholds": bucket_thresholds,
            "candidate_count_buckets": CANDIDATE_COUNT_BUCKETS,
            "features": ALL_FEATURES,
        },
        f"{CACHE_DIR}/matching_model.joblib",
    )
    print(f"\nsaved model + thresholds to {CACHE_DIR}/matching_model.joblib", flush=True)

    # feature importances (permutation is too slow here; use the model's own)
    try:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print("\ntop features:", flush=True)
        for i in order[:15]:
            print(f"  {ALL_FEATURES[i]:30s} {importances[i]:.4f}", flush=True)
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
