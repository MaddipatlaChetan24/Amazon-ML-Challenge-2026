"""
Train the matching classifier and tune a decision threshold for macro F0.5.

Deliberately simple given time constraints: one HistGradientBoostingClassifier
(sklearn, already available -- no new dependency, no install risk) and one
globally-tuned threshold, not a per-entity threshold sweep. A per-entity sweep
(fitting precision/recall from a single entity's own handful of candidates)
is statistically broken -- massively overfits to noise -- so it's not used
here even though it was suggested; see PROGRESS_AND_METHODOLOGY_LOG.md for
the reasoning.

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

    print("\nsweeping threshold for macro F0.5 on validation...", flush=True)
    best_t, best_f05 = 0.5, -1
    for t in np.arange(0.10, 0.96, 0.02):
        f05 = macro_f05_at_threshold(val_df, "prob", "label", t)
        marker = ""
        if f05 > best_f05:
            best_f05, best_t = f05, t
            marker = "  <-- best so far"
        print(f"  threshold={t:.2f}  macro_F0.5={f05:.4f}{marker}", flush=True)

    print(f"\nBEST: threshold={best_t:.2f}  macro_F0.5={best_f05:.4f}", flush=True)

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

    import joblib
    joblib.dump({"model": model, "threshold": float(best_t), "features": ALL_FEATURES},
                f"{CACHE_DIR}/matching_model.joblib")
    print(f"\nsaved model + threshold to {CACHE_DIR}/matching_model.joblib", flush=True)

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
