"""
Gap investigation: does competition between businesses (present on the test
set, mostly absent from sampled validation) cost F0.5?

Input: a feature table built for EVERY training business of one country
(build_features.py --countries India, no --limit). For the fixed validation
businesses of that country it reports macro F0.5
  A. as validation normally sees it: only validation businesses compete
  B. as the test set does: all businesses compete, then 1-to-1 reconciliation
  C. all businesses, no reconciliation
Businesses outside the model's training sample are scored out-of-sample;
those inside it are in-sample (slightly overconfident), so B is if anything
optimistic about how often the right business wins a conflict.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, DATASET_DIR, VAL_ENTITIES_PATH


def macro_f05(pred_pairs: pd.DataFrame, ents, gt_map) -> float:
    pm = pred_pairs.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
    tot = 0.0
    for e in ents:
        t, p = gt_map.get(e, set()), pm.get(e, set())
        if not t:
            tot += 0.0 if p else 1.0
            continue
        tp = len(t & p)
        if tp:
            pr, rc = tp / len(p), tp / len(t)
            tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / len(ents)


def reconcile(df):
    return df.sort_values("prob", ascending=False).drop_duplicates("candidate_entity_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--model", default="matching_model_v8.joblib")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    b = joblib.load(f"{CACHE_DIR}/{a.model}")
    df = pd.read_parquet(a.table)
    df = add_posthoc_features(df, "train")
    df["prob"] = b["model"].predict_proba(df[b["features"]].values)[:, 1]
    df = df[["source1_entity_id", "candidate_entity_id", "label", "prob"]]
    val = set(pd.read_csv(VAL_ENTITIES_PATH, header=None)[0])
    ents = [e for e in df["source1_entity_id"].unique() if e in val]
    gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    gt_map = {r.source1_entity_id: set(r.matched_entity_ids.split(",")) if r.matched_entity_ids else set()
              for r in gt.itertuples() if r.source1_entity_id in val}
    print(f"businesses in table: {df['source1_entity_id'].nunique():,}; validation businesses among them: {len(ents):,}", flush=True)
    thresholds = [a.threshold] if a.threshold else list(np.arange(0.6, 0.86, 0.04))
    for t in thresholds:
        above = df[df["prob"] >= t]
        own = above[above["source1_entity_id"].isin(val)]
        fa = macro_f05(reconcile(own), ents, gt_map)
        fb = macro_f05(reconcile(above), ents, gt_map)
        fc = macro_f05(own, ents, gt_map)
        rec_all = reconcile(above)
        lost = len(own) - rec_all["source1_entity_id"].isin(val).sum()
        print(f"threshold {t:.2f}: A validation-only {fa:.4f} | B all compete + reconcile {fb:.4f} "
              f"| C no reconcile {fc:.4f} | B-A {fb-fa:+.4f} | validation pairs lost to other businesses {lost:,}", flush=True)


if __name__ == "__main__":
    main()
