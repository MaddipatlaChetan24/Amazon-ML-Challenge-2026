"""
Export a self-contained kit for training a stronger re-checker on another GPU
machine (e.g. Kaggle): pair texts + labels, no pipeline code or caches needed.

  train_pairs.parquet  uncertain pairs of TRAINING businesses (never validation):
                       s1_text, c_text, label
  val_band.parquet     uncertain pairs of the fixed validation businesses, with
                       the classifier probability and the MiniLM re-checker score
  val_all.parquet      every validation pair: source1_entity_id, label, prob, n_true
                       (so macro F0.5 can be computed on the remote machine)
  test_band.parquet    uncertain test pairs (row index into the cached test
                       feature table, texts, probability)

Usage:
  python3 export_rechecker_kit.py --part train_val --model matching_model_v8.joblib
  python3 export_rechecker_kit.py --part test --model matching_model_v8.joblib --tag <test features tag> --member cross_encoder_big2_v8
"""
import argparse
import os
import random
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, DATASET_DIR, VAL_ENTITIES_PATH

LO, HI = 0.01, 0.99
# (not imported from cross_encoder: that pulls in torch, whose bundled OpenMP
# runtime clashing with scikit-learn's crashed the classifier's scoring here)


def raw_texts(split: str, ids=None) -> dict:
    out = {}
    for k in (1, 2, 3):
        x = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source{k}.tsv", sep="\t", keep_default_na=False,
                        usecols=["entity_id", "business_name", "business_address"])
        if ids is not None:
            x = x[x["entity_id"].isin(ids)]
        out.update(zip(x["entity_id"], x["business_name"] + " | " + x["business_address"]))
    return out


def export_train_val(model_file, out, n_train_ents, cap, scratch):
    bundle = joblib.load(f"{CACHE_DIR}/{model_file}")
    table = f"{CACHE_DIR}/train_features.parquet"
    val = set(pd.read_csv(VAL_ENTITIES_PATH, header=None)[0])
    ents = [e for e in pq.read_table(table, columns=["source1_entity_id"]).column(0).unique().to_pylist() if e not in val]
    random.seed(7)
    ents = random.sample(ents, min(n_train_ents, len(ents)))
    parts, t0 = [], time.time()
    for c in range(0, len(ents), 100000):
        df = pq.read_table(table, filters=[("source1_entity_id", "in", ents[c:c + 100000])]).to_pandas()
        df = add_posthoc_features(df, "train")
        df["prob"] = bundle["model"].predict_proba(df[bundle["features"]].values)[:, 1]
        parts.append(df.loc[df["prob"].between(LO, HI), ["source1_entity_id", "candidate_entity_id", "label"]])
        print(f"  train chunk {c // 100000 + 1}: {sum(map(len, parts)):,} uncertain pairs ({time.time()-t0:.0f}s)", flush=True)
        del df
    tr = pd.concat(parts)
    if len(tr) > cap:
        tr = tr.sample(cap, random_state=7)

    v = pd.read_parquet(f"{scratch}/v8_val_scored.parquet")
    gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    n_true = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(lambda x: len(x.split(",")) if x else 0)))
    v["n_true"] = v["source1_entity_id"].map(n_true).fillna(0).astype(int)
    band = v[v["prob"].between(LO, HI)].reset_index(drop=True)
    band["minilm"] = np.load(f"{scratch}/ce_big2_v8_val_scores.npy")

    text = raw_texts("train", set(tr["source1_entity_id"]) | set(tr["candidate_entity_id"])
                     | set(band["source1_entity_id"]) | set(band["candidate_entity_id"]))
    for d in (tr, band):
        d["s1_text"] = d["source1_entity_id"].map(text)
        d["c_text"] = d["candidate_entity_id"].map(text)
    tr[["s1_text", "c_text", "label"]].to_parquet(f"{out}/train_pairs.parquet", index=False)
    band[["source1_entity_id", "candidate_entity_id", "s1_text", "c_text", "label", "prob", "minilm"]].to_parquet(
        f"{out}/val_band.parquet", index=False)
    v[["source1_entity_id", "candidate_entity_id", "label", "prob", "n_true"]].to_parquet(f"{out}/val_all.parquet", index=False)
    print(f"train pairs {len(tr):,} (positives {tr['label'].mean()*100:.1f}%), val band {len(band):,}, val all {len(v):,}", flush=True)


def export_test(model_file, out, tag, member):
    c = np.load(f"{CACHE_DIR}/ce_scores_{member}_{tag}.npz")
    idx, minilm = c["idx"], c["score"]
    feats = pq.read_table(f"{CACHE_DIR}/test_features_{tag}.parquet").to_pandas()
    sub = feats.iloc[idx].reset_index(drop=True)
    del feats
    sub = add_posthoc_features(sub, "test")
    bundle = joblib.load(f"{CACHE_DIR}/{model_file}")
    sub["prob"] = bundle["model"].predict_proba(sub[bundle["features"]].values)[:, 1]
    text = raw_texts("test", set(sub["source1_entity_id"]) | set(sub["candidate_entity_id"]))
    pd.DataFrame({"row_idx": idx, "s1_text": sub["source1_entity_id"].map(text).values,
                  "c_text": sub["candidate_entity_id"].map(text).values, "prob": sub["prob"].values,
                  "minilm": minilm}).to_parquet(f"{out}/test_band.parquet", index=False)
    print(f"test band pairs {len(idx):,} exported", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["train_val", "test"], required=True)
    ap.add_argument("--model", default="matching_model_v8.joblib")
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "rechecker_kit"))
    ap.add_argument("--scratch", required=False, default=".")
    ap.add_argument("--train-ents", type=int, default=600000)
    ap.add_argument("--cap", type=int, default=700000)
    ap.add_argument("--tag", default="lanes15-10-10_rr40_x5-10")
    ap.add_argument("--member", default="cross_encoder_big2_v8")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.part == "train_val":
        export_train_val(a.model, a.out, a.train_ents, a.cap, a.scratch)
    else:
        export_test(a.model, a.out, a.tag, a.member)
