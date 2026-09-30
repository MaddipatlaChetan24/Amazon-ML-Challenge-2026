"""
Self-training of the re-checker on France, the country with no training labels.

1. Combine classifier + MiniLM re-checker scores on the exported uncertain test
   pairs; France pairs the combination is very sure about become pseudo-labels
   (>= HI: match, <= LO: non-match).
2. Fine-tune the MiniLM re-checker on them, mixed with labelled training pairs
   so US/India behaviour is kept.
3. Safety check on the fixed US/India validation businesses: report F0.5 with
   the fine-tuned model (must stay close to the current best).
4. Re-score every exported test pair and write a prediction-time score cache +
   member directory, so generate_predictions.py can build the file without
   touching the GPU again.

Uses only the provided test inputs (no external data). Loads torch + small
logistic models only (no gradient-boosting model in this process).
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import cross_encoder as ce
from train_model import CACHE_DIR, DATASET_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--kit", default=f"{CACHE_DIR}/rechecker_kit")
ap.add_argument("--test-kit", default=f"{CACHE_DIR}/rechecker_kit_test")
ap.add_argument("--tag", default="lanes15-10-10_rr40_x5-10_rv3")
ap.add_argument("--base", default="cross_encoder_big2_v8")
ap.add_argument("--out-member", default="cross_encoder_frst_v8")
ap.add_argument("--hi", type=float, default=0.98)
ap.add_argument("--lo", type=float, default=0.02)
ap.add_argument("--per-class", type=int, default=150000)
ap.add_argument("--labelled", type=int, default=300000)
a = ap.parse_args()
t0 = time.time()

tb = pd.read_parquet(f"{a.test_kit}/test_band.parquet")
feat_path = f"{CACHE_DIR}/test_features_{a.tag}.parquet"
n_rows = pq.ParquetFile(feat_path).metadata.num_rows
s1_all = pq.read_table(feat_path, columns=["source1_entity_id"]).column(0).to_numpy()
tb["s1"] = s1_all[tb.row_idx.to_numpy()]
del s1_all
s1c = pd.read_csv(f"{DATASET_DIR}/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"], keep_default_na=False)
tb["country"] = tb.s1.map(dict(zip(s1c.entity_id, s1c.country)))
meta = joblib.load(f"{CACHE_DIR}/{a.base}/combiner.joblib")
tb["comb"] = ce.combine(tb.prob.to_numpy(), tb.minilm.to_numpy(), meta["combiner"])
print(f"test band {len(tb):,} pairs; by country {tb.country.value_counts().to_dict()} ({time.time()-t0:.0f}s)", flush=True)

fr = tb[tb.country == "France"]
pos = fr[fr.comb >= a.hi]; neg = fr[fr.comb <= a.lo]
pos = pos.sample(min(a.per_class, len(pos)), random_state=0); neg = neg.sample(min(a.per_class, len(neg)), random_state=0)
kit = pd.read_parquet(f"{a.kit}/train_pairs.parquet").sample(a.labelled, random_state=3)
train = pd.concat([pos.assign(label=1)[["s1_text", "c_text", "label"]], neg.assign(label=0)[["s1_text", "c_text", "label"]],
                   kit[["s1_text", "c_text", "label"]]]).reset_index(drop=True)
print(f"France pseudo-labels: {len(pos):,} match / {len(neg):,} non-match (of {len(fr):,} France band pairs); "
      f"+ {len(kit):,} labelled training pairs", flush=True)

text = {f"a{i}": t for i, t in enumerate(train.s1_text)}
text.update({f"b{i}": t for i, t in enumerate(train.c_text)})
tok, m = ce.train([(f"a{i}", f"b{i}") for i in range(len(train))], train.label.to_numpy(), text,
                  epochs=1, lr=1e-5, seed=5, base=f"{CACHE_DIR}/{a.base}")
print(f"fine-tuned in {time.time()-t0:.0f}s", flush=True)

# safety check on US/India validation
vb = pd.read_parquet(f"{a.kit}/val_band.parquet")
vt = {f"va{i}": t for i, t in enumerate(vb.s1_text)}; vt.update({f"vb{i}": t for i, t in enumerate(vb.c_text)})
vs = ce.score(tok, m, [(f"va{i}", f"vb{i}") for i in range(len(vb))], vt)
va = pd.read_parquet(f"{a.kit}/val_all.parquet")
pos_in_all = pd.Series(np.arange(len(va)), index=(va.source1_entity_id + "|" + va.candidate_entity_id).values).reindex(
    (vb.source1_entity_id + "|" + vb.candidate_entity_id).values).to_numpy()
nt = va.groupby("source1_entity_id").n_true.first()


def macro_f05(p, t):
    pred = (p >= t).astype(int)
    agg = pd.DataFrame({"e": va.source1_entity_id.values, "pp": pred, "tp": pred * va.label.values}).groupby("e").sum()
    n = nt.reindex(agg.index).values
    pr = np.where(agg.pp > 0, agg.tp / np.maximum(agg.pp, 1), 1.0)
    rc = np.where(n > 0, agg.tp / np.maximum(n, 1), 1.0)
    f = np.where((0.25 * pr + rc) > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    return np.where(n == 0, (agg.pp.values == 0).astype(float), f).mean()


half = vb.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).to_numpy()
X = np.c_[ce.logit(vb.prob.to_numpy()), vs]; y = vb.label.to_numpy()
comb = np.zeros(len(vb))
for h in (0, 1):
    comb[half == h] = LogisticRegression(max_iter=1000).fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
p = va.prob.to_numpy().copy(); p[pos_in_all] = comb
best = max((macro_f05(p, t), t) for t in np.arange(0.5, 0.92, 0.02))
print(f"US/India validation with the self-trained re-checker: {best[0]:.4f} at threshold {best[1]:.2f} "
      f"(current best 0.9763)", flush=True)

# re-score every exported test pair; write member dir + prediction-time score cache
t1 = time.time()
tt = {f"ta{i}": t for i, t in enumerate(tb.s1_text)}; tt.update({f"tb{i}": t for i, t in enumerate(tb.c_text)})
ts = ce.score(tok, m, [(f"ta{i}", f"tb{i}") for i in range(len(tb))], tt)
print(f"re-scored {len(tb):,} test pairs in {time.time()-t1:.0f}s", flush=True)
out = f"{CACHE_DIR}/{a.out_member}"
if os.path.exists(out):
    shutil.rmtree(out)
m.save_pretrained(out); tok.save_pretrained(out)
joblib.dump({"combiner": LogisticRegression(max_iter=1000).fit(X, y), "threshold": float(best[1]),
             "band": meta["band"], "validation_f05": best[0]}, f"{out}/combiner.joblib")
np.savez(f"{CACHE_DIR}/ce_scores_{a.out_member}_{a.tag}.npz", idx=tb.row_idx.to_numpy(), score=ts, n=np.array([n_rows]))
print(f"saved {out} and its test score cache ({time.time()-t0:.0f}s total)", flush=True)
