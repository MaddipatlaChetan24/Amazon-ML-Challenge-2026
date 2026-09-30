"""Package the MiniLM re-checker (cache/ce_big2_model) for the v8 classifier: v8 validation probabilities, band
0.01-0.99, honest 2-fold validation score, combiner + threshold for test."""
import os, shutil, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, joblib
from sklearn.linear_model import LogisticRegression
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import cross_encoder as ce
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, VAL_ENTITIES_PATH, macro_f05_at_threshold
from transformers import AutoTokenizer, AutoModelForSequenceClassification
LO, HI = 0.01, 0.99
OUT = f"{CACHE_DIR}/cross_encoder_big2_v8"

t0 = time.time()
val = pd.read_csv(VAL_ENTITIES_PATH, header=None)[0].tolist()
v = pq.read_table(f"{CACHE_DIR}/train_features.parquet", filters=[("source1_entity_id", "in", val)]).to_pandas()
v = add_posthoc_features(v, "train")
b = joblib.load(f"{CACHE_DIR}/matching_model_v8.joblib")
v["prob"] = b["model"].predict_proba(v[b["features"]].values)[:, 1]
v = v[["source1_entity_id", "candidate_entity_id", "label", "prob"]].reset_index(drop=True)

gt = pd.read_csv(f"{SRC}/../../../dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
n_true = dict(zip(gt.source1_entity_id, gt.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0)))
v["_n_true"] = v.source1_entity_id.map(n_true).fillna(0).astype(int)
print(f"v8 validation scored: {len(v):,} pairs ({time.time()-t0:.0f}s); v8 alone "
      f"{max(macro_f05_at_threshold(v, 'prob', 'label', t) for t in np.arange(0.5, 0.92, 0.02)):.4f}", flush=True)

m = v.prob.between(LO, HI).to_numpy()
band = v[m].reset_index(drop=True)
text = ce.raw_texts("train", set(band.source1_entity_id) | set(band.candidate_entity_id))
tok = AutoTokenizer.from_pretrained(f"{CACHE_DIR}/ce_big2_model")
mdl = AutoModelForSequenceClassification.from_pretrained(f"{CACHE_DIR}/ce_big2_model").to(ce.device()).eval()
t0 = time.time()
s = ce.score(tok, mdl, list(zip(band.source1_entity_id, band.candidate_entity_id)), text)
print(f"re-checked {len(band):,} band pairs in {time.time()-t0:.0f}s", flush=True)

X = np.c_[ce.logit(band.prob.to_numpy()), s]; y = band.label.to_numpy()
half = band.source1_entity_id.map(ce._half).to_numpy()
comb = np.zeros(len(band))
for h in (0, 1):
    comb[half == h] = LogisticRegression().fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
v["p2"] = v.prob
v.loc[m, "p2"] = comb
best = max((macro_f05_at_threshold(v, "p2", "label", t), t) for t in np.arange(0.5, 0.92, 0.02))
print(f"v8 + re-checker (band {LO}-{HI}) validation F0.5: {best[0]:.4f} at threshold {best[1]:.2f}", flush=True)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(f"{CACHE_DIR}/ce_big2_model", OUT)
joblib.dump({"combiner": LogisticRegression().fit(X, y), "threshold": float(best[1]), "band": (LO, HI),
             "validation_f05": best[0]}, f"{OUT}/combiner.joblib")
print(f"packaged {OUT}", flush=True)
