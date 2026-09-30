"""
final_stage.py for a new test table (e.g. with learned-search extras): band
rows and scores come from generate_predictions.py --dump-probs; mpnet scores
are matched by (business, record) key from the exported narrowed test pairs.
"""
import argparse, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
import final_stage as fs
from generate_predictions import pair_keys
from train_model import CACHE_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True)
ap.add_argument("--mpnet-test", default=f"{CACHE_DIR}/mpnet_test_scores.npy")
ap.add_argument("--old-tag", default="lanes15-10-10_rr40_x5-10_rv3")
ap.add_argument("--out", default=f"{CACHE_DIR}/final_probs_v11.npz")
a = ap.parse_args()

vb = pd.read_parquet(f"{fs.K}/val_band.parquet"); vs = np.load(fs.K + "/mpnet_val_scores.npy")
oof, (m2, m3, mg) = fs.fit_all(vb, vs)
va = pd.read_parquet(f"{fs.K}/val_all.parquet")
pos = pd.Series(np.arange(len(va)), index=(va.source1_entity_id + "|" + va.candidate_entity_id).values).reindex(
    (vb.source1_entity_id + "|" + vb.candidate_entity_id).values).to_numpy()
p = va.prob.to_numpy().copy(); p[pos] = oof
f05 = fs.macro_f05_factory(va)
f, thr = max((f05(p, t), t) for t in np.arange(0.4, 0.94, 0.02))
print(f"validation {f:.4f} at threshold {thr:.2f}", flush=True)

d = np.load(a.probs, allow_pickle=True)
p0, ce, s1, cand = d["p0"], d["ce"], d["s1"], d["cand"]
band = np.flatnonzero((p0 >= 0.01) & (p0 <= 0.99))
nar = pd.read_parquet(f"{CACHE_DIR}/kaggle_test_upload/test_band.parquet", columns=["row_idx"])
ts = np.load(a.mpnet_test); assert len(ts) == len(nar)
ot = pq.read_table(f"{CACHE_DIR}/test_features_{a.old_tag}.parquet", columns=["source1_entity_id", "candidate_entity_id"]).take(nar.row_idx.to_numpy())
mpk = pd.Series(ts, index=pair_keys(ot.column(0), ot.column(1)))
mp = mpk.reindex(pair_keys(s1[band], cand[band])).to_numpy()
assert np.isfinite(ce[band]).all()
X2 = np.c_[fs.lg(p0[band]), ce[band]]
s2 = m2.predict_proba(X2)[:, 1]
narrow = ~np.isnan(mp) & (s2 > fs.NARROW_LO) & (s2 < fs.NARROW_HI)
print(f"band {len(band):,}; with mpnet {np.isfinite(mp).sum():,}; narrow used {narrow.sum():,}", flush=True)
s3 = np.full(len(band), np.nan)
s3[narrow] = m3.predict_proba(np.c_[X2, np.nan_to_num(mp)][narrow])[:, 1]
G = fs.group_features(s1[band], p0[band], ce[band], s2, s3, np.nan_to_num(mp), narrow)
fin = s2.copy(); fin[narrow] = mg.predict_proba(G[narrow])[:, 1]
prob = p0.copy(); prob[band] = fin
np.savez(a.out, prob=prob, threshold=np.array([thr]))
print(f"saved {a.out}", flush=True)
