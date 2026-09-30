"""
Final stage for v11 (learned-search candidates): fitted on the validation band
of the rebuilt validation table (valdense10), applied to the test table dumped
by generate_predictions.py --dump-probs. mpnet scores exist only for pairs
already exported to Kaggle; elsewhere the classifier + MiniLM score is used,
exactly as in validation.
"""
import argparse, hashlib, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
import final_stage as fs
from generate_predictions import pair_keys
from train_model import CACHE_DIR, DATASET_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--val-tag", default="valdense10")
ap.add_argument("--probs", required=True)
ap.add_argument("--old-tag", default="lanes15-10-10_rr40_x5-10_rv3")
ap.add_argument("--mpnet-test", default=f"{CACHE_DIR}/mpnet_test_scores.npy")
ap.add_argument("--out", default=f"{CACHE_DIR}/final_probs_v11.npz")
a = ap.parse_args()
K = fs.K

# ---- validation band of the rebuilt table ----
d = pd.read_parquet(f"{CACHE_DIR}/{a.val_tag}_pairs.parquet")
vb = pd.read_parquet(f"{K}/val_band.parquet", columns=["source1_entity_id", "candidate_entity_id", "minilm"])
vb["mp"] = np.load(f"{K}/mpnet_val_scores.npy")
m = np.load(f"{CACHE_DIR}/{a.val_tag}_minilm.npz", allow_pickle=True)
extra = pd.DataFrame({"source1_entity_id": m["s1"], "candidate_entity_id": m["c"], "minilm": m["score"], "mp": np.nan})
sc = pd.concat([vb, extra]).drop_duplicates(["source1_entity_id", "candidate_entity_id"])
band = d[(d.prob >= 0.01) & (d.prob <= 0.99)].merge(sc, on=["source1_entity_id", "candidate_entity_id"], how="left")
assert band.minilm.notna().all()
y = band.label.to_numpy()
X2 = np.c_[fs.lg(band.prob.to_numpy()), band.minilm]
mpv = band.mp.to_numpy(); has = ~np.isnan(mpv)
half = band.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).to_numpy()


def stage(fi):
    m2 = fs.lr().fit(X2[fi], y[fi]); s2 = m2.predict_proba(X2)[:, 1]
    narrow = (s2 > fs.NARROW_LO) & (s2 < fs.NARROW_HI) & has
    X3 = np.c_[X2, np.nan_to_num(mpv)]
    m3 = fs.lr().fit(X3[fi & narrow], y[fi & narrow]); s3 = m3.predict_proba(X3)[:, 1]
    G = fs.group_features(band.source1_entity_id.values, band.prob.to_numpy(), band.minilm.to_numpy(), s2, s3, np.nan_to_num(mpv), narrow)
    mg = fs.hgb().fit(G[fi & narrow], y[fi & narrow])
    return np.where(narrow, mg.predict_proba(G)[:, 1], s2), (m2, m3, mg)


oof = np.zeros(len(band))
for h in (0, 1):
    out, _ = stage(half != h); oof[half == h] = out[half == h]
_, (m2, m3, mg) = stage(np.ones(len(band), bool))
gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
nt = pd.Series(gt.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).values, index=gt.source1_entity_id)
ents = d.source1_entity_id.unique(); ntv = nt.reindex(ents).values


def f05(prob, t):
    x = d[["source1_entity_id", "candidate_entity_id", "label"]].copy(); x["p"] = prob
    x = x[x.p >= t].sort_values("p", ascending=False).drop_duplicates("candidate_entity_id")
    s = x.groupby("source1_entity_id").agg(pp=("label", "size"), tp=("label", "sum")).reindex(ents).fillna(0)
    pr = np.where(s.pp > 0, s.tp / np.maximum(s.pp, 1), 1.0); rc = np.where(ntv > 0, s.tp / np.maximum(ntv, 1), 1.0)
    f = np.where(ntv == 0, (s.pp == 0).astype(float), np.where(s.tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0))
    return f.mean()


pos = pd.Series(np.arange(len(d)), index=(d.source1_entity_id + "|" + d.candidate_entity_id).values).reindex(
    (band.source1_entity_id + "|" + band.candidate_entity_id).values).to_numpy()
p = d.prob.to_numpy().copy(); p[pos] = oof
f, thr = max((f05(p, t), t) for t in np.arange(0.5, 0.92, 0.02))
print(f"validation ({a.val_tag}) final stage: {f:.4f} at threshold {thr:.2f}", flush=True)

# ---- test ----
t = np.load(a.probs, allow_pickle=True)
p0, ce, s1, cand = t["p0"], t["ce"], t["s1"], t["cand"]
bi = np.flatnonzero((p0 >= 0.01) & (p0 <= 0.99))
assert np.isfinite(ce[bi]).all(), "re-checker score missing for some band pairs"
nar = pd.read_parquet(f"{CACHE_DIR}/kaggle_test_upload/test_band.parquet", columns=["row_idx"])
ts = np.load(a.mpnet_test); assert len(ts) == len(nar)
ot = pq.read_table(f"{CACHE_DIR}/test_features_{a.old_tag}.parquet", columns=["source1_entity_id", "candidate_entity_id"]).take(nar.row_idx.to_numpy())
mp = pd.Series(ts, index=pair_keys(ot.column(0), ot.column(1))).reindex(pair_keys(s1[bi], cand[bi])).to_numpy()
TX2 = np.c_[fs.lg(p0[bi]), ce[bi]]
s2 = m2.predict_proba(TX2)[:, 1]
narrow = ~np.isnan(mp) & (s2 > fs.NARROW_LO) & (s2 < fs.NARROW_HI)
s3 = np.full(len(bi), np.nan)
s3[narrow] = m3.predict_proba(np.c_[TX2, np.nan_to_num(mp)][narrow])[:, 1]
G = fs.group_features(s1[bi], p0[bi], ce[bi], s2, s3, np.nan_to_num(mp), narrow)
fin = s2.copy(); fin[narrow] = mg.predict_proba(G[narrow])[:, 1]
prob = p0.copy(); prob[bi] = fin
print(f"test: {len(p0):,} pairs, band {len(bi):,}, mpnet used on {narrow.sum():,}", flush=True)
np.savez(a.out, prob=prob, threshold=np.array([thr]))
print(f"saved {a.out}", flush=True)
