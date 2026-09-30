"""
Rescue final stage: re-decides the uncertain band plus low-probability
learned-search top-5 candidates (re-checked by MiniLM), using the current final
probability, re-checker scores, and the learned-search similarity and rank.
Fitted on validation (valdense10), applied to the test table.

  python3 final_stage_rescue.py --probs ../cache/v11_probs.npz --final ../cache/final_probs_v11.npz
"""
import argparse, hashlib, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
import final_stage as fs
from generate_predictions import pair_keys
from train_model import CACHE_DIR, DATASET_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--probs", required=True); ap.add_argument("--final", required=True)
ap.add_argument("--val-oof", default=f"{CACHE_DIR}/valdense10_oof.parquet")
ap.add_argument("--val-pairs", default=f"{CACHE_DIR}/valdense10_pairs.parquet")
ap.add_argument("--val-minilm", default=f"{CACHE_DIR}/valdense10_minilm.npz")
ap.add_argument("--rescue-k", type=int, default=5)
ap.add_argument("--tag", default="lanes15-10-10_rr40_x5-10_rv3_dn10")
ap.add_argument("--out", default=f"{CACHE_DIR}/final_probs_rescue.npz")
a = ap.parse_args()
K = fs.K; RK = a.rescue_k


def features(ent, prob, p_cur, minilm, mp, dscore, drank):
    df = pd.DataFrame({"e": ent, "ds": dscore, "pc": p_cur})
    g = df.groupby("e")
    return np.c_[fs.lg(prob), fs.lg(p_cur), minilm, mp, dscore, drank, g.ds.transform("max") - df.ds,
                 g.pc.transform("max") - df.pc, g.pc.transform("size"), (df.pc > 0.5).groupby(df.e).transform("sum")]


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0)


# ---- validation ----
o = pd.read_parquet(a.val_oof); d = pd.read_parquet(a.val_pairs)
assert (o.candidate_entity_id.values == d.candidate_entity_id.values).all()
d["p_cur"] = o.p.values
vb = pd.read_parquet(f"{K}/val_band.parquet", columns=["source1_entity_id", "candidate_entity_id", "minilm"]); vb["mp"] = np.load(f"{K}/mpnet_val_scores.npy")
m = np.load(a.val_minilm, allow_pickle=True); r = np.load(f"{CACHE_DIR}/val_rescue_minilm.npz", allow_pickle=True)
sc = pd.concat([vb, pd.DataFrame({"source1_entity_id": m["s1"], "candidate_entity_id": m["c"], "minilm": m["score"], "mp": np.nan}),
                pd.DataFrame({"source1_entity_id": r["s1"], "candidate_entity_id": r["c"], "minilm": r["score"], "mp": np.nan})]
               ).drop_duplicates(["source1_entity_id", "candidate_entity_id"])
d = d.merge(sc, on=["source1_entity_id", "candidate_entity_id"], how="left")
t = pd.read_parquet(f"{CACHE_DIR}/val_dense_topk.parquet").rename(columns={"rank": "drank", "score": "dscore"})
d = d.merge(t, on=["source1_entity_id", "candidate_entity_id"], how="left"); d["dscore"] = d.dscore.astype(float)
act = (((d.prob >= 0.01) & (d.prob <= 0.99)) | (d.drank.notna() & (d.drank < RK) & (d.prob < 0.01))).to_numpy()
A = d[act]
X = features(A.source1_entity_id.values, A.prob.values, A.p_cur.values, A.minilm.values, A.mp.values, A.dscore.values, A.drank.values)
y = A.label.values
half = A.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).values
oof = np.zeros(len(A))
for h in (0, 1):
    oof[half == h] = model().fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
nt = pd.Series(gt.matched_entity_ids.map(lambda s: len(s.split(",")) if s else 0).values, index=gt.source1_entity_id)
ents = d.source1_entity_id.unique(); ntv = nt.reindex(ents).values
d["r"] = d.groupby("source1_entity_id").prob.rank(ascending=False, method="first")


def f05(p, t, topk=0):
    x = d[["source1_entity_id", "candidate_entity_id", "label"]].copy(); x["p"] = p
    if topk:
        x = x[(d.r <= topk).to_numpy() | ((d.drank < RK) & (d.prob < 0.01)).to_numpy()]
    x = x[x.p >= t].sort_values("p", ascending=False).drop_duplicates("candidate_entity_id")
    s = x.groupby("source1_entity_id").agg(pp=("label", "size"), tp=("label", "sum")).reindex(ents).fillna(0)
    pr = np.where(s.pp > 0, s.tp / np.maximum(s.pp, 1), 1.0); rc = np.where(ntv > 0, s.tp / np.maximum(ntv, 1), 1.0)
    return np.where(ntv == 0, (s.pp == 0).astype(float), np.where(s.tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)).mean()


p = d.p_cur.values.copy(); p[np.flatnonzero(act)] = oof
base = max((f05(d.p_cur.values, t), t) for t in np.arange(0.6, 0.9, 0.02))
f, thr = max((f05(p, t), t) for t in np.arange(0.5, 0.9, 0.02))
f10 = f05(p, thr, topk=10)
print(f"validation: current {base[0]:.5f} -> rescue stage {f:.5f} at {thr:.2f}; with top-10 candidates + rescued {f10:.5f}", flush=True)
final_model = model().fit(X, y)

# ---- test ----
T = np.load(a.probs, allow_pickle=True); F = np.load(a.final)
s1, cand, p0, pcur = T["s1"], T["cand"], T["p0"], F["prob"]
c = np.load(f"{CACHE_DIR}/ce_scores_cross_encoder_big2_v8_{a.tag}.npz")
ml = np.full(len(s1), np.nan, dtype=np.float32); ml[c["idx"]] = c["score"]
tt = pd.read_parquet(f"{CACHE_DIR}/test_dense_topk_full.parquet")
ks = pair_keys(s1, cand)
tk = pd.DataFrame({"dr": tt["rank"].values, "dsc": tt["score"].astype(float).values}, index=pair_keys(tt.source1_entity_id.values, tt.candidate_entity_id.values))
tk = tk[~tk.index.duplicated()].reindex(ks)
drank, dscore = tk.dr.to_numpy(), tk.dsc.to_numpy()
nar = pd.read_parquet(f"{CACHE_DIR}/kaggle_test_upload/test_band.parquet", columns=["row_idx"])
ot = pq.read_table(f"{CACHE_DIR}/test_features_lanes15-10-10_rr40_x5-10_rv3.parquet", columns=["source1_entity_id", "candidate_entity_id"]).take(nar.row_idx.to_numpy())
mpk = pd.Series(np.load(f"{CACHE_DIR}/mpnet_test_scores.npy"), index=pair_keys(ot.column(0), ot.column(1)))
tact = np.flatnonzero(((p0 >= 0.01) & (p0 <= 0.99)) | (~np.isnan(drank) & (drank < RK) & (p0 < 0.01)))
print(f"pairs without a re-checker score (left to the model as missing): {np.isnan(ml[tact]).sum():,}", flush=True)
mp = mpk.reindex(ks[tact]).to_numpy()
Xt = features(s1[tact], p0[tact], pcur[tact], ml[tact], mp, dscore[tact], drank[tact])
prob = pcur.copy(); prob[tact] = final_model.predict_proba(Xt)[:, 1]
rescue = (~np.isnan(drank) & (drank < RK) & (p0 < 0.01))
np.savez(a.out, prob=prob, threshold=np.array([thr]), rescue=rescue)
print(f"test: re-decided {len(tact):,} pairs ({rescue.sum():,} rescued); saved {a.out}", flush=True)
