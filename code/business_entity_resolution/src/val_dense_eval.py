"""
End-to-end validation of learned-search extras.
  step a (sklearn): score the rebuilt validation feature table with the classifier
  step b (torch)  : MiniLM re-checker on band pairs not already scored in the kit
  step c (sklearn): final stage (mpnet only where already scored) -> macro F0.5
"""
import sys, os, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(__file__))
from train_model import CACHE_DIR, DATASET_DIR
step, table = sys.argv[1], sys.argv[2]
MODEL = sys.argv[3] if len(sys.argv) > 3 else "matching_model_v8.joblib"
tag = os.path.basename(table).replace(".parquet", "") + ("" if MODEL == "matching_model_v8.joblib" else "_" + MODEL.replace(".joblib", ""))
P = f"{CACHE_DIR}/{tag}_pairs.parquet"; M = f"{CACHE_DIR}/{tag}_minilm.npz"
K = f"{CACHE_DIR}/rechecker_kit"

if step == "a":
    from posthoc_features import add_posthoc_features
    b = joblib.load(f"{CACHE_DIR}/{MODEL}")
    df = add_posthoc_features(pd.read_parquet(table), "train")
    df["prob"] = b["model"].predict_proba(df[b["features"]].values)[:, 1]
    df[["source1_entity_id", "candidate_entity_id", "label", "prob"]].to_parquet(P, index=False)
    print(f"scored {len(df):,} pairs -> {P}", flush=True)

elif step == "b":
    import cross_encoder as ce
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = pd.read_parquet(P)
    vb = pd.read_parquet(f"{K}/val_band.parquet", columns=["source1_entity_id", "candidate_entity_id", "minilm"])
    band = d[(d.prob >= 0.01) & (d.prob <= 0.99)]
    have = pd.MultiIndex.from_frame(vb[["source1_entity_id", "candidate_entity_id"]])
    need = band[~pd.MultiIndex.from_frame(band[["source1_entity_id", "candidate_entity_id"]]).isin(have)]
    print(f"band pairs {len(band):,}; new ones to score {len(need):,}", flush=True)
    mb = f"{CACHE_DIR}/cross_encoder_big2_v8"
    tok = AutoTokenizer.from_pretrained(mb)
    mdl = AutoModelForSequenceClassification.from_pretrained(mb).to(ce.device()).eval()
    text = ce.raw_texts("train", set(need.source1_entity_id) | set(need.candidate_entity_id))
    s = ce.score(tok, mdl, list(zip(need.source1_entity_id, need.candidate_entity_id)), text)
    np.savez(M, s1=need.source1_entity_id.to_numpy(), c=need.candidate_entity_id.to_numpy(), score=s)
    print("saved", M, flush=True)

elif step == "c":
    import final_stage as fs
    d = pd.read_parquet(P)
    vb = pd.read_parquet(f"{K}/val_band.parquet", columns=["source1_entity_id", "candidate_entity_id", "minilm"])
    vb["mp"] = np.load(f"{K}/mpnet_val_scores.npy")
    m = np.load(M, allow_pickle=True)
    extra = pd.DataFrame({"source1_entity_id": m["s1"], "candidate_entity_id": m["c"], "minilm": m["score"], "mp": np.nan})
    sc = pd.concat([vb, extra]).drop_duplicates(["source1_entity_id", "candidate_entity_id"])
    band = d[(d.prob >= 0.01) & (d.prob <= 0.99)].merge(sc, on=["source1_entity_id", "candidate_entity_id"], how="left")
    assert band.minilm.notna().all(), band.minilm.isna().sum()
    gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    nt = pd.Series(gt.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0).values, index=gt.source1_entity_id)
    ents = d.source1_entity_id.unique()
    ntv = nt.reindex(ents)
    # final stage with mpnet only where available
    y = band.label.to_numpy(); import hashlib
    half = band.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).to_numpy()
    X2 = np.c_[fs.lg(band.prob.to_numpy()), band.minilm]; mp = band.mp.to_numpy(); has = ~np.isnan(mp)
    def stage(fi, ai):
        s2 = fs.lr().fit(X2[fi], y[fi]).predict_proba(X2)[:, 1]
        narrow = (s2 > fs.NARROW_LO) & (s2 < fs.NARROW_HI) & has
        X3 = np.c_[X2, np.nan_to_num(mp)]
        s3 = fs.lr().fit(X3[fi & narrow], y[fi & narrow]).predict_proba(X3)[:, 1]
        G = fs.group_features(band.source1_entity_id.values, band.prob.to_numpy(), band.minilm.to_numpy(), s2, s3, np.nan_to_num(mp), narrow)
        mg = fs.hgb().fit(G[fi & narrow], y[fi & narrow])
        return np.where(narrow, mg.predict_proba(G)[:, 1], s2)[ai], s2[ai]
    fin = np.zeros(len(band)); two = np.zeros(len(band))
    for h in (0, 1):
        fin[half == h], two[half == h] = stage(half != h, half == h)
    def f05(prob, t):
        x = d[["source1_entity_id", "candidate_entity_id", "label"]].copy(); x["p"] = prob
        x = x[x.p >= t].sort_values("p", ascending=False).drop_duplicates("candidate_entity_id")
        s = x.groupby("source1_entity_id").agg(pp=("label", "size"), tp=("label", "sum")).reindex(ents).fillna(0)
        n = ntv.values; pr = np.where(s.pp > 0, s.tp / np.maximum(s.pp, 1), 1.0); rc = np.where(n > 0, s.tp / np.maximum(n, 1), 1.0)
        f = np.where(n == 0, (s.pp == 0).astype(float), np.where(s.tp > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0))
        return f.mean()
    key = d.source1_entity_id + "|" + d.candidate_entity_id
    pos = pd.Series(np.arange(len(d)), index=key.values).reindex((band.source1_entity_id + "|" + band.candidate_entity_id).values).to_numpy()
    for name, v in (("classifier + MiniLM", two), ("full final stage", fin)):
        p = d.prob.to_numpy().copy(); p[pos] = v
        print(f"{tag}: {name}: macro F0.5 {max((f05(p, t), t) for t in np.arange(0.5, 0.92, 0.02))}", flush=True)
    tr = set(zip(gt.source1_entity_id, gt.matched_entity_ids))
    g = gt[gt.source1_entity_id.isin(set(ents))]; g = g.assign(m=g.matched_entity_ids.str.split(",")).explode("m"); g = g[g.m != ""]
    have = set(zip(d.source1_entity_id, d.candidate_entity_id))
    print(f"{tag}: blocking recall {np.mean([(a, b) in have for a, b in zip(g.source1_entity_id, g.m)]):.4f}, pairs {len(d):,}", flush=True)
