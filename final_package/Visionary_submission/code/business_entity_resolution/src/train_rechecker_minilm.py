"""Train the MiniLM re-checker (cross-encoder) on uncertain pairs of TRAINING
businesses (never the validation ones).

  python3 train_rechecker_minilm.py 1000000 500000
      -> cache/ce_big2_model  (then run package_rechecker_minilm.py)

Uncertain = classifier probability in 0.01-0.99 under the v7 classifier on the
v7 training feature table (500K businesses per country). If
cache/v7_val_scored.parquet exists, also reports validation macro F0.5."""
import sys, time, random
import numpy as np, pandas as pd, pyarrow.parquet as pq, joblib
from sklearn.linear_model import LogisticRegression
import os
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import cross_encoder as ce
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, VAL_ENTITIES_PATH, macro_f05_at_threshold
N_ENT = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
TABLE = f"{CACHE_DIR}/train_features_v7_lanes15-10-10_rr40_500k.parquet"

t0 = time.time()
val = set(pd.read_csv(VAL_ENTITIES_PATH, header=None)[0])
ents = pq.read_table(TABLE, columns=["source1_entity_id"]).column(0).unique().to_pylist()
train_ents = [e for e in ents if e not in val]
random.seed(0)
pick = random.sample(train_ents, min(N_ENT, len(train_ents)))
b = joblib.load(f"{CACHE_DIR}/matching_model_v7.joblib")
parts, n_pairs = [], 0
for c in range(0, len(pick), 150000):   # chunked: keeps memory low next to the v8 build
    df = pq.read_table(TABLE, filters=[("source1_entity_id", "in", pick[c:c + 150000])]).to_pandas()
    df = add_posthoc_features(df, "train")
    df["prob"] = b["model"].predict_proba(df[b["features"]].values)[:, 1]
    n_pairs += len(df)
    parts.append(df.loc[df.prob.between(0.01, 0.99), ["source1_entity_id", "candidate_entity_id", "label"]])
    print(f"  chunk {c//150000 + 1}: {sum(len(p) for p in parts):,} uncertain pairs so far ({time.time()-t0:.0f}s)", flush=True)
    del df
band = pd.concat(parts)
if len(band) > CAP:
    band = band.sample(CAP, random_state=0)
print(f"{len(pick):,} training businesses, {n_pairs:,} pairs, uncertain {len(band):,} "
      f"(positives {band.label.mean()*100:.1f}%), prep {time.time()-t0:.0f}s", flush=True)
pairs = list(zip(band.source1_entity_id, band.candidate_entity_id))
y = band.label.to_numpy()

VAL = f"{CACHE_DIR}/v7_val_scored.parquet"
HAVE_VAL = os.path.exists(VAL)
v = pd.read_parquet(VAL, columns=["source1_entity_id", "candidate_entity_id", "label", "prob"]) if HAVE_VAL else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "label", "prob"])
vb_mask = v.prob.between(0.05, 0.95).to_numpy()
vb = v[vb_mask]
vpairs = list(zip(vb.source1_entity_id, vb.candidate_entity_id))
text = ce.raw_texts("train", set(band.source1_entity_id) | set(band.candidate_entity_id)
                    | set(vb.source1_entity_id) | set(vb.candidate_entity_id))
t0 = time.time()
tok, m = ce.train(pairs, y, text, epochs=2)
s = ce.score(tok, m, vpairs, text) if len(vpairs) else np.zeros(0)
print(f"trained on {len(pairs):,} pairs + scored {len(vpairs):,} validation pairs in {time.time()-t0:.0f}s", flush=True)
m.save_pretrained(f"{CACHE_DIR}/ce_big2_model"); tok.save_pretrained(f"{CACHE_DIR}/ce_big2_model")

if not HAVE_VAL:
    sys.exit(0)
gt = pd.read_csv(f"{SRC}/../../../dataset/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
n_true = dict(zip(gt.source1_entity_id, gt.matched_entity_ids.map(lambda x: len(x.split(",")) if x else 0)))
v["_n_true"] = v.source1_entity_id.map(n_true).fillna(0).astype(int)
X = np.c_[ce.logit(vb.prob.to_numpy()), s]
half = vb.source1_entity_id.map(ce._half).to_numpy()
comb = np.zeros(len(vb))
for h in (0, 1):   # combiner fit on one half of validation, applied to the other
    lr = LogisticRegression().fit(X[half != h], vb.label.to_numpy()[half != h])
    comb[half == h] = lr.predict_proba(X[half == h])[:, 1]
v["prob2"] = v.prob
v.loc[vb_mask, "prob2"] = comb
for col in ("prob", "prob2"):
    best = max((macro_f05_at_threshold(v, col, "label", t), t) for t in np.arange(0.40, 0.92, 0.02))
    print(f"ALL-validation macro F0.5 {'v7 alone' if col == 'prob' else 'v7 + big cross-encoder'}: {best[0]:.4f} (threshold {best[1]:.2f})", flush=True)
