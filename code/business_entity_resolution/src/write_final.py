"""Write matching_results.tsv straight from the dumped per-pair ids and the
final-stage probabilities (same threshold + 1-to-1 reconciliation + writer as
generate_predictions.py, without recomputing features).

  python3 write_final.py ../cache/v11_probs.npz ../cache/final_probs_v11.npz
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from generate_predictions import reconcile_one_to_one, write_id_list_tsv
from train_model import CACHE_DIR

d = np.load(sys.argv[1], allow_pickle=True)
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 0
f = np.load(sys.argv[2])
prob, thr = f["prob"], float(f["threshold"][0])
assert len(prob) == len(d["s1"])
keep = np.ones(len(prob), bool)
if TOPK:
    # two-stage cascade: the first-stage classifier keeps its top-K candidates per
    # business; only those are passed to the re-checkers / final stage
    r = pd.DataFrame({"s": d["s1"], "p": d["p0"]}).groupby("s")["p"].rank(ascending=False, method="first").to_numpy()
    keep = r <= TOPK
    if "rescue" in f.files:
        keep |= f["rescue"]   # rescued learned-search candidates also go to the matching models
    all_s1_ = pd.read_parquet(f"{CACHE_DIR}/test_source1_clean.parquet")["entity_id"].tolist()
    cp = pd.DataFrame({"source1_entity_id": d["s1"][keep], "candidate_entity_id": d["cand"][keep]})
    cout = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "candidate_pairs.tsv")
    write_id_list_tsv(cp, "candidate_entity_id", cout, all_s1_, "candidate_entity_ids")
    print(f"candidate set: top-{TOPK} per business -> {keep.sum():,} pairs ({keep.sum()/len(all_s1_):.2f}/business)", flush=True)
sel = (prob >= thr) & keep
m = pd.DataFrame({"source1_entity_id": d["s1"][sel], "candidate_entity_id": d["cand"][sel], "prob": prob[sel]})
print(f"threshold {thr:.2f}: pairs above threshold {len(m):,}", flush=True)
m = reconcile_one_to_one(m)
print(f"after 1-to-1 reconciliation {len(m):,}", flush=True)
all_s1 = pd.read_parquet(f"{CACHE_DIR}/test_source1_clean.parquet")["entity_id"].tolist()
out = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "matching_results.tsv")
write_id_list_tsv(m, "candidate_entity_id", out, all_s1, "matched_entity_ids")
print(f"wrote {out}; predicted singleton rate {1 - m.source1_entity_id.nunique()/len(all_s1):.4f}", flush=True)
