"""MiniLM re-checker scores for learned-search top-k test pairs that have none yet
(classifier probability outside the 0.01-0.99 band); added to the per-pair
score cache of the test table so every later stage can use them."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import cross_encoder as ce
from generate_predictions import pair_keys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_model import CACHE_DIR

probs, topk, k, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
MB = "cross_encoder_big2_v8"
d = np.load(probs, allow_pickle=True)
s1, cand = d["s1"], d["cand"]
cache = f"{CACHE_DIR}/ce_scores_{MB}_{tag}.npz"
c = np.load(cache)
assert int(c["n"][0]) == len(s1)
full = np.full(len(s1), np.nan, dtype=np.float32); full[c["idx"]] = c["score"]
t = pd.read_parquet(topk, columns=["source1_entity_id", "candidate_entity_id", "rank"]); t = t[t["rank"] < k]
want = set(pair_keys(t.source1_entity_id.values, t.candidate_entity_id.values).tolist())
keys = pair_keys(s1, cand)
rows = np.flatnonzero(np.isnan(full) & (d["p0"] < 0.01) & np.fromiter((x in want for x in keys.tolist()), bool, len(keys)))
print(f"top-{k} pairs without a re-checker score: {len(rows):,}", flush=True)
tok = AutoTokenizer.from_pretrained(f"{CACHE_DIR}/{MB}")
mdl = AutoModelForSequenceClassification.from_pretrained(f"{CACHE_DIR}/{MB}").to(ce.device()).eval()
text = ce.raw_texts("test", set(s1[rows]) | set(cand[rows]))
full[rows] = ce.score(tok, mdl, list(zip(s1[rows], cand[rows])), text)
done = np.flatnonzero(~np.isnan(full))
np.savez(cache, idx=done, score=full[done], n=np.array([len(s1)]))
print(f"cache now {len(done):,} scored pairs", flush=True)
