"""
Train an extra MiniLM re-checker (different seed) on the exported kit, on the
Mac GPU, then score the validation band and report the ensemble F0.5.
Loads only torch + the kit parquet files: no gradient-boosting model in the
same process (their two OpenMP runtimes crash together on macOS).

  python3 mac_rechecker_seed.py --kit ../cache/rechecker_kit --seed 1 --epochs 1
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import cross_encoder as ce

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(a.kit, f"minilm_seed{a.seed}")
os.makedirs(out, exist_ok=True)

tr = pd.read_parquet(f"{a.kit}/train_pairs.parquet")
vb = pd.read_parquet(f"{a.kit}/val_band.parquet")
# the trainer takes id pairs + an id -> text map; the kit already holds texts
text = {f"a{i}": t for i, t in enumerate(tr["s1_text"])}
text.update({f"b{i}": t for i, t in enumerate(tr["c_text"])})
text.update({f"va{i}": t for i, t in enumerate(vb["s1_text"])})
text.update({f"vb{i}": t for i, t in enumerate(vb["c_text"])})
pairs = [(f"a{i}", f"b{i}") for i in range(len(tr))]
t0 = time.time()
tok, m = ce.train(pairs, tr["label"].to_numpy(), text, epochs=a.epochs, seed=a.seed)
m.save_pretrained(out); tok.save_pretrained(out)
vs = ce.score(tok, m, [(f"va{i}", f"vb{i}") for i in range(len(vb))], text)
np.save(f"{out}/val_scores.npy", vs)
print(f"trained on {len(tr):,} pairs + scored {len(vb):,} validation pairs in {time.time()-t0:.0f}s", flush=True)

va = pd.read_parquet(f"{a.kit}/val_all.parquet")
key_all = va.source1_entity_id + "|" + va.candidate_entity_id
pos = pd.Series(np.arange(len(va)), index=key_all.values).reindex(
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
y = vb.label.to_numpy()
lp = ce.logit(vb.prob.to_numpy())
for name, X in (("classifier + MiniLM (current best)", np.c_[lp, vb.minilm]),
                (f"classifier + MiniLM + MiniLM seed {a.seed}", np.c_[lp, vb.minilm, vs])):
    comb = np.zeros(len(vb))
    for h in (0, 1):
        comb[half == h] = LogisticRegression(max_iter=1000).fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
    p = va.prob.to_numpy().copy(); p[pos] = comb
    best = max((macro_f05(p, t), t) for t in np.arange(0.5, 0.92, 0.02))
    print(f"{name}: validation macro F0.5 {best[0]:.4f} (threshold {best[1]:.2f})", flush=True)
