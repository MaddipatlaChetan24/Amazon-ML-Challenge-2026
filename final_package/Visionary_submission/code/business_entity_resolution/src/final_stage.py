"""
Final stage: combine classifier + MiniLM + mpnet re-checker scores with a
business-aware model (each candidate's score relative to the other
candidates of the same business).

mpnet scores exist only for pairs still uncertain after classifier + MiniLM
(combined score in (NARROW_LO, NARROW_HI)); other pairs keep the
classifier + MiniLM score. Validation emulates exactly that.

  python3 final_stage.py --eval                      # validation only
  python3 final_stage.py --probs ../cache/v9_probs.npz --mpnet-test test_scores.npy
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from train_model import CACHE_DIR

K = f"{CACHE_DIR}/rechecker_kit"
NARROW_LO, NARROW_HI = 0.05, 0.95


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def lr():
    return LogisticRegression(max_iter=1000)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0)


def group_features(ent, prob, minilm, s2, s3, mp, narrow):
    """s2: classifier+MiniLM score; s3: 3-way score (narrow rows only)."""
    s = lg(np.where(narrow, s3, s2))
    d = pd.DataFrame({"e": ent, "s": s, "m": minilm, "v": np.where(narrow, mp, np.nan)})
    g = d.groupby("e")
    return np.c_[lg(prob), minilm, d.v, lg(s2), d.s, g.s.transform("max") - d.s, g.s.rank(ascending=False),
                 g.s.transform("size"), (d.s > 0).groupby(d.e).transform("sum"),
                 g.m.transform("max") - d.m, g.v.transform("max") - d.v]


def macro_f05_factory(va):
    nt = va.groupby("source1_entity_id").n_true.first()

    def f(p, t):
        pred = (p >= t).astype(int)
        agg = pd.DataFrame({"e": va.source1_entity_id.values, "pp": pred, "tp": pred * va.label.values}).groupby("e").sum()
        n = nt.reindex(agg.index).values
        pr = np.where(agg.pp > 0, agg.tp / np.maximum(agg.pp, 1), 1.0)
        rc = np.where(n > 0, agg.tp / np.maximum(n, 1), 1.0)
        f_ = np.where((0.25 * pr + rc) > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
        return np.where(n == 0, (agg.pp.values == 0).astype(float), f_).mean()
    return f


def fit_all(vb, vs):
    """Fit every stage on the whole validation band; returns models + threshold (from OOF)."""
    y = vb.label.to_numpy()
    X2 = np.c_[lg(vb.prob.to_numpy()), vb.minilm]
    half = vb.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).to_numpy()

    def stage(fit_idx, app_idx):
        m2 = lr().fit(X2[fit_idx], y[fit_idx])
        s2 = m2.predict_proba(X2)[:, 1]
        narrow = (s2 > NARROW_LO) & (s2 < NARROW_HI)
        X3 = np.c_[X2, vs]
        m3 = lr().fit(X3[fit_idx & narrow], y[fit_idx & narrow])
        s3 = m3.predict_proba(X3)[:, 1]
        G = group_features(vb.source1_entity_id.values, vb.prob.to_numpy(), vb.minilm.to_numpy(), s2, s3, vs, narrow)
        mg = hgb().fit(G[fit_idx & narrow], y[fit_idx & narrow])
        out = np.where(narrow, mg.predict_proba(G)[:, 1], s2)
        return out[app_idx], (m2, m3, mg)

    oof = np.zeros(len(vb))
    for h in (0, 1):
        oof[half == h], _ = stage(half != h, half == h)
    _, models = stage(np.ones(len(vb), bool), np.ones(len(vb), bool))
    return oof, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--mpnet-val", default=f"{K}/mpnet_val_scores.npy")
    ap.add_argument("--probs", help="npz from generate_predictions.py --dump-probs")
    ap.add_argument("--mpnet-test")
    ap.add_argument("--test-band", default=f"{CACHE_DIR}/rechecker_kit_test/test_band.parquet")
    ap.add_argument("--narrow-test", default=f"{CACHE_DIR}/kaggle_test_upload/test_band.parquet")
    a = ap.parse_args()

    vb = pd.read_parquet(f"{K}/val_band.parquet")
    vs = np.load(a.mpnet_val)
    va = pd.read_parquet(f"{K}/val_all.parquet")
    pos = pd.Series(np.arange(len(va)), index=(va.source1_entity_id + "|" + va.candidate_entity_id).values).reindex(
        (vb.source1_entity_id + "|" + vb.candidate_entity_id).values).to_numpy()
    f05 = macro_f05_factory(va)
    oof, (m2, m3, mg) = fit_all(vb, vs)
    p = va.prob.to_numpy().copy(); p[pos] = oof
    f, thr = max((f05(p, t), t) for t in np.arange(0.4, 0.94, 0.02))
    print(f"validation (mpnet only on narrowed pairs, business-aware final stage): {f:.4f} at threshold {thr:.2f}", flush=True)
    if a.eval:
        return

    # ---- test ----
    tb = pd.read_parquet(a.test_band, columns=["row_idx", "prob", "minilm"])
    nar = pd.read_parquet(a.narrow_test, columns=["row_idx"])
    ts = np.load(a.mpnet_test); assert len(ts) == len(nar)
    mp = pd.Series(ts, index=nar.row_idx.values).reindex(tb.row_idx.values).to_numpy()
    d = np.load(a.probs, allow_pickle=True)
    ent = d["s1"][tb.row_idx.to_numpy()]
    X2 = np.c_[lg(tb.prob.to_numpy()), tb.minilm]
    s2 = m2.predict_proba(X2)[:, 1]
    narrow = ~np.isnan(mp)
    print(f"test band {len(tb):,}; with mpnet score {narrow.sum():,}", flush=True)
    s3 = np.full(len(tb), np.nan)
    s3[narrow] = m3.predict_proba(np.c_[X2, mp][narrow])[:, 1]
    G = group_features(ent, tb.prob.to_numpy(), tb.minilm.to_numpy(), s2, s3, np.nan_to_num(mp), narrow)
    fin = s2.copy()
    fin[narrow] = mg.predict_proba(G[narrow])[:, 1]
    prob = d["prob"].copy()
    prob[tb.row_idx.to_numpy()] = fin
    np.savez(f"{CACHE_DIR}/final_probs.npz", prob=prob, threshold=np.array([thr]))
    print(f"saved final_probs.npz (threshold {thr:.2f})", flush=True)


if __name__ == "__main__":
    main()
