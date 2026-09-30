"""
Second-hop candidate expansion: businesses usually have several records
(copies of the same business in Source 2/3), and the search sometimes finds
only some of them. Records whose normalised, transliterated name is exactly
identical to a record we already matched to a business become extra
candidates for that business; a small classifier decides which to add.
"""
import numpy as np, pandas as pd, scipy.sparse as sp, hashlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from posthoc_features import text_vectors
from train_model import CACHE_DIR

MAX_GROUP = 30


def row_keys(m):
    """Exact-duplicate key per row of a CSR matrix (same normalised text -> same key)."""
    m = m.tocsr(); m.sort_indices()
    ip, ix = m.indptr, m.indices
    return np.array([hash(ix[ip[i]:ip[i+1]].tobytes()) if ip[i+1] > ip[i] else 0 for i in range(m.shape[0])], dtype=np.int64)


class Records:
    def __init__(self, split):
        parts = []
        self.vec = {}
        for s in ("source2", "source3"):
            d = pd.read_parquet(f"{CACHE_DIR}/{split}_{s}_clean.parquet", columns=["entity_id", "country"])
            nm = text_vectors(split, s, "translit"); ad = text_vectors(split, s, "addr")
            kp = f"{CACHE_DIR}/{split}_{s}_namekey.npy"
            if os.path.exists(kp):
                k = np.load(kp)
            else:
                k = row_keys(nm); np.save(kp, k)
            d["key"] = k; parts.append(d)
            self.vec[s] = (nm, ad)
        self.df = pd.concat(parts, ignore_index=True)
        self.df["src_row"] = np.r_[np.arange(len(parts[0])), np.arange(len(parts[1]))]
        self.df = self.df[self.df.key != 0]
        self.df["gsize"] = self.df.groupby(["country", "key"]).entity_id.transform("size")
        self.pos = pd.Series(np.arange(len(self.df)), index=self.df.entity_id.values)
        s1 = pd.read_parquet(f"{CACHE_DIR}/{split}_source1_clean.parquet", columns=["entity_id"]).entity_id
        self.s1pos = pd.Series(np.arange(len(s1)), index=s1.values)
        self.s1vec = (text_vectors(split, "source1", "translit"), text_vectors(split, "source1", "addr"))

    def rec_vecs(self, ids, kind):
        k = 0 if kind == "n" else 1
        s2 = pd.Series(ids).str.startswith("S2-").to_numpy()
        rows = self.df.src_row.to_numpy()[self.pos.reindex(ids).to_numpy()]
        A = self.vec["source2"][k][np.where(s2, rows, 0)]; B = self.vec["source3"][k][np.where(~s2, rows, 0)]
        return (sp.diags(s2.astype(np.float32)) @ A + sp.diags((~s2).astype(np.float32)) @ B).tocsr()

    def s1_vecs(self, ids, kind):
        return self.s1vec[0 if kind == "n" else 1][self.s1pos.reindex(ids).to_numpy()]


def expand(R, pred, retrieved):
    """pred: DataFrame(source1_entity_id, candidate_entity_id, p) of predicted pairs.
    Returns new (business, record) pairs with features."""
    d = R.df
    pk = pred.merge(d[["entity_id", "country", "key", "gsize"]], left_on="candidate_entity_id", right_on="entity_id")
    pk = pk[pk.gsize <= MAX_GROUP]
    new = pk[["source1_entity_id", "candidate_entity_id", "p", "country", "key"]].rename(columns={"candidate_entity_id": "sib"}).merge(
        d[["entity_id", "country", "key", "gsize"]], on=["country", "key"])
    new = new[new.entity_id != new.sib]
    rk = pd.MultiIndex.from_arrays([retrieved.source1_entity_id, retrieved.candidate_entity_id])
    new = new[~pd.MultiIndex.from_arrays([new.source1_entity_id, new.entity_id]).isin(rk)]
    new = new.rename(columns={"entity_id": "cand"})
    # features
    new["sib_addr"] = np.asarray(R.rec_vecs(new.cand.values, "a").multiply(R.rec_vecs(new.sib.values, "a")).sum(1)).ravel()
    agg = new.groupby(["source1_entity_id", "cand"]).agg(n_sib=("sib", "size"), sib_p_max=("p", "max"),
                                                         sib_addr_max=("sib_addr", "max"), gsize=("gsize", "first")).reset_index()
    agg["s1_name"] = np.asarray(R.s1_vecs(agg.source1_entity_id.values, "n").multiply(R.rec_vecs(agg.cand.values, "n")).sum(1)).ravel()
    agg["s1_addr"] = np.asarray(R.s1_vecs(agg.source1_entity_id.values, "a").multiply(R.rec_vecs(agg.cand.values, "a")).sum(1)).ravel()
    npred = pred.groupby("source1_entity_id").size()
    agg["n_pred"] = npred.reindex(agg.source1_entity_id).values
    agg["cand_other"] = agg.groupby("cand").source1_entity_id.transform("size")
    # already predicted for some other business?
    agg["cand_taken"] = agg.cand.isin(set(pred.candidate_entity_id)).astype(int)
    agg["cand_s2"] = agg.cand.str.startswith("S2-").astype(int)
    return agg

FEATS = ["n_sib", "sib_p_max", "sib_addr_max", "gsize", "s1_name", "s1_addr", "n_pred", "cand_other", "cand_taken", "cand_s2"]
