"""
Build the pairwise feature table used to train (and later score) the
matching classifier.

For each Source-1 entity, generates its blocking candidate set (blocking.py,
tuned config), computes the feature vector (features.py) for every
candidate, and -- for splits with ground truth -- labels each pair 1/0.
Negatives come naturally from every blocked candidate NOT in the ground
truth list; combined with the ~26% of Source-2/3 that are pure distractors
(report Section 3.4), this gives a large, realistic hard-negative pool
without any special sampling.

Country iteration is dynamic (`s1['country'].unique()`), not a hardcoded
list -- this is the exact gap flagged in PROGRESS_AND_METHODOLOGY_LOG.md
Section 3.2 for the earlier dev/eval scripts; fixed here since this script
is the one that will actually run against the test set (with France) too.
"""
import heapq
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_country_indexes, get_candidates, row_block_tokens, BLOCK_FIELDS, DERIVED_FIELDS

STORED_BLOCK_FIELDS = [f for f in BLOCK_FIELDS if f not in DERIVED_FIELDS]
from features import compute_features, FEATURE_NAMES, AMBIGUITY_FEATURE_NAMES

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")


def psutil_rss_gb() -> float:
    """Current process RSS in GB -- used only for progress logging, so a
    memory crisis like the one this module previously hit shows up in the
    log immediately instead of silently thrashing for 5+ minutes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

ROW_FIELDS = [
    "entity_id",
    "business_name",
    "business_address",
    "name_core_tokens",
    "name_legal_suffixes",
    "name_is_alias",
    "name_is_website",
    "name_comparison_text",
    "name_has_nonascii",
    "addr_tokens",
    "addr_digit_tokens",
    "addr_digit_tokens_norm",
    "addr_is_empty",
]


def load_clean(split: str, source: str) -> pd.DataFrame:
    return pd.read_parquet(f"{CACHE_DIR}/{split}_{source}_clean.parquet")


class ColumnStore:
    """Lazy row access: keeps one country's Source-2/3 data as plain column
    lists (cheap) plus an entity_id -> integer position index (also cheap --
    an int, not a dict). Row dicts (with frozensets) are constructed ONLY
    when a specific entity_id is actually looked up as a candidate, not
    pre-materialized for the whole country.

    This replaces an earlier version that eagerly built a dict-of-dicts-of-
    frozensets for every row in the country upfront -- for a 6M+ row
    country that meant tens of GB of RAM (observed 38GB+ and still climbing
    on a 3,000-row *test* run, since it materialized the full country
    regardless of how few candidates were actually needed) before ever
    computing a single feature. Since a given Source-2/3 record is typically
    looked up only a handful of times (once per S1 entity that happens to
    block against it), lazy per-access construction is both far cheaper in
    memory and, in aggregate, not meaningfully slower.
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray = None):
        self.cols = {f: df[f].tolist() for f in ROW_FIELDS}
        self.pos = {eid: i for i, eid in enumerate(self.cols["entity_id"])}
        self.embeddings = embeddings  # optional (n_rows, 384) float32, row-aligned to df

    def __contains__(self, eid):
        return eid in self.pos

    def get(self, eid: str) -> dict:
        i = self.pos[eid]
        c = self.cols
        return {
            "entity_id": eid,
            "business_name": c["business_name"][i],
            "business_address": c["business_address"][i],
            "name_core_tokens": frozenset(c["name_core_tokens"][i]),
            "name_legal_suffixes": frozenset(c["name_legal_suffixes"][i]),
            "name_is_alias": c["name_is_alias"][i],
            "name_is_website": c["name_is_website"][i],
            "name_comparison_text": c["name_comparison_text"][i],
            "name_has_nonascii": c["name_has_nonascii"][i],
            "addr_tokens": frozenset(c["addr_tokens"][i]),
            "addr_digit_tokens": frozenset(c["addr_digit_tokens"][i]),
            "addr_digit_tokens_norm": frozenset(c["addr_digit_tokens_norm"][i]),
            "addr_is_empty": c["addr_is_empty"][i],
            "embedding": self.embeddings[i] if self.embeddings is not None else None,
        }


def add_ambiguity_features(df: pd.DataFrame, group_col: str = "source1_entity_id") -> pd.DataFrame:
    """Second-pass, fully-vectorized features comparing each candidate to
    its siblings (other candidates blocked for the same S1 entity). See
    features.py's AMBIGUITY_FEATURE_NAMES docstring for the justification.

    Composite score = a simple weighted blend of the three strongest single
    features (name_jaccard, addr_jaccard, name_jw) -- just the ranking
    basis for *this* pass; the classifier still sees every raw feature
    individually and can learn its own weighting.

    Implementation note: computing "gap to the next-better/worse sibling"
    via a groupby-apply that sorts inside each group would work but is slow
    at this scale (Python-level apply per group). Instead this sorts the
    whole frame ONCE by (group, composite) and uses groupby().shift(), which
    is fully vectorized -- after sorting descending within each group, the
    row immediately above/below a candidate in that ordering *is* its
    nearest-better/nearest-worse sibling.
    """
    df = df.copy()
    composite = 0.5 * df["name_jaccard"] + 0.3 * df["addr_jaccard"] + 0.2 * df["name_jw"]
    df["_composite"] = composite

    g = df.groupby(group_col)["_composite"]
    df["amb_group_size"] = g.transform("size")
    df["amb_group_mean"] = g.transform("mean")
    df["amb_group_std"] = g.transform("std").fillna(0.0)
    std_safe = df["amb_group_std"].replace(0.0, 1.0)
    df["amb_zscore"] = (df["_composite"] - df["amb_group_mean"]) / std_safe
    df["amb_top1_score"] = g.transform("max")
    df["amb_rank"] = g.rank(method="first", ascending=False).astype(int)

    sorted_df = df.sort_values([group_col, "_composite"], ascending=[True, False])
    same_group_above = sorted_df[group_col].eq(sorted_df[group_col].shift(1))
    same_group_below = sorted_df[group_col].eq(sorted_df[group_col].shift(-1))
    next_higher = sorted_df["_composite"].shift(1).where(same_group_above)
    next_lower = sorted_df["_composite"].shift(-1).where(same_group_below)
    # gap_to_better: how far below the nearest better-scoring sibling (0 if
    # this candidate is already the best in its group -- nothing above it)
    sorted_df["amb_gap_to_better"] = (next_higher - sorted_df["_composite"]).fillna(0.0)
    # gap_to_worse: how far ahead of the nearest worse-scoring sibling (0 if
    # this is the single worst in its group)
    sorted_df["amb_gap_to_worse"] = (sorted_df["_composite"] - next_lower).fillna(0.0)

    df = sorted_df.sort_index()
    df.drop(columns=["_composite"], inplace=True)
    return df


CHUNK_ROWS = 20000
PROGRESS_EVERY = 100000
# Per-country state read by _process_range; set before the worker pool is
# forked so every worker inherits it without pickling.
_W = {}


def _process_range(bounds):
    start, end = bounds
    w = _W
    eids, cols, look, emb = w["eids"], w["cols"], w["s1_lookup_cols"], w["s1c_emb"]
    s2_store, s3_store, gt_map = w["s2_store"], w["s3_store"], w["gt_map"]
    out_s1, out_c, out_f, out_l = [], [], [], []
    for i in range(start, end):
        s1id = eids[i]
        s1_row = {f: look[f][i] for f in ROW_FIELDS}
        s1_row["embedding"] = emb[i] if emb is not None else None
        for f in ("name_core_tokens", "name_legal_suffixes", "addr_tokens",
                  "addr_digit_tokens", "addr_digit_tokens_norm"):
            s1_row[f] = frozenset(s1_row[f])
        row_tokens = row_block_tokens({f: cols[f][i] for f in STORED_BLOCK_FIELDS})
        rr = None
        if w["rerank_k"]:
            def rr(scores, q=w["rr_q"][i]):
                rows = np.fromiter((w["rr_row_of"][c] for c in scores), dtype=np.int64, count=len(scores))
                sim = _sparse_dot_rows(w["rr_mat"], rows, q)
                # rounded + tie-broken by row so float summation order can't
                # flip which near-equal candidates make the cut
                order = np.lexsort((rows, -np.round(sim, 5)))
                top = [order[:w["rerank_k"]]]
                k_empty, k_indic = w["rerank_extra"] or (0, 0)
                # Empty-address candidates can only score on the name half of
                # the vector, and transliterated names score low; each gets a
                # few reserved slots ranked only against its own kind.
                ranked_rows = rows[order]
                if k_empty:
                    top.append(order[w["rr_empty"][ranked_rows]][:k_empty])
                if k_indic:
                    top.append(order[w["rr_indic"][ranked_rows]][:k_indic])
                return [w["rr_ids"][rows[j]] for j in np.concatenate(top)]
        cands = get_candidates(row_tokens, w["indexes"], top_k=w["top_k"], lanes=w["lanes"], rerank=rr,
                               extra=w["rev_extra"].get(i))
        true_ids = gt_map.get(s1id, set()) if gt_map is not None else None
        if w.get("extras_only"):
            # only the extra candidates (their features are identical to a full
            # build: score comes from the same retrieval pool)
            ex = set(w["rev_extra"].get(i) or ())
            cands = {c: v for c, v in cands.items() if c in ex}
        for cid, score in cands.items():
            cand_row = s2_store.get(cid) if cid.startswith("S2-") else s3_store.get(cid)
            out_s1.append(s1id)
            out_c.append(cid)
            out_f.append(compute_features(s1_row, cand_row, score))
            if true_ids is not None:
                out_l.append(1 if cid in true_ids else 0)
    feats = np.asarray(out_f, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    return out_s1, out_c, feats, out_l


REV_CHUNK = 50000
REV_PREFILTER = 200
# Tighter pruning for the business index than for forward blocking: pools
# shrink ~15x (median 1,215 -> 76 in India). Measured on v7 forward misses,
# the true business stays in the record's top 3 for 41% (India) / 31% (US)
# of them, vs 44% / 34% with forward pruning.
REV_MAX_DF = {"name_core_tokens": 2000, "addr_tokens": 300, "addr_digit_tokens": 300,
              "addr_digit_tokens_norm": 300, "addr_digit_word": 30}


def _sparse_dot_rows(mat, rows, q) -> np.ndarray:
    """mat[rows] . q for a one-row sparse q. Slicing only q's non-zero columns
    avoids `mat[rows] @ q.T`, which converts a 2M-column transposed vector on
    every call (profiled: most of the re-rank time)."""
    return np.asarray(mat[rows][:, q.indices] @ q.data).ravel()


def _process_reverse(bounds):
    """Record -> business direction: for each Source-2/3 record, the Source-1
    businesses it resembles most. Source 1 is deduplicated, so a record faces
    far fewer look-alikes among businesses than a business faces among
    records. Measured on v7 forward-blocking misses: the true business is in
    the record's top 3 for 34% (US) / 44% (India) of them."""
    start, end = bounds
    w = _W
    out_s1, out_rec = [], []
    for j in range(start, end):
        toks = row_block_tokens({f: w["rev_cols"][f][j] for f in STORED_BLOCK_FIELDS})
        pool = get_candidates(toks, w["rev_index"], top_k=None, fallback=False)
        if not pool:
            continue
        rows = np.fromiter((w["rev_s1_row"][e] for e in pool), dtype=np.int64, count=len(pool))
        if len(rows) > REV_PREFILTER:
            sc = np.fromiter(pool.values(), dtype=np.float64, count=len(pool))
            keep = np.argpartition(-sc, REV_PREFILTER)[:REV_PREFILTER]
            rows = np.sort(rows[keep])  # order-independent, so the result is deterministic
        sim = _sparse_dot_rows(w["rr_q"], rows, w["rr_mat"][j])
        best = np.lexsort((rows, -np.round(sim, 5)))[:w["reverse_k"]]
        out_s1.extend(rows[best].tolist())
        out_rec.extend([j] * len(best))
    return out_s1, out_rec


def load_dense_extra(path: str, k: int) -> dict:
    t = pd.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id", "rank"])
    t = t[t["rank"] < k]
    return t.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()


def build_features_for_split(split: str, max_s1_rows_per_country: int = None, top_k: int = None,
                             lanes: tuple = None, workers: int = 1, rerank_k: int = 0,
                             rerank_extra: tuple = None, reverse_k: int = 0, countries: list = None,
                             only_entities: set = None, dense_extra: dict = None,
                             extras_only: bool = False):
    """split: 'train' or 'test'. max_s1_rows_per_country: for quick benchmarking
    only. top_k: candidate cap per entity -- REQUIRED in practice at full scale
    (median candidate-set size is 3,304-3,534/entity; uncapped, train alone is
    ~12 billion pairs). See blocking.get_candidates docstring for measured
    recall-retention numbers at each cap size."""
    print(f"loading {split} source1/2/3...", flush=True)
    s1 = load_clean(split, "source1")
    s2 = load_clean(split, "source2")
    s3 = load_clean(split, "source3")

    # Embeddings are optional: if the cache wasn't built (embeddings.py's
    # build_embedding_cache), fall back to None everywhere and
    # embedding_cosine degrades to 0.0 for every pair (see features.py) --
    # so this script still runs end-to-end without the embedding step, just
    # without that one feature, rather than crashing.
    # embedding_cosine is filled vectorised afterwards by
    # posthoc_features.add_posthoc_features; copying each country's embeddings
    # into RAM here (~8GB for US train) only to compute it per pair in Python
    # would be wasted memory and time.
    s1_emb_full = s2_emb_full = s3_emb_full = None

    gt_map = None
    if split == "train":
        gt = pd.read_csv(
            f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        gt_map = {
            row.source1_entity_id: set(row.matched_entity_ids.split(","))
            if row.matched_entity_ids.strip()
            else set()
            for row in gt.itertuples(index=False)
        }
    print("loaded.", flush=True)

    # Per-country accumulators are converted to compact numpy arrays as soon
    # as each country finishes (see below) rather than held as one giant
    # Python list-of-lists across all countries. Measured bug this fixes: at
    # test scale (171M pairs), holding every feature as a boxed Python float
    # in nested lists blew up to ~170GB resident (vs. 64GB physical RAM) and
    # the process went into an unrecoverable "stuck" memory-thrashing state
    # -- it worked at training's smaller 29.7M-row scale but not at test's.
    # float32 numpy arrays cut this to ~4 bytes/value instead of Python's
    # ~24-32+ bytes per boxed float plus list overhead.
    all_s1_ids, all_cand_ids, all_feature_arrays, all_labels = [], [], [], []
    if rerank_k and not lanes:
        raise ValueError("rerank_k needs --lanes (the re-rank adds to the lane candidates)")
    countries = [c for c in s1["country"].unique() if not countries or c in countries]
    print(f"countries found: {list(countries)}", flush=True)

    for country in countries:
        t0 = time.time()
        s1_mask = (s1.country == country).to_numpy()
        s2_mask = (s2.country == country).to_numpy()
        s3_mask = (s3.country == country).to_numpy()
        s1c = s1[s1_mask].reset_index(drop=True)
        s1_pos = np.flatnonzero(s1_mask)
        s2c = s2[s2_mask].reset_index(drop=True)
        s3c = s3[s3_mask].reset_index(drop=True)
        s1c_emb = s1_emb_full[s1_mask] if s1_emb_full is not None else None
        s2c_emb = s2_emb_full[s2_mask] if s2_emb_full is not None else None
        s3c_emb = s3_emb_full[s3_mask] if s3_emb_full is not None else None

        if max_s1_rows_per_country and max_s1_rows_per_country < len(s1c):
            # Random sample, not a positional slice -- entity_ids look
            # randomly assigned already, but for an actual training subsample
            # (not just a speed test) it's not worth risking any ordering
            # bias in the source file for a one-line fix.
            sampled = s1c.sample(n=max_s1_rows_per_country, random_state=42)
            if s1c_emb is not None:
                s1c_emb = s1c_emb[sampled.index.to_numpy()]
            s1_pos = s1_pos[sampled.index.to_numpy()]
            s1c = sampled.reset_index(drop=True)
        if only_entities is not None:
            keep = s1c["entity_id"].isin(only_entities).to_numpy()
            s1_pos = s1_pos[keep]
            s1c = s1c[keep].reset_index(drop=True)

        n = len(s1c)
        _W.clear()
        _W.update(
            indexes=build_country_indexes(s2c, s3c),
            s2_store=ColumnStore(s2c, embeddings=s2c_emb),
            s3_store=ColumnStore(s3c, embeddings=s3c_emb),
            eids=s1c["entity_id"].tolist(),
            cols={f: s1c[f].tolist() for f in STORED_BLOCK_FIELDS},
            s1_lookup_cols={f: s1c[f].tolist() for f in ROW_FIELDS},
            s1c_emb=s1c_emb, gt_map=gt_map, top_k=top_k, lanes=lanes, rerank_k=rerank_k,
        )
        if rerank_k:
            # Loaded per country and dropped with _W, so only one country's
            # vectors are resident (all three sources at once measured 46GB).
            from posthoc_features import text_vectors

            def rr_rows(src, rows):
                # name (transliterated) and address char-3gram vectors side by
                # side: one sparse dot product scores both with equal weight.
                return sp.hstack([text_vectors(split, src, "translit")[rows],
                                  text_vectors(split, src, "addr")[rows]]).tocsr().astype(np.float32)

            rr_ids = pd.concat([s2c["entity_id"], s3c["entity_id"]]).tolist()
            both = pd.concat([s2c, s3c])
            _W.update(
                rerank_extra=rerank_extra,
                rr_empty=both["addr_is_empty"].to_numpy(dtype=bool),
                rr_indic=(both["name_has_nonascii"].to_numpy(dtype=bool)
                          & (both["name_core_tokens"].map(len).to_numpy() == 0)),
            )
            del both
            _W.update(
                rr_mat=sp.vstack([rr_rows("source2", np.flatnonzero(s2_mask)),
                                  rr_rows("source3", np.flatnonzero(s3_mask))]).tocsr(),
                rr_q=rr_rows("source1", s1_pos),
                rr_ids=rr_ids, rr_row_of={e: j for j, e in enumerate(rr_ids)},
            )

        _W["rev_extra"] = {}
        if reverse_k:
            if not rerank_k:
                raise ValueError("reverse_k needs --rerank-k (it reuses the re-rank vectors)")
            t_rev = time.time()
            _W.update(
                rev_index=build_country_indexes(s1c, s1c.iloc[0:0], max_df=REV_MAX_DF),
                rev_cols={f: pd.concat([s2c[f], s3c[f]]).tolist() for f in STORED_BLOCK_FIELDS},
                rev_s1_row={e: i for i, e in enumerate(_W["eids"])}, reverse_k=reverse_k,
            )
            n_rec = len(_W["rr_ids"])
            rchunks = [(a, min(a + REV_CHUNK, n_rec)) for a in range(0, n_rec, REV_CHUNK)]
            if workers > 1:
                rpool = multiprocessing.get_context("fork").Pool(workers)
                rresults = rpool.imap(_process_reverse, rchunks)
            else:
                rpool, rresults = None, map(_process_reverse, rchunks)
            rev_extra, n_rev = {}, 0
            for rs1, rrec in rresults:
                for i_s1, j in zip(rs1, rrec):
                    rev_extra.setdefault(i_s1, []).append(_W["rr_ids"][j])
                n_rev += len(rs1)
            if rpool is not None:
                rpool.close(); rpool.join()
            for key in ("rev_index", "rev_cols", "rev_s1_row"):
                _W.pop(key)
            _W["rev_extra"] = rev_extra
            print(f"  reverse pass: {n_rec:,} records -> {n_rev:,} record->business pairs "
                  f"in {time.time()-t_rev:.0f}s", flush=True)

        _W["extras_only"] = extras_only
        if dense_extra:
            # candidates from the learned (bi-encoder) search, added like the
            # reverse-retrieval extras
            n_dense = 0
            for i, e in enumerate(_W["eids"]):
                d = dense_extra.get(e)
                if d:
                    _W["rev_extra"].setdefault(i, []).extend(d)
                    n_dense += len(d)
            print(f"  learned-search extras: {n_dense:,}", flush=True)
        print(f"=== {country}: {n:,} S1 entities, index built in {time.time()-t0:.1f}s ===", flush=True)
        t0 = time.time()
        n_pairs, done_rows, n_pos = 0, 0, 0
        chunks = [(a, min(a + CHUNK_ROWS, n)) for a in range(0, n, CHUNK_ROWS)]
        pool = None
        if workers > 1:
            # fork, not macOS's default spawn: workers inherit the index and
            # stores built above instead of each re-building/copying them.
            pool = multiprocessing.get_context("fork").Pool(workers)
            results = pool.imap(_process_range, chunks)
        else:
            results = map(_process_range, chunks)

        next_report = PROGRESS_EVERY
        for ids1, idsc, feats, labels in results:
            all_s1_ids.extend(ids1)
            all_cand_ids.extend(idsc)
            all_feature_arrays.append(feats)
            if gt_map is not None:
                all_labels.extend(labels)
                n_pos += sum(labels)
            n_pairs += len(ids1)
            done_rows += CHUNK_ROWS
            if done_rows >= next_report:
                next_report += PROGRESS_EVERY
                dt = time.time() - t0
                print(f"    ...{min(done_rows, n):,}/{n:,} S1 rows, {n_pairs:,} pairs so far, "
                      f"{dt:.1f}s ({dt/min(done_rows, n)*1e6:.1f} us/S1row wall), "
                      f"{psutil_rss_gb():.1f}GB RSS", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
        _W_eids = _W["eids"]
        _W.clear()

        dt = time.time() - t0
        print(f"  {country} done: {n_pairs:,} pairs in {dt:.1f}s ({dt/max(n, 1)*1e6:.1f} us/S1row wall), "
              f"{psutil_rss_gb():.1f}GB RSS", flush=True)
        if gt_map is not None:
            n_true = sum(len(gt_map.get(e, ())) for e in _W_eids)
            print(f"  {country} blocking recall: {n_pos:,}/{n_true:,} true matches = "
                  f"{n_pos/max(n_true, 1)*100:.2f}%, {n_pairs/max(n, 1):.1f} candidates/entity", flush=True)

    feat_array = np.vstack(all_feature_arrays) if len(all_feature_arrays) > 1 else all_feature_arrays[0]
    del all_feature_arrays
    feat_df = pd.DataFrame(feat_array, columns=FEATURE_NAMES)
    del feat_array
    feat_df.insert(0, "candidate_entity_id", all_cand_ids)
    feat_df.insert(0, "source1_entity_id", all_s1_ids)
    if gt_map is not None:
        feat_df["label"] = all_labels

    return feat_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="max S1 rows per country, for a quick test run")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="candidate cap per entity (required at full scale -- see build_features_for_split docstring "
        "for measured recall-retention at 100/200/500/1000). No default on purpose: pick deliberately.",
    )
    parser.add_argument("--lanes", default=None,
                        help="name,addr,combined per-lane candidate budgets, e.g. 50,25,25 (overrides --top-k)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (fork); RAM, not cores, is the limit")
    parser.add_argument("--rerank-extra", default=None,
                        help="k_empty,k_indic: extra re-rank slots reserved for empty-address and "
                        "Indic-script-name candidates, e.g. 5,10 (needs --rerank-k)")
    parser.add_argument("--countries", default=None,
                        help="comma-separated subset of countries (default: every country in the data)")
    parser.add_argument("--reverse-k", type=int, default=0,
                        help="also add each record's top-K most similar businesses (record -> business pass)")
    parser.add_argument("--rerank-k", type=int, default=0,
                        help="second stage: add the top-K of the whole retrieved pool by char-3gram "
                        "name+address similarity (needs --lanes)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--only-entities", default=None, help="file with one S1 entity id per line")
    parser.add_argument("--dense-topk", default=None, help="parquet from kaggle_dense.py")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--dense-new", default=None, help="parquet (source1_entity_id, candidate_entity_id) of new pairs: build features for these only")
    args = parser.parse_args()
    lanes = tuple(int(x) for x in args.lanes.split(",")) if args.lanes else None

    df = build_features_for_split(args.split, max_s1_rows_per_country=args.limit, top_k=args.top_k, lanes=lanes,
                                  workers=args.workers, rerank_k=args.rerank_k,
                                  rerank_extra=tuple(int(x) for x in args.rerank_extra.split(",")) if args.rerank_extra else None,
                                  reverse_k=args.reverse_k,
                                  countries=args.countries.split(",") if args.countries else None,
                                  only_entities=set(pd.read_csv(args.only_entities, header=None)[0]) if args.only_entities else None,
                                  dense_extra=(pd.read_parquet(args.dense_new).groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
                                               if args.dense_new else
                                               load_dense_extra(args.dense_topk, args.dense_k) if args.dense_topk else None),
                                  extras_only=bool(args.dense_new))
    print(f"base features done: {len(df):,} rows", flush=True)
    if not args.dense_new:
        df = add_ambiguity_features(df)
    out_path = args.out or os.path.join(CACHE_DIR, f"{args.split}_features.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    print(f"\nWrote {len(df):,} rows to {out_path}", flush=True)
    if "label" in df.columns:
        print(f"label balance: {df['label'].mean()*100:.3f}% positive", flush=True)
