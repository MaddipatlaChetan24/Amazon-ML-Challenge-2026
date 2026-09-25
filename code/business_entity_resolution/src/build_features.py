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
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_country_indexes, get_candidates, BLOCK_FIELDS
from features import compute_features, FEATURE_NAMES, AMBIGUITY_FEATURE_NAMES
from embeddings import load_embeddings

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


def build_features_for_split(split: str, max_s1_rows_per_country: int = None, top_k: int = None):
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
    try:
        s1_emb_full = load_embeddings(f"{split}_source1")
        s2_emb_full = load_embeddings(f"{split}_source2")
        s3_emb_full = load_embeddings(f"{split}_source3")
        print("embeddings loaded.", flush=True)
    except FileNotFoundError:
        s1_emb_full = s2_emb_full = s3_emb_full = None
        print("no embedding cache found -- run embeddings.py first for the "
              "embedding_cosine feature; continuing without it.", flush=True)

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
    countries = s1["country"].unique()
    print(f"countries found: {list(countries)}", flush=True)

    for country in countries:
        out_s1_ids, out_cand_ids, out_features, out_labels = [], [], [], []
        t0 = time.time()
        s1_mask = (s1.country == country).to_numpy()
        s2_mask = (s2.country == country).to_numpy()
        s3_mask = (s3.country == country).to_numpy()
        s1c = s1[s1_mask].reset_index(drop=True)
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
            s1c = sampled.reset_index(drop=True)

        indexes = build_country_indexes(s2c, s3c)
        s2_store = ColumnStore(s2c, embeddings=s2c_emb)
        s3_store = ColumnStore(s3c, embeddings=s3c_emb)

        def get_row(eid):
            return s2_store.get(eid) if eid.startswith("S2-") else s3_store.get(eid)

        n = len(s1c)
        eids = s1c["entity_id"].tolist()
        cols = {f: s1c[f].tolist() for f in BLOCK_FIELDS}
        s1_lookup_cols = {f: s1c[f].tolist() for f in ROW_FIELDS}

        print(f"=== {country}: {n:,} S1 entities, index built in {time.time()-t0:.1f}s ===", flush=True)
        t0 = time.time()
        n_pairs = 0
        PROGRESS_EVERY = 100000

        for i in range(n):
            s1id = eids[i]
            s1_row = {f: s1_lookup_cols[f][i] for f in ROW_FIELDS}
            s1_row["embedding"] = s1c_emb[i] if s1c_emb is not None else None
            s1_row["name_core_tokens"] = frozenset(s1_row["name_core_tokens"])
            s1_row["name_legal_suffixes"] = frozenset(s1_row["name_legal_suffixes"])
            s1_row["addr_tokens"] = frozenset(s1_row["addr_tokens"])
            s1_row["addr_digit_tokens"] = frozenset(s1_row["addr_digit_tokens"])
            s1_row["addr_digit_tokens_norm"] = frozenset(s1_row["addr_digit_tokens_norm"])

            row_tokens = {f: cols[f][i] for f in BLOCK_FIELDS}
            cands = get_candidates(row_tokens, indexes, top_k=top_k)
            true_ids = gt_map.get(s1id, set()) if gt_map is not None else None

            for cid, score in cands.items():
                cand_row = get_row(cid)
                feats = compute_features(s1_row, cand_row, score)
                out_s1_ids.append(s1id)
                out_cand_ids.append(cid)
                out_features.append(feats)
                if true_ids is not None:
                    out_labels.append(1 if cid in true_ids else 0)
                n_pairs += 1

            if (i + 1) % PROGRESS_EVERY == 0:
                dt = time.time() - t0
                print(
                    f"    ...{i+1:,}/{n:,} S1 rows, {n_pairs:,} pairs so far, "
                    f"{dt:.1f}s ({dt/(i+1)*1e6:.1f} us/S1row)",
                    flush=True,
                )

        dt = time.time() - t0
        print(f"  {country} done: {n_pairs:,} pairs in {dt:.1f}s ({dt/n*1e6:.1f} us/S1row)", flush=True)

        # Flush this country to compact arrays now, then drop the raw
        # Python lists (out_features etc. go out of scope next loop iter)
        # so peak memory is ~1 country's worth of boxed floats, not all 3.
        all_feature_arrays.append(np.asarray(out_features, dtype=np.float32))
        all_s1_ids.extend(out_s1_ids)
        all_cand_ids.extend(out_cand_ids)
        if gt_map is not None:
            all_labels.extend(out_labels)
        del out_features, out_s1_ids, out_cand_ids, out_labels
        print(f"  {country} flushed to array, {psutil_rss_gb():.1f}GB RSS", flush=True)

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
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = build_features_for_split(args.split, max_s1_rows_per_country=args.limit, top_k=args.top_k)
    print(f"base features done: {len(df):,} rows; adding ambiguity features...", flush=True)
    df = add_ambiguity_features(df)
    out_path = args.out or os.path.join(CACHE_DIR, f"{args.split}_features.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    print(f"\nWrote {len(df):,} rows to {out_path}", flush=True)
    if "label" in df.columns:
        print(f"label balance: {df['label'].mean()*100:.3f}% positive", flush=True)
