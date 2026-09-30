"""
Multilingual sentence embeddings -- the cross-script bridging feature
planned in RESEARCH_AND_STRATEGY_REPORT.md Section 9 but not built earlier
in the project due to time constraints (see PROGRESS_AND_METHODOLOGY_LOG.md).

Model: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers,
Apache-2.0, ~118M params, 384-dim output) -- comfortably under the contest's
8B-parameter cap, and trained across the languages/scripts this dataset
actually uses (Devanagari, Tamil, Kannada, Telugu, Bengali, Gujarati,
Malayalam, Gurmukhi, Oriya, French), unlike a single-script transliteration
library (tested and rejected earlier in this project -- see
preprocessing.py's docstring history).

Quick validation on real data pulled from this dataset (see chat/commit
history): cosine similarity for the SAME business name across scripts is
0.48 (English vs Hindi, "One Smart Producer...") and 0.145 (English vs
Tamil, "Arihant Foundation..."), both meaningfully above the ~0.07 baseline
for an unrelated pair. The signal is real but moderate, not decisive on its
own -- exactly the profile of a good ADDITIONAL feature for a tree model to
combine with the existing token/character features, not a replacement for
them.

Embeds `name_comparison_text` (not raw `business_name`) -- this is the
already-cleaned field with alias-tail extraction and website
de-concatenation already applied, so the embedding model sees the same
"best guess at the true name" the other features do.
"""
import os

import numpy as np
import pandas as pd

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

SOURCE_FILES = [
    "train_source1", "train_source2", "train_source3",
    "test_source1", "test_source2", "test_source3",
]


def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_embedding_cache(batch_size: int = 512, sources=None):
    """Encode name_comparison_text for every row of every cleaned source
    file and save as {split}_{source}_embeddings.npy, in the SAME row order
    as the corresponding _clean.parquet file (so alignment is a plain
    positional match, no join needed). Run once, after build_clean_cache.py.

    On a machine with a GPU, sentence-transformers will use it
    automatically; this is written to be hardware-agnostic. For the full
    ~26.4M-row corpus, budget real time for this step on CPU -- it was not
    benchmarked at full scale in this environment (see chat history: a
    4-sentence test took ~1.6s after a one-time ~188s model download/load,
    i.e. call this compute-bound and plan capacity accordingly, not
    assume-cheap). Increasing batch_size and running on GPU are the two
    levers if this is slow.
    """
    import time

    model = _get_model()
    print(f"embedding device: {model.device}", flush=True)
    for name in sources or SOURCE_FILES:
        in_path = f"{CACHE_DIR}/{name}_clean.parquet"
        out_path = f"{CACHE_DIR}/{name}_embeddings.npy"
        if os.path.exists(out_path):
            print(f"  {name}: embeddings already exist, skipping", flush=True)
            continue
        texts = pd.read_parquet(in_path, columns=["name_comparison_text"])["name_comparison_text"].fillna("")
        # Encode each distinct name once (names repeat heavily across rows).
        codes, uniques = pd.factorize(texts)
        print(f"  {name}: {len(texts):,} rows, {len(uniques):,} unique names", flush=True)
        # L2-normalised float16, written chunk by chunk: cosine becomes a
        # plain dot product and the full corpus fits on disk (~20GB) without
        # ever holding a float32 copy in RAM.
        uniq_emb = np.lib.format.open_memmap(out_path + ".uniq.tmp.npy", mode="w+", dtype=np.float16,
                                             shape=(len(uniques), EMBEDDING_DIM))
        t0 = time.time()
        chunk = 200_000
        for start in range(0, len(uniques), chunk):
            part = uniques[start:start + chunk].tolist()
            uniq_emb[start:start + len(part)] = model.encode(
                part, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float16)
            done = start + len(part)
            print(f"    {done:,}/{len(uniques):,} unique encoded, {done/(time.time()-t0):,.0f}/s", flush=True)
        out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.float16,
                                        shape=(len(texts), EMBEDDING_DIM))
        for start in range(0, len(texts), 1_000_000):
            out[start:start + 1_000_000] = uniq_emb[codes[start:start + 1_000_000]]
        out.flush()
        del out, uniq_emb
        os.replace(out_path + ".tmp.npy", out_path)
        os.remove(out_path + ".uniq.tmp.npy")
        print(f"  {name}: saved ({len(texts):,}, {EMBEDDING_DIM}) float16 -> {out_path}", flush=True)


def load_embeddings(name: str) -> np.ndarray:
    """name: e.g. 'train_source1'. Returns the (n_rows, 384) L2-normalised
    float16 array (memory-mapped), row-aligned to {name}_clean.parquet."""
    return np.load(f"{CACHE_DIR}/{name}_embeddings.npy", mmap_mode="r")


def add_embedding_cosine(feat_df: pd.DataFrame, split: str, chunk: int = 2_000_000) -> pd.DataFrame:
    """Fill feat_df['embedding_cosine'] in place for every (S1, candidate)
    pair, vectorised. Same value compute_features produces when embeddings
    are loaded at feature-build time, but seconds-to-minutes instead of a
    Python call per pair -- so an existing feature table can gain this
    feature without re-running blocking."""
    pos, embs = {}, {}
    for src in ("source1", "source2", "source3"):
        ids = pd.read_parquet(f"{CACHE_DIR}/{split}_{src}_clean.parquet", columns=["entity_id"])["entity_id"]
        pos[src] = pd.Series(np.arange(len(ids)), index=ids.values)
        embs[src] = load_embeddings(f"{split}_{src}")
    s1_rows = pos["source1"].reindex(feat_df["source1_entity_id"].values).to_numpy()
    cand = feat_df["candidate_entity_id"].values
    is_s2 = pd.Series(cand).str.startswith("S2-").to_numpy()
    c_rows = np.where(is_s2, pos["source2"].reindex(cand).to_numpy(), pos["source3"].reindex(cand).to_numpy())
    out = np.empty(len(feat_df), dtype=np.float32)
    for start in range(0, len(feat_df), chunk):
        sl = slice(start, start + chunk)
        a = np.asarray(embs["source1"][s1_rows[sl].astype(np.int64)], dtype=np.float32)
        b = np.where(is_s2[sl, None],
                     np.asarray(embs["source2"][np.where(is_s2[sl], c_rows[sl], 0).astype(np.int64)], dtype=np.float32),
                     np.asarray(embs["source3"][np.where(is_s2[sl], 0, c_rows[sl]).astype(np.int64)], dtype=np.float32))
        out[sl] = np.einsum("ij,ij->i", a, b)
    feat_df["embedding_cosine"] = out
    return feat_df


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == "__main__":
    build_embedding_cache()
