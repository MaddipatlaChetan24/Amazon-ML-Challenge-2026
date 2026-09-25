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


def build_embedding_cache(batch_size: int = 256, sources=None):
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
    model = _get_model()
    for name in sources or SOURCE_FILES:
        in_path = f"{CACHE_DIR}/{name}_clean.parquet"
        out_path = f"{CACHE_DIR}/{name}_embeddings.npy"
        if os.path.exists(out_path):
            print(f"  {name}: embeddings already exist, skipping", flush=True)
            continue
        df = pd.read_parquet(in_path, columns=["name_comparison_text"])
        texts = df["name_comparison_text"].fillna("").tolist()
        print(f"  {name}: encoding {len(texts):,} texts...", flush=True)
        emb = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        ).astype(np.float32)
        np.save(out_path, emb)
        print(f"  {name}: saved {emb.shape} -> {out_path}", flush=True)


def load_embeddings(name: str) -> np.ndarray:
    """name: e.g. 'train_source1'. Returns the (n_rows, 384) float32 array,
    row-aligned to {name}_clean.parquet."""
    return np.load(f"{CACHE_DIR}/{name}_embeddings.npy")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == "__main__":
    build_embedding_cache()
