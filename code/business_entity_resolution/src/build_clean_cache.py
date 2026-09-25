"""
Preprocessing pass: reads each raw source TSV, applies preprocessing.py to
business_name / business_address, and caches the derived fields as Parquet
so the blocking stage (next step) never has to re-run text cleaning.

Word-segmentation frequencies (used only to de-concatenate website-style
names like 'mkjindustries.com') are built from this contest's own clean
Source-1 text -- no external corpus.

Usage:
    python3 build_clean_cache.py
"""
import os
import sys
import time
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import NameCleaner, clean_address, tokenize

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

FILES = {
    "train_source1": f"{DATASET_DIR}/train/train_source1.tsv",
    "train_source2": f"{DATASET_DIR}/train/train_source2.tsv",
    "train_source3": f"{DATASET_DIR}/train/train_source3.tsv",
    "test_source1": f"{DATASET_DIR}/test/test_source1.tsv",
    "test_source2": f"{DATASET_DIR}/test/test_source2.tsv",
    "test_source3": f"{DATASET_DIR}/test/test_source3.tsv",
}


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def build_word_freq() -> Counter:
    """Corpus = Source-1 (the clean, deduplicated reference source) name AND
    address text, train split only. Large enough (2.2M names + addresses) to
    segment run-on company-name strings without touching outside data."""
    tr1 = load(FILES["train_source1"])
    freq = Counter()
    for name in tr1["business_name"]:
        freq.update(tokenize(name))
    for addr in tr1["business_address"]:
        freq.update(tokenize(addr))
    return freq


def process_file(key: str, path: str, cleaner: NameCleaner):
    t0 = time.time()
    df = load(path)
    n = len(df)

    name_core_tokens = [None] * n
    name_legal_suffixes = [None] * n
    name_is_alias = [False] * n
    name_is_website = [False] * n
    name_comparison_text = [None] * n
    name_has_nonascii = [False] * n

    addr_tokens = [None] * n
    addr_digit_tokens = [None] * n
    addr_digit_tokens_norm = [None] * n
    addr_has_nonascii = [False] * n
    addr_is_empty = [False] * n

    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()

    for i in range(n):
        nc = cleaner.clean(names[i])
        name_core_tokens[i] = sorted(nc["core_tokens"])
        name_legal_suffixes[i] = sorted(nc["legal_suffixes"])
        name_is_alias[i] = nc["is_alias"]
        name_is_website[i] = nc["is_website"]
        name_comparison_text[i] = nc["comparison_text"]
        name_has_nonascii[i] = nc["has_nonascii"]

        ac = clean_address(addrs[i])
        addr_tokens[i] = sorted(ac["tokens"])
        addr_digit_tokens[i] = sorted(ac["digit_tokens"])
        addr_digit_tokens_norm[i] = sorted(ac["digit_tokens_norm"])
        addr_has_nonascii[i] = ac["has_nonascii"]
        addr_is_empty[i] = ac["is_empty"]

    out = pd.DataFrame(
        {
            "entity_id": df["entity_id"],
            "country": df["country"],
            "business_name": df["business_name"],
            "business_address": df["business_address"],
            "name_core_tokens": name_core_tokens,
            "name_legal_suffixes": name_legal_suffixes,
            "name_is_alias": name_is_alias,
            "name_is_website": name_is_website,
            "name_comparison_text": name_comparison_text,
            "name_has_nonascii": name_has_nonascii,
            "addr_tokens": addr_tokens,
            "addr_digit_tokens": addr_digit_tokens,
            "addr_digit_tokens_norm": addr_digit_tokens_norm,
            "addr_has_nonascii": addr_has_nonascii,
            "addr_is_empty": addr_is_empty,
        }
    )

    out_path = os.path.join(CACHE_DIR, f"{key}_clean.parquet")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path)
    print(f"  {key}: {n:,} rows -> {out_path} ({time.time()-t0:.1f}s)")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Building word-segmentation frequency table from train_source1...")
    freq = build_word_freq()
    print(f"  vocabulary size: {len(freq):,} unique tokens")
    cleaner = NameCleaner(freq)

    print("Processing source files...")
    for key, path in FILES.items():
        process_file(key, path, cleaner)

    print("Done.")


if __name__ == "__main__":
    main()
