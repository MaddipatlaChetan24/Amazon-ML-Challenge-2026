"""
Pair features computed vectorised over a finished feature table, instead of
per pair inside build_features.py -- so they can be added (or improved)
without re-running blocking.

name_translit_char3: character-trigram cosine between the two names after
(a) transliterating Indic-script names to Latin (indic-transliteration, MIT)
and (b) dropping legal-form words and the transliteration's trailing
inherent vowel ("silvara" -> "silvar"). Measured on 3,000 real India
cross-script true matches: 57.0% score above the 99th percentile of random
pairs, vs 14.1% for raw transliteration and 18.5% for the multilingual
embedding. For Latin-script pairs it is simply a legal-word-free char-3gram
name similarity.
"""
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import scipy.sparse as sp
from indic_transliteration import sanscript
from sklearn.feature_extraction.text import HashingVectorizer

from embeddings import add_embedding_cosine
from preprocessing import HONORIFIC_PREFIX, SURFACE_TO_CANON, strip_diacritics

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
POSTHOC_FEATURE_NAMES = ["name_translit_char3", "addr_char3", "digit_affix_match", "digit_near_conflict"]

_SCRIPTS = {
    "DEVANAGARI": sanscript.DEVANAGARI, "BENGALI": sanscript.BENGALI, "TAMIL": sanscript.TAMIL,
    "TELUGU": sanscript.TELUGU, "KANNADA": sanscript.KANNADA, "GUJARATI": sanscript.GUJARATI,
    "MALAYALAM": sanscript.MALAYALAM, "GURMUKHI": sanscript.GURMUKHI, "ORIYA": sanscript.ORIYA,
}
_LEGAL = set(SURFACE_TO_CANON) | {"pra", "li", "kampani", "elaelapi"}
_CANDRA_O = str.maketrans({"ॉ": "ो", "ऑ": "ओ", "ૉ": "ો", "ઑ": "ઓ",
                           # Malayalam chillu letters -> consonant + virama
                           "ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്",
                           "ൽ": "ല്", "ൾ": "ള്", "ൿ": "ക്"})
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2 ** 20,
                        alternate_sign=False, norm="l2", lowercase=False, dtype=np.float32)


def _indic_script(text: str):
    for ch in text:
        if ord(ch) > 127 and ch.isalpha():
            first = unicodedata.name(ch, "").split(" ")[0]
            if first in _SCRIPTS:
                return first
    return None


def _translit_word(word: str, script: str) -> list:
    # English loanwords: candra-o (ॉ/ૉ) is the "o" in "logistics"; map to
    # the plain o-sign ITRANS knows, instead of losing the vowel.
    w = sanscript.transliterate(word.translate(_CANDRA_O), _SCRIPTS[script], sanscript.ITRANS)
    # anusvara/candrabindu before a consonant is an "n" in English words
    w = w.replace(".N", "n").replace("M", "n").lower()
    if script == "BENGALI":
        w = w.replace("v", "b")  # one letter for both b and v
    elif script == "TAMIL":
        # Tamil has one letter per voiced/unvoiced pair; English loanwords
        # mostly want the unvoiced sound ("lajisdhighs" -> "lajistiks").
        w = w.replace("dh", "t").replace("gh", "k").replace("bh", "p")
    parts = [p[:-1] if len(p) > 3 and p.endswith("a") else p for p in _NON_ALNUM.sub(" ", w).split()]
    return [p for p in parts if not p.startswith(("praiv", "praibh", "piraiv", "limit"))]


def normalize_name(text: str) -> str:
    # Word by word: ~60K India names mix Latin and Indic words, and the Latin
    # words must not go through transliteration (ITRANS treats capitals as codes).
    text = HONORIFIC_PREFIX.sub("", text or "")
    words = []
    for raw in text.split():
        script = _indic_script(raw)
        if script:
            words.extend(_translit_word(raw, script))
        else:
            words.extend(_NON_ALNUM.sub(" ", strip_diacritics(raw).lower()).split())
    return " ".join(w for w in words if w not in _LEGAL)


def _addr_text(tokens) -> str:
    return " ".join(tokens)


def text_vectors(split: str, src: str, kind: str) -> sp.csr_matrix:
    """Row-aligned (to {split}_{src}_clean.parquet) L2-normalised char-3gram
    vectors, float32, cached to disk. kind: 'translit' (normalised,
    transliterated name) or 'addr' (cleaned address tokens)."""
    path = f"{CACHE_DIR}/{split}_{src}_{kind}_char3.npz"
    if os.path.exists(path):
        return sp.load_npz(path)
    t0 = time.time()
    col, fn = ("name_comparison_text", normalize_name) if kind == "translit" else ("addr_tokens", _addr_text)
    values = pd.read_parquet(f"{CACHE_DIR}/{split}_{src}_clean.parquet", columns=[col])[col]
    keys = values.fillna("") if kind == "translit" else values.map(_addr_text)
    codes, uniques = pd.factorize(keys)
    mat = _HV.transform([fn(u) if kind == "translit" else u for u in uniques])[codes].tocsr().astype(np.float32)
    sp.save_npz(path, mat)
    print(f"  {split}_{src} {kind}: {len(values):,} rows vectorised in {time.time()-t0:.0f}s", flush=True)
    return mat


def add_char3_similarity(feat_df: pd.DataFrame, split: str, kind: str, out_col: str,
                         chunk: int = 2_000_000) -> pd.DataFrame:
    pos, mats = {}, {}
    for src in ("source1", "source2", "source3"):
        ids = pd.read_parquet(f"{CACHE_DIR}/{split}_{src}_clean.parquet", columns=["entity_id"])["entity_id"]
        pos[src] = pd.Series(np.arange(len(ids)), index=ids.values)
        mats[src] = text_vectors(split, src, kind)
    r1 = pos["source1"].reindex(feat_df["source1_entity_id"].values).to_numpy().astype(np.int64)
    cand = feat_df["candidate_entity_id"].values
    is_s2 = pd.Series(cand).str.startswith("S2-").to_numpy()
    rc = np.where(is_s2, pos["source2"].reindex(cand).to_numpy(),
                  pos["source3"].reindex(cand).to_numpy()).astype(np.int64)
    out = np.zeros(len(feat_df), dtype=np.float32)
    for src, mask in (("source2", is_s2), ("source3", ~is_s2)):
        rows = np.flatnonzero(mask)
        for start in range(0, len(rows), chunk):
            sel = rows[start:start + chunk]
            out[sel] = np.asarray(mats["source1"][r1[sel]].multiply(mats[src][rc[sel]]).sum(axis=1)).ravel()
    feat_df[out_col] = out
    return feat_df


def _affix_match(a, b) -> float:
    """1 if the two records share a house/plot number exactly; 1 also when one
    number is a prefix or suffix of the other ("157"/"57", "470"/"47") -- a
    dropped digit. Measured on v5 validation pairs with no exact number
    overlap: 16.5% of model-missed true matches vs 2.9% of hard non-matches.
    (One-digit substitutions were tested too and are MORE common among
    non-matches, so they are not counted.)"""
    for x in a:
        for y in b:
            if x == y or (len(x) >= 2 and len(y) >= 2 and
                          (x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x))):
                return 1.0
    return 0.0


def _near_conflict(a, b) -> float:
    """1 if the records carry house/plot numbers of the same length that are
    close but NOT equal (1..50 apart) and neither number appears on the other
    side -- the signature of planted decoys ("125" vs "128", "6846" vs "6848").
    Measured on v5 validation: 32.8% of false merges and 36.0% of hard
    non-matches, vs 1.3% of correctly matched true pairs."""
    sa, sb = set(a), set(b)
    for x in sa - sb:
        for y in sb - sa:
            if len(x) == len(y) and x.isdigit() and y.isdigit() and 0 < abs(int(x) - int(y)) <= 50:
                return 1.0
    return 0.0


def add_digit_affix_match(feat_df: pd.DataFrame, split: str) -> pd.DataFrame:
    digits = {}
    for src in ("source1", "source2", "source3"):
        x = pd.read_parquet(f"{CACHE_DIR}/{split}_{src}_clean.parquet", columns=["entity_id", "addr_digit_tokens_norm"])
        digits.update(zip(x["entity_id"], x["addr_digit_tokens_norm"]))
    pairs = list(zip(feat_df["source1_entity_id"].values, feat_df["candidate_entity_id"].values))
    feat_df["digit_affix_match"] = np.fromiter((_affix_match(digits[a], digits[b]) for a, b in pairs),
                                               dtype=np.float32, count=len(pairs))
    feat_df["digit_near_conflict"] = np.fromiter((_near_conflict(digits[a], digits[b]) for a, b in pairs),
                                                 dtype=np.float32, count=len(pairs))
    return feat_df


def add_posthoc_features(feat_df: pd.DataFrame, split: str) -> pd.DataFrame:
    for kind, col in (("translit", "name_translit_char3"), ("addr", "addr_char3")):
        t0 = time.time()
        feat_df = add_char3_similarity(feat_df, split, kind, col)
        print(f"{col} added in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    feat_df = add_digit_affix_match(feat_df, split)
    print(f"digit_affix_match + digit_near_conflict added in {time.time()-t0:.0f}s", flush=True)
    if all(os.path.exists(f"{CACHE_DIR}/{split}_{s}_embeddings.npy") for s in ("source1", "source2", "source3")):
        t0 = time.time()
        feat_df = add_embedding_cosine(feat_df, split)
        print(f"embedding_cosine filled in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"WARNING: {split} embeddings missing -- embedding_cosine left as built", flush=True)
    return feat_df
