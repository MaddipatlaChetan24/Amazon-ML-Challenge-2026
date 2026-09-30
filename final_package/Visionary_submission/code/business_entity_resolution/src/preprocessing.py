"""
Text normalization for business_name / business_address fields.

Every rule here is directly justified by a pattern measured in
RESEARCH_AND_STRATEGY_REPORT.md (Section 3): legal-suffix variation and
reordering, the Source-3 alias/"aka" construct, website-domain-style names,
and script switching for India records. Nothing here does any external
lookup -- the word-segmentation dictionary is built from this contest's own
Source-1 (clean) text, not from an outside corpus.
"""
import re
import unicodedata
from collections import Counter

import jellyfish

WORD_RE = re.compile(r"[a-z0-9]+")
DIGIT_RE = re.compile(r"\d+")

ALIAS_PATTERN = re.compile(
    r"\b(?:a/?k/?a|f/?k/?a|d/?b/?a|t/a|formerly known as)\b",
    re.IGNORECASE,
)

WEBSITE_PATTERN = re.compile(r"\.(com|net|org|in|co)\b", re.IGNORECASE)

# Injected source-side noise: Source-2/3 names START with one of these in
# 4.1% / 4.7% of rows (~29-34K each on test) vs 0.10% in Source 1. Stripped in
# the transliterated-name normalisation (posthoc_features.normalize_name),
# which feeds both the second-stage blocking re-rank and a model feature.
HONORIFIC_PREFIX = re.compile(r"^\s*(?:the|mr|dr|smt|shri|sri|m\s*/\s*s)\b\.?\s+", re.IGNORECASE)

# Canonical legal-form -> surface variants (lowercase, punctuation-stripped).
# Removed position-agnostically from name tokens, since word-order
# transposition of the legal suffix is a measured noise pattern
# (e.g. "Memorial Association LLC" -> "Memorial LLC Association").
LEGAL_SUFFIX_GROUPS = {
    "limited": ["limited", "ltd"],
    "private": ["private", "pvt"],
    "incorporated": ["incorporated", "inc"],
    "corporation": ["corporation", "corp"],
    "company": ["company", "co"],
    "llc": ["llc"],
    "llp": ["llp"],
    "lp": ["lp"],
    "pllc": ["pllc"],
    "pc": ["pc"],
    "sarl": ["sarl"],
    "sasu": ["sasu"],
    "sas": ["sas"],
    "eurl": ["eurl"],
    "sci": ["sci"],
    "sa": ["sa"],
    "snc": ["snc"],
}
SURFACE_TO_CANON = {
    surf: canon for canon, surfs in LEGAL_SUFFIX_GROUPS.items() for surf in surfs
}

# Address-term abbreviations, applied token-by-token during address
# tokenization so "rd" and "road" (or "av" and "avenue") become the same
# token for Jaccard/blocking purposes. Covers US/generic English, Indian,
# and French address vocabulary observed in the report's noise taxonomy
# (Section 3.2). This is ordinary bounded linguistic knowledge -- the same
# category as the legal-suffix dictionary above -- not a lookup of any
# specific business record, so it stays within the fair-play boundary
# (report Section 10).
ADDRESS_ABBREV_MAP = {
    # generic English street types
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "pkwy": "parkway", "hwy": "highway",
    "apt": "apartment", "ste": "suite", "bldg": "building",
    # India-specific
    "nr": "near", "opp": "opposite", "extn": "extension", "sec": "sector",
    "blk": "block", "flr": "floor", "no": "number",
    # French
    "rte": "route", "bd": "boulevard", "ch": "chemin", "all": "allee",
    "imp": "impasse", "qu": "quai",
}


def strip_diacritics(text: str) -> str:
    """ASCII-fold Latin diacritics (e.g. French 'é' -> 'e'). Non-Latin
    scripts (Devanagari, Bengali, ...) pass through unchanged -- those are
    bridged later via a multilingual embedding feature, not transliteration.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list:
    """Lowercase alnum tokens. Non-ASCII (script-switched) text yields no
    tokens here by design -- that gap is intentional (see report Section 5)
    and is covered by a separate embedding-based feature downstream."""
    folded = strip_diacritics(text).lower()
    return WORD_RE.findall(folded)


def has_nonascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def extract_digit_tokens(text: str) -> frozenset:
    """Raw digit runs -- script-invariant, survives transliteration."""
    return frozenset(DIGIT_RE.findall(text))


def extract_digit_tokens_normalized(text: str) -> frozenset:
    """Same as above but with leading zeros stripped, to bridge noise like
    '208' vs '0208'."""
    out = set()
    for d in DIGIT_RE.findall(text):
        try:
            out.add(str(int(d)))
        except ValueError:
            out.add(d)
    return frozenset(out)


def phonetic_tokens(tokens: list) -> frozenset:
    """Metaphone code per token. NOT currently wired into NameCleaner.clean()
    or the blocking/feature pipeline -- tested directly against real data
    and found to add no measured value: (1) Jaro-Winkler, already a feature,
    scores 0.965 on the real "Resources"/"Rseouerees" typo pair this was
    meant to help with, so there's no gap there to fill; (2) tested against
    55 real training pairs with zero exact name-token overlap (the actually
    hard cases), phonetic tokens recovered 0 of them -- those cases are
    dominated by script-switching (Devanagari/Tamil/etc.), and Metaphone is
    an English-phonetics algorithm that produces empty output on non-Latin
    tokens just like exact matching does. Left here, documented and unused,
    so this doesn't get re-proposed and re-tested without cause -- it was a
    reasonable hypothesis, tested rigorously, and didn't hold up."""
    return frozenset(jellyfish.metaphone(t) for t in tokens if len(t) > 2)


def strip_legal_suffix(tokens: list) -> tuple:
    """Remove legal-suffix tokens from anywhere in the token list (not just
    the end -- word-order transposition is a measured pattern). Returns
    (core_tokens, canonical_suffixes_found)."""
    core, suffixes = [], []
    for tok in tokens:
        canon = SURFACE_TO_CANON.get(tok)
        if canon:
            suffixes.append(canon)
        else:
            core.append(tok)
    return core, frozenset(suffixes)


def detect_alias(raw_name: str):
    """Detect the Source-3 'aka/fka/dba/t-a/formerly known as' construct.
    Returns (is_alias, tail_text) -- tail_text is the segment after the
    marker, which measured examples show is consistently the true-entity
    name; the segment before the marker is a near-random decoy."""
    m = ALIAS_PATTERN.search(raw_name)
    if not m:
        return False, None
    tail = raw_name[m.end():].strip(" \t.,;:-")
    return True, tail if tail else None


def detect_website(raw_name: str):
    """Detect a domain-style name (e.g. 'mkjindustries.com') and strip the
    TLD. Returns (is_website, stem_text)."""
    m = WEBSITE_PATTERN.search(raw_name)
    if not m:
        return False, None
    stem = raw_name[: m.start()].strip(" \t.,;:-")
    stem = re.sub(r"^(www|http|https)\W*", "", stem, flags=re.IGNORECASE)
    return True, stem if stem else None


def build_word_freq(token_iterable) -> Counter:
    """Build a word-frequency table from this contest's own clean Source-1
    text (name + address tokens). Used only to de-concatenate run-on
    website-style names -- not an external data source."""
    return Counter(token_iterable)


def segment_concatenated(word: str, freq: Counter, total: int, max_word_len: int = 24):
    """Classic DP ('word-break') segmentation of a run-on lowercase string
    into the most probable sequence of dictionary words, using -log(freq)
    as cost. Falls back to a heavily-penalized single-char split for
    out-of-vocabulary stretches so the function always returns a result.
    """
    import math

    n = len(word)
    if n == 0:
        return []
    unseen_cost = math.log(total + n) * 2  # penalty for unknown fragments
    best_cost = [math.inf] * (n + 1)
    best_cut = [0] * (n + 1)
    best_cost[0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            frag = word[j:i]
            c = freq.get(frag)
            cost = best_cost[j] + (
                math.log(total / c) if c else unseen_cost * len(frag)
            )
            if cost < best_cost[i]:
                best_cost[i] = cost
                best_cut[i] = j
    parts = []
    i = n
    while i > 0:
        j = best_cut[i]
        parts.append(word[j:i])
        i = j
    parts.reverse()
    return parts


class NameCleaner:
    """Bundles alias/website detection + legal-suffix stripping into one
    per-row transform, with a word-frequency table for de-concatenation."""

    def __init__(self, word_freq: Counter = None):
        self.word_freq = word_freq or Counter()
        self.total = sum(self.word_freq.values()) or 1

    def clean(self, raw_name: str) -> dict:
        is_alias, alias_tail = detect_alias(raw_name)
        is_website, website_stem = detect_website(raw_name)

        comparison_text = raw_name
        if is_alias and alias_tail:
            comparison_text = alias_tail
        elif is_website and website_stem:
            stem_tokens = tokenize(website_stem)
            if len(stem_tokens) == 1 and len(stem_tokens[0]) > 6:
                segmented = segment_concatenated(
                    stem_tokens[0], self.word_freq, self.total
                )
                comparison_text = " ".join(segmented)
            else:
                comparison_text = website_stem

        tokens = tokenize(comparison_text)
        core_tokens, legal_suffixes = strip_legal_suffix(tokens)

        return {
            "is_alias": is_alias,
            "is_website": is_website,
            "comparison_text": comparison_text,
            "tokens": frozenset(tokens),
            "core_tokens": frozenset(core_tokens),
            "legal_suffixes": legal_suffixes,
            "has_nonascii": has_nonascii(raw_name),
        }


def expand_address_abbrevs(tokens: list) -> list:
    return [ADDRESS_ABBREV_MAP.get(t, t) for t in tokens]


def clean_address(raw_address: str) -> dict:
    tokens = expand_address_abbrevs(tokenize(raw_address))
    return {
        "tokens": frozenset(tokens),
        "digit_tokens": extract_digit_tokens(raw_address),
        "digit_tokens_norm": extract_digit_tokens_normalized(raw_address),
        "has_nonascii": has_nonascii(raw_address),
        "is_empty": raw_address.strip() == "",
    }
