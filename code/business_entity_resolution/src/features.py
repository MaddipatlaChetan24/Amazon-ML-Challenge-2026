"""
Pairwise feature computation for (Source-1 entity, candidate) pairs.

Every feature here is directly justified by a measured pattern in
RESEARCH_AND_STRATEGY_REPORT.md Section 5 / 9.2:
- name/address token Jaccard: measured ~20-36x separation between true and
  random pairs -- the two strongest single features.
- digit-token Jaccard (raw + leading-zero-normalized): script-invariant,
  bridges the ~14% of true matches with zero name-token overlap due to
  transliteration.
- Jaro-Winkler on the cleaned comparison text: catches character-level typos
  that survive tokenization (e.g. "Rsourcaes" vs "Resources").
- legal-suffix match/mismatch: a mismatch (e.g. "Inc" vs "Trust") is itself
  informative, not just noise to strip.
- cand_is_alias / cand_is_website / cand_nonascii: source-fingerprint flags
  from Section 6 -- Source-3-only alias construct, ~4% website-style names,
  transliteration -- so the classifier can learn source-conditioned behavior
  instead of one-size-fits-all thresholds.
- blocking_score: the number of distinct blocking keys shared, already
  computed for free during candidate generation -- a cheap proto-similarity
  the tree model can refine rather than discard.
"""
import jellyfish
import numpy as np

FEATURE_NAMES = [
    "name_jaccard",
    "name_containment_s1_in_cand",
    "name_containment_cand_in_s1",
    "addr_jaccard",
    "digit_jaccard",
    "digit_norm_jaccard",
    "suffix_jaccard",
    "suffix_match",
    "suffix_compatibility",
    "name_jw",
    "addr_jw",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_len_ratio",
    "addr_len_ratio",
    "s1_addr_empty",
    "cand_addr_empty",
    "s1_nonascii",
    "cand_nonascii",
    "cand_is_alias",
    "cand_is_website",
    "cand_is_s2",
    "blocking_score",
    "embedding_cosine",
    "s1_generic_name",
    "cand_generic_name",
]

# Ambiguity features: computed in a second pass (build_features.py,
# add_ambiguity_features) after the base features above, since they compare
# a candidate to its SIBLINGS -- the other candidates blocked for the same
# source1_entity_id -- not just to the S1 entity in isolation. Every feature
# above answers "how similar is this pair," independent of every other
# candidate; these answer "how much does this candidate stand out from the
# competition," which is what actually determines whether a false merge is
# likely. Directly motivated by two measured facts: median blocking
# candidate-set size is in the thousands (so most S1 entities have real
# competition to disambiguate among), and F0.5 punishes a false merge 4x
# harder than a missed match -- so knowing "this candidate barely beat 4
# near-identical others" vs. "this candidate clearly dominates" is exactly
# the signal that should separate a confident prediction from a risky one.
AMBIGUITY_FEATURE_NAMES = [
    "amb_rank",
    "amb_group_size",
    "amb_group_mean",
    "amb_group_std",
    "amb_zscore",
    "amb_gap_to_better",
    "amb_gap_to_worse",
    "amb_top1_score",
]


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union


def _containment(a: frozenset, b: frozenset) -> float:
    """Asymmetric overlap: fraction of A's tokens also present in B. Unlike
    Jaccard, this isn't penalized by B being much larger -- catches cases
    like "Apex" (1 token) vs "Apex Industries" (2 tokens), where Jaccard
    caps at 0.5 even though "Apex" is fully contained in the other name."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb, 1)


def _char_ngram_jaccard(a: str, b: str, n: int) -> float:
    """Character n-gram Jaccard on the cleaned comparison text. Catches
    typos that survive tokenization (shared substrings persist even when a
    letter is scrambled/dropped inside a word), a complementary signal to
    Jaro-Winkler rather than a replacement for it -- n-grams are more
    tolerant of transpositions/insertions, JW weights prefix matches more.
    Deferred in earlier iterations purely for compute cost under a hard
    deadline; added now that isn't the binding constraint."""
    a, b = a.lower(), b.lower()
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    a_ng = {a[i : i + n] for i in range(len(a) - n + 1)}
    b_ng = {b[i : i + n] for i in range(len(b) - n + 1)}
    inter = len(a_ng & b_ng)
    return inter / (len(a_ng) + len(b_ng) - inter)


# Legal-form equivalence groups: forms that represent the SAME underlying
# corporate structure across jurisdictions (US/India/France), even though
# they share no token. E.g. US "LLC" / India "Private"+"Limited" / France
# "SARL"/"EURL" are all limited-liability-company equivalents; this is
# general comparative-business-law knowledge, not a lookup of any specific
# record (same category as the legal-suffix dictionary itself -- see report
# Section 10's fair-play boundary discussion). Grouped from the canonical
# forms in preprocessing.py's LEGAL_SUFFIX_GROUPS.
SUFFIX_COMPAT_GROUPS = [
    {"llc", "limited", "private", "sarl", "eurl"},  # limited-liability company equivalents
    {"incorporated", "corporation", "company", "sa"},  # corporation-style entities
    {"llp", "lp", "pllc", "snc"},  # partnership-style entities
    {"sasu", "sas"},  # French simplified joint-stock company
    {"sci"},  # French civil real-estate company -- structurally distinct, own group
]


def _suffix_compatibility(a: frozenset, b: frozenset) -> float:
    """Graded legal-form compatibility, more informative than exact-match
    alone: 1.0 if any canonical suffix is shared (same as suffix_match),
    0.5 if they're different canonical forms but from the same equivalence
    group (e.g. US LLC vs France SARL -- structurally the same, textually
    unrelated), 0.0 if both sides have a suffix but from different,
    incompatible groups, and a neutral 0.5 if either side has no detected
    suffix at all (can't judge compatibility either way)."""
    if not a or not b:
        return 0.5
    if a & b:
        return 1.0
    for group in SUFFIX_COMPAT_GROUPS:
        if (a & group) and (b & group):
            return 0.5
    return 0.0


# Generic/common name words, taken directly from this dataset's own
# measured top-token frequency lists (report Section 3 / the df-threshold
# EDA), not an invented list -- words so common they carry little
# discriminative power on their own, which is exactly the France 2-letter
# abbreviation flood (CC/PC/AC/LC) and English "Urgent Care"/"Primary Care"-
# style ambiguity problem described in the report's blocking section.
GENERIC_NAME_WORDS = frozenset({
    "center", "partners", "group", "care", "services", "holdings", "associates",
    "health", "india", "industries", "enterprises", "brothers", "ventures",
    "solutions", "trading", "public", "exports", "sri", "and", "of", "the",
    "club", "association", "society", "trust", "foundation", "medical",
    "dental", "clinic", "hospital", "school", "academy", "institute", "church",
})


def _is_generic_name(core_tokens: frozenset, comparison_text: str) -> float:
    """Fraction of core tokens that are generic/common words, treated as a
    continuous [0,1] signal (not a hard boolean) so the tree model can use
    it proportionally rather than as a cliff-edge cutoff. Very short names
    (<=2 characters, e.g. France's literal "CC"/"PC") are flagged at 1.0
    regardless of token content -- length alone is the tell there."""
    stripped = comparison_text.replace(" ", "")
    if len(stripped) <= 2:
        return 1.0
    if not core_tokens:
        return 0.0
    return sum(1 for t in core_tokens if t in GENERIC_NAME_WORDS) / len(core_tokens)


def _embedding_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """See embeddings.py for the model and validation. Zero if either
    embedding is missing/zero (e.g. embedding cache not built -- fails soft
    to 0.0 rather than crashing, so this feature can be added incrementally
    without breaking runs that predate it)."""
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_features(s1_row: dict, cand_row: dict, blocking_score: float) -> list:
    """s1_row / cand_row: plain dicts with the cached preprocessing fields
    (see build_clean_cache.py column list). Returns a list in FEATURE_NAMES
    order (faster to build a flat list than a dict per pair at 10M+ scale)."""
    s1_name_core = s1_row["name_core_tokens"]
    c_name_core = cand_row["name_core_tokens"]
    s1_addr_tok = s1_row["addr_tokens"]
    c_addr_tok = cand_row["addr_tokens"]
    s1_digit = s1_row["addr_digit_tokens"]
    c_digit = cand_row["addr_digit_tokens"]
    s1_digit_norm = s1_row["addr_digit_tokens_norm"]
    c_digit_norm = cand_row["addr_digit_tokens_norm"]
    s1_suffix = s1_row["name_legal_suffixes"]
    c_suffix = cand_row["name_legal_suffixes"]

    name_jw = jellyfish.jaro_winkler_similarity(
        s1_row["name_comparison_text"].lower(), cand_row["name_comparison_text"].lower()
    )
    addr_jw = jellyfish.jaro_winkler_similarity(
        s1_row["business_address"].lower(), cand_row["business_address"].lower()
    )

    s1_comp = s1_row["name_comparison_text"]
    c_comp = cand_row["name_comparison_text"]

    return [
        _jaccard(s1_name_core, c_name_core),
        _containment(s1_name_core, c_name_core),
        _containment(c_name_core, s1_name_core),
        _jaccard(s1_addr_tok, c_addr_tok),
        _jaccard(s1_digit, c_digit),
        _jaccard(s1_digit_norm, c_digit_norm),
        _jaccard(s1_suffix, c_suffix),
        1.0 if s1_suffix == c_suffix else 0.0,
        _suffix_compatibility(s1_suffix, c_suffix),
        name_jw,
        addr_jw,
        _char_ngram_jaccard(s1_comp, c_comp, 3),
        _char_ngram_jaccard(s1_comp, c_comp, 4),
        _len_ratio(s1_row["business_name"], cand_row["business_name"]),
        _len_ratio(s1_row["business_address"], cand_row["business_address"]),
        float(s1_row["addr_is_empty"]),
        float(cand_row["addr_is_empty"]),
        float(s1_row["name_has_nonascii"]),
        float(cand_row["name_has_nonascii"]),
        float(cand_row["name_is_alias"]),
        float(cand_row["name_is_website"]),
        1.0 if cand_row["entity_id"].startswith("S2-") else 0.0,
        float(blocking_score),
        _embedding_cosine(s1_row.get("embedding"), cand_row.get("embedding")),
        _is_generic_name(s1_name_core, s1_comp),
        _is_generic_name(c_name_core, c_comp),
    ]
