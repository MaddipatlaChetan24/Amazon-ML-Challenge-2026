"""
Multi-key union blocking, with a same-field intersection fallback.

Justification (see RESEARCH_AND_STRATEGY_REPORT.md Section 5 and 8): name
token Jaccard and address token Jaccard are strong, largely non-overlapping
signals for true matches. This module builds one inverted index per
(country, field) over Source-2 + Source-3.

A first, cheap pass unions postings from a *pruned* index (tokens with
document frequency > max_df dropped) -- this handles the vast majority of
entities fast, since the median token document-frequency is only 1-6.

Measured failure mode of union-only blocking: a global max_df cutoff can
remove *every* token of a short name if each individual token happens to be
a moderately common word (e.g. "Custom Wealth Services LLC" -- "custom",
"wealth" and "services" can each individually exceed 2,000 occurrences
across 6M+ US records), even though the exact combination is highly
specific. This is common precisely when the address field is empty
(~3.3-3.4% of Source-2/3 rows), leaving no fallback signal.

Fix: for entities whose cheap pass returns too few candidates, fall back to
*intersecting* (not unioning) the FULL, unpruned postings of 2+ of the
entity's own tokens in the same field. Intersecting several individually-
common tokens is still highly precise -- a record sharing "custom" AND
"wealth" AND "services" simultaneously is a strong signal even though each
word alone is common.
"""
import math
from collections import defaultdict, Counter

BLOCK_FIELDS = ["name_core_tokens", "addr_tokens", "addr_digit_tokens", "addr_digit_tokens_norm"]
FALLBACK_FIELDS = ["name_core_tokens", "addr_tokens"]
# Ceiling on a token's FULL (unpruned) posting-list length before it's even
# considered for the fallback intersection. Checking len() on a list is O(1);
# the earlier version instead called set(idx[t]) on every one of an entity's
# tokens regardless of size, which for a common address word like "road" or
# "no" (500K-1.8M postings in India) meant materializing a multi-million-
# element Python set per token per fallback entity -- catastrophically slow
# across tens of thousands of fallback entities. Filtering by length first
# (free) before ever building a set (expensive) fixes this.
FALLBACK_TOKEN_DF_CAP = 50000


def build_full_index(source_dfs: dict, field: str) -> dict:
    """source_dfs: {'S2': df, 'S3': df} already filtered to one country.
    Returns {token: [entity_id, ...]}, unpruned."""
    postings = defaultdict(list)
    for df in source_dfs.values():
        for eid, toks in zip(df["entity_id"], df[field]):
            for t in toks:
                postings[t].append(eid)
    return dict(postings)


def prune_index(full_index: dict, max_df: int) -> dict:
    return {t: ids for t, ids in full_index.items() if len(ids) <= max_df}


# Per-field pruning thresholds. Measured: at max_df=2000, ~38-42% of
# entities have EVERY name_core_token pruned away (short names composed of
# a couple of individually-common-but-jointly-specific words like "Custom
# Wealth Services"), which forces the expensive intersection fallback for
# ~40% of all entities. Raising the name-field threshold reduces that
# "fully silent" rate and lifts recall (measured 98.2% -> 98.8% going from
# 2,000 to 12,000). Pushing it further (tested up to 50,000) backfires: the
# CHEAP union pass itself then routinely processes much larger posting
# lists across nearly all entities (not just the silent ones), which
# measured ~2.5x slower than 12,000 for barely more recall. 10,000 is the
# measured sweet spot -- most of the recall gain, negligible extra cost
# over the 2,000 baseline. addr_tokens/digit fields keep the tighter 2,000
# cap since their silent rate was already low (~8-9%).
DEFAULT_MAX_DF = {
    "name_core_tokens": 10000,
    "addr_tokens": 2000,
    "addr_digit_tokens": 2000,
    "addr_digit_tokens_norm": 2000,
}


def build_country_indexes(s2_country, s3_country, max_df=None) -> dict:
    """max_df: int (applied to all fields) or {field: int}. Defaults to
    DEFAULT_MAX_DF, tuned per-field from measured silent-rate data."""
    if max_df is None:
        max_df_by_field = DEFAULT_MAX_DF
    elif isinstance(max_df, dict):
        max_df_by_field = max_df
    else:
        max_df_by_field = {field: max_df for field in BLOCK_FIELDS}

    source_dfs = {"S2": s2_country, "S3": s3_country}
    full = {field: build_full_index(source_dfs, field) for field in BLOCK_FIELDS}
    pruned = {field: prune_index(full[field], max_df_by_field[field]) for field in BLOCK_FIELDS}
    return {"full": full, "pruned": pruned}


def get_candidates(row_tokens_by_field: dict, indexes: dict, min_candidates: int = 5, top_k: int = None) -> Counter:
    """row_tokens_by_field: {field: iterable_of_tokens} for one S1 row.
    Returns Counter(candidate_id -> proto-similarity score).

    top_k: if set, truncate to the top_k highest-scoring candidates before
    returning. This exists because median blocking candidate-set size was
    measured at 3,304-3,534 per entity -- full feature computation over
    2.2M entities x that scale is ~12 billion pairs (~1-24 hours depending
    on feature cost), not the "single-digit millions" originally assumed.
    Measured retention of already-blocking-found true matches at several
    cap sizes (20K-row sample, US): top-100 keeps 85.9%, top-200 keeps
    88.3%, top-500 keeps 92.1%, top-1000 keeps 94.9% (of the 97.49% overall
    recall, so e.g. top-200 -> ~86.1% effective final recall). No cap size
    is free -- pick based on available compute time, not defaulted silently.

    Fallback triggers per-field when THAT field contributed zero hits from
    its pruned index -- not only when the *total* candidate count is low.
    Measured bug this fixes: an entity can have plenty of candidates from
    its address tokens while its name tokens are *individually* all common
    enough to be pruned (e.g. "Custom Wealth Services" - each word exceeds
    2,000 occurrences on its own). Gating fallback on the global count alone
    let irrelevant address-side candidates mask a completely silent name
    field, hiding a true match whose OWN address happens to be empty (so it
    can only ever be found through the name)."""
    pruned = indexes["pruned"]
    full = indexes["full"]
    scores = Counter()
    field_hit = {}

    for field, tokens in row_tokens_by_field.items():
        index = pruned[field]
        hit = False
        for t in tokens:
            postings = index.get(t)
            if postings:
                # IDF-style weight: a token shared by fewer records is a much
                # stronger signal than one shared by thousands, and this is
                # known for free from len(postings) -- no candidate row
                # lookup needed. Flat +1-per-shared-token scoring (the
                # earlier version) ranked true matches far worse: measured
                # only 85.8% of blocking-found true matches landed in the
                # top-100 by flat count, vs 96.3% for an (expensive, full
                # Jaccard) re-rank. This weighting recovers most of that gap
                # at zero extra cost.
                weight = 1.0 / math.log(math.e + len(postings))
                for eid in postings:
                    scores[eid] += weight
                hit = True
        field_hit[field] = hit

    globally_starved = len(scores) < min_candidates

    for field in FALLBACK_FIELDS:
        if field_hit.get(field, False) and not globally_starved:
            continue  # this field already contributed and we have enough overall
        tokens = list(row_tokens_by_field.get(field, []))
        idx = full[field]
        # Cheap length check first (O(1) on a list) -- never materialize a
        # set for a token whose full posting list is huge.
        usable = [t for t in tokens if t in idx and len(idx[t]) <= FALLBACK_TOKEN_DF_CAP]
        usable.sort(key=lambda t: len(idx[t]))
        sets = [set(idx[t]) for t in usable[:4]]  # smallest few only
        if len(sets) >= 2:
            inter = sets[0].copy()
            for s in sets[1:]:
                inter &= s
                if not inter:
                    break
            # A 2+ token intersection is a strong, precise signal (see
            # module docstring) -- scored above typical single-token IDF
            # weights (which top out around 1.0 for a very rare token) so
            # these survive any downstream top-K truncation by score.
            for eid in inter:
                scores[eid] += 10.0
        elif len(sets) == 1:
            for eid in sets[0]:
                scores[eid] += 0.5

    if top_k is not None and len(scores) > top_k:
        return Counter(dict(scores.most_common(top_k)))

    return scores
