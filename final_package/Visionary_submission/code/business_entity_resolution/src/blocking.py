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
import heapq
import math
from collections import defaultdict, Counter

BLOCK_FIELDS = ["name_core_tokens", "addr_tokens", "addr_digit_tokens", "addr_digit_tokens_norm", "addr_digit_word"]
# Derived at index/query time from two stored fields rather than cached: a
# stored column would be ~120M short strings for one split.
DERIVED_FIELDS = {"addr_digit_word"}


def digit_word_keys(addr_tokens, digit_tokens_norm) -> set:
    """'281|pune'-style keys: a house/plot number paired with each address
    word. Each part alone is usually too common to retrieve on (pruned), the
    pair is specific. Measured: 29.3% of the India true matches that v5
    blocking missed share such a key (df<=50) with their Source-1 record --
    mostly Indic-script names whose only usable signal is a sparse address."""
    words = [w for w in addr_tokens if not w.isdigit()]
    return {f"{d}|{w}" for d in digit_tokens_norm for w in words}


def row_block_tokens(row) -> dict:
    """Per-record blocking tokens for every BLOCK_FIELDS entry; `row` maps the
    stored fields (a dict or a DataFrame row)."""
    out = {f: row[f] for f in BLOCK_FIELDS if f not in DERIVED_FIELDS}
    out["addr_digit_word"] = digit_word_keys(row["addr_tokens"], row["addr_digit_tokens_norm"])
    return out
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
        if field == "addr_digit_word":
            token_lists = map(digit_word_keys, df["addr_tokens"], df["addr_digit_tokens_norm"])
        else:
            token_lists = df[field]
        for eid, toks in zip(df["entity_id"], token_lists):
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
    "addr_digit_word": 50,
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
    for field in DERIVED_FIELDS - set(FALLBACK_FIELDS):
        full[field] = {}  # only the fallback fields need the unpruned index
    return {"full": full, "pruned": pruned, "n_records": len(s2_country) + len(s3_country)}


# Per-lane candidate budgets (name, address, combined). A single combined
# ranking lets neighbours sharing several common address tokens outrank an
# exact business-name match with an empty/partial address; separate lanes
# each keep their own top-K. Measured on 2,000 val entities/country, all
# true matches: recall US 84.8% -> 89.5%, India 79.7% -> 83.0%, with ~68
# candidates/entity instead of ~99 at a single combined top-100.
DEFAULT_LANES = (50, 25, 25)


def get_candidates(row_tokens_by_field: dict, indexes: dict, min_candidates: int = 5, top_k: int = None,
                   lanes: tuple = None, rerank=None, extra=None, fallback: bool = True) -> Counter:
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
    name_lane, addr_lane = Counter(), Counter()
    n_records = indexes.get("n_records", 1)

    for field, tokens in row_tokens_by_field.items():
        index = pruned[field]
        hit = False
        for t in tokens:
            postings = index.get(t)
            if postings:
                if lanes is not None:
                    if field == "name_core_tokens":
                        lw = math.log(n_records / len(postings))
                        for eid in postings:
                            name_lane[eid] += lw
                    else:
                        lw = 1.0 / math.log(math.e + len(postings))
                        for eid in postings:
                            addr_lane[eid] += lw
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

    for field in (FALLBACK_FIELDS if fallback else ()):
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

    if lanes is not None:
        k_name, k_addr, k_comb = lanes
        # Candidates reached only via the intersection fallback carry their
        # fallback score into the name lane (the fallback is name/addr-token based).
        name_score = {eid: v + 0.3 * addr_lane.get(eid, 0.0) for eid, v in name_lane.items()}
        for eid, s in scores.items():
            if eid not in name_lane and eid not in addr_lane:
                name_score[eid] = s
        # Ties are common at the lane cut-off; break them by entity id so the
        # candidate set does not depend on per-process string-hash order.
        keep = set(heapq.nlargest(k_name, name_score, key=lambda e: (name_score[e], e)))
        keep.update(heapq.nlargest(k_addr, addr_lane,
                                   key=lambda e: (addr_lane[e] + 0.05 * name_lane.get(e, 0.0), e)))
        keep.update(heapq.nlargest(k_comb, scores, key=lambda e: (scores[e], e)))
        if rerank is not None and scores:
            # Second stage: re-score the WHOLE retrieved pool with a richer
            # similarity (char-3gram name+address) and add its top picks.
            keep.update(rerank(scores))
        if extra:
            # pairs found in the other direction (record -> business), see
            # build_features' reverse pass; may lie outside this entity's pool
            keep.update(extra)
        return Counter({eid: scores[eid] for eid in sorted(keep)})

    if top_k is not None and len(scores) > top_k:
        return Counter(dict(scores.most_common(top_k)))

    return scores
