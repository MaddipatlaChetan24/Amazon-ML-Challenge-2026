# Progress & Methodology Log
## What's done, what's left, and the reasoning behind each decision

*Living document — update this alongside `RESEARCH_AND_STRATEGY_REPORT.md` (which
holds the EDA/strategy) as the pipeline develops. This file tracks execution status
and the "why / why not" behind each choice, including the metric question raised in
review.*

---

## 1. Status Summary

| Stage | Status |
|---|---|
| Preprocessing (`preprocessing.py`, `build_clean_cache.py`) | **Done** — all 6 source files cleaned and cached to Parquet |
| Blocking (`blocking.py`) | **Working baseline, 97.49% recall confirmed** — 6 iterations (91.1% → 97.49%), ~1.5pp short of the ≥99% target with diminishing returns; decision point on whether to push further or move on (Section 3.1) |
| Feature engineering for matching | **Not started** |
| Matching classifier (GBT) | **Not started** |
| Threshold tuning for F₀.₅ | **Not started** |
| Global 1-to-1 reconciliation pass | **Not started** |
| `candidate_pairs.tsv` / `matching_results.tsv` generation (test set) | **Not started** |
| Singleton-specific and France-specific validation | **Not started** |

---

## 2. Preprocessing — Done

**What was built:** `preprocessing.py` normalizes `business_name` / `business_address`
into structured fields (core tokens with legal suffixes removed, alias-tail
extraction, website-name de-concatenation, digit tokens, script/casing flags).
`build_clean_cache.py` ran this over all 26.4M rows (~5.5 min) and cached the
result as Parquet so every later stage reads pre-cleaned fields instead of
re-parsing raw text.

**Why these specific transforms, not others:**

- **Legal-suffix stripping is position-agnostic** (removes the suffix token
  wherever it appears in the name, not just at the end), because the report's EDA
  found word-order transposition is a real, measured noise pattern (`"Memorial
  Association LLC"` → `"Memorial LLC Association"`). A trailing-only stripper would
  have silently failed on that case.
- **Alias-marker detection (`aka/fka/dba/t-a/formerly known as`) is a dedicated
  step**, not left to generic fuzzy matching, because it's measured as an
  almost-exclusively Source-3 phenomenon (1.8-2.0% of Source-3 names) where the
  segment *before* the marker is a near-random decoy with zero relation to the true
  name. No string-similarity metric would ever bridge that gap; only explicit
  detection + extraction of the tail segment does.
- **Website-name de-concatenation uses a word-segmentation dictionary built from
  this contest's own Source-1 text**, not an external dictionary or library
  (`wordninja`/`unidecode` aren't installed, and pulling one in would be an
  unnecessary new dependency for a problem the provided data already solves). This
  keeps the step fully fair-play compliant — no outside data, just a frequency
  table over data the contest already gave us.
- **Why NOT attempt script transliteration at this stage:** the non-Latin script
  used in India records is genuinely **multi-script, not just Devanagari/Hindi**.
  This went through three passes before landing on a trustworthy number, which is
  itself the point (worth recording as a process lesson, not just a data fact): a
  handful of sampled Devanagari examples in an early draft implied a single-script
  problem; a 2,000-row sample of India Source-2 found 7 scripts; only an
  **exhaustive** pass (every India record, all 6 files, ~10.5M rows) with proper
  Unicode-letter-category filtering (excluding zero-width/formatting characters
  that had inflated an intermediate version of this same check with spurious
  "scripts") found the real picture: **31.11% of all India records** contain
  non-Latin script content, across **9 scripts** — Devanagari, Kannada, Telugu,
  Tamil, Bengali, Gujarati, Malayalam, Gurmukhi, and Oriya (Odisha — missed
  entirely by the 2,000-row sample, since it happened not to include an Odisha
  record) — each keyed to the business's actual state, confirmed both statistically
  and on a specific example the user pointed out (`S3-45067784`, "Arihant
  Foundation Private Limited" in Coimbatore/Tamil Nadu, rendered as
  `அரிஹந்த் ஃபவுண்டேஷன் பிரைவேட் லிமிடெட்` — Tamil, not Devanagari). Also only
  visible at full-population scale: **zero records mix two different non-Latin
  scripts** in one record — script choice is clean and singular per record. This
  finding *reinforces* rather than changes the earlier decision: reliable script
  transliteration would need separate, correctly-detected handling for 9 scripts
  (each with its own transliteration conventions), which is a much larger and more
  fragile undertaking than picking one Devanagari-only library (as was briefly
  considered and rejected earlier in this project). Script-switched cases are
  deliberately left as-is in preprocessing and deferred to the **matching stage**,
  where a single small multilingual sentence embedding (MIT/Apache-licensed, e.g.
  `multilingual-e5-small`, already trained across all of these scripts) provides
  one cosine-similarity feature that bridges meaning uniformly, without needing to
  detect-then-transliterate per script. This is a cleaner boundary: preprocessing
  normalizes surface form; embeddings handle cross-script semantics.

---

## 3. Blocking — In Progress

**What was built:** `blocking.py` builds one inverted index per (country, field)
over Source-2 + Source-3, using 4 token types (name core tokens, address tokens,
raw digit tokens, leading-zero-normalized digit tokens). A candidate for a Source-1
entity is anything sharing a token with it.

**Why token/inverted-index blocking as the *first* method, not TF-IDF cosine or
embedding-based blocking:**
- It's the cheapest and most interpretable option — every candidate has an
  auditable "shared token X" reason, which matters because `candidate_pairs.tsv` is
  explicitly reviewed by the organizers for blocking quality.
- The report's EDA (Section 5) measured name/address token Jaccard as a ~20-36x
  separator between true and random pairs — token blocking is the most direct way
  to exploit exactly that signal.
- It's the same technique the leading open-source tools built for this scale
  (Splink, Sparkly) use as their primary method, not a fallback.
- TF-IDF cosine / embedding-based blocking are **deliberately deferred**, not
  rejected — they're the natural next layer for whatever residual gap remains
  after token blocking is pushed as far as it reasonably goes, since they cost
  more compute and are harder to audit. Building them now, before knowing the
  actual size and shape of the gap, would be premature optimization.

### 3.1 Iteration history (honest log, including the mistakes)

| # | Change | Result | Why |
|---|---|---|---|
| 1 | Union blocking, uniform `max_df=2000` across all 4 token fields | **91.06%** recall, 3.58% of entities got *zero* candidates | Baseline. Diagnosed the zero-candidate cases: short names like `"Custom Wealth Services LLC"` where every individual word (`custom`, `wealth`, `services`) is common enough to exceed the cutoff and get pruned from the index, even though the *combination* is highly specific — especially fatal when the address field is also empty (~3.3% of rows), leaving no fallback signal at all. |
| 2 | Added an intersection-based fallback (full unpruned index, 2+ tokens) for candidate-starved entities | **Killed after 27+ minutes**, still hadn't finished (previous baseline took 6.6 min) | Bug: the fallback converted entire posting lists into Python `set()`s regardless of size — for a common address word like `"road"` or `"no"` (500K-1.8M postings in India), that's materializing a multi-million-element set, repeated across tens of thousands of fallback-triggered entities. Root-caused via direct token/df inspection rather than guessing, then killed the runaway job instead of waiting. |
| 3 | Fixed: filter tokens by posting-list length (an O(1) check) *before* ever building a set | **94.11%** recall, 0.014% zero-candidate rate | Confirmed the specific fix worked on the exact failure case, then validated at full scale. But recall was still short of target, and re-checking the *same* "Custom Wealth Services" example showed it was **still** being missed. |
| 4 | Diagnosed further: fallback was gated on **total** candidate count (`≥5` overall), so an entity with plenty of (irrelevant) address-side candidates never triggered fallback even when its *name* field specifically found nothing. Changed the gate to trigger **per-field**, independent of the global count | **96.9%** recall in the first attempt at measuring this — *but see caveat below* | This is the change that actually fixed the "Custom Wealth Services" case (verified directly: it now returns 403 candidates including the true match). |
| 5 | Noticed name-field pruning at `max_df=2000` leaves **~38-42% of entities** with *every* name token pruned (short names built from moderately-common words), forcing the expensive fallback path for nearly half of all entities. Tuned per-field `max_df` (measured a df/silent-rate curve first, rather than guessing): pushed name-field threshold up to reduce how often fallback is needed at all | Small-sample tests (8K-50K rows): recall up to **98.8%**, cost roughly flat vs. `max_df=2000` at values 5,000-12,000, but **catastrophically slower at 50,000** (the cheap union pass itself starts processing huge posting lists for nearly every entity, not just the deficient ones) | Settled on `name_core_tokens` cap = 10,000 as the measured sweet spot: most of the recall gain, negligible extra cost. |
| — | **Process note:** the first full-scale run reporting "96.9%" (step 4) actually still had `max_df=2000` hardcoded in the *evaluation script* — the per-field tuning from step 5 lived in `blocking.py` but I forgot to remove the override in the script calling it. Caught this by manually re-deriving why a specific "obviously findable" pair was still failing in a supposedly-fixed run, and found the evaluation script wasn't using the code I thought it was. | Corrected and re-run (see next row) | Flagging this plainly rather than quietly restating the number: the 96.9% figure was real, but measured under the *previous* (step-4-only) configuration, not the fully-tuned one. |
| 6 | Corrected full-scale run with the actually-tuned per-field `max_df` applied | **97.49%** recall overall (US 98.62%, India 95.81%), 0.008% zero-candidate rate | This is the confirmed, current number. Two things worth being honest about: (a) the gain over step 4 (96.9% → 97.49%, +0.55pp) is smaller than the small-sample tests suggested (those showed up to 98.8% on an 8K-50K row *US-only* subsample — full-scale India dragged the overall number down, and small samples aren't fully representative); (b) it came at a real cost — average candidates/entity nearly **doubled**, from 2,837 to 5,425, because letting more moderately-common name tokens survive pruning also means the *cheap* union pass pulls in more candidates for entities that already had plenty. |

**Current config:** `name_core_tokens` max_df = 10,000; `addr_tokens` /
`addr_digit_tokens` / `addr_digit_tokens_norm` max_df = 2,000; per-field
intersection fallback for name/address fields when that field individually found
zero hits.

**Where this leaves us relative to the report's ≥99% target:** still **~1.5
points short**, and the last two iterations show clearly diminishing returns —
pushing the token-pruning threshold further costs increasingly more (in candidate
volume and runtime) for less incremental recall. Continuing to tune this one knob
is not the efficient next move. The residual ~2.5% miss rate is exactly the kind of
gap the report earmarked for a *different* mechanism, not more threshold-tuning:

**What's *not yet* tried for blocking, and why it's next in line rather than done
now:** character n-gram / TF-IDF cosine blocking (à la Sparkly) and/or the
multilingual-embedding channel, targeted specifically at the residual miss set
(heavy single-word typos and script-switch cases that survive exact-token blocking
regardless of the df threshold) — no point building either blind before this
checkpoint confirmed what's actually still missing and that it's not just a
threshold-tuning problem.

**Decision point flagged to the user:** whether to keep pushing blocking recall
toward 99% (diminishing returns, rising engineering complexity for n-gram/embedding
blocking) or treat 97.5% as good enough for now and move to feature engineering /
the matching classifier, where there's comparatively more untouched scoring impact
left (the whole precision side of F₀.₅ hasn't been worked on at all yet). Blocking
recall caps the *ceiling*; the matching stage determines how close to that ceiling
the actual leaderboard score lands.

### 3.2 Country-partitioning edge cases — verified exhaustively, not sampled

Blocking partitions candidates by `country` first (Section 8 of the report). Since
that's a hard partition — a candidate in a different country value is *never*
compared at all — it's worth stress-testing directly rather than trusting the
earlier 50,000-row sample:

| Edge case | Result | How verified |
|---|---|---|
| Same name, different countries | Structurally impossible to false-merge | Partition means they're never compared |
| Wrong/missing `country` value | **Does not occur anywhere in the data** | Checked all 6 files' exact string values: zero empty/whitespace values, exactly `{US, India}` in train and `{US, India, France}` in test — no typos or casing variants |
| Country disagreement on a true match (identical name/address, different country label) | **Confirmed absent** | Checked **all 7,638,365** training match edges individually (not a sample) — 0 cross-country mismatches |
| Multinational company matched across countries | Not applicable to this task's definition | Same exhaustive check — the ground truth itself never links a US record to an India record, even implicitly |
| Registration vs. operating country | Not representable | Only one `country` column exists; there's no second field to disagree with |

**Gap this surfaced:** `blocking.py` itself never hardcodes country values (it
partitions on whatever groups it's given), but the *dev/eval scripts*
(`eval_blocking2.py` etc.) hardcode `for country in ["US", "India"]` — correct for
measuring recall against training ground truth (which only has those two), but it
would silently skip France entirely if reused as-is against the test set. Not a bug
yet since no test-facing blocking script exists yet (see Section 5), but flagged
here so it isn't copy-pasted carelessly when that script is written — it needs
`df['country'].unique()`, not a literal list.

---

## 4. Why Blocking Is Evaluated on *Recall*, Not F₀.₅ / Precision / Accuracy

This is worth stating explicitly since it's easy to conflate with the contest's
actual scoring metric.

### 4.1 Recall (pair completeness) is the only metric that matters *for this stage*

Blocking's job is narrowly defined: **propose a candidate set for each Source-1
entity such that, if a true match exists, it's in there somewhere.** It does not
decide which candidates are correct — that's the matching classifier's job,
downstream. So the only question blocking needs to answer is:

> Of all the true match edges that exist, what fraction did I actually place into
> some candidate set?

That fraction — **recall**, also called *pair completeness* in the record-linkage
literature (Papadakis et al., ACM Computing Surveys 2021, cited in the report) — is
exactly and only what blocking should be measured on.

### 4.2 The key asymmetry: recall failures here are permanent; precision failures are not

This is the actual justification, not just convention:

- **If blocking fails to include the true match in the candidate set, no
  classifier — however good — can ever recover it.** That's a hard, permanent
  false negative baked in before the matching model even runs.
- **If blocking includes a wrong candidate (a precision problem), the downstream
  classifier can still fix it** by scoring that pair low and excluding it from
  `matching_results.tsv`. A bloated, noisy candidate set costs compute and makes
  the classifier's job harder, but it is *recoverable*.

Recall failures are irreversible; precision failures are fixable later. That
asymmetry is why blocking is evaluated on recall (with reduction ratio / average
candidate-set size tracked as a secondary **cost** metric, not a correctness
metric), while the *matching* stage afterward is evaluated on the precision-heavy
F₀.₅.

### 4.3 Why not accuracy

Accuracy is meaningless here because of the extreme class imbalance inherent to
entity resolution at this scale: there are ~22.8 trillion *possible* Source-1 ×
(Source-2/3) pairs in training, of which only 7.6 million are true matches. A
blocker that proposed *zero* candidates for every entity would still be "correct"
on well over 99.9999% of all possible pairs (the true negatives) by pure accuracy,
while being completely useless. Recall (and reduction ratio, which tracks how much
of that trillion-pair space got discarded) are the standard metrics for exactly
this reason in the blocking/candidate-generation literature — accuracy simply
doesn't distinguish a useful blocker from a useless one at this scale.

### 4.4 Why not F₀.₅ at the blocking stage

F₀.₅ requires a *binary decision* per pair (matched or not) to compute precision
and recall against. Blocking doesn't make that decision — it only proposes a set of
candidates for a later classifier to decide on. F₀.₅ is the right metric for
`matching_results.tsv` (the final, scored output) precisely because that file
*does* contain binary per-pair decisions. Applying it to blocking's output would
require pretending every candidate is a "predicted match," which isn't what
`candidate_pairs.tsv` claims to be (the README is explicit that this file is never
scored, for exactly this reason).

### 4.5 How the two connect — why the ≥99% recall target in the report isn't arbitrary

Blocking recall sets a **hard ceiling** on the entire pipeline's achievable recall:
if blocking only supplies 96% of true matches, the matching classifier — no matter
how well-tuned — can recover *at most* 96% of them, because the other 4% were never
candidates in the first place. Since F₀.₅ still has a recall term (weighted 1:4
against precision, but non-zero), a capped blocking recall directly caps the
maximum possible final score. That's the whole reason the report set a ≥99%
pair-completeness target: to push this ceiling high enough that it stops being the
binding constraint, leaving precision-tuning (which the classifier and threshold
*can* control) as the main lever on the final score.

---

## 5. What's Left To Do

1. **Finish confirming blocking recall** under the fully-tuned config (run in
   progress); if still short of ~99%, characterize the exact residual miss set
   (already started — see Section 3) and decide whether a TF-IDF/n-gram blocking
   layer is worth the added compute for the remaining gap.
2. **Generate `candidate_pairs.tsv`-shape output** from the tuned blocker for both
   train (for validation) and eventually test.
3. **Feature engineering** on candidate pairs: name/address token Jaccard,
   character-level similarity (Jaro-Winkler / Levenshtein), digit-token overlap,
   legal-suffix match/mismatch, alias-tail comparison, source-pair indicator, and
   the multilingual-embedding cosine feature for cross-script cases.
4. **Train the GBT matching classifier** on the resulting feature table, using the
   ground truth as labels and the ~26% naturally-occurring distractor records as
   free hard negatives.
5. **Tune the decision threshold specifically for macro F₀.₅** on a held-out
   validation split — not accuracy or F1 — per Section 4 above, since this is
   where precision-weighting actually needs to be applied.
6. **Validate singleton behavior and France generalization separately**, not just
   as part of an overall average (per the report's Section 4.1 and Section 9.6),
   since both are disproportionately important slices that an overall macro number
   would hide problems in.
7. **Add the global 1-to-1 reconciliation pass** so no Source-2/3 ID is claimed by
   two different Source-1 rows in the final output, matching the measured
   ground-truth structure (every matched ID has multiplicity exactly 1).
8. **Run the full pipeline on the test set** to produce `matching_results.tsv` and
   `candidate_pairs.tsv`, then validate with `utils/validate_submission.py
   --check-ids`.
