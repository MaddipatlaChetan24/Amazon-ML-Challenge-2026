# Our Approach — Business Entity Resolution
## ML Challenge 2026

---

## 0. The Core Bet

Every serious team at this contest will build *some* version of blocking →
classifier. What we bet on to differentiate isn't a fancier model — it's
**refusing to make a single design decision without measuring it against the
full dataset first**, and treating that discipline as the actual competitive
advantage. Concretely: every threshold, every feature, every architectural
choice below is traceable to a number we measured ourselves, not a default we
assumed. Several of those measurements overturned our own earlier assumptions
(see Section 5) — that's not a failure mode, that's the process working.

---

## 1. Understanding the Problem Before Writing Code

We read the contest as a two-stage pipeline challenge, not a black box, because
the README structures it that way deliberately:

- **`candidate_pairs.tsv` is never scored but is explicitly audited** — the
  organizers care about *why* your blocking recall is what it is, not just the
  final number. So we treated blocking as a first-class deliverable with its
  own measured metric (recall/pair-completeness), not a throwaway intermediate
  step.
- **Macro F₀.₅, computed per entity** — this means a singleton (5.6% of all
  entities) counts exactly as much as an 11-way match. A model that's "usually
  right" but sloppy on singletons underperforms a model that's deliberately
  conservative there. We validated this split explicitly (Section 6).
- **The ≤8B-parameter / MIT-Apache-2.0 constraint plus the no-external-lookup
  rule** together point toward transparent, classical ML — gradient-boosted
  trees on engineered features, not a frontier LLM or an external company
  database. We read this as the organizers testing ML engineering fundamentals,
  not prompt engineering.
- **France appears only in test** — a deliberate generalization probe. We
  treated this as a hard constraint on every design choice: nothing in our
  pipeline is allowed to hard-code `{US, India}` anywhere.

---

## 2. What We Actually Found in the Data (not assumed)

We loaded and statistically analyzed all ~26.4M rows across all 7 files before
writing a line of pipeline code — not a sample. Highlights that directly
shaped the architecture:

| Finding | Number | Why it mattered |
|---|---|---|
| Naive all-pairs comparison | ~22.8 trillion pairs (train) | Blocking isn't optional — it's the only thing that makes this computable |
| Blocking candidate-set size | median 3,304-3,534/entity | The real scale problem is candidate *volume*, not just recall — this drove the `top_k` truncation design |
| True singletons | 5.58% of entities | Given equal macro weight to an 11-way match — a disproportionately important slice |
| Cross-source noise pool | ~26% of every Source-2/3 file has no true match | A free, realistic hard-negative pool for training — no synthetic negative sampling needed |
| India non-Latin script usage | **9 scripts**, 31.11% of records, keyed to state | Discovered via exhaustive re-verification after an initial sampled estimate (7 scripts) *still* missed one entirely (Oriya) — see Section 5 |
| Legal-suffix variety | Position-transposed ("Memorial LLC Association" vs "Memorial Association LLC") | Ruled out trailing-only suffix stripping; drove a position-agnostic design |
| Source-3 alias construct | 1.96% of names, ~0% elsewhere | A structural pattern, not noise — needed explicit detection, not generic fuzzy matching |
| France | Same noise grammar, different vocabulary, verified on real test rows | Confirms a script-agnostic pipeline transfers without special-casing |

Full detail: `RESEARCH_AND_STRATEGY_REPORT.md`.

---

## 3. Architecture

Four deterministic stages plus one small, fast learned component — designed
so the ≤8B/license constraint applies to exactly one well-defined piece, not
the whole system:

```
Raw TSVs
   │
   ▼
[1] Preprocessing        — deterministic text cleaning, zero learned parameters
   │
   ▼
[2] Blocking              — deterministic candidate generation
   │
   ▼
[3] Feature Engineering   — deterministic similarity scoring
   │
   ▼
[4] Classifier             — the ONE learned component (HistGradientBoostingClassifier)
   │
   ▼
[5] Reconciliation + Output
```

This split matters beyond tidiness: because the classifier only ever sees
*normalized similarity scores* (a Jaccard of 0.85, a Jaro-Winkler of 0.9), not
raw text, it never has to generalize across languages or formats — the
deterministic preprocessing layer already did that work. This is *why* France
generalizes without a single France-specific rule anywhere in the codebase.

### 3.1 Preprocessing (`preprocessing.py`)

- Legal-suffix stripping, **position-agnostic** (handles the measured
  reordering pattern)
- Alias-marker detection (`aka/fka/dba/t-a`) with tail extraction — a
  Source-3-specific structural pattern, not generic noise
- Website-domain de-concatenation via a word-segmentation table built from
  this contest's *own* clean Source-1 text (no external dictionary)
- Address-abbreviation expansion (`rd`→`road`, French `rte`→`route`, etc.)
- Diacritic folding via stdlib `unicodedata` (not `unidecode` — same result,
  no GPL-license risk)
- **Deliberately does not attempt script transliteration** — tested directly
  against real data (Section 5) and found unreliable; deferred to a
  semantic-embedding approach instead

### 3.2 Blocking (`blocking.py`)

Multi-key inverted-index blocking (name/address/digit tokens), country-
partitioned (country discovered dynamically from the data — never a hardcoded
list), with:
- Per-field document-frequency pruning, **thresholds tuned from measured
  frequency distributions**, not guessed constants
- A same-field intersection fallback for entities a naive prune would starve
  (fixed a measured bug where common-but-jointly-specific name combinations
  were being silently dropped)
- IDF-weighted scoring (free, uses index metadata already known) for ranking
  candidates before truncation
- `top_k` truncation — required because unbounded candidate volume made
  full-scale feature computation take ~24 hours; the cap and its
  recall-vs-cost tradeoff were measured, not assumed

**Measured result: 97.49% pair completeness (blocking recall) on training
ground truth** (US 98.62%, India 95.81%) — reached after 6 iterations, each
driven by diagnosing a *specific* real failing example, not blind parameter
sweeps.

### 3.3 Feature Engineering (`features.py`)

26 features per candidate pair:
- Token Jaccard + asymmetric containment (name, address)
- Digit-token Jaccard (raw + leading-zero-normalized) — script-invariant by
  construction, survives transliteration
- Jaro-Winkler on cleaned text — catches character-level typos
- Legal-suffix Jaccard + exact-match flag
- Source, alias/website, and non-ASCII indicator flags
- **8 "ambiguity" features** — a candidate's rank, score gap to its nearest
  competitor, and z-score *relative to its sibling candidates for the same
  entity*, not just its own isolated similarity score. Motivated directly by
  measured candidate-set sizes in the thousands and F₀.₅'s 4:1 precision
  weighting — this is the feature set that lets the classifier learn
  "barely won" vs. "clearly dominant," which is exactly the distinction that
  determines false-merge risk.

### 3.4 Classifier (`train_model.py`)

`HistGradientBoostingClassifier` (scikit-learn) — trained in 106 seconds on
23.8M labeled candidate pairs (a stratified subsample, not the full training
set, for tractability under contest time constraints). One threshold, swept
over the full validation range and chosen to maximize **macro F₀.₅ computed
with the exact per-entity aggregation the contest uses** — not accuracy, not
a per-entity threshold sweep (which we identified and rejected as
statistically broken: fitting precision/recall from a single entity's
handful of candidates massively overfits).

**Measured result: macro F₀.₅ = 0.9537** on a held-out, entity-grouped
validation split (threshold 0.64). Singleton subset: 0.9489. Non-singleton
subset: 0.9542. No collapse on the hardest, highest-leverage slice.

### 3.5 Reconciliation + Output (`generate_predictions.py`)

A global 1-to-1 reconciliation pass — enforcing the measured ground-truth
property that every matched id has multiplicity exactly 1 — so the final
output never double-claims a Source-2/3 record across two different Source-1
entities.

---

## 4. Why This Should Win

1. **Every number in this document is measured on the real data, not
   asserted.** The methodology doc can back every claim with a specific
   verification, which is exactly what a reviewed final submission needs to
   survive scrutiny.
2. **Blocking and matching are separately validated**, matching the
   contest's own evaluation structure (pair-completeness for blocking,
   F₀.₅ for matching) — not conflated into one vague "it works" claim.
3. **Genuine France generalization**, not a France-specific hack — verified
   directly against real French test rows showing the identical noise
   grammar, and architecturally guaranteed by the deterministic-preprocessing
   + similarity-feature design.
4. **Full fair-play compliance with clear reasoning at every boundary** — no
   external lookups, no license-incompatible dependencies (checked, not
   assumed — e.g. we caught and avoided a GPL-licensed library), a model
   that's trivially under the parameter cap.
5. **Singleton behavior explicitly validated**, not just hoped for — 5.6% of
   the leaderboard's weight, verified separately from the overall average.
6. **Honest, documented iteration** — the methodology log records real bugs
   found and fixed (a runaway-memory blocking bug, a fallback-gating bug, a
   170GB memory crisis during test inference), each root-caused against a
   specific example rather than patched blind. That's what "reproducible and
   audit-ready" actually looks like, which is what the top-team package
   review is checking for.

---

## 5. What We Got Wrong Along the Way (and why that's worth stating)

We're including this because a submission that only shows the polished end
state is less credible than one that shows the process survived contact with
real data:

- Initially generalized India's script usage as "Devanagari/Hindi" from a
  handful of examples; a 2,000-row sample still found only 7 of the real 9
  scripts; only an exhaustive pass found the true picture (31.11%, 9 scripts,
  state-keyed). Recorded as a standing project rule: verify exhaustively,
  never generalize from a sample.
- A candidate-ranking fallback initially materialized full posting lists as
  Python sets regardless of size, causing a 27+ minute runaway on a job that
  should've taken 6. Root-caused via direct token/df inspection, not guessed.
- A test-inference run hit a genuine memory crisis (~170GB against 64GB
  physical RAM) from accumulating features as native Python lists at full
  scale — worked at training's smaller subsample scale, failed at test's
  full scale. Fixed by switching to `float32` numpy arrays flushed per
  country instead of one global Python list.
- Multiple rounds of externally-sourced "improvement" suggestions were
  evaluated on evidence, not adopted on assertion — several were tested
  directly against real data and rejected when they didn't hold up (e.g.
  Devanagari-only transliteration tested and found to only cover 57% of the
  script problem even where applicable; phonetic/Metaphone tokens tested and
  found to add zero recovery on the actual hard cases).

---

## 6. Honest Current Limitations / Next Steps If Time Allows

- Blocking recall (97.49%) is short of the ≥99% target we set ourselves —
  the residual gap is diagnosed (heavy typos + genuine script-switch cases)
  but not yet closed with a dedicated TF-IDF/n-gram blocking layer.
- **Update:** the multilingual cross-script embedding feature (planned,
  Section 9 of the research report) is now implemented (`embeddings.py`,
  `paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0) and wired into
  `features.py`/`build_features.py` as `embedding_cosine`. Validated
  correct on real data — self-similarity 0.9999999, cross-entity 0.092,
  and same-entity cross-script similarity 0.48 (English↔Hindi) / 0.145
  (English↔Tamil), both meaningfully above the ~0.07 unrelated-pair
  baseline. **Not yet run at full ~26.4M-row corpus scale** — the encoding
  step is compute-bound and wasn't benchmarked end-to-end in the dev
  environment (no GPU); the model this document's F0.5 number refers to
  was trained *before* this feature existed, so it must be retrained once
  the full embedding cache is built.
- The classifier was trained on a stratified subsample (~300K entities) of
  the 2.2M training entities for time tractability, not the full set — on
  a machine without that constraint, `build_features.py --split train`
  (no `--limit`) trains on the full set instead.

None of these are unknown risks — they're specific, scoped, and the first
thing we'd pick up with more time.
