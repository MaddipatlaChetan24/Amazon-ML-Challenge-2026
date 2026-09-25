# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

We built a four-stage deterministic pipeline (preprocessing → multi-key blocking →
feature engineering → gradient-boosted classifier) grounded in an exhaustive
statistical audit of the full 26.4M-row dataset rather than assumptions from
sampling. Blocking achieves 97.49% pair-completeness (recall) on training ground
truth; the final classifier, trained on similarity features (not raw text),
reaches macro F0.5 = 0.9537 on a held-out, entity-grouped validation split, with
no meaningful gap between singleton and non-singleton entities (0.9489 vs. 0.9542).
Because the model consumes only normalized similarity scores, it required no
country-specific logic and generalizes to France (0 training examples) by
construction, not by special-casing.

---

## 2. Methodology

### 2.1 Problem Analysis

Full-corpus (not sampled) EDA findings that directly drove design decisions:

- **Scale:** ~26.4M total rows across 7 files; naive all-pairs comparison is
  ~22.8 trillion pairs for train alone — blocking is mandatory, not optional.
- **Ground truth structure:** mean 3.46 matches/entity, 5.58% true singletons
  (identical between India and US to 3 decimal places, evidence of a single
  country-agnostic generation process), 80.5% of matched entities have hits in
  *both* other sources, and every matched id has multiplicity exactly 1 (never
  claimed by two different Source-1 entities) — enforced in our final output via
  a global 1-to-1 reconciliation pass.
- **Noise taxonomy:** legal-suffix variation with word-order transposition
  ("Memorial Association LLC" ↔ "Memorial LLC Association"), an alias construct
  ("X aka/fka/dba Y") that is almost exclusively a Source-3 phenomenon (1.96% of
  Source-3 names vs. ~0.001% of Source-1/2), website-domain-style concatenated
  names (~4% of Source-2/3), and a per-source casing fingerprint (Source-2
  addresses are 63% fully uppercase vs. 0% for Source-1).
- **Script diversity (India):** non-Latin script content spans **9 scripts**
  (Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Gurmukhi,
  Oriya), each keyed to the business's actual state, affecting 31.11% of all
  India records — verified exhaustively after an initial sampled estimate missed
  2 of the 9 scripts entirely.
- **~26% distractor pool:** roughly a quarter of every Source-2/3 file has no
  true Source-1 counterpart at all — a large, free, naturally-occurring
  hard-negative pool for classifier training.
- **France (test-only, 15% of test Source-1):** verified directly on real test
  rows to exhibit the identical noise grammar (reordering, casing, alias/decoy
  constructs) with a swapped vocabulary (SARL/SASU/SAS/EURL/SCI/SNC legal forms,
  French street abbreviations) — evidence a script/language-agnostic pipeline
  transfers without country-specific rules.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (deterministic candidate generation,
deterministic feature engineering, learned final classification stage only).

**Core Innovation:** Every design decision is tied to a directly measured
pattern in the data rather than a general-purpose default — e.g., blocking
prunes tokens by document frequency using thresholds tuned from the actual
measured token-frequency distribution (not a guessed constant), and the
classifier's "ambiguity" features (a candidate's rank, score gap to its nearest
competitor, and z-score relative to its sibling candidates for the same
Source-1 entity) directly target the precision risk that a precision-weighted
metric like F0.5 penalizes most: picking the wrong candidate among several
similarly-scored options.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:** four token types per record — normalized name core
tokens (legal suffixes stripped position-agnostically, alias tails extracted,
website-style names de-concatenated via a word-segmentation table built from
this contest's own clean Source-1 text), address word tokens (with an
abbreviation-expansion pass: `rd`→`road`, `nr`→`near`, French `rte`→`route`,
etc.), raw digit tokens, and leading-zero-normalized digit tokens (script- and
punctuation-invariant, bridging the ~14% of true matches with zero name-token
overlap due to transliteration). Candidates are partitioned by `country`
first (discovered dynamically from the data at runtime — never a hardcoded
`{US, India}` list — so France is included automatically).

Each country builds one inverted index per token field, pruned to a per-field
document-frequency cap (tuned empirically: 10,000 for name tokens, 2,000 for
address/digit tokens — a name-field cap this high was necessary because at a
naive 2,000 cap, ~38-42% of entities had *every* name token individually common
enough to be pruned, even though the specific word combination was highly
specific). A same-field, multi-token intersection fallback (using the full,
unpruned index) recovers candidates when a field is left completely silent by
pruning — this fixed a measured blocking gap on short, generic names combined
with an empty address field (the only case where such an entity has no other
recoverable signal).

**Candidate pairs generated:** average ~5,425 candidates/entity before
truncation (median 3,304-3,534) — computationally infeasible to carry into
per-pair feature computation at full training scale (~12 billion pairs), so a
`top_k` cap (IDF-weighted score, computed for free during blocking with no
extra candidate-row lookups) truncates to the highest-scoring 100 per entity
before feature computation.

**How true matches were not lost:** blocking recall (pair completeness) was
measured directly against training ground truth at each iteration — 91.06%
(initial union blocking) → 94.11% (fixed a runaway-memory bug in the
intersection fallback) → 96.9%/97.49% (fixed a per-field vs. global fallback
gating bug, then tuned the name-field pruning threshold) — see the project's
`PROGRESS_AND_METHODOLOGY_LOG.md` for the full iteration history, including
the bugs found and how each was root-caused before being fixed, not just
patched.

---

## 4. Matching Model

**Features used** (26 total: 18 pairwise + 2 asymmetric containment + 8
sibling-relative "ambiguity" features):
- **Name features:** token Jaccard, asymmetric containment (both directions —
  catches "Apex" fully contained in "Apex Industries" without Jaccard's size
  penalty), Jaro-Winkler on the cleaned comparison text, legal-suffix Jaccard
  and exact-match flag.
- **Address features:** token Jaccard (post abbreviation-expansion), digit-token
  Jaccard (raw and leading-zero-normalized), Jaro-Winkler, empty-address flags
  for both sides.
- **Other:** source indicator (Source-2 vs. Source-3), alias/website-construct
  flags, non-ASCII flags, name/address length ratios, the blocking proto-score,
  and 8 ambiguity features (rank, group size, group mean/std, z-score, gap to
  the next-better and next-worse sibling candidate, and the group's top score)
  computed in a fully vectorized second pass over the base feature table,
  grouped by `source1_entity_id`.

**Model type:** `HistGradientBoostingClassifier` (scikit-learn — BSD-licensed,
comfortably under the 8B-parameter constraint; trained in 106 seconds on
23.8M labeled candidate pairs).

**Threshold selection method:** a single global threshold swept over [0.10,
0.94] on a held-out validation split (20%, split by `source1_entity_id` via
`GroupShuffleSplit` so no candidate from the same entity leaks across the
split), maximizing macro F0.5 computed with the exact per-entity aggregation
the contest uses (singletons scored 1.0/0.0 for correct/incorrect empty
predictions, not silently dropped from the average). We deliberately did not
use a per-entity threshold sweep (fitting precision/recall from a single
entity's own handful of candidates is a severe overfitting risk given most
entities have only 2-5 true matches) — one well-validated global threshold on
the full validation set is the statistically sound choice under our time
constraints.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), validation:** **0.9537** at threshold 0.64 (singleton
  subset: 0.9489, n=5,696; non-singleton subset: 0.9542, n=54,300).
- **Blocking pair completeness (recall ceiling):** 97.49% on training ground
  truth (US 98.62%, India 95.81%).
- **Common false positives (wrong merges):** [fill in from `output/`
  post-hoc error review, if time permits — generic/short names (e.g. France's
  2-letter abbreviation clusters like "CC"/"PC", each with hundreds of
  same-named records) are the expected highest-risk case, since ambiguity
  features are the primary defense there.]
- **Common false negatives (missed matches):** primarily the residual blocking
  gap (~2.5% of true matches never reach the candidate set) and the
  `top_k=100` truncation cost on entities whose true match ranks outside the
  top 100 by blocking score.

---

## 6. Conclusion

A deterministic, fully-auditable preprocessing and blocking pipeline — every
design choice tied to a directly measured pattern in the full dataset, not a
default or an assumption — feeding a small, fast, license-compliant classifier
achieved strong validation performance (macro F0.5 = 0.9537) without any
country-specific special-casing, which is the property that gives us confidence
in the France generalization. The main lesson learned was operational: several
iterations were needed to get blocking's recall/cost tradeoff right, and each
fix came from directly diagnosing a specific failing example against real data
rather than tuning parameters blind.

---

## Appendix

### A. Code Artefacts

Complete, runnable pipeline ships under `code/business_entity_resolution/`
(`src/`, `README.md`, `requirements.txt`). Entry points, run in order from
`src/`: `build_clean_cache.py` → `build_features.py --split train --limit
150000 --top-k 100` → `train_model.py` → `generate_predictions.py`. See
`code/business_entity_resolution/README.md` for exact commands and the full
module map.

### B. Additional Results

See `RESEARCH_AND_STRATEGY_REPORT.md` (full EDA, literature review) and
`PROGRESS_AND_METHODOLOGY_LOG.md` (iteration-by-iteration execution log,
including bugs found, what was tried and rejected, and why) for full detail
beyond this summary.

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
