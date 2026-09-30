# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Visionary
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Our pipeline matches each Source-1 business to its Source-2/3 records in seven stages:
1. Text normalisation.
2. A **hybrid candidate search**: token retrieval with three ranking lanes, a character-trigram re-rank of the whole
   retrieved pool, reserved slots for empty-address and Indic-script records, reverse retrieval, and a **fine-tuned
   multilingual bi-encoder search**.
3. 38 pairwise features.
4. A gradient-boosted classifier.
5. Two multilingual **cross-encoder re-checkers** on uncertain pairs.
6. A **business-aware final stage**.
7. A global threshold with 1-to-1 reconciliation.

We accepted or rejected every change on one fixed validation set: 99,993 held-out training businesses, scored with
the exact contest metric (macro F0.5 per business, with true matches the search never found counted as misses).
An early version of our own metric ignored those misses and overstated the score by 0.08. Fixing it showed that
candidate search, not the classifier, was the main loss, and that finding shaped the rest of the project.

| | First submission (v1) | Final (v11) |
|---|---|---|
| Candidate-search recall (validation) | 82% | **99.32%** |
| Candidates per business | ~99 | ~73 (every candidate is scored by the classifier) |
| Macro F0.5, validation | 0.878 | **0.9849** |
| Macro F0.5, public leaderboard | 0.864 | 0.976 (v11); final file v12r_full |

The two largest gains were:
- **Learned bi-encoder search:** validation recall 97.02% → 99.32% (India 95.68% → 99.32%), F0.5 0.9777 → 0.9837.
- **Cross-encoder re-checker:** F0.5 0.959 → 0.976.

---

## 2. Methodology

### 2.1 Problem Analysis

These findings come from the full corpus, not samples, and each one drove a design decision:
- **Scale:** ~26.4M rows. Comparing all pairs would mean trillions of pairs, so candidate search is mandatory.
- **Ground truth:** 3.46 matches per business on average; **89% of businesses have ≥2 true records**, and the metric
  counts every one, so missing one copy of four costs ~6% for that business. 5.58% are singletons (they score 1 only
  if predicted empty). No record belongs to two businesses, which we enforce with 1-to-1 reconciliation. About 26% of
  Source-2/3 records match nothing (distractors).
- **Decoys:** the same name often belongs to several distinct businesses. Near-duplicates differ only in house number
  (e.g. 200 vs 20 Linden Ave) or legal form, and some records share an address but not a business (multi-tenant
  buildings).
- **Scripts:** 31% of India records contain one of nine Indic scripts, and ~60K names mix Latin and Indic words. Before
  transliteration and the learned search, cross-script copies were the largest single cause of search misses.
- **Noise:** word-order shuffles, legal-suffix variants, honorific prefixes (4–5% of Source-2/3 names vs 0.1% of
  Source-1), aliases (aka/fka/dba), website-style names, character substitutions (`Banga1ore`, `R0cky`), abbreviations
  (e.g. "SSF" for "Sunrise Sun Foods"), invented replacement names at the same address, and partial or empty
  addresses.
- **France (15% of test Source-1, no training data):** no rule is keyed to a country. Indexes are built for whatever
  countries appear in the data, the features are similarity measures, and both re-checkers and the learned search are
  multilingual.

### 2.2 Solution Strategy

**Approach:** hybrid candidate search → learned pairwise classifier → cross-encoder re-checking → business-aware final
stage → reconciliation.

**How we decided:** every change was measured before adoption, on (a) candidate-search recall against ground truth and
(b) the exact contest F0.5 on the same held-out businesses. After each round we broke validation loss down by error
type (search miss / found but rejected / false positive) and targeted the largest bucket. Section 5 lists ideas we
rejected on that evidence.

---

## 3. Candidate Generation (Blocking)

All search is **per country**: a business is compared only with records from its own country.

1. **Token retrieval with three ranking lanes.** Inverted indexes cover name tokens (legal suffixes stripped, aliases
   and website names split), address tokens (abbreviations expanded), raw and normalised digit tokens, and
   **number+word composite keys** (e.g. `119|cotton`, document frequency ≤ 50). The retrieved pool (thousands of
   records) is ranked three ways, and we keep the top 15 by name score, top 10 by address score and top 10 by combined
   score. Separate lanes fix a measured failure: in one combined ranking, neighbours sharing common address words
   outranked exact-name matches with empty addresses.
2. **Character-trigram re-rank of the whole pool.** Every pooled record is re-scored by the cosine of trigram vectors of
   the transliterated name plus the address; the top 40 are added. Indic scripts are transliterated with the rule-based
   `indic-transliteration` library (MIT) plus measured per-script fixes.
3. **Reserved slots.** The 5 best empty-address records and the 10 best Indic-script-name records get their own slots.
   Both rank poorly on the combined vector but are often true copies.
4. **Reverse retrieval.** Each record retrieves its top-3 businesses, and those pairs are added.
5. **Learned search (bi-encoder).** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M) is fine-tuned on 1.5M
   (business, true record) text pairs from training businesses, excluding validation ones. Training uses in-batch
   negatives (batch 256, symmetric InfoNCE, scale 30, one epoch, ~47 min on 2 × T4). Every business and record becomes
   a 384-d vector, and the **top 10 nearest records in the same country** are added as candidates.
   Alone, top-10 already finds 97.9% of true matches. Combined with stages 1–4 it recovers **7,954 of the 10,306** true
   matches that stages 1–4 missed, at only 4.5 extra candidates per business. It helps most on cross-script copies,
   abbreviations and heavy typos.

| Candidate search (validation) | Recall | Candidates / business |
|---|---|---|
| Top-100 single ranking (v1) | 84.8% US / 79.7% India | ~99 |
| Stages 1–4 | 98.37% US / 95.68% India | 60 / 68 |
| **Stages 1–5 (final)** | **99.32% US / 99.32% India** | **64 / 73** |

**Candidate set (final).** `candidate_pairs.tsv` is exactly the set the classifier scores: all search candidates
(stages 1–5), ~73 per business out of millions of records per country, with a validation recall of 99.32%. We also
measured a cascade that passes only the classifier's top-10 to the re-checkers (same F0.5, 10.7 per business). We
submit the full set because the classifier itself is an ML model and runs on all of these candidates.

---

## 4. Matching Model

**Features (38):**
- **Name:** token Jaccard, containment both ways, Jaro-Winkler, character 3-/4-gram Jaccard, length ratio, legal-suffix
  Jaccard / exact match / compatibility, generic-name fraction, and transliterated-name trigram cosine.
- **Address:** token Jaccard, raw and normalised digit Jaccard, Jaro-Winkler, length ratio, empty flags, and trigram
  cosine.
- **House numbers:** `digit_affix_match` (e.g. 119 vs 00119) and `digit_near_conflict` (same-length numbers 1–50 apart
  that aren't shared, the typical decoy signature).
- **Semantic:** multilingual MiniLM embedding cosine of names.
- **Record flags:** source, alias, website, non-ASCII, blocking score.
- **Ambiguity (per business):** rank, group size/mean/std, z-score, gaps to neighbouring candidates, group maximum.

**Classifier:** scikit-learn `HistGradientBoostingClassifier` (lr 0.12, 127 leaves, depth 16, early stopping), trained
on ~29M labelled pairs from 500K businesses per country (validation businesses excluded).

**Re-checker 1 — MiniLM cross-encoder** (`paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, 118M):
- Fine-tuned on 500K uncertain pairs of training businesses.
- Reads both raw texts ("name | address") jointly and re-scores every pair with classifier probability 0.01–0.99.
- A logistic combiner merges its logit with the classifier logit.
- Validation: 0.959 → **0.9763**.

**Re-checker 2 — mpnet cross-encoder** (`paraphrase-multilingual-mpnet-base-v2`, Apache-2.0, 278M):
- Trained on 700K uncertain training pairs on 2 × T4 (2 epochs, AMP).
- Scores pairs that are still uncertain after re-checker 1 (combined score 0.05–0.95).

**Business-aware final stage:**
- A small gradient-boosted model over the classifier and both re-checker scores, plus each pair's score *relative to
  the other candidates of the same business* (gap to best, rank, number above 0.5, per-re-checker gaps).
- Fitted out-of-fold on validation halves split by business.
- +0.0014 over re-checker 1 alone (0.9763 → 0.9777).

**Classifier retraining with learned-search candidates.** The learned search was also run for the 900K
non-validation training businesses. Its top-10 candidates were added to the training table and the classifier was
retrained, so it scores these candidates correctly (validation 0.98369 → 0.98448).

**Rescue step.** Learned-search top-5 candidates that the classifier scored below 0.01 (already in the candidate set) are re-checked by MiniLM. A
final gradient-boosted model then re-decides them together with the uncertain band, using the current probability,
both re-checker scores, the learned-search cosine and rank, and per-business gaps (0.98448 → **0.98487**).

**Decision rule:** a global threshold of 0.74, chosen by sweeping the exact contest F0.5 on validation. Then global 1-to-1
reconciliation: each record keeps only its highest-scoring business.

**Validation protocol:** 99,993 held-out training businesses, never used to train the classifier, re-checkers or
learned search. The re-checker combiners and final stage are fitted out-of-fold (two halves by business hash).
F0.5 is computed per business against the **full** ground truth.

---

## 5. Results & Error Analysis

| Version | Change | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | token search, top-100 cap | 0.878 | 0.864 |
| v5 | three lanes + trigram re-rank + transliteration (+ stacking) | 0.949 | 0.936 |
| v7 | number+word keys, house-number features, MiniLM re-checker | 0.971 | 0.955 |
| v8 | reserved empty-address / Indic slots, 500K-pair re-checker | 0.9763 | 0.9685 |
| v9 | + reverse retrieval | ≈0.9763* | 0.970 |
| v10 | + mpnet re-checker + business-aware final stage | 0.9777 | — |
| v11 | + learned bi-encoder search (top-10) | 0.9837 | 0.976 |
| v11_k10 | + top-10 candidate cascade (candidate set 68 → 10 per business) | 0.9837 | — |
| v12 | + classifier retrained with learned-search candidates | 0.9845 | — |
| **v12r_full (final)** | **+ rescue step for low-scoring learned-search candidates; all search candidates scored** | **0.9849** | — |

\* reverse retrieval only changes test-time candidates.

**Remaining loss (v10 breakdown, validation):** search misses 0.0101, found but rejected 0.0082, extra false positives
0.0024, predictions on singletons 0.0011. v11 targets the first bucket: search recall rose from 97.02% to 99.32%.

**Typical remaining errors:**
- **Found but rejected:** an invented name at the exact address (e.g. "RIZARIZAVERA | 941 Hesters Crossing Rd" for
  "Jackson, Price & Hernandez LLC"), or the correct name with an empty address. These look exactly like a neighbour in
  the same building, or a same-name business elsewhere, so accepting them adds about as many false positives as it
  fixes.
- **False positives:** same-name businesses with near-identical addresses, and very short or generic names.

**Rejected ideas (measured on validation):**

| Idea | Result |
|---|---|
| Phonetic keys | Recovered 0 of 55 sampled misses |
| Rule-based post-processing | Lowered F0.5 |
| Per-bucket thresholds, calibration | No gain |
| Second MiniLM seed | No gain |
| Second-hop search (near-duplicates of already-matched records) | +0.00005: 169K new pairs, only 1,205 true, because identical names are shared by many businesses |
| Invented-name / token-rarity features | +0.0001 |
| France self-training with pseudo-labels | Neutral on US/India; France unverifiable, so not submitted |
| XLM-R cross-encoder | Did not learn |
| Larger LLMs | Excluded by the license rule |

**Leaderboard vs validation:** the leaderboard has run 0.006–0.008 below validation, mostly because of France (no
labels). The gap halved when search improved, so search changes transfer well to the test set.

---

## 6. Conclusion

The decisive lessons were about measurement:
- A metric that ignored search misses hid the real bottleneck. Measured correctly, candidate search was the largest
  loss at every stage of the project.
- A **learned bi-encoder search** gave the largest late gain: +0.006 F0.5 from only ~4.5 extra candidates per business.
- **Cross-encoders reading both records jointly** fixed most of what hand-built features could not.

**Fair play and licensing:**
- Only the provided data is used: no external lookups, APIs, geocoding or external training data.
- Legal-form lists, address abbreviations and transliteration rules are general linguistic knowledge.
- Pretrained models (all Apache-2.0, each ≤ 278M parameters, far below 8B):
  - `paraphrase-multilingual-MiniLM-L12-v2`: name embeddings, MiniLM re-checker, learned search.
  - `paraphrase-multilingual-mpnet-base-v2`: mpnet re-checker.
- Every model is fine-tuned only on the training split; the classifier is trained from scratch.

---

## Appendix

### A. Code Artefacts

The runnable pipeline is in `code/business_entity_resolution/`: `src/`, `README.md` (exact commands, in order, with
timings), `requirements.txt` and `val_entities.txt` (the fixed validation businesses). Main modules:
- `build_features.py` / `blocking.py`: candidate search.
- `features.py`, `posthoc_features.py`: features.
- `train_model.py`: classifier.
- `train_rechecker_minilm.py`, `package_rechecker_minilm.py`, `kaggle_rechecker.py`: re-checkers.
- `export_dense_kit.py`, `kaggle_dense.py`: learned search.
- `final_stage_v11.py`: final stage.
- `generate_predictions.py`: scoring, reconciliation, output files.

### B. Additional Results

`CURRENT_ARCHITECTURE_AND_EDGE_CASES.md` has the full loss breakdown and real examples of every edge-case type.
`PROGRESS_AND_METHODOLOGY_LOG.md` is the iteration log, and `RESEARCH_AND_STRATEGY_REPORT.md` covers EDA and the
literature review.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
