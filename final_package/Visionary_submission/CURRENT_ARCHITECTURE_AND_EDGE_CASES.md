# Current Architecture & Edge Cases — status 2026-09-27 10:00 IST

This covers the pipeline as it runs now, where it loses points (measured), and the edge cases behind those losses.
Every number is measured on the **fixed validation set: 99,993 training businesses (US + India) held out, scored with the exact contest metric** (macro F0.5 per business, blocking misses counted), unless marked "leaderboard".

---

## 1. Scoreboard

| File | What changed | Validation | Leaderboard |
|---|---|---|---|
| v1 | first pipeline | 0.878 | 0.864 |
| v5 + stacking | multi-lane search, char-3gram re-rank, transliteration | 0.949 | 0.936 |
| v7 + MiniLM re-checker | number+word keys, house-number features, cross-encoder | 0.971 | 0.955 |
| v8 + re-checker | reserved slots for empty-address / Indic-script records | 0.9763 | 0.9685 |
| **v9_ce** | + reverse retrieval (record → business) at test time | ~0.9763* | **0.970** (current) |
| v10 (ready, not uploaded) | + mpnet re-checker + business-aware final stage | 0.9777 | — |
| v11 (in progress) | + learned (bi-encoder) search candidates | ? | — |

\* reverse retrieval only changes test-time candidates, so validation can't show its effect.

Top of leaderboard: ~0.991. One upload left before the 22:00 deadline.

---

## 2. Pipeline (v10)

```
Source 1 business ──► 1. clean ──► 2. candidate search (~60 records) ──► 3. pair features (38)
                                                                              │
     6. threshold 0.74 + 1-to-1 reconcile ◄── 5. final stage ◄── 4. classifier + re-checkers
```

### 1. Cleaning (`preprocessing.py`)
- Lowercase, strip punctuation and accents, split out legal suffixes (LLC, Inc, Pvt Ltd, Private Limited, LLP…), and flag aliases and websites.
- Addresses: tokens, digit tokens, and normalised digit tokens (leading zeros and `#` stripped, "No." handled).
- Transliteration of Indic scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati…) to Latin, word by word, using `indic-transliteration` (MIT) with per-script fixes (candra-o, chillu letters, Bengali b/v, Tamil consonants).

### 2. Candidate search (`blocking.py`, `build_features.py`)
Everything is done **per country**: a business is only compared with records from the same country.
- **Inverted indexes** on name words, address words, raw and normalised digits, and **number+word composite keys** (e.g. `119|cotton`, max document frequency 50).
- **Three ranking lanes** — name-led (IDF + 0.3 × address), address-led (+ 0.05 × name), combined — keeping top 15 / 10 / 10 from each.
- **Re-rank the whole retrieved pool** by character-trigram similarity (transliterated name vector + address vector, sparse dot product), keeping the top 40.
- **Reserved slots:** 5 for the best empty-address records, 10 for the best Indic-script-name records. These rank poorly on the combined vector but are often true copies.
- **Reverse retrieval (test only):** each record searches for its top-3 businesses, and those pairs are added.
- Ties are broken deterministically by entity id, so runs are reproducible byte for byte.
- **Result:** ~60 candidates per business. Validation recall is **97.02%** (US 98.37%, India **95.68%**).

### 3. Pair features (`features.py`, `posthoc_features.py`, `embeddings.py`)
There are 38 features:
- Name: token Jaccard, fuzzy ratios, transliterated char-3gram cosine, legal-suffix agreement, alias/website flags.
- Address: token overlap, char-3gram cosine, empty-address flags.
- Digits: shared digit tokens, `digit_affix_match` (e.g. 119 vs 00119), and `digit_near_conflict` (same-length numbers 1–50 apart that aren't shared, the typical decoy signature).
- Multilingual MiniLM embedding cosine of names.
- **Ambiguity features:** the candidate's rank within its business, gap to the best candidate, and number of close competitors.

### 4. Classifier + re-checkers
- **Classifier:** HistGradientBoosting (lr 0.12, 127 leaves, depth 16, early stopping), trained on 500K businesses per country.
- **Re-checker 1 — MiniLM-L12 multilingual cross-encoder** (Apache-2.0, 118M): fine-tuned on 700K uncertain pairs. It re-scores every pair whose classifier probability is between 0.01 and 0.99 (5.2M test pairs). A logistic combiner merges it with the classifier logit.
- **Re-checker 2 — paraphrase-multilingual-mpnet cross-encoder** (Apache-2.0, 278M): trained on Kaggle T4×2 GPUs. It scores the 1.26M test pairs that are still uncertain (0.05–0.95) after re-checker 1.

### 5. Final stage (`final_stage.py`)
A small gradient-boosted model per pair. Inputs:
- The classifier score and both re-checker scores.
- **Business-aware features:** the pair's score relative to the business's other candidates (gap to best, rank, number above 0.5, competitor gaps per re-checker).

It adds +0.0014 over re-checker 1 alone.

### 6. Decision
- Global threshold 0.74 (tuned on validation).
- **1-to-1 reconciliation:** each record goes only to its highest-scoring business. Ground truth never assigns a record to two businesses.
- **Output:** 5.69M predicted pairs; 5.99% of businesses get an empty prediction (training ground truth has 5.58% singletons).

---

## 3. Where the points are lost (validation, v10 = 0.9777, total loss 0.0223)

| Error type | Businesses affected | F0.5 points lost |
|---|---|---|
| **Search never found a true copy** | 7,550 | **0.0101** |
| **Found but scored below threshold (false negative)** | 7,504 | **0.0082** |
| Extra wrong record predicted (false positive) | 1,097 | 0.0024 |
| Singleton (no true match) but we predicted one | 105 | 0.0011 |
| Wrong pick (FP + FN) | 104 | 0.0003 |
| Search miss + FP | 76 | 0.0002 |

Key structural facts:
- **89% of businesses have ≥2 true records** (about 3.5 on average), and the metric counts every copy. Missing 1 of 4 costs about 6% for that business. **0.0177 of the 0.0223 loss** comes from these multi-record businesses.
- **India is the weak country:** the search misses 7,490 true records in India vs 2,816 in the US.
- **Leaderboard sits ~0.007 below validation**, mostly France (15% of test businesses, no training labels at all).

---

## 4. Edge cases (real validation examples)

### A. The search misses a true copy
| # | Pattern | Example (business → missed record) |
|---|---|---|
| A1 | **Script change + shortened address** | "Star Builders Limited, 119 Cotton Street, Kolkata" → `স্টার বিল্ডার্স লিমিটেড \| 119, KOLKATA, HOWRAH` |
| A2 | **Script change + mixed script inside the name** | "Green Properties LLP" → `ग्रीन Properties एलएलपी \| 173, NEW DELHI` |
| A3 | **Abbreviation / acronym** | "Sunrise Sun Foods Limited" → `SSF \| Shop No.F-8, Rohit Shopping Complex…` (full, identical address) |
| A4 | **Heavy typo + word reorder** | "Green Properties LLP" → `LLP GREEN PRPERTIIS \| 173, NEW DELHI` |
| A5 | **Typo + empty address** | "Rocky Travel LLC, 36 Galena Blvd" → `ROCKY TRSVAEL LLC \| (empty)` |
| A6 | **Name variant + address reduced to one number** | "Bangalore (india) LLP, No. 175, 6th Main…" → `Bangalore (india) (L.L.P.) \| No B3/175, Bangalore, KA` |
| A7 | **Accent-mangled name + wrong house number** | "Big Diner, 200 Linden Ave" → `Big Dér \| 20 LINDEN AVENUE, RIALTO` |
| A8 | **Script change + address typo** | "Good Management Pvt Ltd, B-24 Okhla…" → `गुड मैनेजमेंट प्राइवेट लिमिटेड \| … Industrial Arfa…` |

Common thread: the missed copy shares little *surface text* with the business. It is in another script, abbreviated, heavily typo'd, or has a truncated or empty address, while ~60 other records share more words.

### B. Found but rejected (scored below threshold)
| # | Pattern | Example | Score |
|---|---|---|---|
| B1 | **Invented/unrelated name at the exact address** | "Jackson, Price & Hernandez LLC, 941 Hesters Crossing Rd" → `RIZARIZAVERA \| 00941 HESTERS CROSSING RD` | 0.58 |
| B2 | same | "Brunhilde's Coffee, 311 Salamanca Ct" → `Jaxkelo \| 31 Salamanca Court` | 0.60 |
| B3 | same | "X T & H Bancorp, 206 11th St" → `Umbraverajax \| 206 11th St, Nashville` | 0.68 |
| B4 | **Correct name, empty address** | "Environmental Initiative PC" → `Environmental Initiative-PC \| (empty)` | 0.24 |
| B5 | same | "Jj (india) Private Limited" → `Jj (india) Private Limited \| (empty)` | 0.68 |
| B6 | **Name without legal words, empty address** | "1207 Second Plaza Apartments LP" → `1207 SECOND PLAZA APARTMENTS \| (empty)` | 0.58 |

Why these are hard: B1–B3 look exactly like **a different business in the same building**, and B4–B6 look exactly like **a same-name business elsewhere**. Both kinds of decoy are common, so accepting these adds about as many false positives as it fixes. Invented names sometimes *are* accepted (e.g. "Avilyra | 24 Lincoln Ave" at 0.85).

### C. Decoys (false positives)
- **House number off by a little:** 200 vs 20 Linden Ave, 941 vs 914. `digit_near_conflict` was added for this and was the main v7 gain.
- **Same common name, different city/branch:** "Lakshmi Infrastructure Private Limited", "Star Builders". The re-checkers handle most of them.
- **Same address, different business** (multi-tenant buildings).

### D. Data-format edge cases (handled)
- Address field order shuffled ("Madhya Pradesh, SHOP NO…" or state first). Token-based features are order-independent.
- State written as code / full name / native script (MP, Madhya Pradesh, दिल्ली). Tokens are normalised and transliterated.
- House number formats `119`, `#119`, `##119`, `No. 119`, `00119`, `1-73` vs `173`, `V-7-158/3` vs `V-158/3`. Normalised digit tokens and affix match handle these.
- Character noise: `Banga1ore`, `R0cky`, `Sec0nd`, `Pdvsage`. Char-trigram features plus the cross-encoders tolerate it.
- Legal suffix variants: Ltd / Limited / (LTD) / Pvt / Private / L.L.P. / -LLP.
- Word shuffles: "Lakshmi Private Private Infrastructure Limited", "X Bnacopr & H T".
- Empty business name or empty address on either side: flags, plus the reserved empty-address search slots.

### E. Unseen country (France)
- No country names appear in any rule. Indexes are built for whatever countries exist in the data, and both re-checkers are multilingual.
- Not specially handled: French legal forms (SARL, SAS, SA, EURL) and street words (rue, avenue, boulevard, bis/ter house numbers). The threshold is tuned on US/India.
- Estimated France score is ~0.93–0.95 (inferred from the gap between validation and leaderboard). This can't be measured without labels.

---

## 5. Tried and rejected (measured, validation)

| Idea | Gain | Why rejected |
|---|---|---|
| Rule-based post-processing | ≤ 0 | Hurt precision |
| Sibling similarity (candidate vs other predicted records) as feature | ~0 | Re-checker already covers it |
| **Second-hop search** (near-duplicates of already-matched records) | +0.00005 | 169K new pairs, only 1,205 true: identical names are shared by many different businesses |
| Invented-name / token-rarity features in final stage | +0.0001 | Re-checkers already use the signal |
| Word coverage, phonetic keys | ~0 | — |
| Per-bucket thresholds, calibration | ~0 | — |
| Second MiniLM seed | 0 (0.9762) | — |
| France self-training (pseudo-labels) | 0 on US/India; France unknown | 12% of French answers change, unverifiable |
| XLM-R cross-encoder | did not learn at lr 3e-5 | — |
| Llama-class LLM re-checker | — | License (must be MIT/Apache, ≤ 8B) |

---

## 6. In progress: learned search (v11)

- **Model:** multilingual MiniLM-L12 bi-encoder (Apache-2.0).
- **Training:** fine-tuned on 1.5M (business, true record) pairs with in-batch negatives (batch 256, symmetric InfoNCE, scale 30). It trained cleanly (loss 0.018 → 0.001).
- **Search:** every business and record becomes a 384-d vector, and the top-50 nearest records per business are found in the same country.
- **Integration:** the top-k are added as extra candidates next to the existing search, then go through the same features, classifier, re-checkers and final stage. The code is written and verified: it reproduces v10 exactly when no extras are added.
- **Status:** the first Kaggle run crashed out of GPU memory at the search step (fixed). A rerun is pending to get validation recall.
- **Target:** recover a large share of the 10,306 missed true records (0.0101 of the loss), especially Indic-script copies (A1, A2, A8) and abbreviations (A3).

---

## 7. Open questions where outside ideas would help most

1. **B-type false negatives (0.0082):** how to separate "invented name at the exact address = same business" from "neighbour in the same building", and "same name, no address = same business" from "same name elsewhere". Is there a signal in the data we aren't using?
2. **Abbreviations (A3):** cheap ways to generate acronym keys (SSF ↔ Sunrise Sun Foods) for search.
3. **France:** any rule-safe way (no external data) to adapt thresholds or features to French legal forms and street words.
4. **Singletons:** 5.99% predicted empty vs 5.58% in training. Is the test singleton rate different?
