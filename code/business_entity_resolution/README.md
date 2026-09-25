# Business Entity Resolution — Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
the raw `dataset/` files. Every stage is deterministic code except the final
classifier (a gradient-boosted tree, trained once and saved).

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline stages (run in order)

All scripts live in `src/` and are run from that directory. Paths to the
dataset are currently hardcoded near the top of each script to this
environment's path; update `DATASET_DIR` in `build_clean_cache.py` and
`build_features.py` if running elsewhere.

```bash
cd src

# 1. Preprocessing: clean business_name / business_address for all 6 source
#    files, cache derived fields (tokens, legal suffixes, alias/website
#    flags, digit tokens) to Parquet. ~5-6 min on the dev machine this was
#    built on (18 cores, no GPU); purely CPU-bound text processing, so
#    scales with core count.
python3 build_clean_cache.py

# 1b. (Recommended, optional) Build the multilingual embedding cache -- the
#     cross-script bridging feature (embedding_cosine) for the ~31% of
#     India records that use non-Latin script (9 different scripts,
#     verified exhaustively -- see RESEARCH_AND_STRATEGY_REPORT.md Section
#     3.1). Skippable: if this step isn't run, embedding_cosine silently
#     defaults to 0.0 for every pair and the rest of the pipeline runs
#     unchanged (see features.py's _embedding_cosine docstring). NOT
#     benchmarked at full ~26.4M-row scale in the dev environment -- this
#     is compute-bound (sentence encoding), so a GPU machine will be much
#     faster than CPU-only; increase embeddings.py's batch_size if you have
#     the memory for it. If you hit an ImportError about is_offline_mode
#     from sentence-transformers, see the huggingface_hub note in
#     requirements.txt.
python3 embeddings.py

# 2. Build the training feature table: blocking (multi-key inverted index,
#    tuned per-field max_df) + pairwise features (now including
#    embedding_cosine if step 1b ran) + ambiguity (sibling-relative)
#    features, labeled against train_ground_truth.tsv. --limit and --top-k
#    are explicit on purpose, no silent defaults -- see blocking.py /
#    build_features.py docstrings for the measured recall-vs-cost tradeoff
#    behind the top_k choice (100/200/500/1000 were measured; higher
#    recovers more of blocking's 97.49% recall ceiling but costs more
#    compute). --limit 150000 below was a time-constrained compromise in
#    the dev environment (a 300K-entity stratified subsample instead of
#    the full 2.2M) -- on a machine without that constraint, drop --limit
#    entirely to train on the full training set, and consider a higher
#    --top-k (e.g. 500) for a better recall/cost tradeoff.
python3 build_features.py --split train --limit 150000 --top-k 100 \
    --out ../cache/train_features.parquet

# 3. Train the matching classifier (HistGradientBoostingClassifier --
#    scikit-learn, BSD-licensed, far under the 8B-parameter cap) and tune
#    one global decision threshold for macro F0.5 on a held-out validation
#    split (grouped by source1_entity_id, so no candidate from the same
#    entity leaks across the split).
python3 train_model.py

# 4. Run the full pipeline on the test set (all countries, discovered
#    dynamically from the data -- not a hardcoded {US, India} list, so
#    France is included automatically) and write both output files.
#    --top-k should match (or exceed) whatever was used in step 2 --
#    otherwise the model sees a different candidate-density distribution at
#    inference than it was trained on.
python3 generate_predictions.py --top-k 100

# 5. Validate before submitting.
cd ../../..  # back to student_resource/
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### A note on memory at full scale

`build_features_for_split` flushes each country's features to a compact
`float32` numpy array as soon as that country finishes, specifically because
an earlier version that held everything as native Python lists across all
countries hit ~170GB resident memory (against 64GB physical RAM) at the
full test-set scale (~171M pairs) and went into an unrecoverable
memory-thrashing state -- it had worked fine at the smaller training
subsample scale, which is what made the bug easy to miss initially. Watch
the `{country} flushed to array, X GB RSS` log lines; if RSS grows
unboundedly rather than staying roughly flat across countries, something
regressed.

## Module map

| File | Role |
|---|---|
| `src/preprocessing.py` | Text normalization: legal-suffix stripping (position-agnostic), alias-marker detection ("aka/fka/dba/t-a"), website-domain de-concatenation (word segmentation via a frequency table built from this contest's own Source-1 text), address-abbreviation expansion, digit-token extraction. No external lookups. |
| `src/build_clean_cache.py` | Runs `preprocessing.py` over all 6 raw TSVs, caches derived fields to Parquet. |
| `src/embeddings.py` | Optional: builds the multilingual sentence-embedding cache (`paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0) that powers the `embedding_cosine` feature -- the cross-script bridging signal for India's 9-script non-Latin content. Fails soft if not run (feature defaults to 0.0). |
| `src/blocking.py` | Multi-key inverted-index blocking (name/address/digit tokens), per-field document-frequency pruning, a same-field intersection fallback for candidate-starved entities, IDF-weighted scoring, and `top_k` truncation (required at full scale -- see module docstring for the measured recall-vs-volume tradeoff). |
| `src/features.py` | Pairwise similarity features (Jaccard, containment, Jaro-Winkler, digit overlap, legal-suffix match, embedding cosine) and sibling-relative "ambiguity" features (rank, gap-to-next-best/worst, z-score within a candidate group). |
| `src/build_features.py` | Orchestrates blocking + feature computation + labeling into one feature table per split. |
| `src/train_model.py` | Trains the classifier, tunes the F0.5 threshold, saves `cache/matching_model.joblib`. |
| `src/generate_predictions.py` | Runs the full pipeline on `test/`, applies the model + threshold, reconciles to a 1-to-1 assignment (no Source-2/3 id claimed by two different Source-1 rows, matching the measured ground-truth structure), writes both output TSVs. |

## Design rationale

See `../../RESEARCH_AND_STRATEGY_REPORT.md` (dataset EDA and strategy) and
`../../PROGRESS_AND_METHODOLOGY_LOG.md` (execution log, including what was
tried, measured, and rejected, with reasoning) for the full justification
behind every design choice above.
