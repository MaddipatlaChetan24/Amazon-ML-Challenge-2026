# Business Entity Resolution — Pipeline

These steps rebuild `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw `dataset/` files.
Every path is resolved relative to the script that uses it. Put `dataset/` next to `code/` (so it sits at
`student_resource/dataset/`), and all intermediate files go to `code/business_entity_resolution/cache/`.

## Setup

```bash
pip install -r requirements.txt
```

The pipeline was built and run on a MacBook Pro (18 CPU cores, 64 GB RAM, Apple GPU via MPS). The three GPU-heavy
steps (mpnet re-checker, learned search, and optionally the MiniLM re-checker) ran on Kaggle (2 × Tesla T4). They
also run on any CUDA machine, or on MPS/CPU if you can wait longer.
RAM is the limit on a 64 GB machine, not cores: keep `--workers` at 4 or below.
Never load torch and scikit-learn's gradient-boosting model in the same process when both use OpenMP on macOS,
because the two runtimes can crash each other. The scripts below are split to avoid that.

## Pipeline overview

```
clean → candidate search (token lanes + char-3gram re-rank + reserved slots + reverse retrieval
        + learned bi-encoder search) → 38 pair features → gradient-boosted classifier
      → MiniLM cross-encoder on uncertain pairs → mpnet cross-encoder on still-uncertain pairs
      → business-aware final stage → threshold 0.74 + 1-to-1 reconciliation → output files
```

Validation always uses the same 99,993 held-out training businesses (`cache/val_entities.txt`) and the exact contest
metric: macro F0.5 per business, with true matches the search never found counted as misses.

## Steps (from `src/`)

```bash
mkdir -p cache && cp val_entities.txt cache/   # the fixed 99,993 validation businesses
cd src

# 1. Normalise all six source files -> cache/*_clean.parquet  (~10 min)
python3 build_clean_cache.py

# 2. Multilingual name embeddings (paraphrase-multilingual-MiniLM-L12-v2) -> cache/*_embeddings.npy  (~50 min on MPS)
python3 embeddings.py

# 3. Training feature table: 500K businesses per country, candidate search + features + labels  (~1.5 h)
python3 build_features.py --split train --limit 500000 --lanes 15,10,10 --rerank-k 40 --rerank-extra 5,10 \
    --workers 4 --out ../cache/train_features.parquet

# 4. Classifier (HistGradientBoosting), trained on every business except the fixed validation ones;
#    reports the blocking ceiling and validation macro F0.5  (~15 min)
python3 train_model.py --max-iter 3000 --learning-rate 0.12 --max-leaf-nodes 127 --max-depth 16 \
    --out matching_model_v8.joblib

# 5. MiniLM re-checker (cross-encoder, Apache-2.0).
#    5a trains it on 500K uncertain pairs of training businesses (never validation ones); the pairs
#       are taken from the v7 classifier and feature table (steps 3-4 without --rerank-extra, saved
#       as train_features_v7_lanes15-10-10_rr40_500k.parquet / matching_model_v7.joblib)  (~3 h on MPS)
#    5b scores the validation band (classifier probability 0.01-0.99) and fits the logistic combiner
#       -> cache/cross_encoder_big2_v8/
python3 train_rechecker_minilm.py 1000000 500000
python3 package_rechecker_minilm.py

# 6. Kit for the GPU re-checker: 700K training pairs + validation band  -> cache/rechecker_kit/
python3 export_rechecker_kit.py --part train_val --model matching_model_v8.joblib

# 7. mpnet re-checker on a CUDA GPU (Kaggle T4 x2: ~3 h), trained on the kit; writes
#    rechecker_model/ and val_scores.npy (copy val_scores.npy to cache/rechecker_kit/mpnet_val_scores.npy)
python3 kaggle_rechecker.py --data ../cache/rechecker_kit --out <dir>

# 8. Learned search (bi-encoder) on a CUDA GPU (Kaggle T4 x2: ~2 h). Needs the text kit:
python3 export_dense_kit.py                      # -> cache/dense_kit/
python3 kaggle_dense.py --data ../cache/dense_kit --out <dir> --k 30
#    -> <dir>/train_topk.parquet (validation businesses) and <dir>/test_topk.parquet (all test businesses)
#    copy them to cache/val_dense_topk.parquet and cache/test_dense_topk.parquet

# 9. Test candidates without the learned search + classifier + MiniLM re-checker (~4 h: ~2.9 h features,
#    ~1.3 h re-checker). Caches cache/test_features_lanes15-10-10_rr40_x5-10_rv3.parquet.
python3 generate_predictions.py --lanes 15,10,10 --rerank-k 40 --rerank-extra 5,10 --reverse-k 3 \
    --model matching_model_v8.joblib --cross-encoder cross_encoder_big2_v8 --workers 4

# 10. Uncertain test pairs for the mpnet re-checker -> cache/rechecker_kit_test/, cache/kaggle_test_upload/
python3 export_rechecker_kit.py --part test --model matching_model_v8.joblib \
    --tag lanes15-10-10_rr40_x5-10_rv3 --member cross_encoder_big2_v8 --out ../cache/rechecker_kit_test
#    keep pairs whose classifier+MiniLM score is in (0.05, 0.95) -> cache/kaggle_test_upload/test_band.parquet
#    (the 7-line filter is in the section "Narrowed test band" below), then on the GPU:
python3 kaggle_rechecker.py --data ../cache/rechecker_kit --score-only <dir>/rechecker_model \
    --test ../cache/kaggle_test_upload/test_band.parquet --out <dir>
#    copy <dir>/test_scores.npy to cache/mpnet_test_scores.npy

# 11. Final submission (v11): add the learned-search top-10 candidates to the test table, score the
#     new pairs, apply the business-aware final stage, write and validate both files (~2.5 h).
#     Six commands, listed below.
```

### Step 11 in detail

```bash
# (a) new pairs = learned-search top-10 not already candidates
python3 make_new_pairs.py ../cache/test_features_lanes15-10-10_rr40_x5-10_rv3.parquet ../cache/test_dense_topk.parquet 10 \
    ../cache/test_new_pairs.parquet
# (b) features for the new pairs only; identical to a full rebuild (verified on validation)
python3 build_features.py --split test --lanes 15,10,10 --workers 4 --dense-new ../cache/test_new_pairs.parquet \
    --out ../cache/test_newonly.parquet
# (c) merge with the step-9 table, recompute per-business ambiguity features, restore full-rebuild row order
python3 merge_extra_features.py ../cache/test_features_lanes15-10-10_rr40_x5-10_rv3.parquet \
    ../cache/test_newonly.parquet ../cache/test_features_lanes15-10-10_rr40_x5-10_rv3_dn10.parquet
# (d) classifier + MiniLM (already-scored pairs reused by pair key); dump per-pair scores
python3 generate_predictions.py --lanes 15,10,10 --rerank-k 40 --rerank-extra 5,10 --reverse-k 3 \
    --dense-topk ../cache/test_dense_topk.parquet --dense-k 10 --seed-ce-tag lanes15-10-10_rr40_x5-10_rv3 \
    --model matching_model_v8.joblib --cross-encoder cross_encoder_big2_v8 --workers 4 --dump-probs ../cache/v11_probs.npz
# (e) final stage, fitted on the rebuilt validation table (see "Validation of the learned search")
python3 final_stage_v11.py --probs ../cache/v11_probs.npz
# (f) threshold + 1-to-1 reconciliation + both output files
python3 generate_predictions.py --lanes 15,10,10 --rerank-k 40 --rerank-extra 5,10 --reverse-k 3 \
    --dense-topk ../cache/test_dense_topk.parquet --dense-k 10 \
    --model matching_model_v8.joblib --cross-encoder cross_encoder_big2_v8 --workers 4 --final-probs ../cache/final_probs_v11.npz
cd ../../.. && python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### Validation of the learned search (needed by step 11e)

```bash
python3 build_features.py --split train --only-entities ../cache/val_entities.txt --lanes 15,10,10 --rerank-k 40 \
    --rerank-extra 5,10 --workers 4 --dense-topk ../cache/val_dense_topk.parquet --dense-k 10 --out ../cache/valdense10.parquet
python3 val_dense_eval.py a ../cache/valdense10.parquet   # classifier
python3 val_dense_eval.py b ../cache/valdense10.parquet   # MiniLM on new uncertain pairs
python3 val_dense_eval.py c ../cache/valdense10.parquet   # -> macro F0.5 0.9837, blocking recall 0.9932
```

### Narrowed test band (step 10)

```python
import pandas as pd, joblib, cross_encoder as ce
tb = pd.read_parquet("../cache/rechecker_kit_test/test_band.parquet")
m = joblib.load("../cache/cross_encoder_big2_v8/combiner.joblib")
c = ce.combine(tb.prob.to_numpy(), tb.minilm.to_numpy(), m["combiner"])
tb[(c > 0.05) & (c < 0.95)][["row_idx", "s1_text", "c_text"]].reset_index(drop=True) \
  .to_parquet("../cache/kaggle_test_upload/test_band.parquet", index=False)
```


## Final submission (v12r_full): retrained classifier + rescue step

Steps 1–11 give v11. The final file adds three changes on top.

**Candidate set.** `candidate_pairs.tsv` holds every search candidate (stages 1–5, ~73 per business), which is
exactly what the classifier scores. Learned-search top-5 candidates that the classifier scores below 0.01 are
re-checked by MiniLM and re-decided by the rescue stage; they are already in the candidate set. (A variant that passes
only the classifier's top 10 to the re-checkers gives the same validation F0.5; pass `10` instead of `1000000` to
write_final.py.)

```bash
# 12. Learned search for the training businesses (Kaggle GPU, ~1.7 h). Queries = the 900,005 non-validation
#     businesses of the training table; same model and settings as step 8.
#     -> cache/trainq_out/train_topk.parquet
#     (kernel: cache/kernel_trainq/run_trainq.py; data: cache/dense_trainq/)
# 13. Add those candidates to the training table and retrain the classifier
python3 make_new_pairs.py ../cache/train_features.parquet ../cache/trainq_out/train_topk.parquet 10 ../cache/train_new_pairs.parquet
python3 build_features.py --split train --only-entities ../cache/train_nonval_entities.txt --lanes 15,10,10 --workers 4 \
    --dense-new ../cache/train_new_pairs.parquet --out ../cache/train_newonly.parquet
python3 merge_extra_features.py ../cache/train_features.parquet ../cache/train_newonly.parquet ../cache/train_features_dn10.parquet
python3 train_model.py --max-iter 3000 --learning-rate 0.12 --max-leaf-nodes 127 --max-depth 16 \
    --table train_features_dn10.parquet --out matching_model_v12.joblib
#     (train_nonval_entities.txt = businesses of train_features.parquet not in val_entities.txt)
# 14. Validation with the retrained classifier (final stage 0.98448)
for s in a b c; do python3 val_dense_eval.py $s ../cache/valdense10.parquet matching_model_v12.joblib; done
# 15. MiniLM for rescued validation / test pairs (learned-search top-k, classifier probability < 0.01)
python3 rescue_minilm_val.py                  # validation (writes cache/val_rescue_minilm.npz)
python3 score_rescue_test.py ../cache/v11_probs.npz ../cache/test_dense_topk_full.parquet 5 lanes15-10-10_rr40_x5-10_rv3_dn10
# 16. Test scoring with the retrained classifier, final stage, rescue stage, top-10 cascade, output files
python3 generate_predictions.py --lanes 15,10,10 --rerank-k 40 --rerank-extra 5,10 --reverse-k 3 \
    --dense-topk ../cache/test_dense_topk.parquet --dense-k 10 \
    --model matching_model_v12.joblib --cross-encoder cross_encoder_big2_v8 --workers 4 --dump-probs ../cache/v12_probs.npz
python3 final_stage_v11.py --val-tag valdense10_matching_model_v12 --probs ../cache/v12_probs.npz --out ../cache/final_probs_v12.npz
python3 val_oof_final.py valdense10_matching_model_v12       # validation out-of-fold scores for the rescue stage
python3 final_stage_rescue.py --probs ../cache/v12_probs.npz --final ../cache/final_probs_v12.npz \
    --val-oof ../cache/valdense10_matching_model_v12_oof.parquet --val-pairs ../cache/valdense10_matching_model_v12_pairs.parquet \
    --val-minilm ../cache/valdense10_matching_model_v12_minilm.npz --out ../cache/final_probs_v12r.npz
python3 write_final.py ../cache/v12_probs.npz ../cache/final_probs_v12r.npz 1000000   # writes both output files
```

Validation (99,993 held-out businesses): v11 0.98369 → retrained classifier 0.98448 → + rescue **0.98487**.

## Module map

| File | Role |
|---|---|
| `preprocessing.py`, `build_clean_cache.py` | Name/address normalisation (legal suffixes, aliases, websites, abbreviations, digit tokens). |
| `embeddings.py` | Multilingual name embeddings and the `embedding_cosine` feature. |
| `posthoc_features.py` | Indic-script transliteration, char-3gram vectors, `name_translit_char3`, `addr_char3`, `digit_affix_match`, `digit_near_conflict`. |
| `blocking.py` | Inverted-index retrieval, three ranking lanes, hooks for re-rank and extra candidates. |
| `build_features.py` | Candidate search (lanes, re-rank, reserved slots, reverse retrieval, learned-search extras) + features; `--dense-new` builds features for extra pairs only. |
| `features.py` | Pairwise and per-business ambiguity features. |
| `train_model.py` | Classifier, fixed validation split, exact contest metric. |
| `cross_encoder.py`, `train_rechecker_minilm.py`, `package_rechecker_minilm.py` | MiniLM re-checker training, scoring, logistic combiner. |
| `export_rechecker_kit.py`, `kaggle_rechecker.py` | Pair kits and the GPU (mpnet) re-checker. |
| `export_dense_kit.py`, `kaggle_dense.py` | Learned (bi-encoder) search: fine-tune, encode, nearest-record search per country. |
| `val_dense_eval.py`, `dense_eval.py` | Validation of the learned-search candidates (recall and end-to-end F0.5). |
| `make_new_pairs.py`, `merge_extra_features.py` | New learned-search pairs; merges their features into an existing test table. |
| `final_stage.py`, `final_stage_keyed.py`, `final_stage_v11.py` | Business-aware final stage (v10 table / keyed / v11 table). |
| `final_stage_rescue.py`, `score_rescue_test.py`, `write_final.py` | Rescue stage, its MiniLM scores, and the top-10 cascade writer for both output files. |
| `generate_predictions.py` | Test scoring, re-checker stages (scores cached per pair), threshold, 1-to-1 reconciliation, both output files. |

## Determinism

Candidate selection breaks score ties by entity id, and every model uses a fixed seed. Rebuilding the step-9 test
table reproduced the earlier submission byte for byte.
