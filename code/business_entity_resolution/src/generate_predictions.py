"""
Run the full pipeline on the test set and write matching_results.tsv and
candidate_pairs.tsv in the exact format the README specifies.

Steps: build the test feature table (all 3 countries, dynamic iteration --
no hardcoded country list, so France is included automatically), score every
pair with the trained model, apply the tuned threshold, then run a global
1-to-1 reconciliation pass (per the measured ground-truth property that every
matched S2/S3 id has multiplicity exactly 1 -- see
PROGRESS_AND_METHODOLOGY_LOG.md Section 4) so no candidate is claimed by two
different Source-1 rows in the final output. Every Source-1 test entity gets
exactly one output row, even ones blocking found zero candidates for (empty
match list) -- required by the README's validation rules.
"""
import os
import sys
import time

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_features import build_features_for_split
from features import FEATURE_NAMES, AMBIGUITY_FEATURE_NAMES
from build_features import add_ambiguity_features

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")

ALL_FEATURES = FEATURE_NAMES + AMBIGUITY_FEATURE_NAMES


def reconcile_one_to_one(matches: pd.DataFrame) -> pd.DataFrame:
    """matches: rows with columns source1_entity_id, candidate_entity_id,
    prob -- already filtered to prob >= threshold. If the same
    candidate_entity_id appears under multiple source1_entity_id, keep only
    the highest-probability assignment (matches the measured ground-truth
    property: every matched id has multiplicity exactly 1)."""
    matches = matches.sort_values("prob", ascending=False)
    return matches.drop_duplicates(subset="candidate_entity_id", keep="first")


def write_id_list_tsv(df: pd.DataFrame, id_col: str, out_path: str, all_s1_ids, header_id_col: str):
    """df: columns [source1_entity_id, id_col]. Writes one row per S1 id in
    all_s1_ids (required -- every test entity must appear even with an empty
    list), comma-joined, no duplicates, tab-separated."""
    grouped = df.groupby("source1_entity_id")[id_col].apply(lambda s: ",".join(sorted(set(s))))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header_id_col}\n")
        for s1id in all_s1_ids:
            ids = grouped.get(s1id, "")
            f.write(f"{s1id}\t{ids}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="candidate cap per entity -- should match (or exceed) whatever was used to build "
        "the training feature table (build_features.py's --top-k), otherwise the model sees a "
        "different candidate-density distribution at inference than it was trained on.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("loading trained model...", flush=True)
    bundle = joblib.load(f"{CACHE_DIR}/matching_model.joblib")
    model, threshold = bundle["model"], bundle["threshold"]
    print(f"  threshold = {threshold}", flush=True)

    print(f"\nbuilding test feature table (all countries, top_k={args.top_k})...", flush=True)
    t0 = time.time()
    feat_df = build_features_for_split("test", top_k=args.top_k)
    print(f"base test features: {len(feat_df):,} rows in {time.time()-t0:.1f}s", flush=True)

    feat_df = add_ambiguity_features(feat_df)
    print("ambiguity features added.", flush=True)

    # candidate_pairs.tsv -- every candidate the blocker produced, pre-threshold
    all_s1 = pd.read_parquet(f"{CACHE_DIR}/test_source1_clean.parquet")["entity_id"].tolist()
    candidate_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    write_id_list_tsv(feat_df, "candidate_entity_id", candidate_out, all_s1, "candidate_entity_ids")
    print(f"wrote {candidate_out}", flush=True)

    print("\nscoring candidates with trained model...", flush=True)
    t0 = time.time()
    probs = model.predict_proba(feat_df[ALL_FEATURES].values)[:, 1]
    feat_df["prob"] = probs
    print(f"scored in {time.time()-t0:.1f}s", flush=True)

    matched = feat_df[feat_df["prob"] >= threshold][["source1_entity_id", "candidate_entity_id", "prob"]]
    print(f"pairs above threshold (pre-reconciliation): {len(matched):,}", flush=True)
    matched = reconcile_one_to_one(matched)
    print(f"pairs after 1-to-1 reconciliation: {len(matched):,}", flush=True)

    matching_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    write_id_list_tsv(matched, "candidate_entity_id", matching_out, all_s1, "matched_entity_ids")
    print(f"wrote {matching_out}", flush=True)

    n_singleton_pred = sum(1 for s1id in all_s1 if s1id not in set(matched["source1_entity_id"]))
    print(f"\npredicted singleton rate: {n_singleton_pred/len(all_s1)*100:.2f}% "
          f"(training ground truth was 5.58% -- sanity check, not a guarantee)", flush=True)


if __name__ == "__main__":
    main()
