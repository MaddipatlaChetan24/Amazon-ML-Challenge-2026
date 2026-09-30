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
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_features import build_features_for_split, add_ambiguity_features, load_dense_extra
from features import FEATURE_NAMES, AMBIGUITY_FEATURE_NAMES
from posthoc_features import POSTHOC_FEATURE_NAMES, add_posthoc_features
from train_model import bucket_for_count

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output")

ALL_FEATURES = FEATURE_NAMES + AMBIGUITY_FEATURE_NAMES + POSTHOC_FEATURE_NAMES


def pair_keys(s1, cand) -> np.ndarray:
    """int64 key per (S1 id, S2/S3 id) pair; ids are 'S<k>-<number>' with number < 2^30."""
    import pyarrow as pa, pyarrow.compute as pc
    a = pc.cast(pc.utf8_slice_codeunits(pa.array(s1), 3), pa.int64()).to_numpy()
    c = pa.array(cand)
    b = pc.cast(pc.utf8_slice_codeunits(c, 3), pa.int64()).to_numpy()
    src = pc.starts_with(c, "S3-").to_numpy(zero_copy_only=False).astype(np.int64)
    return (a << 31) | (src << 30) | b


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
    parser.add_argument("--lanes", default=None,
                        help="name,addr,combined per-lane candidate budgets, e.g. 50,25,25 (overrides --top-k); "
                        "must match what build_features.py used for the training table")
    parser.add_argument("--model", default="matching_model.joblib", help="model file name inside cache/")
    parser.add_argument("--rerank-k", type=int, default=0, help="must match the training table's --rerank-k")
    parser.add_argument("--rerank-extra", default=None, help="must match the training table's --rerank-extra")
    parser.add_argument("--reverse-k", type=int, default=0, help="must match the training table's --reverse-k")
    parser.add_argument("--cross-encoder", default=None,
                        help="directory inside cache/ written by cross_encoder.py; re-scores uncertain pairs")
    parser.add_argument("--dense-topk", default=None, help="test_topk.parquet from kaggle_dense.py (learned-search extras)")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--seed-ce-tag", default=None, help="reuse re-checker scores cached for this older test table tag")
    parser.add_argument("--dump-probs", default=None,
                        help="save final per-pair probabilities (+ entity ids) to this npz, for final_stage.py")
    parser.add_argument("--final-probs", default=None,
                        help="npz from final_stage.py: replaces per-pair probabilities and the threshold")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes for test feature building")
    parser.add_argument(
        "--rebuild-features", action="store_true",
        help="recompute the test feature table even if a cached one exists for this --top-k "
        "(the cache is invalidated automatically by top_k via the filename, but not by other "
        "pipeline changes like blocking.py/features.py edits -- pass this after such a change).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lanes = tuple(int(x) for x in args.lanes.split(",")) if args.lanes else None
    tag = f"lanes{'-'.join(map(str, lanes))}" if lanes else f"top{args.top_k}"
    if args.rerank_k:
        tag += f"_rr{args.rerank_k}"
    rerank_extra = tuple(int(x) for x in args.rerank_extra.split(",")) if args.rerank_extra else None
    if rerank_extra:
        tag += f"_x{'-'.join(map(str, rerank_extra))}"
    if args.reverse_k:
        tag += f"_rv{args.reverse_k}"
    if args.dense_topk:
        tag += f"_dn{args.dense_k}"
    features_cache_path = f"{CACHE_DIR}/test_features_{tag}.parquet"

    print("loading trained model...", flush=True)
    bundle = joblib.load(f"{CACHE_DIR}/{args.model}")
    model = bundle["model"]
    global_threshold = bundle["threshold"]
    bucket_thresholds = bundle.get("bucket_thresholds")  # None if an older global-only model was loaded
    if bucket_thresholds:
        print(f"  per-candidate-count-bucket thresholds: {bucket_thresholds}", flush=True)
    else:
        print(f"  no bucket thresholds in this model file -- falling back to global threshold "
              f"= {global_threshold}", flush=True)

    if os.path.exists(features_cache_path) and not args.rebuild_features:
        print(f"\nloading cached test features from {features_cache_path} ...", flush=True)
        t0 = time.time()
        feat_df = pd.read_parquet(features_cache_path)
        print(f"loaded {len(feat_df):,} rows in {time.time()-t0:.1f}s "
              f"(pass --rebuild-features to recompute from scratch)", flush=True)
    else:
        print(f"\nbuilding test feature table (all countries, top_k={args.top_k})...", flush=True)
        t0 = time.time()
        feat_df = build_features_for_split("test", top_k=args.top_k, lanes=lanes, workers=args.workers,
                                           rerank_k=args.rerank_k, rerank_extra=rerank_extra,
                                           reverse_k=args.reverse_k,
                                           dense_extra=load_dense_extra(args.dense_topk, args.dense_k) if args.dense_topk else None)
        print(f"base test features: {len(feat_df):,} rows in {time.time()-t0:.1f}s", flush=True)

        feat_df = add_ambiguity_features(feat_df)
        print("ambiguity features added.", flush=True)

        t0 = time.time()
        feat_df.to_parquet(features_cache_path)
        print(f"cached test features to {features_cache_path} in {time.time()-t0:.1f}s "
              f"(future runs at this --top-k will load instantly instead of recomputing)", flush=True)

    feat_df = add_posthoc_features(feat_df, "test")

    # candidate_pairs.tsv -- every candidate the blocker produced, pre-threshold
    all_s1 = pd.read_parquet(f"{CACHE_DIR}/test_source1_clean.parquet")["entity_id"].tolist()
    candidate_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    write_id_list_tsv(feat_df, "candidate_entity_id", candidate_out, all_s1, "candidate_entity_ids")
    print(f"wrote {candidate_out}", flush=True)

    print("\nscoring candidates with trained model...", flush=True)
    t0 = time.time()
    if "stage1" in bundle:
        # stacked model (train_stacked.py): stage-1 probabilities + their
        # per-entity group statistics feed the stage-2 model
        from train_stacked import add_prob_group_features
        p1 = bundle["stage1"]["model"].predict_proba(feat_df[bundle["stage1_features"]].values)[:, 1]
        feat_df = add_prob_group_features(feat_df, p1)
    probs = model.predict_proba(feat_df[bundle.get("features", ALL_FEATURES)].values)[:, 1]
    feat_df["prob"] = probs
    print(f"scored in {time.time()-t0:.1f}s", flush=True)

    if args.cross_encoder:
        # third stage: re-score uncertain pairs with cross-encoder(s).
        # combiner.joblib is either one stage {band, combiner, members?} or
        # {"stages": [...]} ordered wide -> narrow; a narrower stage (e.g. a
        # slower ensemble on the core band) overrides the wider one.
        import cross_encoder as ce
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        ce_dir = f"{CACHE_DIR}/{args.cross_encoder}"
        meta = joblib.load(f"{ce_dir}/combiner.joblib")
        stages = meta.get("stages") or [{"band": meta["band"], "combiner": meta["combiner"],
                                         "members": meta.get("members", [args.cross_encoder])}]
        p0 = feat_df["prob"].to_numpy().copy()
        member_band = {}
        for st in stages:
            for mb in st["members"]:
                lo, hi = member_band.get(mb, (1.0, 0.0))
                member_band[mb] = (min(lo, st["band"][0]), max(hi, st["band"][1]))
        wide = np.zeros(len(p0), dtype=bool)
        for lo, hi in member_band.values():
            wide |= (p0 >= lo) & (p0 <= hi)
        idx = np.flatnonzero(wide)
        s1v = feat_df["source1_entity_id"].to_numpy()
        cv = feat_df["candidate_entity_id"].to_numpy()
        t0 = time.time()
        text = ce.raw_texts("test", set(s1v[idx]) | set(cv[idx]))
        scores = {}
        for mb, (lo, hi) in member_band.items():
            sel = idx[(p0[idx] >= lo) & (p0[idx] <= hi)]
            full = np.full(len(p0), np.nan, dtype=np.float32)
            # Scores are cached per (re-checker, test feature table): row order
            # of the cached table is fixed, so an ensemble run only computes
            # the members/pairs it has not scored before.
            score_cache = f"{CACHE_DIR}/ce_scores_{mb}_{tag}.npz"
            if os.path.exists(score_cache):
                c = np.load(score_cache)
                if len(c["n"]) and int(c["n"][0]) == len(p0):
                    full[c["idx"]] = c["score"]
            elif args.seed_ce_tag:
                # reuse scores of pairs already scored for another test table
                old = np.load(f"{CACHE_DIR}/ce_scores_{mb}_{args.seed_ce_tag}.npz")
                import pyarrow.parquet as pq
                ot = pq.read_table(f"{CACHE_DIR}/test_features_{args.seed_ce_tag}.parquet",
                                   columns=["source1_entity_id", "candidate_entity_id"]).take(old["idx"])
                ok = pd.Series(old["score"], index=pair_keys(ot.column(0), ot.column(1)))
                ok = ok[~ok.index.duplicated()]
                nk = pair_keys(s1v[sel], cv[sel])
                full[sel] = ok.reindex(nk).to_numpy()
                print(f"  {mb}: seeded {np.isfinite(full[sel]).sum():,} of {len(sel):,} scores from {args.seed_ce_tag}", flush=True)
            need = sel[np.isnan(full[sel])]
            if len(need):
                tok = AutoTokenizer.from_pretrained(f"{CACHE_DIR}/{mb}")
                mdl = AutoModelForSequenceClassification.from_pretrained(f"{CACHE_DIR}/{mb}").to(ce.device()).eval()
                full[need] = ce.score(tok, mdl, list(zip(s1v[need], cv[need])), text)
                del mdl
                done = np.flatnonzero(~np.isnan(full))
                np.savez(score_cache, idx=done, score=full[done], n=np.array([len(p0)]))
            scores[mb] = full
            print(f"  {mb}: {len(sel):,} pairs ({len(need):,} newly scored, {len(sel)-len(need):,} from cache) "
                  f"({time.time()-t0:.0f}s)", flush=True)
        new_p = p0.copy()
        for st in stages:
            lo, hi = st["band"]
            sel = np.flatnonzero((p0 >= lo) & (p0 <= hi))
            new_p[sel] = ce.combine(p0[sel], [scores[mb][sel] for mb in st["members"]], st["combiner"])
        feat_df["prob"] = new_p
        global_threshold, bucket_thresholds = meta["threshold"], None
        print(f"cross-encoder stage(s) done in {time.time()-t0:.0f}s; threshold {global_threshold:.2f}", flush=True)

    if args.dump_probs:
        extra = {}
        if args.cross_encoder:
            extra = {"p0": p0, "ce": scores[list(scores)[0]], "cand": feat_df["candidate_entity_id"].to_numpy()}
        np.savez(args.dump_probs, prob=feat_df["prob"].to_numpy(), s1=feat_df["source1_entity_id"].to_numpy(), **extra)
        print(f"saved per-pair probabilities to {args.dump_probs}", flush=True)
    if args.final_probs:
        fp = np.load(args.final_probs)
        assert len(fp["prob"]) == len(feat_df)
        feat_df["prob"] = fp["prob"]
        global_threshold, bucket_thresholds = float(fp["threshold"][0]), None
        print(f"final-stage probabilities loaded; threshold {global_threshold:.2f}", flush=True)

    if bucket_thresholds:
        # Bucket by candidate count -- same definition used in training
        # (see train_model.py), so an entity's threshold depends only on how
        # much blocking "competition" it has, which is known at inference
        # time (no label needed).
        group_size = feat_df.groupby("source1_entity_id")["candidate_entity_id"].transform("size")
        entity_threshold = group_size.apply(lambda n: bucket_thresholds.get(bucket_for_count(n), global_threshold))
    else:
        entity_threshold = global_threshold

    matched = feat_df[feat_df["prob"] >= entity_threshold][["source1_entity_id", "candidate_entity_id", "prob"]]
    print(f"pairs above threshold (pre-reconciliation): {len(matched):,}", flush=True)
    matched = reconcile_one_to_one(matched)
    print(f"pairs after 1-to-1 reconciliation: {len(matched):,}", flush=True)

    matching_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    write_id_list_tsv(matched, "candidate_entity_id", matching_out, all_s1, "matched_entity_ids")
    print(f"wrote {matching_out}", flush=True)

    matched_s1_ids = set(matched["source1_entity_id"])
    n_singleton_pred = sum(1 for s1id in all_s1 if s1id not in matched_s1_ids)
    print(f"\npredicted singleton rate: {n_singleton_pred/len(all_s1)*100:.2f}% "
          f"(training ground truth was 5.58% -- sanity check, not a guarantee)", flush=True)


if __name__ == "__main__":
    main()
