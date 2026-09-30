#!/bin/bash
# usage: bash make_final_zip.sh <team_name> <submission_folder e.g. submissions/v11>
set -euo pipefail
TEAM="$1"; SUB="$2"
cd "$(dirname "$0")"
OUT="final_package/${TEAM}_submission"
rm -rf final_package; mkdir -p "$OUT/output" "$OUT/code/business_entity_resolution/src"
cp "$SUB/matching_results.tsv" "$SUB/candidate_pairs.tsv" "$OUT/output/"
B=code/business_entity_resolution
cp -R $B/src/. "$OUT/code/business_entity_resolution/src/"; rm -rf "$OUT/code/business_entity_resolution/src/__pycache__"
cp $B/README.md $B/requirements.txt $B/val_entities.txt "$OUT/code/business_entity_resolution/"
cp Documentation_template.md "$OUT/"
cp CURRENT_ARCHITECTURE_AND_EDGE_CASES.md "$OUT/" 2>/dev/null || true
python3 utils/validate_submission.py --matching "$OUT/output/matching_results.tsv" --candidate "$OUT/output/candidate_pairs.tsv" --test-dir dataset/test --check-ids
(cd final_package && rm -f "${TEAM}_submission.zip" && zip -qr "${TEAM}_submission.zip" "${TEAM}_submission")
ls -la "final_package/${TEAM}_submission.zip"
unzip -l "final_package/${TEAM}_submission.zip" | tail -3
