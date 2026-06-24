#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-..}"
OUTPUT="${2:-icd_linking_transfer.tar.zst}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/entity_linking/data/source"
rsync -a \
  --exclude data \
  --exclude chroma \
  --exclude models \
  --exclude experiments \
  --exclude logs \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude entity_linking \
  "$WORKSPACE/entity_linking/" "$STAGE/entity_linking/"
cp "$WORKSPACE/processed_alignment_data/terms/who_icd10.jsonl" \
  "$STAGE/entity_linking/data/source/"
cp "$WORKSPACE/processed_alignment_data/terms/icd11_mms.jsonl" \
  "$STAGE/entity_linking/data/source/"
cp "$WORKSPACE/processed_alignment_data/maps/icd10_icd11.jsonl" \
  "$STAGE/entity_linking/data/source/"
cp "$WORKSPACE/processed_alignment_data/audit/validation.json" \
  "$STAGE/entity_linking/data/source/"
cp "$WORKSPACE/processed_alignment_data/audit/output_verification.json" \
  "$STAGE/entity_linking/data/source/"
cp "$WORKSPACE/processed_alignment_data/manifest.json" \
  "$STAGE/entity_linking/data/source/"

(cd "$STAGE/entity_linking" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | \
  while IFS= read -r file; do sha256sum "$file"; done \
  > SHA256SUMS)
tar -C "$STAGE" -cf - entity_linking | zstd -f -T0 -10 -o "$OUTPUT"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT")" && pwd)"
OUTPUT_NAME="$(basename "$OUTPUT")"
(cd "$OUTPUT_DIR" && sha256sum "$OUTPUT_NAME" > "$OUTPUT_NAME.sha256")
echo "Created $OUTPUT and $OUTPUT.sha256"
