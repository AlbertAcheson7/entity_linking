#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

BASE_CONFIG="configs/icd10_to_icd11_directional_multiview.yaml"
FINETUNE_CONFIG="configs/icd10_to_icd11_directional_multiview_finetune.yaml"

echo "Directional PubMedBERT multiview: base retrieval"
"$PYTHON" -m icd_linker.cli evaluate --config "$BASE_CONFIG" --variant base

echo "Directional PubMedBERT multiview: mine hard negatives"
"$PYTHON" -m icd_linker.cli mine-negatives --config "$FINETUNE_CONFIG"

echo "Directional PubMedBERT multiview: train"
"$PYTHON" -m icd_linker.cli train --config "$FINETUNE_CONFIG"

echo "Directional PubMedBERT multiview: finetuned retrieval"
"$PYTHON" -m icd_linker.cli evaluate --config "$FINETUNE_CONFIG" --variant finetuned

echo "Done."