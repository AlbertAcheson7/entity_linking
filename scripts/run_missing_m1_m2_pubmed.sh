#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

PUBMED_FINETUNE_CONFIG="configs/icd10_to_icd11_matrix_pubmedbert_finetune.yaml"
DIRECTIONAL_CONFIG="configs/icd10_to_icd11_directional.yaml"
DIRECTIONAL_FINETUNE_CONFIG="configs/icd10_to_icd11_directional_finetune.yaml"

eval_base() {
  local config="$1"
  "$PYTHON" -m icd_linker.cli evaluate --config "$config" --variant base
}

finetune_and_eval() {
  local config="$1"
  "$PYTHON" -m icd_linker.cli mine-negatives --config "$config"
  "$PYTHON" -m icd_linker.cli train --config "$config"
  "$PYTHON" -m icd_linker.cli evaluate --config "$config" --variant finetuned
}

echo "M1 supplement: PubMedBERT finetune"
finetune_and_eval "$PUBMED_FINETUNE_CONFIG"

echo "M2 supplement: PubMedBERT directional base"
eval_base "$DIRECTIONAL_CONFIG"

echo "M2 supplement: PubMedBERT directional finetune"
finetune_and_eval "$DIRECTIONAL_FINETUNE_CONFIG"

echo "Done."
