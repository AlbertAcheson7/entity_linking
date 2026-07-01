#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

BCE_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce.yaml"
BCE_FINETUNE_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce_finetune.yaml"
BGE_CONFIG="configs/snomed_icd10cm_option_a_matrix_bge_m3.yaml"
PUBMED_CONFIG="configs/snomed_icd10cm_option_a_matrix_pubmedbert.yaml"
PUBMED_FINETUNE_CONFIG="configs/snomed_icd10cm_option_a_matrix_pubmedbert_finetune.yaml"
DIRECTIONAL_CONFIG="configs/snomed_icd10cm_option_a_directional.yaml"
DIRECTIONAL_FINETUNE_CONFIG="configs/snomed_icd10cm_option_a_directional_finetune.yaml"

BCE_ENTITY_MAX_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce_C_entity_max.yaml"
BCE_NAME_ONLY_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce_D_name_only.yaml"
BCE_CONTEXT_ONLY_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce_D_context_only.yaml"
BCE_PATH_ONLY_CONFIG="configs/snomed_icd10cm_option_a_matrix_bce_D_path_only.yaml"

run_prepare_once() {
  "$PYTHON" -m icd_linker.cli prepare --config "$BCE_CONFIG"
}

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

run_prepare_once

echo "M0: backbone model, direct retrieval"
eval_base "$BCE_CONFIG"
eval_base "$BGE_CONFIG"
eval_base "$PUBMED_CONFIG"

echo "M4: target view/fusion strategy, using BCE configs"
eval_base "$BCE_NAME_ONLY_CONFIG"
eval_base "$BCE_CONTEXT_ONLY_CONFIG"
eval_base "$BCE_PATH_ONLY_CONFIG"
eval_base "$BCE_ENTITY_MAX_CONFIG"
echo "M4 reference: BCE multi-view best_view was already run in M0."

echo "M1: finetune vs direct retrieval"
finetune_and_eval "$BCE_FINETUNE_CONFIG"
finetune_and_eval "$PUBMED_FINETUNE_CONFIG"

echo "M3: negative strategy"
echo "Current code supports mined top-k hard negatives plus in-batch negatives during training."
echo "Separate in-batch-only / mixed-negative configs need code or config support before running."

echo "M2: directional projection"
eval_base "$DIRECTIONAL_CONFIG"
finetune_and_eval "$DIRECTIONAL_FINETUNE_CONFIG"

echo "Done."
