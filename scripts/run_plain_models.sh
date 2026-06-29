#!/usr/bin/env bash
set -euo pipefail
#  -e 任何一行命令失败（返回非 0），脚本立即退出，防止错误累积。
#  -u 使用未定义的变量时直接报错，防止拼写错误。
# -o pipefail 管道中任何一个命令失败，整个管道返回非 0，防止错误被掩盖。

# Minimal sequential experiments:
# M0 backbone -> M4 target fusion -> M1 finetune -> M3 negatives -> M2 projection
#
# This script intentionally avoids the full Cartesian product.
# Each stage fixes the current best setting before testing the next variable.

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

BCE_CONFIG="configs/icd10_to_icd11_matrix_bce.yaml"
BCE_FINETUNE_CONFIG="configs/icd10_to_icd11_matrix_bce_finetune.yaml"
BGE_CONFIG="configs/icd10_to_icd11_matrix_bge_m3.yaml"
PUBMED_CONFIG="configs/icd10_to_icd11_matrix_pubmedbert.yaml"
DIRECTIONAL_CONFIG="configs/icd10_to_icd11_directional.yaml"
DIRECTIONAL_FINETUNE_CONFIG="configs/icd10_to_icd11_directional_finetune.yaml"

BCE_ENTITY_MAX_CONFIG="configs/icd10_to_icd11_matrix_bce_C_entity_max.yaml"
BCE_NAME_ONLY_CONFIG="configs/icd10_to_icd11_matrix_bce_D_name_only.yaml"
BCE_CONTEXT_ONLY_CONFIG="configs/icd10_to_icd11_matrix_bce_D_context_only.yaml"
BCE_PATH_ONLY_CONFIG="configs/icd10_to_icd11_matrix_bce_D_path_only.yaml"

run_prepare_once() {
  # 数据预处理。因此数据切分和清洗只在最开始跑一次。
  "$PYTHON" -m icd_linker.cli prepare --config "$BCE_CONFIG"
}

eval_base() {  
  # local makes config visible only inside this function.
  # "$1" is the first argument passed to eval_base.
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

echo "M4: target view/fusion strategy, using BCE configs currently available"
eval_base "$BCE_NAME_ONLY_CONFIG"
eval_base "$BCE_CONTEXT_ONLY_CONFIG"
eval_base "$BCE_PATH_ONLY_CONFIG"
eval_base "$BCE_ENTITY_MAX_CONFIG"
echo "M4 reference: BCE multi-view best_view was already run in M0."

echo "M1: finetune vs direct retrieval, using current selected BCE baseline config"
finetune_and_eval "$BCE_FINETUNE_CONFIG"

echo "M3: negative strategy"
echo "Current code supports mined top-k hard negatives plus in-batch negatives during training."
echo "Separate in-batch-only / mixed-negative configs need code or config support before running."

echo "M2: directional projection"
eval_base "$DIRECTIONAL_CONFIG"
finetune_and_eval "$DIRECTIONAL_FINETUNE_CONFIG"

echo "Done."
