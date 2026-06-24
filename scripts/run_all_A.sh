#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p logs

CONFIGS=(
  "configs/icd10_to_icd11_matrix_bce.yaml"
  "configs/icd10_to_icd11_matrix_bge_m3.yaml"
  "configs/icd10_to_icd11_matrix_pubmedbert.yaml"
)

run_stage() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name" | tee -a "logs/${name}.log"
  "$@" 2>&1 | tee -a "logs/${name}.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "logs/${name}.log"
}

run_stage prepare_A python -m icd_linker.cli prepare \
  --config "configs/icd10_to_icd11.yaml"

for config in "${CONFIGS[@]}"; do
  model_name="$(basename "$config" .yaml)"
  run_stage "evaluate_A_${model_name}" python -m icd_linker.cli evaluate \
    --config "$config" \
    --variant base
done
