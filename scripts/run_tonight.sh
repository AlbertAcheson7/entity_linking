#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/icd10_to_icd11.yaml}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p logs

run_stage() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name" | tee -a "logs/${name}.log"
  "$@" 2>&1 | tee -a "logs/${name}.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "logs/${name}.log"
}

run_stage doctor python -m icd_linker.cli doctor --config "$CONFIG"   
run_stage prepare python -m icd_linker.cli prepare --config "$CONFIG"   
run_stage build_base python -m icd_linker.cli build-index --config "$CONFIG" --variant base
run_stage evaluate_base python -m icd_linker.cli evaluate --config "$CONFIG" --variant base
run_stage rerank_base python -m icd_linker.cli evaluate --config "$CONFIG" --variant base --rerank
run_stage mine_negatives python -m icd_linker.cli mine-negatives --config "$CONFIG"
run_stage train python -m icd_linker.cli train --config "$CONFIG"
run_stage build_finetuned python -m icd_linker.cli build-index --config "$CONFIG" --variant finetuned
run_stage evaluate_finetuned python -m icd_linker.cli evaluate --config "$CONFIG" --variant finetuned
run_stage rerank_finetuned python -m icd_linker.cli evaluate --config "$CONFIG" --variant finetuned --rerank
run_stage compare python -m icd_linker.cli compare --config "$CONFIG"
