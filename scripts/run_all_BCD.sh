#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

mkdir -p logs

CONFIG_SPECS=(
  "bce:configs/icd10_to_icd11_matrix_bce.yaml"
  "bge_m3:configs/icd10_to_icd11_matrix_bge_m3.yaml"
  "pubmedbert:configs/icd10_to_icd11_matrix_pubmedbert.yaml"
)

GENERATED_CONFIGS=()
cleanup() {
  if ((${#GENERATED_CONFIGS[@]})); then
    rm -f "${GENERATED_CONFIGS[@]}"
  fi
}
trap cleanup EXIT

run_stage() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name" | tee -a "logs/${name}.log"
  "$@" 2>&1 | tee -a "logs/${name}.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "logs/${name}.log"
}

make_config() {
  local base_config="$1"
  local output_config="$2"
  local experiments_dir="$3"
  local query_text_key="$4"
  local ranking_unit="$5"
  local aggregation="$6"
  local target_views="$7"

  python - "$base_config" "$output_config" "$experiments_dir" \
    "$query_text_key" "$ranking_unit" "$aggregation" "$target_views" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, output_config, experiments_dir = sys.argv[1:4]
query_text_key, ranking_unit, aggregation, target_views = sys.argv[4:8]

with Path(base_config).open(encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["paths"]["experiments_dir"] = experiments_dir
retrieval = cfg.setdefault("retrieval", {})
retrieval["backend"] = "matrix"
retrieval["query_text_key"] = query_text_key
retrieval["ranking_unit"] = ranking_unit
retrieval["aggregation"] = aggregation
retrieval["target_view_keys"] = target_views.split(",")

with Path(output_config).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
PY
}

run_eval() {
  local model_slug="$1"
  local base_config="$2"
  local experiment_slug="$3"
  local query_text_key="$4"
  local ranking_unit="$5"
  local aggregation="$6"
  local target_views="$7"

  local generated_config="configs/.run_BCD_${model_slug}_${experiment_slug}.yaml"
  local experiments_dir="experiments/matrix_${model_slug}/${experiment_slug}"
  make_config "$base_config" "$generated_config" "$experiments_dir" \
    "$query_text_key" "$ranking_unit" "$aggregation" "$target_views"
  GENERATED_CONFIGS+=("$generated_config")

  run_stage "evaluate_${experiment_slug}_${model_slug}" \
    python -m icd_linker.cli evaluate \
      --config "$generated_config" \
      --variant base
}

if [[ "${SKIP_PREPARE:-0}" != "1" ]]; then
  run_stage prepare_BCD python -m icd_linker.cli prepare \
    --config "configs/icd10_to_icd11.yaml"
fi

for spec in "${CONFIG_SPECS[@]}"; do
  model_slug="${spec%%:*}"
  base_config="${spec#*:}"

  # B: use source context as the query; target still expands name/context/path.
  run_eval "$model_slug" "$base_config" "B_context_query" \
    "query_context_text" "view_record" "none" \
    "name_text,context_text,path_text"

  # C: aggregate expanded target views back to entity scores with max pooling.
  run_eval "$model_slug" "$base_config" "C_entity_max" \
    "query_name_text" "entity" "max" \
    "name_text,context_text,path_text"

  # D: single-view ablations on the target side.
  run_eval "$model_slug" "$base_config" "D_name_only" \
    "query_name_text" "view_record" "none" \
    "name_text"
  run_eval "$model_slug" "$base_config" "D_context_only" \
    "query_name_text" "view_record" "none" \
    "context_text"
  run_eval "$model_slug" "$base_config" "D_path_only" \
    "query_name_text" "view_record" "none" \
    "path_text"
done
