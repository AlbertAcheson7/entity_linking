#!/usr/bin/env bash
set -euo pipefail

# 第0个版本：Chroma base 版本

# 1. 原始 ICD-10 / ICD-11 / map 数据 -> data/prepared/
python -m icd_linker.cli prepare \
  --config configs/icd10_to_icd11_chroma.yaml

# 2. 用 base embedding 编码 target name_text/context_text，写入 Chroma 索引
python -m icd_linker.cli build-index \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant base

# 3. 用 Chroma 检索 test.jsonl，输出到 experiments/chroma/base/
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant base

# 4. 可选：在 Chroma 候选后面加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant base \
  --rerank


# 第0个版本：Chroma finetune 版本

# 1. 用 Chroma base 检索结果挖 hard negatives -> train_with_negatives.jsonl
python -m icd_linker.cli mine-negatives \
  --config configs/icd10_to_icd11_chroma.yaml

# 2. 用 query_context_text 和 target context_text 做对比学习训练
python -m icd_linker.cli train \
  --config configs/icd10_to_icd11_chroma.yaml

# 3. 用 finetuned embedding 编码 target name_text/context_text，写入 Chroma 索引
python -m icd_linker.cli build-index \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant finetuned

# 4. 用 Chroma 检索 test.jsonl，输出到 experiments/chroma/finetuned/
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant finetuned

# 5. 可选：在 finetuned Chroma 候选后面加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_chroma.yaml \
  --variant finetuned \
  --rerank


# 第一个模型：matrix BCE base 版本

# 1. 原始 ICD-10 / ICD-11 / map 数据 -> data/prepared/
python -m icd_linker.cli prepare \
  --config configs/icd10_to_icd11_matrix_bce.yaml

# 2. matrix backend 没有单独的 build-index 命令；
#    evaluate 内部会：
#    - 读取 data/prepared/target_terms.jsonl
#    - 取 target name_text/context_text/path_text
#    - 用 BCE embedding 编码 target 文本
#    - 读取 data/prepared/test.jsonl
#    - 用 BCE embedding 编码 query_name_text
#    - 计算 query_embeddings @ target_embeddings.T
#    - 按相似度从高到低排序，写入 experiments/matrix_bce/base/
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce.yaml \
  --variant base

# 3. 可选：对 matrix BCE 候选加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce.yaml \
  --variant base \
  --rerank


# 第一个模型：matrix BCE base 版本的结构消融/查询变体

# B_context_query：evaluate 内部编码 query_context_text 和 target name_text/context_text/path_text，
# 再用 query_embeddings @ target_embeddings.T 排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce_B_context_query.yaml \
  --variant base

# C_entity_max：evaluate 内部编码 query_name_text 和 target name_text/context_text/path_text，
# 同一 term_uid 的多个 target 文本得分取最大值后排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce_C_entity_max.yaml \
  --variant base

# D_name_only：evaluate 内部编码 query_name_text 和 target name_text，再按点积相似度排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce_D_name_only.yaml \
  --variant base

# D_context_only：evaluate 内部编码 query_name_text 和 target context_text，再按点积相似度排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce_D_context_only.yaml \
  --variant base

# D_path_only：evaluate 内部编码 query_name_text 和 target path_text，再按点积相似度排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce_D_path_only.yaml \
  --variant base


# 第一个模型：matrix BCE finetune 版本

# 1. 用 matrix BCE base 检索结果挖 hard negatives
python -m icd_linker.cli mine-negatives \
  --config configs/icd10_to_icd11_matrix_bce.yaml

# 2. 训练共享编码器：query_context_text 和 target context_text 进同一个 AutoModel
python -m icd_linker.cli train \
  --config configs/icd10_to_icd11_matrix_bce.yaml

# 3. matrix backend 没有单独的 build-index 命令；
#    evaluate 内部会加载 finetuned_model_dir，重新编码 target 和 query，
#    再计算 query_embeddings @ target_embeddings.T 排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce.yaml \
  --variant finetuned

# 4. 可选：对 finetuned matrix BCE 候选加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_matrix_bce.yaml \
  --variant finetuned \
  --rerank


# 第二个模型：directional PubMedBERT base 版本

# 1. 原始 ICD-10 / ICD-11 / map 数据 -> data/prepared/
python -m icd_linker.cli prepare \
  --config configs/icd10_to_icd11_directional.yaml

# 2. base directional 没有 projection 文件；
#    evaluate 内部编码 target context_text 和 query_context_text，
#    query/target 都只走 AutoModel + L2 normalize，
#    再计算 query_embeddings @ target_embeddings.T 排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_directional.yaml \
  --variant base

# 3. 可选：对 directional base 候选加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_directional.yaml \
  --variant base \
  --rerank


# 第二个模型：directional PubMedBERT finetune 版本

# 1. 用 directional base 检索结果挖 hard negatives
python -m icd_linker.cli mine-negatives \
  --config configs/icd10_to_icd11_directional.yaml

# 2. 训练定向双编码器：query 走 source_projection，target 走 target_projection
python -m icd_linker.cli train \
  --config configs/icd10_to_icd11_directional.yaml

# 3. evaluate 加载 finetuned_model_dir 和 directional_projection.pt；
#    target context_text 走 target_projection，query_context_text 走 source_projection，
#    再计算 query_embeddings @ target_embeddings.T 排序
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_directional.yaml \
  --variant finetuned

# 4. 可选：对 finetuned directional 候选加 BCE reranker
python -m icd_linker.cli evaluate \
  --config configs/icd10_to_icd11_directional.yaml \
  --variant finetuned \
  --rerank
