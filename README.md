# 20260620 ICD-10 → ICD-11 MMS 实体链接（Entity Linking）

基于 BCE 稠密检索（Dense Retrieval）、ChromaDB、BCE 重排序（Reranking）以及对比学习微调（Contrastive Fine-tuning）的两阶段实体链接系统。

## 数据

项目使用 `medical_alignment_pipeline` 生成的标准化数据文件：

```text
data/source/
├── who_icd10.jsonl
├── icd11_mms.jsonl
├── icd10_icd11.jsonl
├── validation.json
├── output_verification.json
└── manifest.json
```

仅使用 `direction=ICD10_TO_ICD11_MMS` 的映射记录。

数据按照 ICD-10 的 `source_term_uid` 分组，并使用随机种子 `42` 按照 `80% / 10% / 10%` 划分为训练集、验证集和测试集。

## 服务器环境配置

```bash
cd entity_linking
conda env create -f environment.yml
conda activate icd-linking
pip install -e .
export HF_ENDPOINT=https://hf-mirror.com
```

检查 GPU、依赖项、源数据文件以及磁盘空间：

```bash
python -m icd_linker.cli doctor --config configs/icd10_to_icd11.yaml
```

将准备好的 Mac 数据传输压缩包复制到服务器，校验校验和并解压到 `data/source`：

```bash
rsync -P icd_linking_transfer.tar.zst* user@server:/path/to/work/
cd /path/to/work
sha256sum -c icd_linking_transfer.tar.zst.sha256
tar --use-compress-program=unzstd -xf /path/to/icd_linking_transfer.tar.zst
(cd entity_linking && sha256sum -c SHA256SUMS)
```

## 运行今晚的实验

推荐将各阶段独立执行，便于失败后恢复：

```bash
python -m icd_linker.cli prepare --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli build-index --config configs/icd10_to_icd11.yaml --variant base
python -m icd_linker.cli evaluate --config configs/icd10_to_icd11.yaml --variant base --rerank
python -m icd_linker.cli mine-negatives --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli train --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli build-index --config configs/icd10_to_icd11.yaml --variant finetuned
python -m icd_linker.cli evaluate --config configs/icd10_to_icd11.yaml --variant finetuned --rerank
python -m icd_linker.cli compare --config configs/icd10_to_icd11.yaml
```

或者直接执行完整实验流程：

```bash
bash scripts/run_tonight.sh
```

训练配置采用：

- 微批次（Micro-batch）：8 个查询（queries）
- 每个查询使用：5 个困难负样本（hard negatives）
- 梯度累积（Gradient Accumulation）：4 步

这样可以获得等效于 **32 个查询的参数更新批次大小**，同时避免在 24GB GPU 上发生显存溢出（OOM）。

## 查询单个 ICD-10 实体

```bash
python -m icd_linker.cli link \
  --config configs/icd10_to_icd11.yaml \
  --variant finetuned \
  --text "Cholera due to Vibrio cholerae O1, biovar eltor" \
  --top-k 10 \
  --rerank
```

## 输出目录

```text
data/prepared/       标准化视图以及训练/验证/测试数据
chroma/base/         基线模型索引（不可变）
chroma/finetuned/    微调模型索引
models/finetuned/    最优对比学习模型检查点
experiments/base/    基线实验指标与预测结果
experiments/finetuned/
experiments/comparison.json
logs/
```

完整权威的 Term 数据不会存储在 Chroma 中。

Chroma 中仅保留精简元数据（Metadata）。

所有检索得到的 `term_uid` 都可以通过以下文件解析获取完整信息：

```text
data/prepared/target_terms.jsonl
```

## 内存占用估算

对于全部 **708,101** 个术语：

- 单个 768 维 float32 向量嵌入约占用 **2.03 GiB**
- Name View 与 Context View 双视图嵌入约占用 **4.05 GiB**（不包含索引开销）

因此，双视图 Chroma 索引建议预留：

**12–30 GB RAM**

本次首个实验仅包含 **48,800 个术语**，预计内存占用将远低于 **10 GB RAM**。

# ICD-10 → ICD-11 MMS Entity Linking

Two-stage entity linking with BCE dense retrieval, ChromaDB, BCE reranking,
and contrastive fine-tuning.

## Data

The project consumes the normalized files produced by
`medical_alignment_pipeline`:

```text
data/source/
├── who_icd10.jsonl
├── icd11_mms.jsonl
├── icd10_icd11.jsonl
├── validation.json
├── output_verification.json
└── manifest.json
```

Only mappings with `direction=ICD10_TO_ICD11_MMS` are used. Sources are split
80/10/10 with seed 42, grouped by ICD-10 `source_term_uid`.

## Server setup

```bash
cd entity_linking
conda env create -f environment.yml
conda activate icd-linking
pip install -e .
export HF_ENDPOINT=https://hf-mirror.com
```

Preflight the GPU, dependencies, source files, and disk space:

```bash
python -m icd_linker.cli doctor --config configs/icd10_to_icd11.yaml
```

Copy the prepared Mac transfer archive to the server, verify its checksum, and
extract it into `data/source`:

```bash
rsync -P icd_linking_transfer.tar.zst* user@server:/path/to/work/
cd /path/to/work
sha256sum -c icd_linking_transfer.tar.zst.sha256
tar --use-compress-program=unzstd -xf /path/to/icd_linking_transfer.tar.zst
(cd entity_linking && sha256sum -c SHA256SUMS)
```

## Run tonight's experiment

Run stages independently (recommended for recovery):

```bash
python -m icd_linker.cli prepare --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli build-index --config configs/icd10_to_icd11.yaml --variant base
python -m icd_linker.cli evaluate --config configs/icd10_to_icd11.yaml --variant base --rerank
python -m icd_linker.cli mine-negatives --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli train --config configs/icd10_to_icd11.yaml
python -m icd_linker.cli build-index --config configs/icd10_to_icd11.yaml --variant finetuned
python -m icd_linker.cli evaluate --config configs/icd10_to_icd11.yaml --variant finetuned --rerank
python -m icd_linker.cli compare --config configs/icd10_to_icd11.yaml
```

Or execute the same sequence:

```bash
bash scripts/run_tonight.sh
```

The training configuration uses a micro-batch of 8 queries, five hard
negatives per query, and four gradient-accumulation steps. This gives an
effective 32-query update while avoiding an out-of-memory failure on a
24 GB GPU.

## Query one ICD-10 entity

```bash
python -m icd_linker.cli link \
  --config configs/icd10_to_icd11.yaml \
  --variant finetuned \
  --text "Cholera due to Vibrio cholerae O1, biovar eltor" \
  --top-k 10 \
  --rerank
```

## Outputs

```text
data/prepared/       normalized views and train/validation/test files
chroma/base/         immutable baseline collections
chroma/finetuned/    fine-tuned collections
models/finetuned/    best contrastive checkpoint
experiments/base/    baseline metrics and predictions
experiments/finetuned/
experiments/comparison.json
logs/
```

The authoritative full Term data remains outside Chroma. Chroma metadata is
deliberately small. Every retrieved `term_uid` can be resolved through
`data/prepared/target_terms.jsonl`.

## Memory estimate

For all 708,101 terms, one 768-dimensional float32 embedding uses about
2.03 GiB; name/context views use about 4.05 GiB before index overhead.
Allow 12–30 GB RAM for dual-view Chroma indexes. The first experiment contains
48,800 terms and is expected to use far less than 10 GB RAM.
