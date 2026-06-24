"""使用基础 BCE 向量检索器为训练集挖掘困难负例。

输入：
    1. cfg：由 cli.py 调用 load_config(--config) 读取 YAML 后传入；
    2. prepared_dir/train.jsonl。每行至少包含 source_term_uid、
       query_name_text、query_context_text 和 positive_target_uids；
    3. base_chroma_dir 中预先构建好的 target-name_base 和
       target-context_base 集合。也就是说，运行本步骤前必须先执行：
       build-index --variant base。

检索过程：
    1. Retriever(cfg, "base", rerank=False) 加载配置中的未微调 BCE
       embedding，并打开由同一基础模型构建的 Chroma 索引；
    2. 名称视图和上下文视图分别检索 retrieval.per_view_top_k 个候选；
    3. Retriever 使用 RRF 融合两个候选列表，并保留
       retrieval.fused_top_k 个候选；
    4. 排除该查询的全部真实目标 positive_target_uids；
    5. 从剩余候选中按融合排名截取
       training.hard_negatives_per_query 个困难负例。

这里没有使用 BCE reranker，也没有使用微调后的 embedding。负例数量并不等于
底层向量检索的候选数量：通常会先多检索一些候选，再过滤正例并截取所需负例。

输出：
    1. train_with_negatives.jsonl：在原训练记录上增加 hard_negative_uids。
    2. negative_mining_manifest.json：记录样本数、每条查询的负例数等元数据。

目标：
    使用基础检索器找出“看起来相关、但并非正确映射”的候选概念，使后续对比学习
    能够学习区分语义接近的 ICD 概念，而不只是区分随机且容易的负例。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .io_utils import read_jsonl, write_json, write_jsonl
from .retrieval import Retriever


def mine_negatives(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """用基础索引检索并保存困难负例，返回负例挖掘清单。"""
    prepared = Path(cfg["paths"]["prepared_dir"])

    # cfg 来自 cli.py 中的 load_config(args.config)。variant="base" 会加载
    # cfg["models"]["embedding"] 指定的未微调 BCE embedding，并连接
    # cfg["paths"]["base_chroma_dir"] 中已经构建好的基础索引。
    # rerank=False 表示不加载、也不使用 BCE reranker。
    retriever = Retriever(cfg, "base", rerank=False)

    # 最终为每条查询保存多少个困难负例。当前示例配置为 5。
    requested = cfg["training"]["hard_negatives_per_query"]
    rows = []
    train_rows = list(read_jsonl(prepared / "train.jsonl"))

    # 这是一次送入向量模型的查询条数，只影响编码速度和显存/内存占用，
    # 不决定每条查询最终得到多少个候选或负例。
    batch_size = cfg["retrieval"]["query_batch_size"] # 128

    # 外层再次按 query_batch_size 切分训练记录，避免一次把全部查询交给检索器。
    for start in range(0, len(train_rows), batch_size):
        batch = train_rows[start:start + batch_size]

        # top_k=fused_top_k 要求 Retriever 返回融合后的较大候选池。
        # Retriever 内部仍会先让名称、上下文两个视图分别检索 per_view_top_k，
        # 再使用 RRF 融合。当前配置中二者都是 50。
        results = retriever.retrieve_batch(
            [
                (row["query_name_text"], row["query_context_text"])
                for row in batch
            ],
            top_k=cfg["retrieval"]["fused_top_k"],
        )
        for row, candidates in zip(batch, results):
            positives = set(row["positive_target_uids"])

            # candidates 已按双视图融合分数从高到低排列。排名靠前但不属于任何
            # 真实目标的概念，就是与查询语义接近、较容易混淆的困难负例。
            # 先过滤全部正例，再截取 requested 个；并非直接只检索 requested 个。
            negatives = [
                item["term_uid"] for item in candidates
                if item["term_uid"] not in positives
            ][:requested]
            if len(negatives) < requested:
                # 不静默接受负例不足，否则不同查询的训练难度和 batch 结构会不一致。
                raise RuntimeError(
                    f"not enough negatives for {row['source_term_uid']}"
                )
            rows.append({**row, "hard_negative_uids": negatives})

    count = write_jsonl(prepared / "train_with_negatives.jsonl", rows)

    # 安全检查：训练查询与测试查询的 source_term_uid 不得重叠。
    train_ids = {row["source_term_uid"] for row in rows}
    test_ids = {
        row["source_term_uid"] for row in read_jsonl(prepared / "test.jsonl")
    }
    if train_ids & test_ids:
        raise AssertionError("test leakage into hard-negative mining")
    manifest = {
        "records": count,
        "negatives_per_query": requested,
        "variant": "base",
        "test_sources_used": 0,
    }
    # manifest 便于追踪本次生成数据的规模和关键配置。
    write_json(prepared / "negative_mining_manifest.json", manifest)
    return manifest
