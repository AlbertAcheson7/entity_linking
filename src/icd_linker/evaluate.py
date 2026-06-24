"""在已构建的目标术语向量索引上评估实体链接/术语映射效果。

典型调用：
1. evaluate --config ... --variant base
   仅使用 base embedding 检索，并用 RRF 融合名称视图和上下文视图的排名。
# TODO: RRF融合是啥？
2. evaluate --config ... --variant base --rerank
   先执行完全相同的向量检索与 RRF 融合，再用 reranker 对候选重新排序。
# TODO: 所以我之前在服务器上跑的那一次，结果相当于是一共有四个metric。

输入：
- 配置文件中的索引目录、模型、检索参数和实验输出目录；
- prepared/<split>.jsonl（默认 test.jsonl），每行包含查询的名称/上下文，
  以及一个或多个正确目标 positive_target_uids；
- build-index 已写入 Chroma 的 target-name 和 target-context 两个集合；
- rerank=True 时还会加载配置中的 reranker 模型。

# TODO: 所以说在最后Metric的时候是对他的target U ID进行的Metric比较，那你在模型中呢你模型中你希望他直接给你输入的是U ID还是说输入的是 target name和context name呢？

输出：
- experiments/<variant>/<split>_retrieval_predictions.jsonl，或
  <split>_reranked_predictions.jsonl：逐查询的正确答案和候选排名；
- 同名 *_metrics.json：Recall@K、AllTargetRecall@K、MRR 等汇总指标；
- evaluate() 同时返回该指标 JSON，CLI 会将它打印到终端/日志。

# TODO: 需要在服务器中查看输出文件夹中的信息,是不是模型rerank之后的输出的结果有存储呢？.以及输出的四个metric.json.

# TODO: 添加评价指标ACC@k， recall@k,hit@k.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .io_utils import read_jsonl, write_json, write_jsonl
from .metrics import RankingMetrics
from .retrieval import Retriever


def evaluate(
    cfg: Dict[str, Any], variant: str, rerank: bool,
    split: str = "test",
) -> Dict[str, Any]:
    """评估指定 embedding 版本在某个数据切分上的全目标库检索效果。"""
    prepared = Path(cfg["paths"]["prepared_dir"])
    experiment_dir = Path(cfg["paths"]["experiments_dir"]) / variant
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # variant=base 会加载基础 embedding 和 base Chroma 索引；
    # --rerank 只控制是否加载交叉编码 reranker，不会切换向量索引。
    retriever = Retriever(cfg, variant, rerank=rerank)

    # 配置的 output_top_k=10 主要用于普通 link 推理；评估固定至少保留 50
    # 个候选，才能正确计算 Recall/AllTargetRecall@50 和 MRR。
    max_k = max(50, cfg["retrieval"]["output_top_k"])
    metrics = RankingMetrics((1, 5, 10, 50))
    predictions = []

    # 默认读取 prepare.py 生成的 data/prepared/test.jsonl。
    # 每行是一条源术语查询，可对应一个或多个正确的目标术语 UID。
    rows = list(read_jsonl(prepared / f"{split}.jsonl"))
    batch_size = cfg["retrieval"]["query_batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]

        # Retriever 分别用 query_name_text 查询 target-name 索引、用
        # query_context_text 查询 target-context 索引，然后以 RRF 融合排名。
        # 若 rerank=True，还会用 query_context_text 与候选目标 context_text
        # 逐对打分，并按 rerank_score 从高到低重新排列前 50 个候选。
        results = retriever.retrieve_batch(
            [
                (row["query_name_text"], row["query_context_text"])
                for row in batch
            ],
            top_k=max_k,
        )
        for row, candidates in zip(batch, results):
            ranked = [item["term_uid"] for item in candidates]
            positives = set(row["positive_target_uids"])

            # 指标含义：
            # Recall@K：Top-K 中是否至少命中一个正确目标，再对所有查询取平均；
            #            因此它本质上是 Hit@K/查询级命中率。
            # AllTargetRecall@K：Top-K 覆盖了该查询全部正确目标中的多少比例，
            #                    再对所有查询取平均，适合一对多映射。
            # MRR：第一个正确目标排名的倒数，再对所有查询取平均。
            metrics.add(ranked, positives)

            # 保存完整逐查询结果，便于检查错例、候选分数和两个视图的原始排名。
            predictions.append({
                "source_term_uid": row["source_term_uid"],
                "positive_target_uids": sorted(positives),
                "candidates": candidates,
            })
    suffix = "reranked" if rerank else "retrieval"
    metric_values = metrics.result()
    if rerank:
        # 这两个字段是便于报告展示的别名；数值与 rerank 后的 Recall@1/@5
        # 完全相同，并不是额外计算出的另一套指标。
        metric_values["Hit@1_after_reranking"] = metric_values["Recall@1"]
        metric_values["Hit@5_after_reranking"] = metric_values["Recall@5"]
    output = {
        "variant": variant,
        "split": split,
        "reranked": rerank,
        "embedding_model": (
            cfg["models"]["embedding"]
            if variant == "base" else cfg["paths"]["finetuned_model_dir"]
        ),
        "fusion_strategy": cfg["retrieval"].get("fusion_strategy", "rrf"),
        "reranker_model": cfg["models"]["reranker"] if rerank else None,
        "metrics": metric_values,
    }

    # 两次 base 评估写入不同文件，因此不会互相覆盖：
    # test_retrieval_* 保存纯向量检索结果，test_reranked_* 保存重排结果。
    write_jsonl(experiment_dir / f"{split}_{suffix}_predictions.jsonl", predictions)
    write_json(experiment_dir / f"{split}_{suffix}_metrics.json", output)
    return output


def link_text(
    cfg: Dict[str, Any], variant: str, text: str,
    top_k: int, rerank: bool,
) -> Dict[str, Any]:
    """链接一段临时输入文本；这是 CLI 的 link 命令，不参与数据集评估。"""
    retriever = Retriever(cfg, variant, rerank=rerank)

    # 临时文本没有单独构造名称和上下文，因此同一 text 同时查询两个视图。
    candidates = retriever.retrieve(text, text, top_k=top_k)
    return {
        "query": text,
        "variant": variant,
        "reranked": rerank,
        "candidates": candidates,
    }
