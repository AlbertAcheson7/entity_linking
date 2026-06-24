from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from .io_utils import (
    load_lookup, read_jsonl, seeded_split, write_json, write_jsonl,
)
from .text_views import build_views

"""
TODO: 这个脚本还是有问题，原文件中的同义词description定义上位词全部都没有被编码在context text中.
"""
def _prepared_term(term: dict, lookup: Dict[str, dict], max_chars: int) -> dict:
    """将原始术语转换为后续检索、训练共用的标准结构。"""
    # build_views 会清洗文本，并分别生成：
    # 1. name_text：仅术语名称，适合精确、短文本语义匹配；
    # 2. context_text：名称 + 编码 + 同义词 + 描述 + 定义 + 上位概念，
    #    适合利用更丰富的语义上下文进行召回。
    views = build_views(term, lookup, max_chars)

    # 保留追踪术语身份、版本和来源所需的字段，便于索引和结果回溯。
    metadata = {
        key: term.get(key)
        for key in (
            "term_uid", "terminology", "version", "code", "class_kind",
            "active", "is_map_derived",
        )
    }
    return {**metadata, **views}


def prepare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """预处理源/目标术语和映射关系，并生成训练、验证、测试数据。"""
    # 这个脚本最后得到的所有输出都在 entity_linking/data/prepared
    source_dir = Path(cfg["paths"]["source_dir"])
    prepared_dir = Path(cfg["paths"]["prepared_dir"])
    prepared_dir.mkdir(parents=True, exist_ok=True)
    inputs = cfg["input_files"]
    source_path = source_dir / inputs["source_terms"]
    target_path = source_dir / inputs["target_terms"]
    maps_path = source_dir / inputs["maps"]

    source_lookup = load_lookup(source_path)
    target_lookup = load_lookup(target_path)
    expected_source = cfg["data"]["expected_source_terms"]
    expected_target = cfg["data"]["expected_target_terms"]

    # 用配置中的预期数量检查输入是否完整，尽早发现文件缺失、解析失败或版本错误。
    if len(source_lookup) != expected_source:
        raise ValueError(f"source terms {len(source_lookup)} != {expected_source}")
    if len(target_lookup) != expected_target:
        raise ValueError(f"target terms {len(target_lookup)} != {expected_target}")

    max_chars = cfg["data"]["context_max_characters"]

    # 源术语和目标术语使用相同规则生成双文本视图，保证检索时语义表示一致。
    prepared_sources = {
        uid: _prepared_term(term, source_lookup, max_chars)
        for uid, term in source_lookup.items()
    }
    prepared_targets = {
        uid: _prepared_term(term, target_lookup, max_chars)
        for uid, term in target_lookup.items()
    }
    write_jsonl(
        prepared_dir / "source_terms.jsonl",
        (prepared_sources[uid] for uid in sorted(prepared_sources)),
    )
    write_jsonl(
        prepared_dir / "target_terms.jsonl",
        (prepared_targets[uid] for uid in sorted(prepared_targets)),
    )

    # positives 以 source_uid 为键、目标 UID 集合为值，例如：
    # {"ICD10:A00-A09": {"ICD11:目标1", "ICD11:目标2", ...}}。
    # 因此一对多映射不会被拆成多个训练查询；写入数据集时，同一源术语的
    # 所有正确目标会统一保存到 positive_target_uids 数组中。
    # 使用 set 还能自动去除映射文件中的重复正例对。
    positives = defaultdict(set)
    map_rows = 0
    for record in read_jsonl(maps_path):
        # direction 表示“用哪套术语作为查询、链接到哪套目标术语”。
        # 当前配置为 ICD10_TO_ICD11_MMS，即输入 ICD-10 概念，检索对应的
        # ICD-11 MMS 概念。它描述的是映射方向，不是术语树的上下级方向。
        # 映射文件可能包含其他任务方向，因此这里只保留当前配置指定的方向。
        if record.get("direction") != cfg["data"]["map_direction"]:
            continue
        source_uid = record["source_term_uid"]
        target_uid = record["target_term_uid"]

        # 每条映射的两端必须存在于本次加载的术语版本中，否则无法用于训练或评估。
        if source_uid not in source_lookup or target_uid not in target_lookup:
            raise ValueError(
                f"map endpoint absent: {source_uid} -> {target_uid}"
            )
        positives[source_uid].add(target_uid)
        map_rows += 1
    if not positives:
        raise ValueError("no positive mappings after direction filtering")

    split_cfg = cfg["data"]["split"]

    # 按“源术语 UID”而不是按映射行切分。
    # 这样同一个源术语的所有目标正例只会出现在一个集合中，避免数据泄漏。
    split_ids = seeded_split(
        list(positives), split_cfg["train"], split_cfg["validation"],
        cfg["seed"],
    )

    # 防御性检查：训练、验证、测试集合中的源术语必须两两不重叠。
    # 此阶段的数据集只有查询文本及其全部正例目标，没有硬负例。
    # 后续 mine-negatives 阶段会从基础检索模型返回的高排名错误候选中挖掘
    # hard_negative_uids，并生成 train_with_negatives.jsonl 供训练使用；
    # 验证集和测试集仍只保留正例，用于全目标库检索评估。
    if any(split_ids[a] & split_ids[b] for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test")
    )):
        raise AssertionError("source leakage between splits")

    split_counts = {}
    for split_name, source_ids in split_ids.items():
        rows = []
        for uid in sorted(source_ids):
            # 每行代表一个查询源术语；一个查询可以有多个正确的目标术语。
            rows.append({
                "source_term_uid": uid,
                "query_name_text": prepared_sources[uid]["name_text"],
                "query_context_text": prepared_sources[uid]["context_text"],
                "positive_target_uids": sorted(positives[uid]),
            })
        split_counts[split_name] = write_jsonl(
            prepared_dir / f"{split_name}.jsonl", rows
        )

    # 记录本次预处理的规模、切分结果和关键参数，便于复现及快速核对产物。
    manifest = {
        "source_terms": len(source_lookup),
        "target_terms": len(target_lookup),
        "filtered_map_rows": map_rows,
        "unique_positive_pairs": sum(len(v) for v in positives.values()),
        "mapped_sources": len(positives),
        "splits": split_counts,
        "seed": cfg["seed"],
        "direction": cfg["data"]["map_direction"],
    }
    write_json(prepared_dir / "prepare_manifest.json", manifest)
    return manifest
