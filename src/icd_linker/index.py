"""将预处理后的 ICD 术语编码为向量，并构建可持久化的 Chroma 检索索引。

输入：
1. cfg：项目配置，提供 prepared_dir、Chroma 输出目录、嵌入模型、
   批大小、距离函数和集合名称等参数；
2. variant：base 或 finetuned，决定使用基础模型还是微调模型；
3. prepared_dir/source_terms.jsonl 和 target_terms.jsonl。每条术语至少需要
   term_uid、name_text、context_text，以及用于结果回溯的术语元数据。

输出：
1. Chroma 持久化目录中的 4 个向量集合：
   source-name、source-context、target-name、target-context；
2. 同目录下的 index_manifest.json，记录模型版本和各集合的记录数量；
3. build_index 的返回值：{集合名: 记录数量}。

其中下游实体链接主要查询 target 的两个集合；source 集合可用于源术语侧的
检索、分析或扩展任务。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .io_utils import read_jsonl, write_json
from .models import load_embedding


def _metadata(term: dict, view_type: str) -> dict:
    """提取写入 Chroma 的轻量元数据，供检索后识别并回溯原始术语。"""
    # Chroma 元数据应使用简单标量；这里统一转换类型，避免 None 或复杂对象
    # 导致写入失败。完整的文本内容则通过 collection.add 的 documents 保存。
    values = {
        "term_uid": term["term_uid"],
        "terminology": term["terminology"],
        "version": str(term.get("version", "")),
        "code": str(term.get("code", "")),
        "class_kind": str(term.get("class_kind", "")),
        "active": bool(term.get("active", True)),
        "is_map_derived": bool(term.get("is_map_derived", False)),
        "view_type": view_type,
    }
    return values


def build_index(cfg: Dict[str, Any], variant: str) -> Dict[str, int]:
    """用指定版本的嵌入模型构建源术语和目标术语的双视图向量索引。"""
    import chromadb

    prepared = Path(cfg["paths"]["prepared_dir"])

    # base 和 finetuned 使用独立目录，避免两种模型生成的向量相互覆盖。
    chroma_dir = Path(
        cfg["paths"]["base_chroma_dir"]
        if variant == "base" else cfg["paths"]["finetuned_chroma_dir"]
    )
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    model = load_embedding(cfg, variant) # TODO: 这个方法链接到了models. PY这个脚本，这个脚本里边的方法也需要搞明白。这个地方指定了，如果说变量传入了fine tune的话，是需要微调模型的
    batch_size = cfg["index"]["batch_size"] # TODO: Batch size不应该是根据bc embedding这个模型确定的吗？还可以自己指定吗
    suffix = variant

    # 每套术语分别建立“名称”和“上下文”索引，共得到 2 × 2 = 4 个集合。
    # name_text 较短，强调名称相似度；context_text 还包含编码、同义词、
    # 描述、定义、上位概念等信息，适合更丰富的语义匹配。
    # 三元组含义依次为：输入文件侧、文本视图、配置中的集合名称键。
    specs = [
        ("source", "name", "source_name"),
        ("source", "context", "source_context"),
        ("target", "name", "target_name"),
        ("target", "context", "target_context"),
    ]
    counts = {}
    for side, view, config_key in specs:
        collection_name = f"{cfg['index']['collections'][config_key]}_{suffix}"

        # 构建过程采用“删除后重建”，保证重复执行时不会残留旧模型向量，
        # 也不会因为相同 ID 已存在而产生重复或冲突。
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": cfg["index"]["distance"]},
        )

        # prepare.py 已把原始术语统一整理为 source_terms.jsonl 和
        # target_terms.jsonl，因此本文件不再负责文本清洗或字段拼接。
        terms = list(read_jsonl(prepared / f"{side}_terms.jsonl"))
        text_key = f"{view}_text"

        # 分批编码可以控制显存/内存占用；同一批文本、向量、ID、元数据
        # 必须严格按相同顺序写入，确保每条向量能正确对应原术语。
        for start in range(0, len(terms), batch_size):
            batch = terms[start:start + batch_size]
            texts = [item[text_key] for item in batch]
            embeddings = model.encode(texts, batch_size, show_progress=False)
            collection.add(
                ids=[f"{item['term_uid']}::{view}" for item in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[_metadata(item, view) for item in batch],
            )

        # 写入完成后核对数量，防止批处理或持久化过程中静默漏写数据。
        count = collection.count()
        if count != len(terms):
            raise RuntimeError(
                f"{collection_name}: Chroma count {count} != {len(terms)}"
            )
        counts[collection_name] = count

    # manifest 是本次索引构建的“收据”，便于确认用了哪个模型，以及
    # 每个集合实际写入了多少条术语。
    write_json(
        chroma_dir / "index_manifest.json",
        {
            "variant": variant,
            "embedding_model": (
                cfg["models"]["embedding"]
                if variant == "base" else cfg["paths"]["finetuned_model_dir"]
            ),
            "collections": counts,
        },
    )
    return counts
