"""向量模型与重排模型的统一适配层。

本模块不直接读取或写入数据文件。它接收字符串或字符串列表，输出：
    - EmbeddingAdapter.encode：形状为 [文本数, 向量维度] 的 float32 单位向量。
    - BCEReranker.scores：查询与各候选文本的相关性分数列表。

目标：
    屏蔽 BCEmbedding 与 Hugging Face 模型在初始化、编码接口和返回格式上的差异，
    为索引、检索、负例挖掘和评估提供稳定一致的模型调用方式。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


class EmbeddingAdapter:
    """文本向量模型的最小统一接口。"""

    def encode(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def encode_query(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        return self.encode(texts, batch_size, show_progress)

    def encode_target(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        return self.encode(texts, batch_size, show_progress)


class BCEEmbeddingAdapter(EmbeddingAdapter):
    """将 BCEmbedding 的 EmbeddingModel 适配为统一的 encode 接口。"""

    def __init__(self, model_name: str, device: str | None = None):
        from BCEmbedding import EmbeddingModel

        kwargs = {"model_name_or_path": model_name, "use_fp16": True}
        if device is not None:
            kwargs["device"] = device

        # 不同 BCEmbedding 版本的构造参数可能不同；通过签名过滤不支持的参数。
        signature = inspect.signature(EmbeddingModel)
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        self.model = EmbeddingModel(**kwargs)

    def encode(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        """把一批文本编码为 L2 归一化后的 float32 二维数组。"""
        kwargs = {"batch_size": batch_size, "show_progress_bar": show_progress}

        # 同样兼容不同版本 encode() 的参数列表。
        signature = inspect.signature(self.model.encode)
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        values = self.model.encode(list(texts), **kwargs)
        result = np.asarray(values, dtype=np.float32)
        return normalize_and_check(result)


class SentenceTransformerEmbeddingAdapter(EmbeddingAdapter):
    """Use sentence-transformers models through the same encode interface."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = np.asarray(values, dtype=np.float32)
        return normalize_and_check(result)


class DirectionalHFEmbeddingAdapter(EmbeddingAdapter):
    """Hugging Face encoder with optional source/target projection heads."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        query_max_length: int = 256,
        target_max_length: int = 384,
        projection_path: str | Path | None = None,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.functional = F
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.query_max_length = query_max_length
        self.target_max_length = target_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        self.source_projection = None
        self.target_projection = None

        if projection_path is not None and Path(projection_path).exists():
            state = torch.load(projection_path, map_location=self.device)
            hidden_size = int(state["hidden_size"])
            projection_dim = int(state["projection_dim"])
            self.source_projection = nn.Linear(
                hidden_size, projection_dim, bias=False
            ).to(self.device)
            self.target_projection = nn.Linear(
                hidden_size, projection_dim, bias=False
            ).to(self.device)
            self.source_projection.load_state_dict(state["source_projection"])
            self.target_projection.load_state_dict(state["target_projection"])
            self.source_projection.eval()
            self.target_projection.eval()

    def _encode_side(
        self,
        texts: Sequence[str],
        batch_size: int,
        max_length: int,
        projection,
    ) -> np.ndarray:
        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                hidden = self.encoder(
                    **tokens, return_dict=True
                ).last_hidden_state[:, 0]
                if projection is not None:
                    hidden = projection(hidden)
                hidden = self.functional.normalize(hidden, dim=1)
                vectors.append(hidden.cpu().numpy())
        return normalize_and_check(np.concatenate(vectors, axis=0))

    def encode(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        return self.encode_query(texts, batch_size, show_progress)

    def encode_query(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        return self._encode_side(
            texts, batch_size, self.query_max_length, self.source_projection
        )

    def encode_target(
        self, texts: Sequence[str], batch_size: int, show_progress: bool = False
    ) -> np.ndarray:
        return self._encode_side(
            texts, batch_size, self.target_max_length, self.target_projection
        )


def normalize_and_check(values: np.ndarray) -> np.ndarray:
    """校验向量数值，并逐行做 L2 归一化，便于用点积计算余弦相似度。"""
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("embeddings contain NaN/Inf or invalid dimensions")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding detected")
    values = values / norms
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-3):
        raise ValueError("embedding normalization failed")
    return values.astype(np.float32)


def load_embedding(cfg: Dict[str, Any], variant: str) -> EmbeddingAdapter:
    """根据 variant 加载基础模型或本地微调模型。"""
    adapter = cfg["models"].get("embedding_adapter", "bce")
    device = cfg["models"].get("embedding_device")
    if variant == "base":
        model_name = cfg["models"]["embedding"]
    elif variant == "finetuned":
        # BCEmbedding 可直接读取本地 Hugging Face checkpoint；这样基础版与微调版
        # 使用完全相同的池化和归一化逻辑，评估结果才具有可比性。
        model_name = cfg["paths"]["finetuned_model_dir"]
    else:
        raise ValueError(f"unknown variant: {variant}")
    if adapter == "bce":
        return BCEEmbeddingAdapter(model_name, device=device)
    if adapter in {"sentence_transformers", "sentence-transformers"}:
        return SentenceTransformerEmbeddingAdapter(model_name, device=device)
    if adapter in {"directional_hf", "hf_directional"}:
        train_cfg = cfg.get("training", {})
        projection_path = None
        if variant == "finetuned":
            projection_path = Path(cfg["paths"]["finetuned_model_dir"]) / (
                cfg["models"].get("projection_file", "directional_projection.pt")
            )
        return DirectionalHFEmbeddingAdapter(
            model_name,
            device=device,
            query_max_length=train_cfg.get("query_max_length", 256),
            target_max_length=train_cfg.get("target_max_length", 384),
            projection_path=projection_path,
        )
    raise ValueError(f"unknown embedding_adapter: {adapter}")


class BCEReranker:
    """对“查询-候选文本”对进行精细相关性打分的 BCEmbedding 重排器。"""

    def __init__(self, model_name: str):
        from BCEmbedding import RerankerModel

        kwargs = {"model_name_or_path": model_name, "use_fp16": True}

        # 兼容不接受 use_fp16 或任意关键字参数的旧版本。
        signature = inspect.signature(RerankerModel)
        accepts_kwargs = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        self.model = RerankerModel(**kwargs)

    def scores(
        self, query: str, passages: Sequence[str], batch_size: int
    ) -> List[float]:
        """输入一个查询和多个候选文本，按原候选顺序返回相关性分数。"""
        pairs = [[query, passage] for passage in passages]
        if hasattr(self.model, "compute_score"):
            # 新版接口可直接对文本对批量打分。
            method = self.model.compute_score
            signature = inspect.signature(method)
            kwargs = {}
            if "batch_size" in signature.parameters:
                kwargs["batch_size"] = batch_size
            scores = method(pairs, **kwargs)
        else:
            # 旧版 rerank() 返回的是重排后的索引和分数，需要还原为原始候选顺序。
            result = self.model.rerank(query, list(passages))
            ids = result.get("rerank_ids", list(range(len(passages))))
            ranked_scores = result["rerank_scores"]
            score_by_id = dict(zip(ids, ranked_scores))
            scores = [score_by_id[index] for index in range(len(passages))]
        if np.isscalar(scores):
            # 单候选时某些版本返回标量，这里统一为列表。
            scores = [scores]
        return [float(value) for value in scores]
