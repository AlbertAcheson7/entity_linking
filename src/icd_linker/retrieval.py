from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .io_utils import load_lookup
from .models import BCEReranker, EmbeddingAdapter, load_embedding


class Retriever:
    def __init__(self, cfg: Dict[str, Any], variant: str, rerank: bool = False):
        import chromadb

        self.cfg = cfg
        self.variant = variant
        self.model: EmbeddingAdapter = load_embedding(cfg, variant)
        chroma_path = (
            cfg["paths"]["base_chroma_dir"]
            if variant == "base" else cfg["paths"]["finetuned_chroma_dir"]
        )
        client = chromadb.PersistentClient(path=chroma_path)
        collections = cfg["index"]["collections"]
        self.name_collection = client.get_collection(
            f"{collections['target_name']}_{variant}"
        )
        self.context_collection = client.get_collection(
            f"{collections['target_context']}_{variant}"
        )
        self.targets = load_lookup(
            Path(cfg["paths"]["prepared_dir"]) / "target_terms.jsonl"
        )
        self.reranker = (
            BCEReranker(cfg["models"]["reranker"]) if rerank else None
        )

    def _query_view(self, collection, text: str, top_k: int) -> List[str]:
        return self._query_view_batch(collection, [text], top_k)[0]

    def _query_view_batch(
        self, collection, texts: Sequence[str], top_k: int
    ) -> List[List[str]]:
        embeddings = self.model.encode(
            texts, batch_size=self.cfg["retrieval"]["query_batch_size"]
        )
        result = collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return [
            [metadata["term_uid"] for metadata in row]
            for row in result["metadatas"]
        ]

    def _fuse(
        self, name_ids: Sequence[str], context_ids: Sequence[str],
        query_context: str, top_k: int,
    ) -> List[dict]:
        retrieval = self.cfg["retrieval"]
        fused_top = retrieval["fused_top_k"]
        strategy = retrieval.get("fusion_strategy", "rrf")
        if strategy not in {"rrf", "best_view"}:
            raise ValueError(f"unknown fusion_strategy: {strategy}")
        scores = defaultdict(float)
        ranks = defaultdict(dict)
        for view, ids in (("name", name_ids), ("context", context_ids)):
            for rank, uid in enumerate(ids, 1):
                ranks[uid][view] = rank
                if strategy == "rrf":
                    rrf_k = retrieval["rrf_constant"]
                    scores[uid] += 1.0 / (rrf_k + rank)
                else:
                    # Treat name/context as alternative target views. If a UID
                    # appears in both views, keep its best single-view rank
                    # instead of adding evidence from both lists.
                    scores[uid] = max(scores[uid], 1.0 / rank)
        candidates = sorted(scores, key=scores.get, reverse=True)[:fused_top]
        if self.reranker:
            passages = [self.targets[uid]["context_text"] for uid in candidates]
            rerank_scores = self.reranker.scores(
                query_context, passages, retrieval["reranker_batch_size"]
            )
            rerank_lookup = dict(zip(candidates, rerank_scores))
            candidates = [
                uid for uid, _ in sorted(
                    zip(candidates, rerank_scores),
                    key=lambda item: item[1], reverse=True,
                )
            ]
        else:
            rerank_lookup = {}
        return [
            {
                "rank": rank,
                "term_uid": uid,
                "code": self.targets[uid].get("code", ""),
                "name": self.targets[uid]["name_text"],
                "fusion_strategy": strategy,
                "fusion_score": scores[uid],
                "rrf_score": scores[uid] if strategy == "rrf" else None,
                "view_ranks": ranks[uid],
                "rerank_score": rerank_lookup.get(uid),
            }
            for rank, uid in enumerate(candidates[:top_k], 1)
        ]

    def retrieve(
        self, query_name: str, query_context: str,
        top_k: int | None = None,
    ) -> List[dict]:
        return self.retrieve_batch(
            [(query_name, query_context)], top_k=top_k
        )[0]

    def retrieve_batch(
        self, queries: Sequence[Tuple[str, str]],
        top_k: int | None = None,
    ) -> List[List[dict]]:
        if not queries:
            return []
        retrieval = self.cfg["retrieval"]
        per_view = retrieval["per_view_top_k"]
        output_top = top_k or retrieval["output_top_k"]
        name_results = self._query_view_batch(
            self.name_collection, [item[0] for item in queries], per_view
        )
        context_results = self._query_view_batch(
            self.context_collection, [item[1] for item in queries], per_view
        )
        return [
            self._fuse(name_ids, context_ids, query_context, output_top)
            for (query_name, query_context), name_ids, context_ids
            in zip(queries, name_results, context_results)
        ]
