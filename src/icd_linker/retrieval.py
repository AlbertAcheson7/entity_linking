from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .io_utils import load_lookup, read_jsonl
from .models import BCEReranker, EmbeddingAdapter, load_embedding


def _query_text(query: Any, key: str) -> str:
    if isinstance(query, Mapping):
        return str(query.get(key) or "")
    name, context = query
    if key == "query_name_text":
        return name
    if key == "query_context_text":
        return context
    if key == "query_path_text":
        return context
    raise ValueError(f"query key {key!r} requires dict query rows")


def _target_view_key(view: str) -> str:
    return view if view.endswith("_text") else f"{view}_text"


def _view_name(view_key: str) -> str:
    return view_key[:-5] if view_key.endswith("_text") else view_key


class ChromaRetriever:
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
        embeddings = self.model.encode_query(
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
            self.name_collection,
            [_query_text(item, "query_name_text") for item in queries],
            per_view,
        )
        context_results = self._query_view_batch(
            self.context_collection,
            [_query_text(item, "query_context_text") for item in queries],
            per_view,
        )
        return [
            self._fuse(
                name_ids,
                context_ids,
                _query_text(query, "query_context_text"),
                output_top,
            )
            for query, name_ids, context_ids
            in zip(queries, name_results, context_results)
        ]


class MatrixRetriever:
    """Exhaustive matrix retrieval over expanded target text views.

    With ranking_unit=view_record, each target view is a separate ranked item:
    B1::name, B1::context, B1::path. A hit is counted when any returned record's
    term_uid is a gold target. With ranking_unit=entity or aggregation=max, the
    best view score is collapsed back to one candidate per target entity.
    """

    def __init__(self, cfg: Dict[str, Any], variant: str, rerank: bool = False):
        self.cfg = cfg
        self.variant = variant
        self.model: EmbeddingAdapter = load_embedding(cfg, variant)
        prepared = Path(cfg["paths"]["prepared_dir"])
        self.targets = load_lookup(prepared / "target_terms.jsonl")
        self.target_terms = list(read_jsonl(prepared / "target_terms.jsonl"))
        self.reranker = (
            BCEReranker(cfg["models"]["reranker"]) if rerank else None
        )
        self.target_records = self._expand_targets()
        self.target_embeddings = self._encode_targets()

    def _retrieval_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("retrieval", {})

    def _expand_targets(self) -> List[dict]:
        retrieval = self._retrieval_cfg()
        view_keys = retrieval.get(
            "target_view_keys", ["name_text", "context_text"]
        )
        records = []
        for term in self.target_terms:
            for view in view_keys:
                key = _target_view_key(view)
                text = str(term.get(key) or "").strip()
                if not text:
                    continue
                records.append({
                    "record_id": f"{term['term_uid']}::{_view_name(key)}",
                    "term_uid": term["term_uid"],
                    "view_type": _view_name(key),
                    "text": text,
                })
        if not records:
            raise ValueError("no target view records were built")
        return records

    def _encode_targets(self) -> np.ndarray:
        retrieval = self._retrieval_cfg()
        batch_size = self.cfg["index"].get(
            "batch_size", retrieval.get("query_batch_size", 128)
        )
        texts = [record["text"] for record in self.target_records]
        return self.model.encode_target(texts, batch_size, show_progress=True)

    def _top_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        limit = min(top_k, len(scores))
        if limit <= 0:
            return np.array([], dtype=np.int64)
        indices = np.argpartition(-scores, limit - 1)[:limit]
        return indices[np.argsort(-scores[indices])]

    def _record_candidates(
        self, scores: np.ndarray, top_k: int
    ) -> List[dict]:
        indices = self._top_indices(scores, top_k)
        candidates = []
        for rank, index in enumerate(indices, 1):
            record = self.target_records[int(index)]
            target = self.targets[record["term_uid"]]
            candidates.append({
                "rank": rank,
                "term_uid": record["term_uid"],
                "record_id": record["record_id"],
                "view_type": record["view_type"],
                "code": target.get("code", ""),
                "name": target["name_text"],
                "retrieval_backend": "matrix",
                "ranking_unit": "view_record",
                "score": float(scores[int(index)]),
                "rerank_score": None,
            })
        return candidates

    def _entity_candidates(
        self, scores: np.ndarray, top_k: int
    ) -> List[dict]:
        best: Dict[str, Tuple[float, dict]] = {}
        for score, record in zip(scores, self.target_records):
            current = best.get(record["term_uid"])
            value = float(score)
            if current is None or value > current[0]:
                best[record["term_uid"]] = (value, record)
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        candidates = []
        for rank, (uid, (score, record)) in enumerate(ranked[:top_k], 1):
            target = self.targets[uid]
            candidates.append({
                "rank": rank,
                "term_uid": uid,
                "record_id": record["record_id"],
                "best_view_type": record["view_type"],
                "code": target.get("code", ""),
                "name": target["name_text"],
                "retrieval_backend": "matrix",
                "ranking_unit": "entity",
                "aggregation": "max",
                "score": score,
                "rerank_score": None,
            })
        return candidates

    def _rerank(
        self, query_context: str, candidates: List[dict], top_k: int
    ) -> List[dict]:
        if not self.reranker:
            return candidates[:top_k]
        retrieval = self._retrieval_cfg()
        passages = [
            self.targets[item["term_uid"]]["context_text"]
            for item in candidates
        ]
        scores = self.reranker.scores(
            query_context, passages, retrieval["reranker_batch_size"]
        )
        ranked = [
            {**candidate, "rerank_score": score}
            for candidate, score in sorted(
                zip(candidates, scores),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        for rank, candidate in enumerate(ranked[:top_k], 1):
            candidate["rank"] = rank
        return ranked[:top_k]

    def retrieve(
        self, query_name: str, query_context: str,
        top_k: int | None = None,
    ) -> List[dict]:
        return self.retrieve_batch(
            [(query_name, query_context)], top_k=top_k
        )[0]

    def retrieve_batch(
        self, queries: Sequence[Any],
        top_k: int | None = None,
    ) -> List[List[dict]]:
        if not queries:
            return []
        retrieval = self._retrieval_cfg()
        query_key = retrieval.get("query_text_key", "query_name_text")
        output_top = top_k or retrieval["output_top_k"]
        ranking_unit = retrieval.get("ranking_unit", "view_record")
        aggregation = retrieval.get("aggregation", "none")
        if ranking_unit not in {"view_record", "entity"}:
            raise ValueError(f"unknown ranking_unit: {ranking_unit}")
        if aggregation not in {"none", "max"}:
            raise ValueError(f"unknown aggregation: {aggregation}")

        texts = [_query_text(item, query_key) for item in queries]
        query_embeddings = self.model.encode_query(
            texts,
            batch_size=retrieval["query_batch_size"],
            show_progress=False,
        )
        score_matrix = query_embeddings @ self.target_embeddings.T
        results = []
        for query, scores in zip(queries, score_matrix):
            if ranking_unit == "entity" or aggregation == "max":
                candidates = self._entity_candidates(scores, output_top)
            else:
                candidates = self._record_candidates(scores, output_top)
            results.append(self._rerank(
                _query_text(query, "query_context_text"),
                candidates,
                output_top,
            ))
        return results


class Retriever:
    """Factory wrapper that keeps the old test-facing class shape."""

    _fuse = ChromaRetriever._fuse

    def __new__(
        cls, cfg: Dict[str, Any], variant: str, rerank: bool = False
    ) -> ChromaRetriever | MatrixRetriever:
        backend = cfg.get("retrieval", {}).get("backend", "chroma")
        if backend == "chroma":
            return ChromaRetriever(cfg, variant, rerank=rerank)
        if backend == "matrix":
            return MatrixRetriever(cfg, variant, rerank=rerank)
        raise ValueError(f"unknown retrieval backend: {backend}")
