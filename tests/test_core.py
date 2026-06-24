from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from icd_linker.io_utils import seeded_split
from icd_linker.metrics import RankingMetrics
from icd_linker.text_views import build_views
from icd_linker.retrieval import Retriever


class CoreTests(unittest.TestCase):
    def test_split_has_no_overlap_and_is_deterministic(self):
        ids = [f"id-{i}" for i in range(100)]
        first = seeded_split(ids, 0.8, 0.1, 42)
        second = seeded_split(ids, 0.8, 0.1, 42)
        self.assertEqual(first, second)
        self.assertFalse(first["train"] & first["validation"])
        self.assertFalse(first["train"] & first["test"])
        self.assertFalse(first["validation"] & first["test"])
        self.assertEqual(sum(map(len, first.values())), 100)

    def test_context_deduplicates_and_resolves_parent(self):
        parent = {
            "term_uid": "p", "name": "Infectious diseases",
        }
        term = {
            "term_uid": "x", "name": "Cholera", "code": "A00",
            "synonyms": ["Cholera", "Asiatic cholera", "Asiatic cholera"],
            "index_terms": ["Epidemic cholera"],
            "description": "Acute intestinal infection",
            "definitions": ["Acute intestinal infection"],
            "parent_ids": ["p"], "path_names": [],
        }
        views = build_views(term, {"p": parent}, 6000)
        self.assertEqual(views["name_text"], "Cholera")
        self.assertIn("Code: A00", views["context_text"])
        self.assertEqual(views["context_text"].count("Asiatic cholera"), 1)
        self.assertIn("Parent concepts: Infectious diseases", views["context_text"])

    def test_context_includes_raw_icd_hierarchy_fields(self):
        term = {
            "term_uid": "x",
            "name": "Cholera due to Vibrio cholerae 01, biovar eltor",
            "code": "A00.1",
            "description": "Cholera due to Vibrio cholerae 01, biovar eltor",
            "raw_fields": {
                "classKind": "category",
                "chapter": "I",
                "depth": "2",
            },
        }
        views = build_views(term, {}, 6000)
        self.assertIn("Class kind: category", views["context_text"])
        self.assertIn("ICD chapter: I", views["context_text"])
        self.assertIn("Hierarchy depth: 2", views["context_text"])

    def test_ranking_metrics_multi_target(self):
        metrics = RankingMetrics((1, 5, 10, 50))
        metrics.add(["wrong", "target-b", "target-a"], {"target-a", "target-b"})
        result = metrics.result()
        self.assertEqual(result["Recall@1"], 0.0)
        self.assertEqual(result["Recall@5"], 1.0)
        self.assertEqual(result["MRR"], 0.5)
        self.assertEqual(result["AllTargetRecall@5"], 1.0)

    def test_rrf_fusion(self):
        retriever = object.__new__(Retriever)
        retriever.cfg = {
            "retrieval": {
                "fusion_strategy": "rrf",
                "fused_top_k": 3, "rrf_constant": 60,
                "reranker_batch_size": 8,
            }
        }
        retriever.targets = {
            "a": {"name_text": "A", "context_text": "A"},
            "b": {"name_text": "B", "context_text": "B"},
            "c": {"name_text": "C", "context_text": "C"},
        }
        retriever.reranker = None
        result = retriever._fuse(["a", "b"], ["b", "c"], "query", 3)
        self.assertEqual(result[0]["term_uid"], "b")
        self.assertEqual({x["term_uid"] for x in result}, {"a", "b", "c"})

    def test_best_view_fusion_does_not_boost_overlap(self):
        retriever = object.__new__(Retriever)
        retriever.cfg = {
            "retrieval": {
                "fusion_strategy": "best_view",
                "fused_top_k": 3, "rrf_constant": 60,
                "reranker_batch_size": 8,
            }
        }
        retriever.targets = {
            "a": {"name_text": "A", "context_text": "A"},
            "b": {"name_text": "B", "context_text": "B"},
            "c": {"name_text": "C", "context_text": "C"},
        }
        retriever.reranker = None
        result = retriever._fuse(["a", "b"], ["b", "c"], "query", 3)
        self.assertEqual(result[0]["term_uid"], "a")
        self.assertEqual(result[1]["term_uid"], "b")
        self.assertIsNone(result[0]["rrf_score"])
        self.assertEqual(result[0]["fusion_strategy"], "best_view")


if __name__ == "__main__":
    unittest.main()
